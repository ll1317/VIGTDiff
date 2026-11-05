
import os
import torch.nn as nn
import torch as th
from step_sample import create_named_schedule_sampler
import numpy as np
import math
import torch
import torch.nn.functional as F
from Modules import *


class CoDiffu(nn.Module):
    def __init__(self, args):
        super(CoDiffu, self).__init__()
        self.hidden_size = args.hidden_size
        self.schedule_sampler_name = args.schedule_sampler_name
        self.diffusion_steps = args.diffusion_steps

        self.noise_schedule = args.noise_schedule
        betas = self.get_betas(self.noise_schedule, self.diffusion_steps)
        betas = np.array(betas, dtype=np.float64)
        self.betas = betas
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()
        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])

        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        self.posterior_mean_coef1 = (betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod))
        self.posterior_mean_coef2 = ((1.0 - self.alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - self.alphas_cumprod))
        self.posterior_variance = (betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod))

        self.num_timesteps = int(self.betas.shape[0])
        self.schedule_sampler = create_named_schedule_sampler(self.schedule_sampler_name, self.num_timesteps)

        self.rescale_timesteps = args.rescale_timesteps
        self.time_embed = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size * 4),
            SiLU(),
            nn.Linear(self.hidden_size * 4, self.hidden_size),
        )
        self.att = Transformer_rep(args)
        self.lambda_history = args.lambda_history
        self.lambda_intent = args.lambda_intent
        self.dropout = nn.Dropout(args.dropout)
        self.norm_diffu_rep = LayerNorm(self.hidden_size)
        self.item_num = args.item_num + 1
        self.item_embeddings = nn.Embedding(self.item_num, self.hidden_size)
        self.embed_dropout = nn.Dropout(args.emb_dropout)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.loss_ce = nn.CrossEntropyLoss()

        # =====Global
        self.n_clusters = int(args.num_cluster)
        self.n_iters = args.num_iter
        self.mlp = nn.Sequential(nn.Linear(self.hidden_size * args.max_len, self.n_clusters))

        # ===== short
        self.short_window = int(args.short_window)
        self.fusion_alpha = float(args.fusion_alpha)
        self.short_noise_scale = 0.09
        self.short_gate_scale = 0.70
        self.div_top_m = 25
        # short no longer uses prototype quantization; keep guidance token-level
        self.anchor_gate = nn.Sequential(
            nn.Linear(2 * self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 1)
        )
        self.short_token_gate = nn.Sequential(
            nn.Linear(2 * self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 1)
        )
        self.short_core_fuse = nn.Sequential(
            nn.Linear(2 * self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size)
        )
        self.short_noise_fuse = nn.Sequential(
            nn.Linear(2 * self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size)
        )
        self.short_norm = LayerNorm(self.hidden_size, eps=1e-12)
        self.short_clean_norm = LayerNorm(self.hidden_size, eps=1e-12)
        nn.init.constant_(self.short_token_gate[-1].bias, -1.0)
        nn.init.constant_(self.anchor_gate[-1].bias, 0.5)

    def timestep_embedding(self, timesteps, dim, max_period=10000):
        half = dim // 2
        freqs = th.exp(
            -math.log(max_period) * th.arange(start=0, end=half, dtype=th.float32) / half
        ).to(device=timesteps.device)
        args = timesteps[:, None].float() * freqs[None]
        embedding = th.cat([th.cos(args), th.sin(args)], dim=-1)
        if dim % 2:
            embedding = th.cat([embedding, th.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def get_betas(self, noise_schedule, diffusion_steps):
        betas = get_named_beta_schedule(noise_schedule, diffusion_steps)
        return betas

    def q_sample(self, x_start, t, noise=None, mask=None):
        if noise is None:
            noise = th.randn_like(x_start)

        assert noise.shape == x_start.shape
        x_t = (
            extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

        if mask is None:
            return x_t
        else:
            mask = th.broadcast_to(mask.unsqueeze(dim=-1), x_start.shape)
            return th.where(mask == 0, x_start, x_t)

    def _scale_timesteps(self, t):
        if self.rescale_timesteps:
            return t.float() * (1000.0 / self.num_timesteps)
        return t

    def q_posterior_mean_variance(self, x_start, x_t, t):
        assert x_start.shape == x_t.shape
        posterior_mean = (
            extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        assert (posterior_mean.shape[0] == x_start.shape[0])
        return posterior_mean

    def p_mean_variance(self, rep_item, x_t, t, mask_seq, guidance):
        model_output = self.denoise(rep_item, x_t, self._scale_timesteps(t), mask_seq, guidance)
        x_0 = model_output  # predicted clean state; same semantic space as item embeddings
        model_log_variance = extract_into_tensor(self.posterior_variance, t, x_t.shape)
        model_mean = self.q_posterior_mean_variance(x_start=x_0, x_t=x_t, t=t)
        return model_mean, model_log_variance, x_0

    def p_sample(self, item_rep, noise_x_t, t, mask_seq, guidance):
        """Sample one reverse-diffusion step without inference-time drift."""
        model_mean, model_log_variance, _ = self.p_mean_variance(
            item_rep, noise_x_t, t, mask_seq, guidance
        )

        noise = th.randn_like(noise_x_t)
        nonzero_mask = (
            (t != 0)
            .float()
            .view(-1, *([1] * (len(noise_x_t.shape) - 1)))
        )
        sample_xt = (
            model_mean
            + nonzero_mask * th.exp(0.5 * model_log_variance) * noise
        )
        return sample_xt

    def reverse_p_sample(self, item_rep, noise_x_t, mask_seq, guidance):
        device = item_rep.device
        indices = list(range(self.num_timesteps))[::-1]
        for i in indices:
            t = th.tensor([i] * item_rep.shape[0], device=device)
            with th.no_grad():
                noise_x_t = self.p_sample(item_rep, noise_x_t, t, mask_seq, guidance)
        return noise_x_t

    def denoise(self, item_rep, x_t, t, mask_seq, guidance):
        emb_t = self.time_embed(self.timestep_embedding(t, self.hidden_size))
        x_t = x_t + emb_t

        # guidance should be [B, L, D]
        if guidance.shape[1] < item_rep.shape[1]:
            pad_len = item_rep.shape[1] - guidance.shape[1]
            pad = torch.zeros(
                guidance.shape[0], pad_len, guidance.shape[2],
                device=guidance.device, dtype=guidance.dtype
            )
            guidance = torch.cat([pad, guidance], dim=1)

        guidance = guidance * mask_seq.unsqueeze(-1)

        res = self.att(
            item_rep * self.lambda_history
            + 0.001 * x_t.unsqueeze(1)
            + guidance * self.lambda_intent,
            mask_seq
        )
        res = self.norm_diffu_rep(self.dropout(res))
        out = res[:, -1, :]
        return out

    def diffu(self, item_rep, item_tag, mask_seq, guidance):
        noise = th.randn_like(item_tag)
        t, weights = self.schedule_sampler.sample(item_rep.shape[0], item_tag.device)
        x_t = self.q_sample(item_tag, t, noise=noise)
        x_0 = self.denoise(item_rep, x_t, self._scale_timesteps(t), mask_seq, guidance)
        return x_0

    def loss_diffu_ce(self, rep_diffu, labels):
        scores = torch.matmul(rep_diffu, self.item_embeddings.weight.t())
        return self.loss_ce(scores, labels.squeeze(-1))

    def loss_posterior_dispersion(self, rep_diffu, labels, main_intent):
        labels = labels.view(-1).long()

        if main_intent.dim() == 3:
            main_intent = main_intent[:, -1, :]
        if main_intent.dim() != 2:
            raise ValueError(
                'main_intent must have shape [B, D] or [B, L, D], '
                f'but got {tuple(main_intent.shape)}'
            )
        if main_intent.size(0) != rep_diffu.size(0):
            raise ValueError(
                'main_intent and rep_diffu must have the same batch size, '
                f'but got {main_intent.size(0)} and {rep_diffu.size(0)}'
            )
        item_weight = self.item_embeddings.weight.detach()
        item_num = item_weight.size(0)

        if item_num <= 2:
            return rep_diffu.new_zeros(())
        scores = torch.matmul(rep_diffu, item_weight.t()).clone()
        scores[:, 0] = -1e9
        scores.scatter_(1, labels.unsqueeze(1), -1e9)

        top_m = min(int(self.div_top_m), item_num - 2)
        if top_m <= 1:
            return rep_diffu.new_zeros(())

        top_scores, top_items = torch.topk(scores, k=top_m, dim=-1)
        prob = torch.softmax(top_scores, dim=-1)
        top_emb = item_weight[top_items]  # [B, M, D]

        posterior_center = torch.sum(
            prob.unsqueeze(-1) * top_emb,
            dim=1,
            keepdim=True
        )
        candidate_offset = top_emb - posterior_center
        main_dir = F.normalize(
            main_intent.detach(),
            dim=-1,
            eps=1e-8
        ).unsqueeze(1)  # [B, 1, D]

        parallel_component = (
            (candidate_offset * main_dir).sum(dim=-1, keepdim=True)
            * main_dir
        )
        tangent_component = candidate_offset - parallel_component

        tangent_squared_distance = tangent_component.pow(2).sum(dim=-1)
        tangent_dispersion = (
            prob * tangent_squared_distance
        ).sum(dim=-1)
        tangent_dispersion = tangent_dispersion / float(self.hidden_size)


        return -tangent_dispersion.mean()

    def diffu_rep_pre(self, rep_diffu):
        scores = torch.matmul(rep_diffu, self.item_embeddings.weight.t())
        return scores

    # =====Global
    def intent_cluster(self, input, n_clusters):
        X = input.view(input.shape[0], -1)
        centers = X[torch.randperm(X.size(0))[:n_clusters]].to(input.device)
        labels = self.mlp(X)
        labels = F.gumbel_softmax(labels, tau=0.1, hard=True)

        for i in range(self.n_clusters):
            if torch.sum(labels[:, i]) == 0:
                centers[i] = X[torch.randint(0, X.size(0), (1,))]
            else:
                centers[i] = torch.mean(X[labels[:, i].bool()], dim=0)
        return centers.view(n_clusters, input.shape[1], input.shape[-1]), torch.argmax(labels, dim=1)

    def _single_scale_short_intent(self, input, mask_seq, K):
        B, L, D = input.shape
        K = min(int(K), L)

        short_input = input[:, -K:, :]
        short_mask = mask_seq[:, -K:].float()

        # 1) robust anchor: last item + recent summary
        anchor_k = min(3, K)
        recent_summary = short_input[:, -anchor_k:, :].mean(dim=1, keepdim=True)  # [B,1,D]
        last_item = short_input[:, -1:, :]                                         # [B,1,D]
        anchor_conf = torch.sigmoid(
            self.anchor_gate(torch.cat([last_item, recent_summary], dim=-1))
        )                                                                          # [B,1,1]
        anchor = anchor_conf * last_item + (1.0 - anchor_conf) * recent_summary    # [B,1,D]
        anchor_expand = anchor.expand(-1, K, -1)                                   # [B,K,D]

        # 2) token-level core/noise split
        token_gate_input = torch.cat([short_input, anchor_expand], dim=-1)          # [B,K,2D]
        token_gate = self.short_gate_scale * torch.sigmoid(self.short_token_gate(token_gate_input))
        token_gate = token_gate * short_mask.unsqueeze(-1)

        short_core_tokens = token_gate * short_input
        short_noise_tokens = (1.0 - token_gate) * short_input * short_mask.unsqueeze(-1)

        # 3) aggregate core / noise with anchor-guided attention
        core_logits = (short_core_tokens @ anchor.transpose(-1, -2)).squeeze(-1) / math.sqrt(D)
        core_logits = core_logits.masked_fill(short_mask == 0, -1e4)
        core_attn = torch.softmax(core_logits, dim=1) * short_mask
        core_attn = core_attn / (core_attn.sum(dim=1, keepdim=True) + 1e-8)
        short_core = torch.sum(short_core_tokens * core_attn.unsqueeze(-1), dim=1)  # [B,D]

        noise_logits = (short_noise_tokens @ anchor.transpose(-1, -2)).squeeze(-1) / math.sqrt(D)
        noise_logits = noise_logits.masked_fill(short_mask == 0, -1e4)
        noise_attn = torch.softmax(noise_logits, dim=1) * short_mask
        noise_attn = noise_attn / (noise_attn.sum(dim=1, keepdim=True) + 1e-8)
        short_noise = torch.sum(short_noise_tokens * noise_attn.unsqueeze(-1), dim=1)  # [B,D]

        # 4) clean short reconstruction
        anchor_vec = anchor.squeeze(1)
        short_core = self.short_core_fuse(torch.cat([short_core, anchor_vec], dim=-1))
        short_core = self.short_norm(short_core)

        short_noise = self.short_noise_fuse(torch.cat([short_noise, anchor_vec], dim=-1))

        # ===== local differential volatility-aware noise suppression =====
        # Only use adjacent changes inside the current short window.
        # Higher volatility means the recent behaviors fluctuate more, so subtract stronger noise.
        if K > 1:
            delta = short_input[:, 1:, :] - short_input[:, :-1, :]                 # [B,K-1,D]
            delta_mask = short_mask[:, 1:] * short_mask[:, :-1]                   # [B,K-1]
            delta_norm = delta.norm(dim=-1) * delta_mask                          # [B,K-1]
            valid_delta = delta_mask.sum(dim=1, keepdim=True).clamp(min=1.0)      # [B,1]
            local_volatility = delta_norm.sum(dim=1, keepdim=True) / valid_delta  # [B,1]
            local_volatility = torch.sigmoid(local_volatility).clamp(0.0, 1.0)    # [B,1]
        else:
            local_volatility = torch.zeros(B, 1, device=input.device, dtype=input.dtype)

        adaptive_noise_scale = self.short_noise_scale * local_volatility           # [B,1]
        short_clean = short_core - adaptive_noise_scale * torch.tanh(short_noise)  # [B,D]
        short_clean = self.short_clean_norm(short_clean)                           # [B,D]

        # 5) token-level residual short guidance
        short_clean_rep = short_clean.unsqueeze(1).repeat(1, K, 1)                 # [B,K,D]
        batch_intent = short_input + token_gate * (short_clean_rep - short_input)  # [B,K,D]
        batch_intent = batch_intent * short_mask.unsqueeze(-1)

        scale_score = 1.0 - local_volatility                                       # [B,1]
        gate_mean = token_gate.mean()

        return batch_intent, scale_score, gate_mean

    def short_intent_cluster(self, input, n_clusters, mask_seq):
        B, L, D = input.shape

        raw_scales = [2, self.short_window, self.short_window + 2]
        scale_list = []
        for scale in raw_scales:
            scale = max(1, min(int(scale), L))
            if scale not in scale_list:
                scale_list.append(scale)

        max_k = max(scale_list)
        scale_intents = []
        scale_scores = []
        gate_means = []

        for scale in scale_list:
            scale_intent, scale_score, gate_mean = self._single_scale_short_intent(
                input, mask_seq, scale
            )
            if scale_intent.shape[1] < max_k:
                pad_len = max_k - scale_intent.shape[1]
                pad = torch.zeros(
                    B, pad_len, D,
                    device=scale_intent.device,
                    dtype=scale_intent.dtype
                )
                scale_intent = torch.cat([pad, scale_intent], dim=1)

            scale_intents.append(scale_intent)
            scale_scores.append(scale_score)
            gate_means.append(gate_mean)

        scale_stack = torch.stack(scale_intents, dim=1)              # [B,S,max_k,D]
        score_stack = torch.cat(scale_scores, dim=1)                 # [B,S]
        scale_weights = torch.softmax(score_stack, dim=1)            # [B,S]
        batch_intent = (scale_stack * scale_weights[:, :, None, None]).sum(dim=1)  # [B,max_k,D]

        if self.training and torch.rand(1).item() < 0.01:
            print("short gate mean:", torch.stack(gate_means).mean().item())

        return None, batch_intent

    def forward(self, sequence, tag, train_flag):
        item_embeddings = self.item_embeddings(sequence)
        item_embeddings = self.embed_dropout(item_embeddings)
        item_embeddings = self.LayerNorm(item_embeddings)

        self.centroids, self.labels = self.intent_cluster(item_embeddings, self.n_clusters)
        long_intent = self.centroids[self.labels]     # [B,L,D]

        mask_seq = (sequence > 0).float()

        short_centers, short_intent = self.short_intent_cluster(
            item_embeddings.detach(), self.n_clusters, mask_seq
        )

        if short_intent.shape[1] < item_embeddings.shape[1]:
            pad_len = item_embeddings.shape[1] - short_intent.shape[1]
            pad = torch.zeros(
                short_intent.shape[0], pad_len, short_intent.shape[2],
                device=short_intent.device, dtype=short_intent.dtype
            )
            short_intent_full = torch.cat([pad, short_intent], dim=1)
        else:
            short_intent_full = short_intent

        fused_intent = self.fusion_alpha * long_intent + (1.0 - self.fusion_alpha) * short_intent_full
        main_intent = fused_intent[:, -1, :]

        if train_flag:
            tag_emb = self.item_embeddings(tag.squeeze(-1))
            rep_diffu = self.diffu(item_embeddings, tag_emb, mask_seq, fused_intent)
            return rep_diffu, self.centroids, main_intent

        noise_x_t = th.randn_like(item_embeddings[:, -1, :])
        rep_diffu = self.reverse_p_sample(
            item_embeddings, noise_x_t, mask_seq, fused_intent
        )
        return rep_diffu, self.centroids
