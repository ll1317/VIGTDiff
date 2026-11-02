import torch.nn as nn
import torch.optim as optim
import datetime
import torch
import numpy as np
import copy
from Modules import KMeans
import pickle
import torch.nn.functional as F
import os
import time as Time


def optimizers(model, args):
    if args.optimizer.lower() == 'adam':
        return optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer.lower() == 'sgd':
        return optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
    else:
        raise ValueError


def cal_hr(label, predict, ks):
    max_ks = max(ks)
    _, topk_predict = torch.topk(predict, k=max_ks, dim=-1)
    hit = label == topk_predict
    hr = [hit[:, :ks[i]].sum().item() / label.size()[0] for i in range(len(ks))]
    return hr


def cal_ndcg(label, predict, ks):
    max_ks = max(ks)
    _, topk_predict = torch.topk(predict, k=max_ks, dim=-1)
    hit = (label == topk_predict).int()
    ndcg = []
    for k in ks:
        max_dcg = dcg(torch.tensor([1] + [0] * (k - 1)))
        predict_dcg = dcg(hit[:, :k])
        ndcg.append((predict_dcg / max_dcg).mean().item())
    return ndcg


def dcg(hit):
    log2 = torch.log2(torch.arange(1, hit.size()[-1] + 1) + 1).unsqueeze(0)
    rel = (hit / log2).sum(dim=-1)
    return rel


def hrs_and_ndcgs_k(scores, labels, ks):
    metrics = {}
    ndcg = cal_ndcg(labels.clone().detach().to('cpu'), scores.clone().detach().to('cpu'), ks)
    hr = cal_hr(labels.clone().detach().to('cpu'), scores.clone().detach().to('cpu'), ks)
    for k, ndcg_temp, hr_temp in zip(ks, ndcg, hr):
        metrics['HR@%d' % k] = hr_temp
        metrics['NDCG@%d' % k] = ndcg_temp
    return metrics


def get_item_embedding_weight(model):
    """Return the learned item-embedding matrix for normal/DataParallel models."""
    if isinstance(model, nn.DataParallel):
        return model.module.item_embeddings.weight.detach()
    return model.item_embeddings.weight.detach()


def ild_per_user(topk_predict, item_weight, ks):
    """
    Compute per-user Intra-List Distance (ILD@K).

    ILD@K(u) = 1 / (K * (K - 1))
               * sum_{i in R_u^K} sum_{j in R_u^K, j != i}
                 ||e_i - e_j||_2
    """
    result = {}

    for k in ks:
        rec_items = topk_predict[:, :k].long()
        rec_emb = F.embedding(rec_items, item_weight)

        if k > 1:
            pairwise_dist = torch.cdist(rec_emb, rec_emb, p=2)
            ild_user = pairwise_dist.sum(dim=(1, 2)) / float(k * (k - 1))
        else:
            ild_user = torch.zeros(
                rec_emb.size(0),
                device=rec_emb.device,
                dtype=rec_emb.dtype
            )

        result['ILD@%d' % k] = ild_user

    return result


def normalized_exposure_gini(exposure_counts):
    """
    Compute the normalized Gini coefficient of catalog-wide item exposure.

    exposure_counts must contain every real catalog item, including items
    with zero exposure. Padding item 0 should be excluded before calling.
    """
    counts = np.asarray(exposure_counts, dtype=np.float64).reshape(-1)
    n_items = counts.size

    if n_items <= 1:
        return 0.0
    if np.any(counts < 0):
        raise ValueError("Exposure counts must be non-negative.")

    total_exposure = counts.sum()
    if total_exposure <= 0:
        return 0.0

    exposure_prob = np.sort(counts / total_exposure)
    ranks = np.arange(1, n_items + 1, dtype=np.float64)

    gini = np.sum(
        (2.0 * ranks - n_items - 1.0) * exposure_prob
    ) / float(n_items - 1)

    return float(np.clip(gini, 0.0, 1.0))


def _sanitize_diagnostic_tag(tag):
    """Make a filesystem-safe tag without changing any experiment logic."""
    safe_chars = []
    for ch in str(tag):
        if ch.isalnum() or ch in ('-', '_', '.'):
            safe_chars.append(ch)
        else:
            safe_chars.append('_')
    return ''.join(safe_chars)


def _print_and_log(message, logger=None):
    print(message)
    if logger is not None:
        logger.info(message)


def save_ild_diagnostics(
        diagnostic_dir,
        diagnostic_tag,
        sequences,
        labels,
        top20,
        ild_by_k,
        item_weight,
        logger=None
):
    """
    Save final-test recommendations and per-user ILD values.

    This is diagnostics only. It does not change training, validation,
    early stopping, checkpoint selection, ranking, or metric computation.
    """
    os.makedirs(diagnostic_dir, exist_ok=True)
    safe_tag = _sanitize_diagnostic_tag(diagnostic_tag)
    save_path = os.path.join(diagnostic_dir, safe_tag + '.npz')

    np.savez_compressed(
        save_path,
        sequences=sequences,
        labels=labels,
        top20=top20,
        item_weight=item_weight,
        ild5_user=ild_by_k[5],
        ild10_user=ild_by_k[10],
        ild20_user=ild_by_k[20]
    )

    _print_and_log(
        '[ILD diagnostics] Saved: {}'.format(os.path.abspath(save_path)),
        logger
    )
    return save_path


def _recompute_ild_with_fixed_embeddings(top20, item_weight, k, batch_size=1024):
    """Recompute per-user ILD for a recommendation list in a fixed embedding space."""
    top20 = np.asarray(top20)
    item_weight = np.asarray(item_weight)
    outputs = []

    weight_tensor = torch.from_numpy(item_weight).float()
    for start in range(0, top20.shape[0], batch_size):
        batch_items = torch.from_numpy(top20[start:start + batch_size, :k]).long()
        batch_emb = F.embedding(batch_items, weight_tensor)
        if k > 1:
            pairwise_dist = torch.cdist(batch_emb, batch_emb, p=2)
            batch_ild = pairwise_dist.sum(dim=(1, 2)) / float(k * (k - 1))
        else:
            batch_ild = torch.zeros(batch_items.size(0), dtype=batch_emb.dtype)
        outputs.append(batch_ild.numpy())

    return np.concatenate(outputs, axis=0)


def compare_ild_diagnostics(reference_path, current_path, logger=None):
    """
    Compare an A-only diagnostic file with the current A+loss file.

    Set environment variable ILD_COMPARE_PATH to the A-only .npz path.
    The comparison is skipped safely when the path is absent or incompatible.
    """
    if not reference_path:
        _print_and_log(
            '[ILD diagnostics] ILD_COMPARE_PATH is not set; saved current run only.',
            logger
        )
        return

    if not os.path.isfile(reference_path):
        _print_and_log(
            '[ILD diagnostics] Reference file not found: {}'.format(reference_path),
            logger
        )
        return

    required_keys = {
        'sequences', 'labels', 'top20',
        'ild5_user', 'ild10_user', 'ild20_user'
    }

    with np.load(reference_path) as reference_data, np.load(current_path) as current_data:
        missing_reference = required_keys.difference(reference_data.files)
        missing_current = required_keys.difference(current_data.files)
        if missing_reference or missing_current:
            _print_and_log(
                '[ILD diagnostics] Comparison skipped because required arrays are missing. '
                'reference_missing={} current_missing={}'.format(
                    sorted(missing_reference), sorted(missing_current)
                ),
                logger
            )
            return

        ref_sequences = reference_data['sequences']
        cur_sequences = current_data['sequences']
        ref_labels = reference_data['labels'].reshape(-1)
        cur_labels = current_data['labels'].reshape(-1)
        ref_top20 = reference_data['top20']
        cur_top20 = current_data['top20']

        same_shapes = (
            ref_sequences.shape == cur_sequences.shape
            and ref_labels.shape == cur_labels.shape
            and ref_top20.shape == cur_top20.shape
        )
        same_order = (
            same_shapes
            and np.array_equal(ref_sequences, cur_sequences)
            and np.array_equal(ref_labels, cur_labels)
        )

        header = (
            '\n========== ILD / Ranking Change Diagnostics ==========' 
            '\nReference: {}'
            '\nCurrent:   {}'
            '\nSame test-user order: {}'
        ).format(
            os.path.abspath(reference_path),
            os.path.abspath(current_path),
            same_order
        )
        _print_and_log(header, logger)

        if not same_order:
            _print_and_log(
                'Comparison stopped: test sequence/label order is not identical.',
                logger
            )
            _print_and_log(
                '========================================================\n',
                logger
            )
            return

        if ref_top20.ndim != 2 or ref_top20.shape[1] < 20:
            _print_and_log(
                'Comparison stopped: top20 arrays must have shape [N, >=20].',
                logger
            )
            _print_and_log(
                '========================================================\n',
                logger
            )
            return

        # Segment-level changes answer whether only one ranking interval changed.
        segments = [
            ('Rank1-5', 0, 5),
            ('Rank6-10', 5, 10),
            ('Rank11-20', 10, 20)
        ]
        _print_and_log('\n[Segment-level recommendation changes]', logger)
        for segment_name, start, end in segments:
            ref_segment = ref_top20[:, start:end]
            cur_segment = cur_top20[:, start:end]
            segment_size = end - start

            position_change_rate = float(np.mean(ref_segment != cur_segment))
            replaced_counts = np.asarray([
                segment_size - len(set(ref_row.tolist()).intersection(cur_row.tolist()))
                for ref_row, cur_row in zip(ref_segment, cur_segment)
            ], dtype=np.float64)

            _print_and_log(
                '{}: position_change_rate={:.6f}, '
                'mean_replaced_items={:.6f}, changed_user_rate={:.6f}'.format(
                    segment_name,
                    position_change_rate,
                    float(replaced_counts.mean()),
                    float(np.mean(replaced_counts > 0))
                ),
                logger
            )

        _print_and_log('\n[Top-K set changes, target movement, and per-user ILD changes]', logger)
        delta_means = []
        for k in (5, 10, 20):
            ref_k = ref_top20[:, :k]
            cur_k = cur_top20[:, :k]

            replaced_counts = np.asarray([
                k - len(set(ref_row.tolist()).intersection(cur_row.tolist()))
                for ref_row, cur_row in zip(ref_k, cur_k)
            ], dtype=np.float64)

            ref_hit = np.asarray([
                label in row for label, row in zip(ref_labels, ref_k)
            ], dtype=bool)
            cur_hit = np.asarray([
                label in row for label, row in zip(cur_labels, cur_k)
            ], dtype=bool)

            moved_in_rate = float(np.mean((~ref_hit) & cur_hit))
            moved_out_rate = float(np.mean(ref_hit & (~cur_hit)))

            ref_ild = reference_data['ild{}_user'.format(k)].reshape(-1)
            cur_ild = current_data['ild{}_user'.format(k)].reshape(-1)
            delta = cur_ild - ref_ild
            delta_means.append(float(delta.mean()))

            tolerance = 1e-8
            positive_rate = float(np.mean(delta > tolerance))
            unchanged_rate = float(np.mean(np.abs(delta) <= tolerance))
            negative_rate = float(np.mean(delta < -tolerance))

            _print_and_log(
                'K={}: changed_user_rate={:.6f}, mean_replaced_items={:.6f}, '
                'target_moved_in_rate={:.6f}, target_moved_out_rate={:.6f}'.format(
                    k,
                    float(np.mean(replaced_counts > 0)),
                    float(replaced_counts.mean()),
                    moved_in_rate,
                    moved_out_rate
                ),
                logger
            )
            _print_and_log(
                '     ILD_ref={:.9f}, ILD_cur={:.9f}, delta_mean={:+.9f}, '
                'delta_median={:+.9f}, delta_std={:.9f}'.format(
                    float(ref_ild.mean()),
                    float(cur_ild.mean()),
                    float(delta.mean()),
                    float(np.median(delta)),
                    float(delta.std())
                ),
                logger
            )
            _print_and_log(
                '     user_delta_positive={:.6f}, unchanged={:.6f}, negative={:.6f}'.format(
                    positive_rate,
                    unchanged_rate,
                    negative_rate
                ),
                logger
            )

        # Learned-embedding ILD can change even when the recommendation set does not.
        # When both files contain item_weight, decompose total ILD change into:
        #   (1) ranking/list change under the fixed A-only embedding geometry;
        #   (2) embedding-geometry change for the current recommendation list.
        if 'item_weight' in reference_data.files and 'item_weight' in current_data.files:
            ref_weight = reference_data['item_weight']
            cur_weight = current_data['item_weight']
            if ref_weight.shape == cur_weight.shape:
                _print_and_log(
                    '\n[ILD change decomposition: ranking effect vs embedding-geometry effect]',
                    logger
                )
                for k in (5, 10, 20):
                    ref_list_ref_geometry = _recompute_ild_with_fixed_embeddings(
                        ref_top20, ref_weight, k
                    )
                    cur_list_ref_geometry = _recompute_ild_with_fixed_embeddings(
                        cur_top20, ref_weight, k
                    )
                    cur_list_cur_geometry = _recompute_ild_with_fixed_embeddings(
                        cur_top20, cur_weight, k
                    )

                    ranking_delta_fixed_geometry = (
                        cur_list_ref_geometry - ref_list_ref_geometry
                    )
                    geometry_delta_current_list = (
                        cur_list_cur_geometry - cur_list_ref_geometry
                    )

                    _print_and_log(
                        'K={}: ranking_delta_under_reference_geometry={:+.9f}, '
                        'geometry_delta_for_current_list={:+.9f}, '
                        'combined={:+.9f}'.format(
                            k,
                            float(ranking_delta_fixed_geometry.mean()),
                            float(geometry_delta_current_list.mean()),
                            float(
                                ranking_delta_fixed_geometry.mean()
                                + geometry_delta_current_list.mean()
                            )
                        ),
                        logger
                    )
            else:
                _print_and_log(
                    '\n[ILD decomposition skipped] item_weight shapes differ: {} vs {}'.format(
                        ref_weight.shape, cur_weight.shape
                    ),
                    logger
                )
        else:
            _print_and_log(
                '\n[ILD decomposition skipped] item_weight is absent in one diagnostic file.',
                logger
            )

        equal_after_rounding = (
            round(delta_means[0], 2)
            == round(delta_means[1], 2)
            == round(delta_means[2], 2)
        )
        nearly_identical_exact = (
            max(delta_means) - min(delta_means) <= 1e-8
        )
        _print_and_log(
            '\n[Delta consistency check] exact_deltas={}; '
            'same_after_2_decimals={}; nearly_identical_exact={}'.format(
                ['{:+.9f}'.format(value) for value in delta_means],
                equal_after_rounding,
                nearly_identical_exact
            ),
            logger
        )
        _print_and_log(
            '========================================================\n',
            logger
        )


def LSHT_inference(model, args, data_loader):
    device = args.device
    model = model.to(device)
    with torch.no_grad():
        test_metrics_dict = {'HR@5': [], 'NDCG@5': [], 'HR@10': [], 'NDCG@10': [], 'HR@20': [], 'NDCG@20': []}
        test_metrics_dict_mean = {}
        for test_batch in data_loader:
            test_batch = [x.to(device) for x in test_batch]

            rep_diffu = model(test_batch[0], test_batch[1], train_flag=False)
            scores_rec_diffu = model.diffu_rep_pre(rep_diffu)
            metrics = hrs_and_ndcgs_k(scores_rec_diffu, test_batch[1], [5, 10, 20])
            for k, v in metrics.items():
                test_metrics_dict[k].append(v)
    for key_temp, values_temp in test_metrics_dict.items():
        values_mean = round(np.mean(values_temp) * 100, 4)
        test_metrics_dict_mean[key_temp] = values_mean
    print(test_metrics_dict_mean)


# def contra_loss(diffu,labels,num_cluster,criterion):
#     centers=diffu[torch.randperm(len(diffu))[:num_cluster]]
#     for i in range(num_cluster):
#         if torch.sum(labels == i) == 0:
#             centers[i] = diffu[torch.randint(0, diffu.size(0), (1,))]
#         else:
#             centers[i] = torch.mean(diffu[labels == i], dim=0)

#     distance=None
#     # centers=torch.stack(centers)
#     for i in range(num_cluster):
#         centers = torch.cat((centers[i:i + 1], centers[0:i],
#         centers[i + 1:]), 0)
#         distance_= torch.einsum('nc,kc->nk', [
#     nn.functional.normalize(diffu[labels==i], dim=1),
#     nn.functional.normalize(centers, dim=1)
# ])

#         if distance is None:
#             distance = F.softmax(distance_, dim=1)
#         else:
#             distance = torch.cat((distance, F.softmax(distance_, dim=1)),
#                                             0)
#             idx = torch.zeros(distance.shape[0], dtype=torch.long).cuda()

#     loss = criterion(distance, idx)
#     return loss

def model_train(tra_data_loader, val_data_loader, test_data_loader, model, args, logger):
    epochs = args.epochs
    device = args.device
    metric_ks = args.metric_ks
    model = model.to(device)
    is_parallel = args.num_gpu > 1
    if is_parallel:
        model = nn.DataParallel(model)

    # ===== print model running config =====
    raw_model = model.module if isinstance(model, nn.DataParallel) else model
    lambda_div = 0.0075

    # Final-test diagnostics only. These environment variables do not alter
    # model training or evaluation logic.
    diagnostic_dir = os.environ.get("ILD_DIAG_DIR", "ild_diagnostics")
    default_diagnostic_tag = "{}_loss_M{}_lambda{}".format(
        getattr(args, "dataset", "dataset"),
        getattr(raw_model, "div_top_m", "NA"),
        str(lambda_div).replace('.', 'p')
    )
    diagnostic_tag = os.environ.get("ILD_DIAG_TAG", default_diagnostic_tag)
    diagnostic_reference_path = os.environ.get("ILD_COMPARE_PATH", "").strip()

    running_config_msg = (
        "\n========== Running Config Check ==========\n"
        "Dataset: {}\n"
        "Device: {}\n"
        "Batch size: {}\n"
        "LR: {}\n"
        "Epochs: {}\n"
        "Eval interval: {}\n"
        "Patience: {}\n"
        "\nA module:\n"
        "  short_window = {}\n"
        "  short_noise_scale = {}\n"
        "  short_gate_scale = {}\n"
        "  fusion_alpha = {}\n"
        "\nTangential dispersion loss:\n"
        "  div_top_m = {}\n"
        "  lambda_div = {}\n"
        "\nFinal-test diagnostics only:\n"
        "  diagnostic_dir = {}\n"
        "  diagnostic_tag = {}\n"
        "  compare_reference = {}\n"
        "\nDiffusion:\n"
        "  diffusion_steps = {}\n"
        "  lambda_history = {}\n"
        "  lambda_intent = {}\n"
        "==========================================\n"
    ).format(
        getattr(args, "dataset", "unknown"),
        getattr(args, "device", "unknown"),
        getattr(args, "batch_size", "unknown"),
        getattr(args, "lr", "unknown"),
        getattr(args, "epochs", "unknown"),
        getattr(args, "eval_interval", "unknown"),
        getattr(args, "patience", "unknown"),

        getattr(raw_model, "short_window", "NA"),
        getattr(raw_model, "short_noise_scale", "NA"),
        getattr(raw_model, "short_gate_scale", "NA"),
        getattr(raw_model, "fusion_alpha", "NA"),

        getattr(raw_model, "div_top_m", "NA"),
        lambda_div,

        diagnostic_dir,
        diagnostic_tag,
        diagnostic_reference_path if diagnostic_reference_path else "NOT_SET",

        getattr(raw_model, "diffusion_steps", "NA"),
        getattr(raw_model, "lambda_history", "NA"),
        getattr(raw_model, "lambda_intent", "NA"),
    )

    print(running_config_msg, flush=True)
    logger.info(running_config_msg)

    optimizer = optimizers(model, args)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.decay_step, gamma=args.gamma)
    best_metrics_dict = {'Best_HR@5': 0, 'Best_NDCG@5': 0, 'Best_HR@10': 0, 'Best_NDCG@10': 0, 'Best_HR@20': 0,
                         'Best_NDCG@20': 0}
    best_epoch = {'Best_epoch_HR@5': 0, 'Best_epoch_NDCG@5': 0, 'Best_epoch_HR@10': 0, 'Best_epoch_NDCG@10': 0,
                  'Best_epoch_HR@20': 0, 'Best_epoch_NDCG@20': 0}
    bad_count = 0
    # criterion = nn.CrossEntropyLoss().to(args.device)
    for epoch_temp in range(epochs):
        print('Epoch: {}'.format(epoch_temp))
        logger.info('Epoch: {}'.format(epoch_temp))
        model.train()
        metrics_dict = {'HR@5': [], 'NDCG@5': [], 'HR@10': [], 'NDCG@10': [], 'HR@20': [], 'NDCG@20': []}
        flag_update = 0
        train_start = Time.time()
        for index_temp, train_batch in enumerate(tra_data_loader):
            train_batch = [x.to(device) for x in train_batch]
            optimizer.zero_grad()
            diffu_rep, code, main_intent = model(
                train_batch[0],
                train_batch[1],
                train_flag=True
            )

            loss_ce = model.loss_diffu_ce(
                diffu_rep,
                train_batch[1]
            )

            loss_div = model.loss_posterior_dispersion(
                diffu_rep,
                train_batch[1],
                main_intent
            )

            loss_all = (
                    loss_ce
                    + lambda_div * loss_div
            )

            loss_all.backward()

            # loss_all = model.loss_diffu_ce(diffu_rep, train_batch[1])
            # loss_all.backward()

            optimizer.step()
            # if index_temp % int(len(tra_data_loader) / 5 + 1) == 0:
            #     print('[%d/%d] Loss: %.4f' % (index_temp, len(tra_data_loader), loss_all.item()))
            #     logger.info('[%d/%d] Loss: %.4f' % (index_temp, len(tra_data_loader), loss_all.item()))
            if index_temp % int(len(tra_data_loader) / 5 + 1) == 0:
                msg = (
                        '[%d/%d] '
                        'CE: %.4f | '
                        'loss_div: %.4f | '
                        'Loss: %.4f'
                        % (
                            index_temp,
                            len(tra_data_loader),
                            loss_ce.item(),
                            loss_div.item(),
                            loss_all.item()
                        )
                )

                print(msg)
                logger.info(msg)

        print("loss in epoch {}: {}".format(epoch_temp, loss_all.item()))
        print("Train cost: " + Time.strftime("%H: %M: %S", Time.gmtime(Time.time() - train_start)))

        lr_scheduler.step()

        if epoch_temp != 0 and epoch_temp % args.eval_interval == 0:
            print('start predicting: ', datetime.datetime.now())
            logger.info('start predicting: {}'.format(datetime.datetime.now()))
            model.eval()
            with torch.no_grad():
                metrics_dict = {'HR@5': [], 'NDCG@5': [], 'HR@10': [], 'NDCG@10': [], 'HR@20': [], 'NDCG@20': []}
                # metrics_dict_mean = {}
                eval_start = Time.time()

                for val_batch in val_data_loader:
                    val_batch = [x.to(device) for x in val_batch]
                    rep_diffu, code = model(val_batch[0], val_batch[1], train_flag=False)

                    scores_rec_diffu = model.diffu_rep_pre(rep_diffu)  ### inner_production

                    metrics = hrs_and_ndcgs_k(scores_rec_diffu, val_batch[1], metric_ks)
                    for k, v in metrics.items():
                        metrics_dict[k].append(v)
            # if epoch_temp % 10 == 0:
            #   torch.save(rep_diffu, str(epoch_temp)+'generation_nv.pt')
            for key_temp, values_temp in metrics_dict.items():
                values_mean = round(np.mean(values_temp) * 100, 4)
                if values_mean > best_metrics_dict['Best_' + key_temp]:
                    flag_update = 1
                    bad_count = 0
                    best_metrics_dict['Best_' + key_temp] = values_mean
                    best_epoch['Best_epoch_' + key_temp] = epoch_temp
            # if epoch_temp % 10 == 0:
            #    torch.save(code, str(epoch_temp)+'code_nc.pt')
            print("Evalution cost: " + Time.strftime("%H: %M: %S", Time.gmtime(Time.time() - eval_start)))

            if flag_update == 0:
                bad_count += 1
            else:
                print(best_metrics_dict)
                print(best_epoch)
                logger.info(best_metrics_dict)
                logger.info(best_epoch)
                best_model = copy.deepcopy(model)
            if bad_count >= args.patience:
                break

    logger.info(best_metrics_dict)
    logger.info(best_epoch)

    if args.eval_interval > epochs:
        best_model = copy.deepcopy(model)

    top_100_item = []
    with torch.no_grad():
        test_metrics_dict = {'HR@5': [], 'NDCG@5': [], 'HR@10': [], 'NDCG@10': [], 'HR@20': [], 'NDCG@20': []}
        test_metrics_dict_mean = {}

        # Extra metrics copied from the right-side file.
        # They are computed only in final testing and do not affect training,
        # validation, early stopping, best-model selection, original HR/NDCG,
        # or original Top-100 generation.
        diversity_ks = [5, 10, 20]
        diversity_user_count = 0
        ild_sums = {'ILD@%d' % k: 0.0 for k in diversity_ks}
        coverage_item_sets = {k: set() for k in diversity_ks}

        item_weight = get_item_embedding_weight(best_model)
        catalog_size = max(int(item_weight.size(0)) - 1, 1)

        exposure_counts = {
            k: np.zeros(catalog_size, dtype=np.int64)
            for k in diversity_ks
        }

        # Diagnostics only.
        padding_top5 = 0
        padding_top10 = 0
        padding_top20 = 0

        # Save per-user final-test data for ILD/ranking-change checks.
        diagnostic_sequences = []
        diagnostic_labels = []
        diagnostic_top20 = []
        diagnostic_ild = {k: [] for k in diversity_ks}

        for test_batch in test_data_loader:
            test_batch = [x.to(device) for x in test_batch]
            rep_diffu, code = best_model(test_batch[0], test_batch[1], train_flag=False)
            scores_rec_diffu = best_model.diffu_rep_pre(rep_diffu)  ### Inner Production
            # scores_rec_diffu = best_model.routing_rep_pre(rep_diffu)   ### routing

            _, indices = torch.topk(scores_rec_diffu, k=100)
            top_100_item.append(indices)

            metrics = hrs_and_ndcgs_k(scores_rec_diffu, test_batch[1], metric_ks)
            for k, v in metrics.items():
                test_metrics_dict[k].append(v)

            # The original ranking above is kept unchanged for HR/NDCG and Top-100.
            # For diversity metrics only, mask padding id 0 so ILD/Coverage/Gini
            # are computed on real recommendable items.
            diversity_scores = scores_rec_diffu.clone()
            diversity_scores[:, 0] = float('-inf')
            _, diversity_indices = torch.topk(
                diversity_scores,
                k=max(diversity_ks),
                dim=-1
            )

            padding_top5 += (diversity_indices[:, :5] == 0).sum().item()
            padding_top10 += (diversity_indices[:, :10] == 0).sum().item()
            padding_top20 += (diversity_indices[:, :20] == 0).sum().item()

            ild_batch = ild_per_user(
                diversity_indices,
                item_weight,
                diversity_ks
            )
            diversity_user_count += test_batch[1].size(0)
            for metric_name, user_values in ild_batch.items():
                ild_sums[metric_name] += user_values.sum().item()

            # Diagnostics-only storage; no effect on any metric or ranking.
            diagnostic_sequences.append(
                test_batch[0].detach().cpu().numpy()
            )
            diagnostic_labels.append(
                test_batch[1].view(-1).detach().cpu().numpy()
            )
            diagnostic_top20.append(
                diversity_indices[:, :20].detach().cpu().numpy()
            )
            for k in diversity_ks:
                diagnostic_ild[k].append(
                    ild_batch['ILD@%d' % k].detach().cpu().numpy()
                )

            for k in diversity_ks:
                recommended_ids = (
                    diversity_indices[:, :k]
                    .detach()
                    .cpu()
                    .reshape(-1)
                    .tolist()
                )

                coverage_item_sets[k].update(recommended_ids)

                batch_exposure = np.bincount(
                    np.asarray(recommended_ids, dtype=np.int64) - 1,
                    minlength=catalog_size
                )
                exposure_counts[k] += batch_exposure[:catalog_size]

    print("padding in Top-5:", padding_top5)
    print("padding in Top-10:", padding_top10)
    print("padding in Top-20:", padding_top20)
    print("catalog_size:", catalog_size)
    print("args.item_num:", args.item_num)

    logger.info("padding in Top-5: {}".format(padding_top5))
    logger.info("padding in Top-10: {}".format(padding_top10))
    logger.info("padding in Top-20: {}".format(padding_top20))
    logger.info("catalog_size: {}".format(catalog_size))
    logger.info("args.item_num: {}".format(args.item_num))

    for key_temp, values_temp in test_metrics_dict.items():
        values_mean = round(np.mean(values_temp) * 100, 4)
        test_metrics_dict_mean[key_temp] = values_mean

    diversity_metrics_dict_mean = {}
    for k in diversity_ks:
        diversity_metrics_dict_mean['ILD@%d' % k] = round(
            ild_sums['ILD@%d' % k] / max(diversity_user_count, 1),
            6
        )
        diversity_metrics_dict_mean['CatalogCoverage@%d(%%)' % k] = round(
            len(coverage_item_sets[k]) / float(catalog_size) * 100.0,
            4
        )
        diversity_metrics_dict_mean['ExposureGini@%d' % k] = round(
            normalized_exposure_gini(exposure_counts[k]),
            6
        )

    # ===== Final-test ILD/ranking diagnostics only =====
    diagnostic_sequences = np.concatenate(diagnostic_sequences, axis=0)
    diagnostic_labels = np.concatenate(diagnostic_labels, axis=0)
    diagnostic_top20 = np.concatenate(diagnostic_top20, axis=0)
    diagnostic_ild = {
        k: np.concatenate(diagnostic_ild[k], axis=0)
        for k in diversity_ks
    }

    # Internal consistency: the saved per-user means must reproduce the
    # already-computed aggregate ILD values.
    consistency_lines = [
        '\n========== Current-Run ILD Consistency Check ==========',
        'top20 shape: {}'.format(diagnostic_top20.shape),
        'K-specific slices: Top-5={}, Top-10={}, Top-20={}'.format(
            diagnostic_top20[:, :5].shape,
            diagnostic_top20[:, :10].shape,
            diagnostic_top20[:, :20].shape
        )
    ]
    for k in diversity_ks:
        saved_mean = float(diagnostic_ild[k].mean())
        aggregate_mean = float(
            ild_sums['ILD@%d' % k] / max(diversity_user_count, 1)
        )
        consistency_lines.append(
            'ILD@{} saved_user_mean={:.9f}, aggregate_mean={:.9f}, '
            'abs_error={:.12f}'.format(
                k,
                saved_mean,
                aggregate_mean,
                abs(saved_mean - aggregate_mean)
            )
        )
    consistency_lines.append(
        '======================================================'
    )
    consistency_message = '\n'.join(consistency_lines)
    print(consistency_message)
    logger.info(consistency_message)

    current_diagnostic_path = save_ild_diagnostics(
        diagnostic_dir=diagnostic_dir,
        diagnostic_tag=diagnostic_tag,
        sequences=diagnostic_sequences,
        labels=diagnostic_labels,
        top20=diagnostic_top20,
        ild_by_k=diagnostic_ild,
        item_weight=item_weight.detach().cpu().numpy(),
        logger=logger
    )

    compare_ild_diagnostics(
        reference_path=diagnostic_reference_path,
        current_path=current_diagnostic_path,
        logger=logger
    )

    print('Test------------------------------------------------------')
    logger.info('Test------------------------------------------------------')
    print(test_metrics_dict_mean)
    logger.info(test_metrics_dict_mean)

    print('Diversity: ILD / Catalog Coverage / Exposure Gini---------------')
    logger.info('Diversity: ILD / Catalog Coverage / Exposure Gini---------------')
    print(diversity_metrics_dict_mean)
    logger.info(diversity_metrics_dict_mean)

    print('Best Eval---------------------------------------------------------')
    logger.info('Best Eval---------------------------------------------------------')
    print(best_metrics_dict)
    print(best_epoch)
    logger.info(best_metrics_dict)
    logger.info(best_epoch)

    if args.diversity_measure:
        path_data = '../datasets/data/category/' + args.dataset + '/id_category_dict.pkl'
        with open(path_data, 'rb') as f:
            id_category_dict = pickle.load(f)
        id_top_100 = torch.cat(top_100_item, dim=0).tolist()
        category_list_100 = []
        for id_top_100_temp in id_top_100:
            category_temp_list = []
            for id_temp in id_top_100_temp:
                category_temp_list.append(id_category_dict[id_temp])
            category_list_100.append(category_temp_list)
        category_list_100.append(category_list_100)
        path_data_category = '../datasets/data/category/' + args.dataset + '/DiffuRec_top100_category.pkl'
        with open(path_data_category, 'wb') as f:
            pickle.dump(category_list_100, f)

    return best_model, test_metrics_dict_mean
