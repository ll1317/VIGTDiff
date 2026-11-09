import torch.nn as nn
import torch.optim as optim
import torch
import numpy as np
import copy
import torch.nn.functional as F


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
    if isinstance(model, nn.DataParallel):
        return model.module.item_embeddings.weight.detach()
    return model.item_embeddings.weight.detach()


def ild_per_user(topk_predict, item_weight, ks):
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


def model_train(tra_data_loader, val_data_loader, test_data_loader, model, args, logger):
    epochs = args.epochs
    device = args.device
    metric_ks = args.metric_ks
    model = model.to(device)

    if args.num_gpu > 1:
        model = nn.DataParallel(model)

    lambda_div = float(args.lambda_div)

    optimizer = optimizers(model, args)
    lr_scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=args.decay_step,
        gamma=args.gamma
    )

    best_metrics_dict = {
        'Best_HR@5': 0,
        'Best_NDCG@5': 0,
        'Best_HR@10': 0,
        'Best_NDCG@10': 0,
        'Best_HR@20': 0,
        'Best_NDCG@20': 0
    }
    best_epoch = {
        'Best_epoch_HR@5': 0,
        'Best_epoch_NDCG@5': 0,
        'Best_epoch_HR@10': 0,
        'Best_epoch_NDCG@10': 0,
        'Best_epoch_HR@20': 0,
        'Best_epoch_NDCG@20': 0
    }

    bad_count = 0
    best_model = copy.deepcopy(model)

    for epoch_temp in range(epochs):
        model.train()

        for train_batch in tra_data_loader:
            train_batch = [x.to(device) for x in train_batch]
            optimizer.zero_grad()

            diffu_rep, _, main_intent = model(
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

            loss_all = loss_ce + lambda_div * loss_div
            loss_all.backward()
            optimizer.step()

        print(
            'Epoch {} | CE: {:.4f} | TPTD: {:.4f} | Loss: {:.4f}'.format(
                epoch_temp,
                loss_ce.item(),
                loss_div.item(),
                loss_all.item()
            )
        )
        logger.info(
            'Epoch {} | CE: {:.4f} | TPTD: {:.4f} | Loss: {:.4f}'.format(
                epoch_temp,
                loss_ce.item(),
                loss_div.item(),
                loss_all.item()
            )
        )

        lr_scheduler.step()

        if epoch_temp != 0 and epoch_temp % args.eval_interval == 0:
            model.eval()
            metrics_dict = {
                'HR@5': [],
                'NDCG@5': [],
                'HR@10': [],
                'NDCG@10': [],
                'HR@20': [],
                'NDCG@20': []
            }

            with torch.no_grad():
                for val_batch in val_data_loader:
                    val_batch = [x.to(device) for x in val_batch]
                    rep_diffu, _ = model(
                        val_batch[0],
                        val_batch[1],
                        train_flag=False
                    )

                    scores_rec_diffu = model.diffu_rep_pre(rep_diffu)
                    metrics = hrs_and_ndcgs_k(
                        scores_rec_diffu,
                        val_batch[1],
                        metric_ks
                    )

                    for k, v in metrics.items():
                        metrics_dict[k].append(v)

            val_metrics = {
                key: round(np.mean(values) * 100, 4)
                for key, values in metrics_dict.items()
            }
            print('Validation:', val_metrics)
            logger.info('Validation: {}'.format(val_metrics))

            flag_update = 0
            for key_temp, values_mean in val_metrics.items():
                if values_mean > best_metrics_dict['Best_' + key_temp]:
                    flag_update = 1
                    bad_count = 0
                    best_metrics_dict['Best_' + key_temp] = values_mean
                    best_epoch['Best_epoch_' + key_temp] = epoch_temp

            if flag_update == 0:
                bad_count += 1
            else:
                best_model = copy.deepcopy(model)

            if bad_count >= args.patience:
                break

    logger.info('Best validation metrics: {}'.format(best_metrics_dict))
    logger.info('Best epochs: {}'.format(best_epoch))

    if args.eval_interval > epochs:
        best_model = copy.deepcopy(model)

    # ==================== Final Test ====================
    test_metrics_dict = {
        'HR@5': [],
        'NDCG@5': [],
        'HR@10': [],
        'NDCG@10': [],
        'HR@20': [],
        'NDCG@20': []
    }

    diversity_ks = [5, 10, 20]
    ild_sums = {'ILD@%d' % k: 0.0 for k in diversity_ks}
    diversity_user_count = 0

    item_weight = get_item_embedding_weight(best_model)

    best_model.eval()
    with torch.no_grad():
        for test_batch in test_data_loader:
            test_batch = [x.to(device) for x in test_batch]

            rep_diffu, _ = best_model(
                test_batch[0],
                test_batch[1],
                train_flag=False
            )
            scores_rec_diffu = best_model.diffu_rep_pre(rep_diffu)

            # HR / NDCG
            metrics = hrs_and_ndcgs_k(
                scores_rec_diffu,
                test_batch[1],
                metric_ks
            )
            for k, v in metrics.items():
                test_metrics_dict[k].append(v)

            # ILD
            diversity_scores = scores_rec_diffu.clone()
            diversity_scores[:, 0] = float('-inf')
            _, diversity_indices = torch.topk(
                diversity_scores,
                k=max(diversity_ks),
                dim=-1
            )

            ild_batch = ild_per_user(
                diversity_indices,
                item_weight,
                diversity_ks
            )

            diversity_user_count += test_batch[1].size(0)
            for metric_name, user_values in ild_batch.items():
                ild_sums[metric_name] += user_values.sum().item()

    test_metrics_dict_mean = {
        key: round(np.mean(values) * 100, 4)
        for key, values in test_metrics_dict.items()
    }

    ild_metrics_dict_mean = {
        'ILD@%d' % k: round(
            ild_sums['ILD@%d' % k] / max(diversity_user_count, 1),
            6
        )
        for k in diversity_ks
    }

    final_metrics = {
        **test_metrics_dict_mean,
        **ild_metrics_dict_mean
    }

    print('Test:', final_metrics)
    logger.info('Test: {}'.format(final_metrics))

    return best_model, final_metrics
