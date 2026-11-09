# VIGTDiff

Implementation of **VIGTDiff**.

## Training

### Toys

```bash
python main.py --dataset toys --short_window 4 --fusion_alpha 0.85 --lambda_div 0.0075 --num_cluster 32 --lambda_intent 0.4 --lambda_history 1
```

### Beauty

```bash
python main.py --dataset amazon_beauty --short_window 3 --fusion_alpha 0.80 --lambda_div 0.0075 --num_cluster 32 --lambda_intent 0.4 --lambda_history 0.6
```

### Steam

```bash
python main.py --dataset steam --short_window 4 --fusion_alpha 0.85 --lambda_div 0.05 --num_cluster 32 --lambda_intent 0.4 --lambda_history 0.6
```

### ML-1M

```bash
python main.py --dataset ml-1m --short_window 5 --fusion_alpha 0.85 --lambda_div 0.005 --num_cluster 8 --lambda_intent 0.6 --lambda_history 1
```
