import math

def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if epoch < args.warmup_epoch:
        lr = args.lr * epoch / args.warmup_epoch
    else:
        lr = args.lr * 0.5 * (1. + math.cos(math.pi * (epoch - args.warmup_epoch) / (args.epoch - args.warmup_epoch)))
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr
