import torch
import torch.nn.functional as F


def train_one_client(
    model,
    dataloader,
    epochs=1,
    lr=1e-3,
    device="cpu",
    grad_clip=1.0,
    pos_weight=None,
    weight_decay=1e-4,
    use_scheduler=True
):
    model = model.to(device)
    model.train()

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))
    else:
        scheduler = None

    if pos_weight is not None:
        pos_weight = torch.tensor(float(pos_weight), device=device, dtype=torch.float32)
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        loss_fn = torch.nn.BCEWithLogitsLoss()

    for _ in range(epochs):
        for x, y in dataloader:
            x = x.to(device)
            y = y.float().to(device)

            logits = model(x)
            loss = loss_fn(logits, y)

            opt.zero_grad(set_to_none=True)
            loss.backward()

            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            opt.step()

        if scheduler is not None:
            scheduler.step()

    return model


def get_update(global_state, local_state):
    update = {}
    for k in global_state.keys():
        update[k] = local_state[k] - global_state[k]
    return update


def apply_update(global_state, agg_update, lr_global=1.0):
    new_state = {}
    for k in global_state.keys():
        new_state[k] = global_state[k] + lr_global * agg_update[k]
    return new_state
