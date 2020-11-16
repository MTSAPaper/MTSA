import torch
from decimal import *
from ignite.engine import Engine


def create_trainer(model, optimizer, scheduler, loss_fn, max_grad_norm, device):
    model.to(device)

    def _update(engine, batch):
        model.train()
        optimizer.zero_grad()
        x, y = prepare_batch(batch, device)
        v_outputs, a_outputs, y_hat, w_s = model(*x)
        v_loss = loss_fn[0](x[0], x[1], v_outputs)
        a_loss = loss_fn[0](x[0], x[1], a_outputs)
        s_loss = loss_fn[1](y_hat, y)
        loss = v_loss + a_loss + w_s * s_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        scheduler.step()
        return loss
    return Engine(_update)


def create_evaluator(model, metrics, device):
    metrics = metrics or {}
    if device:
        model.to(device)

    def _inference(engine, batch):
        model.eval()
        with torch.no_grad():
            x, y = prepare_batch(batch, device)
            y_pred = model(*x)
            return y_pred[2], y
    evaluator = Engine(_inference)
    for name, metric in metrics.items():
        metric.attach(evaluator, name)
    return evaluator


def log_training_loss(trainer):
    print(f'Epoch[{trainer.state.epoch}]\tIteration: {trainer.state.iteration}\tLoss: {trainer.state.output:.4f}')


def log_results(evaluator, loader, split):
    evaluator.run(loader)
    metrics = evaluator.state.metrics
    metrics = form(metrics)
    print(f"{split} Results:\t"
          f"{metrics['acc2']}\t{metrics['f1']}\t"
          f"{metrics['mae']}\t{metrics['corr']}\t"
          f"{metrics['acc7']}\t{metrics['acc5']}")


def score_fn(evaluator):
    score = sum(evaluator.state.metrics.values()) - 2 * evaluator.state.metrics['mae']
    return score


def prepare_batch(batch, device):
    for i in range(len(batch)):
        batch[i] = batch[i].to(device)
    return batch[:-1], batch[-1]


def form(metrics):
    getcontext().prec = 3
    metrics['acc2'] = 100 * Decimal(f"{metrics['acc2']:.3f}")
    metrics['acc5'] = 100 * Decimal(f"{metrics['acc5']:.3f}")
    metrics['acc7'] = 100 * Decimal(f"{metrics['acc7']:.3f}")
    metrics['f1'] = 100 * Decimal(f"{metrics['f1']:.3f}")
    metrics['mae'] = Decimal(f"{metrics['mae']:.3f}")
    metrics['corr'] = Decimal(f"{metrics['corr']:.3f}")
    return metrics
