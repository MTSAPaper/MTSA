import random

import numpy as np
from torch.nn import MSELoss
from ignite.engine import Events
from ignite.handlers import ModelCheckpoint, EarlyStopping
from transformers import AdamW, get_linear_schedule_with_warmup

from handler import *
from param import model_param, train_param
from model import Model
from metric import metric
from dataset import train_loader, dev_loader


def train():
    set_seed(train_param.seed)
    model = Model(model_param)
    optimizer = AdamW(model.parameters(), lr=train_param.lr, eps=1e-8)
    update_steps = train_param.epoch * len(train_loader)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=update_steps)
    loss_fn = [translate, MSELoss()]
    device = torch.device(f'cuda:{train_param.device}')
    trainer = create_trainer(model, optimizer, scheduler, loss_fn, train_param.grad_norm, device)
    train_evaluator = create_evaluator(model, metric, device)
    dev_evaluator = create_evaluator(model, metric, device)
    trainer.add_event_handler(Events.ITERATION_COMPLETED(every=train_param.interval), log_training_loss)
    trainer.add_event_handler(Events.EPOCH_COMPLETED, log_results, *(train_evaluator, train_loader, 'Train'))
    trainer.add_event_handler(Events.EPOCH_COMPLETED, log_results, *(dev_evaluator, dev_loader, 'Dev'))
    es_handler = EarlyStopping(patience=train_param.patience, score_function=score_fn, trainer=trainer)
    dev_evaluator.add_event_handler(Events.COMPLETED, es_handler)
    ckpt_handler = ModelCheckpoint(train_param.save_path, '', score_function=score_fn,
                                   score_name='score', require_empty=False)
    dev_evaluator.add_event_handler(Events.COMPLETED, ckpt_handler, {'model': model, 'param': model_param})
    print(f'Start running {train_param.save_path.split("/")[-1]} at device: {train_param.device}\t'
          f'lr: {train_param.lr}')
    trainer.run(train_loader, max_epochs=train_param.epoch)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.device_count() > 0:
        torch.cuda.manual_seed_all(seed)


def translate(t_inputs, masks, d_outputs):
    t_inputs = t_inputs[:, 1:].unsqueeze(2)
    masks = masks[:, 1:]
    loss = -torch.log(torch.gather(d_outputs.transpose(0, 1), 2, t_inputs)).squeeze(2)
    loss = torch.sum(loss * masks) / torch.sum(masks)
    return loss


if __name__ == '__main__':
    train()
