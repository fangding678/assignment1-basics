import math

import numpy as np
import numpy.typing as npt
import torch
from torch.optim.sgd import SGD
from typing import Union, Iterable, Any, Dict, Optional, Callable
from typing_extensions import TypeAlias
import argparse

ParamsT: TypeAlias = Union[Iterable[torch.Tensor], Iterable[Dict[str, Any]]]


class AdamW(torch.optim.Optimizer):
    def __init__(self, params: ParamsT, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        defaults: Dict[str, Any] = dict({'lr': lr, 'betas': betas, 'eps': eps, 'weight_decay': weight_decay})
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            weight_decay = group['weight_decay']
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                t = state.get('t', 0) + 1
                m = state.get('m', torch.zeros_like(p.data))
                v = state.get('v', torch.zeros_like(p.data))
                grad = p.grad.data
                m = beta1 * m + (1 - beta1) * grad
                v = beta2 * v + (1 - beta2) * torch.square(grad)
                lr_t = lr * math.sqrt(1 - beta2**t) / (1 - beta1**t)
                p.data -= lr_t * m / (torch.sqrt(v) + eps)
                p.data *= (1 - lr * weight_decay)
                state['t'] = t
                state['m'] = m
                state['v'] = v
        return loss


def get_lr_cosine_schedule(it, max_learning_rate, min_learning_rate, warmup_iters, cosine_cycle_iters):
    lr = min_learning_rate
    if it < warmup_iters:
        lr = it / warmup_iters * max_learning_rate
    elif it < cosine_cycle_iters:
        lr = min_learning_rate + (1 + math.cos((it - warmup_iters) / (cosine_cycle_iters - warmup_iters) * math.pi)) * (
                    max_learning_rate - min_learning_rate) / 2
    return lr


def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float, eps=1e-6) -> None:
    grads = [p.grad for p in parameters if p.grad is not None]
    total_norm = torch.linalg.norm(torch.stack([torch.linalg.norm(g.detach()) for g in grads]))
    if total_norm > max_l2_norm:
        coef = max_l2_norm / (total_norm + eps)
        for g in grads:
            g.detach().mul_(coef)


def get_batch_data(dataset: npt.NDArray, batch_size: int, context_length: int, device: str) -> tuple[
    torch.Tensor, torch.Tensor]:
    ix = torch.randint(0, len(dataset) - context_length, (batch_size, ))
    x = torch.stack([torch.from_numpy(dataset[i: i + context_length]) for i in ix])
    y = torch.stack([torch.from_numpy(dataset[i + 1: i + 1 + context_length]) for i in ix])
    return x.to(device), y.to(device)


def save_checkpoint(model, optimizer, iteration, out):
    obj = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'iteration': iteration}
    torch.save(obj, out)


def load_checkpoint(src, model, optimizer):
    obj = torch.load(src)
    model.load_state_dict(obj['model'])
    if optimizer and 'optimizer' in obj:
        optimizer.load_state_dict(obj['optimizer'])
    return obj['iteration']


def resource_use(batch=1, vocab_size=50257, context_length=1024, num_layers=48, d_model=1600, num_heads=25):
    d_ff = 4 * d_model
    vocab_param = vocab_size * d_model
    norm_param = 2 * d_model
    att_param = 4 * d_model * d_model
    ffn_param = 3 * d_model * d_ff
    total_param = vocab_param + num_layers * (norm_param + att_param + ffn_param)
    norm_activation = 2 * context_length * d_model
    att_activation = 4 * context_length * d_model + context_length * context_length
    ffn_activation = context_length * d_model + 2 * context_length * d_ff
    total_activation = num_layers * (norm_activation + att_activation + ffn_activation)
    print('total_param', total_param)
    print('total_activation', total_activation)
    print('total mem(GB)', (total_param * 16 + batch * total_activation * 4) / 1024 / 1024 / 1024)
    pass


def train_demo(lr=1.):
    weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
    opt = SGD([weights], lr=lr)
    print('#' * 30, lr, '#' * 30)
    for t in range(10):
        opt.zero_grad()                 # Reset the gradients for all learnable parameters.
        loss = (weights**2).mean()      # Compute a scalar loss value.
        print(loss.cpu().item())
        if t % 30 == 1:
            print(t, weights)
            print(opt.param_groups)
        loss.backward()                 # Run backward pass, which computes gradients.
        opt.step()                      # Run optimizer step.


def test_train():
    # train_demo()
    # train_demo(lr=1e1)
    # train_demo(lr=1e2)
    # train_demo(lr=1e3)

    resource_use()


def get_args():
    parser = argparse.ArgumentParser(description="Train a Transformer model argument.")
    parser.add_argument("--vocab_size", type=int, default=10000, help="Model vocab_size")
    parser.add_argument("--context_length", type=int, default=128, help="Model context_length")
    parser.add_argument("--num_layers", type=int, default=12, help="Model num_layers")
    parser.add_argument("--num_heads", type=int, default=8, help="Model num_heads")
    parser.add_argument("--d_model", type=int, default=128, help="Model dimension d_model")
    parser.add_argument("--d_ff", type=int, default=512, help="Model dimension d_ff")

    return parser


def _train(args):
    vocab_size = args.vocab_size
    context_length = args.context_length
    num_layers = args.num_layers
    num_heads = args.num_heads
    d_model = args.d_model
    d_ff = args.d_ff
    pass


if __name__ == '__main__':
    parser = get_args()
    args = parser.parse_args()
    _train(args)
    pass




