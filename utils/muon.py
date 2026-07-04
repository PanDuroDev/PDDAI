import torch


def zeropower_via_newtonschulz5(G, steps=5):
    """Newton-Schulz iteration for orthogonalization (Muon core).
    Works for both square and non-square matrices."""
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.clone()
    X = X / (X.norm() + 1e-8)
    for _ in range(steps):
        S = X.T @ X
        X = a * X + b * X @ S + c * X @ S @ S
    return X


class Muon(torch.optim.Optimizer):
    """
    Muon optimizer for 2D weights + AdamW for others.
    Usage:
        param_groups = [
            {'params': hidden_2d, 'use_muon': True, 'lr': 0.02, 'weight_decay': 0.01},
            {'params': other, 'use_muon': False, 'lr': 3e-4, 'betas': (0.9, 0.95), 'weight_decay': 0.01},
        ]
        opt = Muon(param_groups)
    """

    def __init__(self, param_groups):
        defaults = {'use_muon': False, 'lr': 3e-4, 'betas': (0.9, 0.95),
                     'weight_decay': 0.01, 'momentum': 0.95, 'nesterov': True}
        super().__init__(param_groups, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            use_muon = group.get('use_muon', False)
            lr = group['lr']
            wd = group.get('weight_decay', 0.0)

            if use_muon:
                momentum = group.get('momentum', 0.95)
                nesterov = group.get('nesterov', True)
                for p in group['params']:
                    if p.grad is None:
                        continue
                    g = p.grad.data
                    state = self.state[p]

                    if 'momentum_buffer' not in state:
                        state['momentum_buffer'] = torch.zeros_like(g)
                    buf = state['momentum_buffer']

                    buf.mul_(momentum).add_(g)

                    if nesterov:
                        g = g.add(buf, alpha=momentum)
                    else:
                        g = buf

                    if wd > 0:
                        p.data.mul_(1 - lr * wd)

                    if g.ndim >= 2:
                        g_ortho = zeropower_via_newtonschulz5(g)
                        p.data.add_(g_ortho, alpha=-lr)
                    else:
                        p.data.add_(g, alpha=-lr)
            else:
                betas = group.get('betas', (0.9, 0.95))
                for p in group['params']:
                    if p.grad is None:
                        continue
                    g = p.grad.data
                    state = self.state[p]

                    if 'step' not in state:
                        state['step'] = 0
                        state['exp_avg'] = torch.zeros_like(g)
                        state['exp_avg_sq'] = torch.zeros_like(g)

                    state['step'] += 1
                    exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                    b1, b2 = betas

                    exp_avg.mul_(b1).add_(g, alpha=1 - b1)
                    exp_avg_sq.mul_(b2).addcmul_(g, g, value=1 - b2)

                    step = state['step']
                    bias_corr1 = 1 - b1 ** step
                    bias_corr2 = 1 - b2 ** step

                    denom = (exp_avg_sq.sqrt() / bias_corr2 ** 0.5).add_(1e-8)

                    if wd > 0:
                        p.data.mul_(1 - lr * wd)
                    p.data.addcdiv_(exp_avg / bias_corr1, denom, value=-lr)
