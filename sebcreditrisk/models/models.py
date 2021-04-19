import numpy as np
import torch
from scipy.stats import norm


class tasche(torch.nn.Module):
    def __init__(self, a_0, b_0):
        super(tasche, self).__init__()
        self.alpha = torch.nn.Parameter(torch.tensor(a_0))
        self.beta = torch.nn.Parameter(torch.tensor(b_0))

    def forward(self, px, fn_inv):
        x = torch.tensor(fn_inv)
        px = torch.tensor(px)

        pdx = 1 / (1 + torch.exp(self.alpha + self.beta * x))
        pd = torch.sum(pdx * px)
        pxd = pdx * px / pd

        pn = 1 - pd
        pnx = 1 - pdx
        pxn = pnx * px / pn

        ar = torch.sum(pxn[:-1] * torch.flip(torch.cumsum(torch.flip(pxd[1:], [0]), dim=0), [0])) - torch.sum(pxn[1:] * torch.cumsum(pxd[:-1], dim=0))

        return torch.hstack((pd, ar))


def qmm(m):
    px = np.sum(m, axis=1) / np.sum(m)
    pd = np.sum(m[:,-1]) / np.sum(m)
    pn = 1 - pd

    pdx = m[:,-1] / np.sum(m, axis=1)
    pnx = 1 - pdx
    pxd = pdx * px / pd
    pxn = pnx * px / pn

    ar = np.sum(pxn[:-1] * np.flip(np.cumsum(np.flip(pxd[1:])))) - np.sum(pxn[1:] * np.cumsum(pxd[:-1]))

    fn = (2 * np.flip(np.cumsum(np.flip(pxn))) - pxn)/2
    fn_inv = norm.ppf(fn)

    mu = np.mean(fn_inv)
    tau = np.std(fn_inv)
    c = np.sqrt(2) * norm.ppf((ar + 1) / 2)
    sigma = np.sqrt((tau ** 2) / (1 + pd * pn * c ** 2))
    mu_n = mu + pd * sigma * c
    mu_d = mu - pn * sigma * c

    alpha_0 = (mu_d ** 2 - mu_n ** 2) / (2 * sigma ** 2) + np.log(pn / pd)
    beta_0 = (mu_n - mu_d) / sigma ** 2 

    tasche_smoother = tasche(alpha_0, beta_0)

    mse = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(tasche_smoother.parameters(), lr=0.1)
    epoch = 0
    stop = False
    loss_prev = 0

    while not stop:
        epoch = epoch + 1
        optimizer.zero_grad()
        res = tasche_smoother(px, fn_inv)
        loss = mse(torch.tensor([pd, ar]), res)
        loss.backward()
        optimizer.step()
        loss_current = loss.item()
        stop = np.abs(loss_prev - loss_current) / loss_current < 10 ** (-10) or epoch >= 1000
        loss_prev = loss_current

    alpha = tasche_smoother.alpha.item()
    beta = tasche_smoother.beta.item()

    return 1 / (1 + np.exp(alpha + beta * fn_inv))
