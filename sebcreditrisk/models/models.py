import numpy as np
import torch
from scipy.stats import norm
from sebcreditrisk.utils import get_px, get_pdx, get_pd


class Tasche(torch.nn.Module):
    '''
    Class for optimizing parameters (alpha and beta) for quasi moment matching based on Tasche, 2013
    '''
    def __init__(self, a_0, b_0):
        super(Tasche, self).__init__()
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
    '''
    Fucntion for quasi moment matching. The formulation can be found in Tasche, 2013
    '''
    px = get_px(m)
    pd = get_pd(m)
    pn = 1 - pd

    pdx = get_pdx(m)
    pnx = 1 - pdx
    pxd = pdx * px / pd
    pxn = pnx * px / pn

    # Accuracy ratio
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

    tasche_smoother = Tasche(alpha_0, beta_0)

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
        stop = np.abs(loss_prev - loss_current) / np.abs(loss_current) < 10 ** (-10) or epoch >= 1000
        loss_prev = loss_current

    alpha = tasche_smoother.alpha.item()
    beta = tasche_smoother.beta.item()

    return 1 / (1 + np.exp(alpha + beta * fn_inv))


class MertonVasicek(torch.nn.Module):
    '''
    Simple class for calibrating the rho factor for Merton-Vasicek model

    Attributes:
    dim: int
        1 for the case of single rho factor for the whole portfolio, otherwise the number of ratings
    rho: float or arraylike
        Sensitivity factor
    '''
    def __init__(self, dim, rho0):
        '''
        Constructor for the class

        Parameters:
        dim: int
        rho0: float
            initial value for rho in (0,1)
        '''
        super(MertonVasicek, self).__init__()
        self.dim = dim
        self.rho = torch.nn.Parameter(torch.ones(1, self.dim).double()*rho0)
        
    def forward(self, odf_ttc, z):
        if not torch.is_tensor(odf_ttc):
            odf_ttc = torch.tensor(odf_ttc.copy()).reshape(-1)
        if not torch.is_tensor(z):
            z = torch.tensor(z.copy()).reshape(-1,1)

        N = torch.distributions.normal.Normal(0,1)
        pd = N.cdf((N.icdf(odf_ttc) + z@torch.sqrt(self.rho)) / torch.sqrt(1 - self.rho)) + torch.tensor(10e-10)
        return pd

    def get_rho():
        return self.rho.item()


def nllloss(pd, odf):
    if len(odf.shape) == 1:
        pd = pd.reshape(-1)
    loss = - torch.sum(odf * torch.log(pd) + (1 - odf) * torch.log(1 - pd))
    return loss


def fit_mv(transitions, z, rating_level_rho=False, lr=0.001, rho0=0.1):
    px = np.array([get_px(m) for m in transitions])

    if not rating_level_rho:
        odf_h = np.array([get_pd(m) for m in transitions])
        odf_ttc = np.mean(odf_h)
        dim = 1
    else:
        odf_h = np.vstack([qmm(m) for m in transitions])
        odf_ttc = np.mean(odf_h, axis=0)
        dim =  odf_ttc.shape[0]

    odf_h = torch.tensor(odf_h)

    MV = MertonVasicek(dim, rho0)
    MV.train()
    optimizer = torch.optim.Adam(MV.parameters(), lr=lr)
    epoch = 0
    stop = False
    loss_prev = 10

    while not stop:
        epoch = epoch + 1
        optimizer.zero_grad()
        pd = MV(odf_ttc, z)
        #print(pd)
        loss = nllloss(pd, odf_h)
        loss.backward()
        optimizer.step()
        loss_current = loss.item()
        stop = np.abs(loss_prev - loss_current) / np.abs(loss_current) < 10 ** (-10) or epoch >= 10000
        loss_prev = loss_current
        #print(loss_current, MV.rho)

    return MV
