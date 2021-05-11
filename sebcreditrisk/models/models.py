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
    Fucntion for quasi moment matching. The formulation can be found in Tasche, 2013.

    Parameters:
    m: ndarray
        Transition matrix (count)

    Return:
    Smoothed PD vector (arraylike)
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
        stop = np.abs(loss_prev - loss_current) < 1e-10 * np.abs(loss_prev) or epoch >= 10000
        loss_prev = loss_current

    alpha = tasche_smoother.alpha.item()
    beta = tasche_smoother.beta.item()

    return 1 / (1 + np.exp(alpha + beta * fn_inv))


class MertonVasicek(torch.nn.Module):
    '''
    Simple class for calibrating the rho factor for (multifactor) Merton-Vasicek model

    Attributes:
    rho_dim: int
        1 for the case of single rho factor for the whole portfolio, otherwise the number of ratings
    z_dim: int
        Number of indicators, 1 for single factor model
    I: ndarray
        Sign indicator for correlation between PD and Z
    w: narray
        Weights for indicators (weights will go through softmax transformation so that they sum to 1)
    rho: float or arraylike
        Sensitivity factor
    '''

    def __init__(self, rho_dim, z_dim, I, rho0):
        '''
        Constructor for the class

        Parameters:
        rho_dim: int, required
        z_dim: int, required
        I: ndarray, required
        rho0: float, required
            initial value for rho in (0,1)
        '''

        super(MertonVasicek, self).__init__()
        self.rho_dim = rho_dim
        self.z_dim = z_dim
        self.I = I
        self.w = torch.nn.Parameter(torch.ones(1, self.z_dim).double())
        self.rho = torch.nn.Parameter(torch.ones(1, self.rho_dim).double()*rho0)
        
    def forward(self, odf_ttc, z):
        '''
        Forward function for calculating PD

        Parameters:
        odf_ttc: arraylike, float, required
            Through-the-cycle mean ODF or PD, float if dim = 1, array of of length dim otherwise
        z: arraylike, ndarray, required
            Economic indicators (see fit_mv function)
        '''

        # Transfrom numpy inputs to tensors
        if not torch.is_tensor(odf_ttc):
            odf_ttc = torch.tensor(odf_ttc.copy()).reshape(-1)
        if not torch.is_tensor(z):
            z = torch.tensor(z.copy()).reshape(-1, self.z_dim)

        # Applying softmax to raw weights
        w_n = torch.sqrt(torch.nn.functional.softmax(self.w, dim=1))

        # MV, 10e-10 is added to the pd to make sure the optimization process is more numerically stable
        N = torch.distributions.normal.Normal(0,1)
        pd = N.cdf((N.icdf(odf_ttc) + (z * w_n) @ self.I * torch.sqrt(self.rho)) / torch.sqrt(1 - self.rho)) + torch.tensor(10e-10)

        return pd

    def get_rho(self):
        return self.rho.detach().numpy()

    def get_weight(self):
        return torch.sqrt(torch.nn.functional.softmax(self.w, dim=1)).detach().numpy()

    def get_pd_ttc(self, odf, z):
        '''
        Reverse function to eliminate the Z factor from historic PD

        Parameters:
        odf: ndarray, required
            Observed default frequency (smoothed by qmm in the case of seperate rho)
        z: ndarray, required
        '''

        if not torch.is_tensor(odf):
            odf = torch.tensor(odf.copy())
        if not torch.is_tensor(z):
            z = torch.tensor(z.copy()).reshape(-1, self.z_dim)

        w_n = torch.sqrt(torch.nn.functional.softmax(self.w, dim=1))

        N = torch.distributions.normal.Normal(0,1)
        odf_trend_removed = N.cdf(N.icdf(odf) * torch.sqrt(1 - self.rho) - (z * w_n) @ self.I * torch.sqrt(self.rho))
        if self.rho_dim ==1:
            pd_ttc = torch.mean(odf_trend_removed)
        else:
            pd_ttc = torch.mean(odf_trend_removed, axis=0)

        return pd_ttc.detach().numpy()


def nllloss(pd, odf):
    '''
    Negative loglikelihood loss

    Parameters:
    pd: ndarray, required
        Predicted PD
    odf: ndarray, required
        Historical ODF 
    '''

    if odf.ndim == 1:
        pd = pd.reshape(-1)
    loss = - torch.sum(odf * torch.log(pd) + (1 - odf) * torch.log(1 - pd))
    return loss


def fit_mv(transitions, z, rating_level_rho=False, rho0=0.05, lr=0.00001, max_epochs=100000):
    '''Function for fitting MV model

    Parameters:
    transitions: list, required
        List of yearly transition matrices in chronological order
    z: ndarray, required
        Economic indicators, array of length N (years) for single indicator, matrix of N (year) x M  (indicators) for multiple indicators
    rating_level_rho: bool, optional
        Set to True to fit separate rho for each rating
    rho0: float, optional
        Starting value for rho
    lr: float, optional
        Learning rate for rho
    max_epochs: int, optional
        Maximum number of training iterations
    '''

    # Setting inputs for MV (get dimension, and correlation sign for Z factors, calculate historical ODF)
    if z.ndim == 1:
        z_dim = 1
        z = z.reshape(-1,1)
    else:
        z_dim = z.shape[1]

    if not rating_level_rho:
        odf_h = np.array([get_pd(m) for m in transitions])
        odf_ttc = np.mean(odf_h)
        rho_dim = 1
        I = torch.tensor(np.sign([np.corrcoef(z[:,i], odf_h)[0,1] for i in range(z_dim)])).reshape(-1,1)
    else:
        odf_h = np.vstack([get_pdx(m) for m in transitions])
        odf_ttc = np.mean(odf_h, axis=0)
        rho_dim =  odf_ttc.shape[0]
        I = torch.tensor(np.sign([[np.corrcoef(odf_h[:,i], z[:,j])[0,1] for i in range(rho_dim)] for j in range(z_dim)]))

    # Historical ODF
    odf_h = torch.tensor(odf_h)

    MV = MertonVasicek(rho_dim, z_dim, I, rho0)
    MV.train()

    # Training
    optimizer = torch.optim.Adam([{'params': MV.w, 'lr':0.001}, {'params': MV.rho}], lr=lr)
    epoch = 0
    stop = False
    loss_prev = 10

    while not stop:
        epoch = epoch + 1
        optimizer.zero_grad()
        pd = MV(odf_ttc, z)
        loss = nllloss(pd, odf_h)
        loss.backward()
        optimizer.step()
        loss_current = loss.item()
        stop = np.abs(loss_prev - loss_current) < 1e-10 * np.abs(loss_prev) or epoch >= max_epochs
        loss_prev = loss_current

    return MV


def portfolio_rmse(MV, transitions, z, rating_level_rho=False):
    '''
    Calculate RMSE of predicted PD compared to historical ODF

    Parameters:
    MV: MertonVasicek, required
        Fitted MV model
    transitions: list, required
        List of observed transition matrices
    z: ndarray, required
    rating_level_rho: bool, optional
        Set to True for MV model that has separate rho for different ratings

    Return:
    rmse: float, arraylike
        RMSE of the predicted PD
    '''

    MV.eval()

    px = np.array([get_px(m) for m in transitions])
    pd_true = np.array([get_pd(m) for m in transitions])

    if not rating_level_rho:
        odf = pd_true
    else:
        odf = np.vstack([qmm(m) for m in transitions])

    pd_ttc = MV.get_pd_ttc(odf, z)    

    if not rating_level_rho:
        pd_pred = MV(pd_ttc, z).detach().numpy()
    else:
        pd_pred = np.sum(MV(pd_ttc, z).detach().numpy() * px, axis=1)

    rmse = np.sqrt(np.mean((pd_pred - pd_true) ** 2))

    return rmse
