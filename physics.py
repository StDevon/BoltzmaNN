import torch
import numpy as np
from cubic_spline import TorchCubicSpline
from parameters import massRatio, g_m1, Msq, m1, mDM, g_x

from pinn import compute_gradients, smooth_max


device = "cuda" if torch.cuda.is_available() else "cpu"


# dof from arXiv:1606.07494
dof_arr = np.array(
    [
        [0.00, 10.71, 1.00228],
        [0.50, 10.74, 1.00029],
        [1.00, 10.76, 1.00048],
        [1.25, 11.09, 1.00505],
        [1.60, 13.68, 1.02159],
        [2.00, 17.61, 1.02324],
        [2.15, 24.07, 1.05423],
        [2.20, 29.84, 1.07578],
        [2.40, 47.83, 1.06118],
        [2.50, 53.04, 1.04690],
        [3.00, 73.48, 1.01778],
        [4.00, 83.10, 1.00123],
        [4.30, 85.56, 1.00389],
        [4.60, 91.97, 1.00887],
        [5.00, 102.17, 1.00750],
        [5.45, 104.98, 1.00023],
    ]
)

# Prepare data
gx = dof_arr[:, 0] - 3  # log10(T/GeV) points
gy = dof_arr[:, 1]  # g points

hy_np = dof_arr[:, 1] / dof_arr[:, 2]  # h points

# Compute log10(hy)
log_hy = np.log10(hy_np)

hslog_spline_torch = TorchCubicSpline(gx, log_hy)
dloghdlogT_spline_torch = hslog_spline_torch.derivative()


def g_tilde_torch(T):
    return (1 / 3) * dloghdlogT_spline_torch(torch.log10(T))


gy_spline_torch = TorchCubicSpline(gx, gy)


def g_rho_torch(T):
    return gy_spline_torch(torch.log10(T))


mPl = torch.tensor(1e19, dtype=torch.float64)


def Hubble_torch(T):
    return torch.sqrt(8 * (np.pi**3) * g_rho_torch(T) / (90)) * T**2 / mPl


def Hubble_tilde_torch(T):
    return Hubble_torch(T) / (1 + g_tilde_torch(T))


def feq(q):
    return q**2 / (torch.exp(q) - 1.0)


def collisionTerm_torch(x, q, f, feq):
    # in complete analogy to PyBolt collision term
    terms_Elim1 = torch.stack(
        [
            x * torch.ones_like(q),
            x * massRatio + q,
            x**2
            * (1 + 4 * q**2 / (x**2 * (1 - massRatio**2)) - massRatio**2)
            / (4 * q),
        ],
        dim=1,
    )
    Elim1 = smooth_max(terms_Elim1, dim=1)

    terms_Elim2 = torch.stack(
        [x - q, x * massRatio * torch.ones_like(q), x**2 / (4 * q)], dim=1
    )
    Elim2 = smooth_max(terms_Elim2, dim=1)

    prefactor = (2 * g_m1 * Msq * x / m1) / (8 * torch.pi * q**3)
    log_term = torch.log((1 + torch.exp(-Elim1)) / (1 + torch.exp(-Elim2)))
    return prefactor * log_term * (f - feq)


def BE_residue(f, x, q):
    eq = torch.sqrt((mDM * x / m1) ** 2 + q**2)
    df_dx, df_dq = compute_gradients(f, x, q)
    CollTerm = collisionTerm_torch(x, q, f(x, q), feq(q))
    free_BE = x * df_dx - (
        g_tilde_torch(m1 / x) * (df_dq * q - 2 * f(x, q))
    )  # both sides multiplied by x

    return free_BE + (q**2) * CollTerm / (2 * g_x * eq) / Hubble_tilde_torch(m1 / x)
