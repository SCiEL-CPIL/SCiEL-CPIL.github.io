"""Chebyshev collocation approach to minimize uncertainty product upper bound."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_DEADLINE = time.time() + 170


def _chebyshev_setup(N):
    """
    Set up Chebyshev collocation points, weights, and derivative matrices.
    We work on [-L, L] with N+1 Chebyshev points.
    
    Returns:
        x: collocation points (N+1,)
        w: quadrature weights (N+1,)
        D1: first derivative matrix (N+1, N+1)
        D2: second derivative matrix (N+1, N+1)
    """
    N = int(N)
    L = 12.0  # domain half-width - wider to capture tails better
    
    # Chebyshev points on [-1, 1]
    j = np.arange(N + 1)
    t = np.cos(np.pi * j / N)
    t = t[::-1]  # ascending order
    x = L * t
    
    # Weights for Chebyshev quadrature
    w = np.ones(N + 1) * np.pi / N
    w[0] *= 0.5
    w[-1] *= 0.5
    w = L * w  # scale to [-L, L]
    
    # Derivative matrices
    # Standard Chebyshev differentiation matrix on [-1, 1]
    c = np.ones(N + 1)
    c[0] = 0.5
    c[-1] = 0.5
    c = c * (-1.0) ** np.arange(N + 1)
    
    X = np.tile(t, (N + 1, 1))
    dX = X - X.T
    dX[np.abs(dX) < 1e-15] = 1.0  # avoid division by zero
    
    # Off-diagonal
    D = (c[:, None] / c[None, :]) / (dX + np.eye(N + 1))
    # Diagonal
    D = D - np.diag(D.sum(axis=1))
    
    # Scale to [-L, L]
    D1 = D / L
    
    # D2 = D1 @ D1
    D2 = D1 @ D1
    
    return x, w, D1, D2


def _build_functional(N, L=12.0):
    """
    Build the quadratic forms for position and momentum variances.
    
    g(x) = P(x) * exp(-x^2/2) where P is a polynomial of degree <= 12
    with coefficients (a0, a4, a8, a12) where a12 is determined by g(0)=0.
    
    We parameterize by (a0, a4, a8) and compute a12 = -(a0 + 3*a4 + 105*a8)/945.
    """
    x, w, D1, D2 = _chebyshev_setup(N)
    
    # Hermite polynomials H_n(x) for n = 0, 4, 8, 12
    def H0(x): return np.ones_like(x)
    def H4(x): return x**4 - 6*x**2 + 3
    def H8(x): return x**8 - 28*x**6 + 210*x**4 - 420*x**2 + 105
    def H12(x): return x**12 - 66*x**10 + 1155*x**8 - 6930*x**6 + 13860*x**4 - 7560*x**2 + 945
    
    # Basis functions: phi_n(x) = H_n(x) * exp(-x^2/2)
    exp_factor = np.exp(-x**2 / 2.0)
    phi0 = H0(x) * exp_factor
    phi4 = H4(x) * exp_factor
    phi8 = H8(x) * exp_factor
    phi12 = H12(x) * exp_factor
    
    phis = np.array([phi0, phi4, phi8, phi12])  # shape (4, N+1)
    
    # M_0
    M_0 = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            M_0[i, j] = np.sum(w * phis[i] * phis[j])
    
    # M_pos
    x2 = x**2
    M_pos = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            M_pos[i, j] = np.sum(w * x2 * phis[i] * phis[j])
    
    # M_mom: <phi_i, -D2 phi_j>
    # -D2 phi_j is the action of the momentum-squared operator on phi_j
    M_mom = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            D2_phi_j = D2 @ phis[j]
            M_mom[i, j] = -np.sum(w * phis[i] * D2_phi_j)
    
    return M_0, M_pos, M_mom


def _bound_and_grad(params, M_0, M_pos, M_mom):
    """
    Compute bound and gradient analytically.
    params = (a0, a4, a8)
    a12 = -(a0 + 3*a4 + 105*a8)/945
    """
    a0, a4, a8 = params
    a12 = -(a0 + 3.0 * a4 + 105.0 * a8) / 945.0
    a = np.array([a0, a4, a8, a12])
    
    N = a @ M_0 @ a
    P = a @ M_pos @ a
    Q = a @ M_mom @ a
    
    if N <= 1e-30 or P <= 0 or Q <= 0 or not np.isfinite(N) or not np.isfinite(P) or not np.isfinite(Q):
        return 1e18, np.array([1e18, 1e18, 1e18])
    
    bound = np.sqrt(P * Q / (N * N))
    
    # da/dparams (4x3)
    da_dparams = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [-1.0/945.0, -3.0/945.0, -105.0/945.0]
    ])
    
    # dN/dparams = 2 * (M_0 @ a)^T @ da_dparams
    M0a = M_0 @ a
    Mposa = M_pos @ a
    Mmoma = M_mom @ a
    
    dN_dparams = 2.0 * (da_dparams @ M0a)  # shape (3,)
    dP_dparams = 2.0 * (da_dparams @ Mposa)
    dQ_dparams = 2.0 * (da_dparams @ Mmoma)
    
    # d(P*Q/N^2)/dparams
    # = (dP*Q + P*dQ)/N^2 - 2*P*Q*dN/N^3
    dPQN2 = (dP_dparams * Q + P * dQ_dparams) / (N * N) - 2.0 * P * Q * dN_dparams / (N * N * N)
    
    grad = dPQN2 / (2.0 * bound)
    
    if not np.all(np.isfinite(grad)):
        grad = np.array([1e18, 1e18, 1e18])
    
    return bound, grad


def construct():
    # Use Chebyshev collocation with high resolution
    N = 300  # number of intervals, N+1 points
    
    try:
        M_0, M_pos, M_mom = _build_functional(N)
    except Exception:
        # Fallback to seed values
        return np.array([-1.13, 0.04, 0.000308642])
    
    def _objective_and_grad(x):
        v, g = _bound_and_grad(x, M_0, M_pos, M_mom)
        return v, g
    
    def _objective(x):
        v, _ = _bound_and_grad(x, M_0, M_pos, M_mom)
        return v
    
    # Generate diverse starting points
    starts = [
        np.array([-1.13, 0.04, 0.000308642]),  # seed
        np.array([-1.0, 0.05, 0.0004]),
        np.array([-1.5, 0.06, 0.0006]),
        np.array([-0.5, 0.02, 0.0002]),
        np.array([-2.0, 0.08, 0.001]),
        np.array([0.0, 0.0, 0.0]),
        np.array([-0.8, 0.03, 0.0003]),
        np.array([-1.3, 0.045, 0.00035]),
        np.array([-1.2, 0.05, 0.0004]),
        np.array([-0.9, 0.035, 0.0003]),
    ]
    
    rng = np.random.default_rng(42)
    for _ in range(25):
        starts.append(rng.normal(0.0, 0.7, 3))
    
    best_val = np.inf
    best_x = None
    
    for s in starts:
        if time.time() > _DEADLINE - 25:
            break
        try:
            res = minimize(_objective_and_grad, s, method='SLSQP', jac=True,
                           options={'maxiter': 5000, 'ftol': 1e-16, 'gtol': 1e-15,
                                    'eps': 1e-9})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_x = res.x.copy()
        except Exception:
            pass
    
    # Polish with tighter tolerances
    if best_x is not None and time.time() < _DEADLINE - 15:
        try:
            res = minimize(_objective_and_grad, best_x, method='SLSQP', jac=True,
                           options={'maxiter': 8000, 'ftol': 1e-18, 'gtol': 1e-16,
                                    'eps': 1e-10})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_x = res.x.copy()
        except Exception:
            pass
    
    if best_x is None:
        return np.array([-1.13, 0.04, 0.000308642])
    
    return np.array(best_x)
# EVOLVE-BLOCK-END
