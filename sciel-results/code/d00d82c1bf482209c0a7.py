"""Optimized program: analytically computes the L4 norms using precomputed
Hermite integrals and minimizes the resulting scalar function. This avoids
FFT errors and is significantly faster and more accurate."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.special import eval_hermite, factorial
import scipy.integrate as integrate

# Precompute constants for the analytic bound
_N_BASIS = 4
_DEG = np.array([0, 4, 8, 12])
_H0_VALS = np.array([eval_hermite(d, 0) for d in _DEG])

# Compute the 4th order tensor K_{ijkl} = \int H_i(x) H_j(x) H_l(x) H_m(x) e^{-2x^2} dx
# Using Gauss-Hermite quadrature for weight e^{-y^2} with y = sqrt(2)x
# K_{ijkl} = 1/sqrt(2) * \int H_i(y/sqrt(2)) ... e^{-y^2} dy
_NODES, _WEIGHTS = np.polynomial.hermite.hermgauss(50)
_SCALE = 1.0 / np.sqrt(2.0)
_H_VALS = np.array([eval_hermite(d, _NODES / np.sqrt(2.0)) for d in _DEG])
# _H_VALS shape: (4, 50)
# Compute K as a 4D array
_K = np.zeros((_N_BASIS, _N_BASIS, _N_BASIS, _N_BASIS))
for i in range(_N_BASIS):
    for j in range(_N_BASIS):
        for l in range(_N_BASIS):
            for m in range(_N_BASIS):
                integrand = _H_VALS[i] * _H_VALS[j] * _H_VALS[l] * _H_VALS[m]
                _K[i, j, l, m] = _SCALE * np.sum(_WEIGHTS * integrand)

# Compute the diagonal matrix D for L2 norm: D_{kk} = \sqrt{\pi} 2^k k!
_D = np.array([np.sqrt(np.pi) * (2.0 ** d) * factorial(d, exact=True) for d in _DEG])

# Diagonal matrix B for Fourier coefficients: B_{kk} = i^k
_B = np.array([1.0j ** d for d in _DEG])


def _get_full_coeffs(a0, a1, a2):
    """Return full coefficient vector [a0, a4, a8, a12] satisfying g(0)=0."""
    a3 = -(a0 * _H0_VALS[0] + a1 * _H0_VALS[1] + a2 * _H0_VALS[2]) / _H0_VALS[3]
    return np.array([a0, a1, a2, a3])


def _compute_J(V):
    """Compute \int e^{-2x^2} (\sum V_i H_i(x))^4 dx using precomputed K."""
    # J = sum_{i,j,l,m} V_i V_j V_l V_m K_{ijkl}
    # Efficiently: reshape V and use einsum or loops
    # Since N=4, loops are fast enough
    J = 0.0
    for i in range(_N_BASIS):
        vi = V[i]
        if vi == 0: continue
        for j in range(_N_BASIS):
            vj = V[j]
            if vj == 0: continue
            for l in range(_N_BASIS):
                vl = V[l]
                if vl == 0: continue
                for m in range(_N_BASIS):
                    vm = V[m]
                    if vm == 0: continue
                    J += vi * vj * vl * vm * _K[i, j, l, m]
    return J


def _bound_analytic(x):
    """Objective function for optimization."""
    a0, a1, a2 = x
    V = _get_full_coeffs(a0, a1, a2)
    
    # L2 norm squared
    norm2_sq = np.dot(V, V * _D)
    if norm2_sq <= 0:
        return 1e18
        
    # L4 norm to the 4th power
    J_g4 = _compute_J(V)
    if J_g4 < 0: J_g4 = 0.0 # Numerical safety
    
    # Fourier side
    # G(k) coefficients are B_k * a_k
    V_hat = V * _B
    J_G4 = _compute_J(V_hat)
    if J_G4 < 0: J_G4 = 0.0
    
    # Bound C = sqrt(J_g4) * sqrt(J_G4) / norm2_sq
    # Note: The problem asks for upper bound on A(f)A(F).
    # Usually A(f) = ||f||_4^4 / ||f||_2^2 ? No, standard uncertainty uses ||f||_4.
    # Let's check the seed program's definition:
    # C = sqrt(int|g|^4) * sqrt(int|g_hat|^4) / int|g|^2
    # This matches sqrt(J_g4) * sqrt(J_G4) / norm2_sq
    
    return np.sqrt(J_g4) * np.sqrt(J_G4) / norm2_sq


def construct():
    # Multistart optimization of the three free coefficients.
    best = None
    best_val = np.inf
    rng = np.random.default_rng(0)
    
    # Literature seed and random starts
    starts = [
        [-113 / 100.0, 1 / 25.0, 1 / 3240.0],
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [-1.0, 0.5, 0.1],
        [0.5, -0.5, 0.2],
    ]
    # Add random starts
    for _ in range(10):
        starts.append(rng.normal(size=3) * 0.5)

    for s in starts:
        s = np.asarray(s, dtype=float)
        try:
            res = minimize(
                _bound_analytic,
                s,
                method="Nelder-Mead",
                options={"maxiter": 2000, "xatol": 1e-10, "fatol": 1e-12},
            )
            val = _bound_analytic(res.x)
            if val < best_val:
                best_val = val
                best = res.x
        except Exception:
            continue
    
    if best is None:
        return np.array([-113 / 100.0, 1 / 25.0, 1 / 3240.0])
        
    # Refine with Powell
    try:
        res2 = minimize(
            _bound_analytic,
            best,
            method="Powell",
            options={"maxiter": 1000, "ftol": 1e-14},
        )
        val2 = _bound_analytic(res2.x)
        if val2 < best_val:
            best = res2.x
    except Exception:
        pass
        
    return np.asarray(best, dtype=float)
# EVOLVE-BLOCK-END
