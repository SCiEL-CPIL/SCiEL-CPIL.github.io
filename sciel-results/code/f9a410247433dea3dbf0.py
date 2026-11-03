import numpy as np
from scipy.optimize import minimize, differential_evolution
import time


def construct():
    t_start = time.time()
    deadline = 200.0  # safety margin from 210s

    # Hermite polynomial values at 0 for the constraint P(0) = 0
    H0_0 = 1.0
    H4_0 = 12.0
    H8_0 = 1680.0
    H12_0 = 665280.0

    # Coefficients for H0, H4, H8, H12 as arrays indexed by power//2
    # H0(x) = 1
    # H4(x) = 12 - 48x^2 + 16x^4
    # H8(x) = 1680 - 32256x^2 + 23040x^4 - 4608x^6 + 256x^8
    # H12(x) = 3991680 - 12355200x^2 + 13271040x^4 - 5806080x^6 + 1032192x^8 - 98304x^10 + 4096x^12
    H0_arr = np.array([1.0])
    H4_arr = np.array([12.0, -48.0, 16.0])
    H8_arr = np.array([1680.0, -32256.0, 23040.0, -4608.0, 256.0])
    H12_arr = np.array([3991680.0, -12355200.0, 13271040.0, -5806080.0,
                        1032192.0, -98304.0, 4096.0])

    # Precompute ∫ x^{2k} e^{-πx²} dx for k=0..12
    # = (2k-1)!! / (2π)^k
    E_int = np.zeros(13)
    E_int[0] = 1.0
    for k in range(1, 13):
        df = 1.0
        for j in range(1, 2 * k, 2):
            df *= j
        E_int[k] = df / (2.0 * np.pi) ** k

    # Signs for Q: Q[m] gets coefficient from P[2m] * (-1)^m
    signs = np.array([(-1.0) ** m for m in range(7)])

    def objective(params):
        a0, a4, a8 = params
        a12 = -(a0 * H0_0 + a4 * H4_0 + a8 * H8_0) / H12_0

        # Build P as array of coefficients for powers 0, 2, 4, ..., 12
        P = np.zeros(7)
        P[0] += a0 * H0_arr[0]
        for m in range(3):
            P[m] += a4 * H4_arr[m]
        for m in range(5):
            P[m] += a8 * H8_arr[m]
        for m in range(7):
            P[m] += a12 * H12_arr[m]

        # ||g||₂² = sum_{i,j} P[i]*P[j] * E_int[i+j]
        P2 = np.convolve(P, P)
        norm_sq = np.dot(P2, E_int)

        if norm_sq <= 0:
            return 1e30

        # Build Q[m] = P[m] * (-1)^m
        Q = P * signs
        Q2 = np.convolve(Q, Q)
        norm_sq_hat = np.dot(Q2, E_int)

        if norm_sq_hat <= 0:
            return 1e30

        return norm_sq * norm_sq_hat

    # Initial guess from the seed program
    x0 = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])

    best_result = None
    best_score = np.inf

    # First, try differential evolution for global optimization
    if time.time() < deadline - 60:
        try:
            bounds = [(-5.0, 5.0), (-1.0, 1.0), (-0.1, 0.1)]
            de_result = differential_evolution(
                objective,
                bounds,
                maxiter=1000,
                tol=1e-12,
                seed=42,
                polish=True,
                init='sobol',
                popsize=15
            )
            if de_result.fun < best_score:
                best_score = de_result.fun
                best_result = de_result.x.copy()
        except Exception:
            pass

    # Then, try multiple local optimizations from different starting points
    start_points = [
        x0,
        np.array([-1.0, 0.1, 0.01]),
        np.array([0.0, 0.0, 0.0]),
        np.array([-0.5, 0.05, 0.005]),
        np.array([-1.2, 0.05, 0.001]),
        np.array([-0.8, 0.04, 0.002]),
        np.array([-1.1, 0.08, 0.003]),
        np.array([-0.9, 0.06, 0.004]),
    ]

    for sp in start_points:
        if time.time() > deadline - 30:
            break

        try:
            result = minimize(
                objective,
                sp,
                method='Nelder-Mead',
                options={'maxiter': 20000, 'xatol': 1e-14, 'fatol': 1e-16,
                         'adaptive': True, 'initial_simplex': None}
            )

            if result.fun < best_score:
                best_score = result.fun
                best_result = result.x.copy()

        except Exception:
            continue

    # Final high-precision polish pass with BFGS if time permits
    if best_result is not None and time.time() < deadline - 10:
        try:
            result = minimize(
                objective,
                best_result,
                method='BFGS',
                options={'maxiter': 10000, 'gtol': 1e-16, 'disp': False}
            )
            if result.fun < best_score:
                best_score = result.fun
                best_result = result.x.copy()
        except Exception:
            pass

    # Another NM polish with very tight tolerance
    if best_result is not None and time.time() < deadline - 5:
        try:
            result = minimize(
                objective,
                best_result,
                method='Nelder-Mead',
                options={'maxiter': 50000, 'xatol': 1e-16, 'fatol': 1e-18,
                         'adaptive': True}
            )
            if result.fun < best_score:
                best_score = result.fun
                best_result = result.x.copy()
        except Exception:
            pass

    # If no valid result, fall back to seed
    if best_result is None:
        return np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])

    return best_result
