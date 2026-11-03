import numpy as np
from scipy.optimize import minimize, minimize_scalar, differential_evolution
import time


def construct():
    t_start = time.time()
    deadline = 205.0  # 5s safety margin from 210s

    # Hermite polynomial values at 0 for the constraint
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
    # m ranges 0..6
    signs = np.array([(-1.0) ** m for m in range(7)])

    def objective(params):
        a0, a4, a8 = params
        a12 = -(a0 * H0_0 + a4 * H4_0 + a8 * H8_0) / H12_0

        # Build P as array of coefficients for powers 0, 2, 4, ..., 12
        # P[2m] = a0*H0[2m] + a4*H4[2m] + a8*H8[2m] + a12*H12[2m]
        # Index m corresponds to power 2m
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

        # Build Q[m] = P[m] * (-1)^m
        Q = P * signs
        Q2 = np.convolve(Q, Q)
        norm_sq_hat = np.dot(Q2, E_int)

        return norm_sq * norm_sq_hat

    def objective_log(params):
        """Log of the objective for better conditioning"""
        val = objective(params)
        if val <= 0:
            return 0.0
        return np.log(val)

    # Initial guess from the seed program
    x0 = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])

    best_result = None
    best_score = np.inf

    # Try multiple starting points
    start_points = [
        x0,
        np.array([-1.0, 0.1, 0.01]),
        np.array([0.0, 0.0, 0.0]),
        np.array([-0.5, 0.05, 0.005]),
        np.array([-1.2, 0.05, 0.001]),
        np.array([-0.8, 0.04, 0.002]),
        np.array([-1.1, 0.08, 0.003]),
        np.array([-0.9, 0.06, 0.004]),
        np.array([-1.13, 0.04, 0.0003]),
        np.array([-1.14, 0.039, 0.00031]),
        np.array([-1.12, 0.041, 0.00029]),
        np.array([-1.0, 0.04, 0.0003]),
        np.array([-1.2, 0.05, 0.0004]),
        np.array([-0.9, 0.03, 0.0002]),
    ]

    for sp in start_points:
        if time.time() > deadline - 60:
            break

        try:
            result = minimize(
                objective,
                sp,
                method='Nelder-Mead',
                options={'maxiter': 30000, 'xatol': 1e-15, 'fatol': 1e-17,
                         'adaptive': True, 'initial_simplex': None}
            )

            if result.fun < best_score:
                best_score = result.fun
                best_result = result.x.copy()

        except Exception:
            continue

    # Also try BFGS from the best NM result
    if best_result is not None and time.time() < deadline - 30:
        try:
            result = minimize(
                objective,
                best_result,
                method='BFGS',
                options={'maxiter': 20000, 'gtol': 1e-18, 'disp': False}
            )
            if result.fun < best_score:
                best_score = result.fun
                best_result = result.x.copy()
        except Exception:
            pass

    # Try L-BFGS-B
    if best_result is not None and time.time() < deadline - 20:
        try:
            result = minimize(
                objective,
                best_result,
                method='L-BFGS-B',
                options={'maxiter': 20000, 'ftol': 1e-18, 'gtol': 1e-18}
            )
            if result.fun < best_score:
                best_score = result.fun
                best_result = result.x.copy()
        except Exception:
            pass

    # Try Powell's method
    if best_result is not None and time.time() < deadline - 15:
        try:
            result = minimize(
                objective,
                best_result,
                method='Powell',
                options={'maxiter': 20000, 'xtol': 1e-15, 'ftol': 1e-18}
            )
            if result.fun < best_score:
                best_score = result.fun
                best_result = result.x.copy()
        except Exception:
            pass

    # Final high-precision polish pass with Nelder-Mead
    if best_result is not None and time.time() < deadline - 5:
        try:
            result = minimize(
                objective,
                best_result,
                method='Nelder-Mead',
                options={'maxiter': 100000, 'xatol': 1e-18, 'fatol': 1e-20,
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
