import numpy as np
import scipy.special
import time
from scipy.optimize import minimize
from scipy.special import roots_genlaguerre

def construct():
    start_time = time.time()
    deadline = start_time + 580.0

    def time_remaining():
        return deadline - time.time()

    try:
        m, n = 12, 1
        N_ROOTS = 12

        seed_roots = np.array([4.0 * (i + 1) for i in range(N_ROOTS)])

        def compute_bound(roots):
            """
            Computes a proxy for the uncertainty bound C4.
            Uses the function f(x) = exp(-x^2/4) * prod(x^2 - r_i).
            Minimizes sigma_x * sigma_k.
            """
            roots = np.sort(np.asarray(roots, dtype=float))
            if np.any(roots <= 0):
                return 1e10

            try:
                coeffs = np.poly(roots)
            except Exception:
                return 1e10

            # 1. Position Space Moments (x^2)
            n_quad = 100
            nodes_0, weights_0 = roots_genlaguerre(n_quad, -0.5)
            Q_vals_0 = np.polyval(coeffs, 2.0 * nodes_0)
            norm_sq = (2.0 ** -0.5) * np.sum(weights_0 * Q_vals_0**2)
            
            if norm_sq < 1e-15:
                return 1e10

            nodes_2, weights_2 = roots_genlaguerre(n_quad, 0.5)
            Q_vals_2 = np.polyval(coeffs, 2.0 * nodes_2)
            mean_x2_int = (2.0 ** 0.5) * np.sum(weights_2 * (2.0 * nodes_2) * Q_vals_2**2)
            mean_x2 = mean_x2_int / norm_sq
            sigma_x = np.sqrt(mean_x2)

            # 2. Frequency Space Moments (k^2)
            N_fft = 8192
            L = 20.0
            x_fft = np.linspace(-L, L, N_fft)
            dx = x_fft[1] - x_fft[0]
            
            y_fft = x_fft**2
            Q_fft = np.polyval(coeffs, y_fft)
            f_fft = np.exp(-x_fft**2 / 4.0) * Q_fft
            
            F = np.fft.fft(f_fft)
            F = np.fft.fftshift(F)
            k = np.fft.fftshift(np.fft.fftfreq(N_fft, d=dx))
            k_rad = 2.0 * np.pi * k
            
            F_sq = np.abs(F)**2
            dk_rad = k_rad[1] - k_rad[0]
            
            norm_k_rad = np.sum(F_sq) * dk_rad
            if norm_k_rad < 1e-15:
                return 1e10
            
            k2_F_sq = k_rad**2 * F_sq
            mean_k2_rad = np.sum(k2_F_sq) * dk_rad / norm_k_rad
            sigma_k = np.sqrt(mean_k2_rad)
            
            return sigma_x * sigma_k

        def objective(params):
            S, alpha = params
            if S <= 0:
                return 1e10
            try:
                nodes = roots_genlaguerre(N_ROOTS, alpha)[0]
                roots = S * nodes
                if np.any(roots <= 0):
                    return 1e10
                return compute_bound(roots)
            except Exception:
                return 1e10

        x0 = [2.0, 0.0]
        
        best_score = compute_bound(seed_roots)
        best_roots = seed_roots

        if time_remaining() > 30:
            try:
                res = minimize(
                    objective,
                    x0,
                    method='Nelder-Mead',
                    options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10}
                )
                if res.fun < best_score:
                    best_score = res.fun
                    S, alpha = res.x
                    nodes = roots_genlaguerre(N_ROOTS, alpha)[0]
                    best_roots = S * nodes
            except Exception:
                pass
        
        if time_remaining() > 60:
            try:
                res2 = minimize(
                    objective,
                    [S, alpha],
                    method='Nelder-Mead',
                    options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12}
                )
                if res2.fun < best_score:
                    best_score = res2.fun
                    S, alpha = res2.x
                    nodes = roots_genlaguerre(N_ROOTS, alpha)[0]
                    best_roots = S * nodes
            except Exception:
                pass

        best_roots = np.sort(best_roots)
        best_roots = np.maximum(best_roots, 1e-6)
        
        if len(best_roots) != N_ROOTS or np.any(~np.isfinite(best_roots)):
            return [4.0 * (i + 1) for i in range(N_ROOTS)]

        return [float(r) for r in best_roots]

    except Exception:
        return [4.0 * (i + 1) for i in range(N_ROOTS)]

if __name__ == "__main__":
    construct()
