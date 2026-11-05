"""
Third autocorrelation inequality: minimize 2*N*max(conv(h,h))/sum(h)^2 over h in R^N, N=400.

Strategy:
- Represent h as a linear combination of the first M DCT-II basis functions.
  Low-frequency DCT components produce signals with slowly varying autocorrelation,
  which helps minimize the peak of the autoconvolution relative to the sum squared.
- Use a smooth log-sum-exp (soft-max) surrogate for the maximum of the autoconvolution
  to enable gradient-based optimization.
- Global search with differential evolution on the DCT coefficients.
- Local refinement with L-BFGS-B on the DCT coefficients.
- Final polish: direct coordinate descent on h itself for a few iterations.
"""
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.fftpack import dct
import time

N = 400

def compute_value(h):
    """Compute the task value: 2*N*max(conv(h,h))/sum(h)^2"""
    h = np.asarray(h, dtype=float)
    s = np.sum(h)
    if abs(s) < 1e-12 or not np.all(np.isfinite(h)):
        return np.inf
    conv = np.convolve(h, h)
    return abs(2.0 * N * np.max(conv) / (s ** 2))

def make_h_from_dct(coeffs):
    """
    Convert DCT coefficient vector to step heights h of length N.
    coeffs: array of length M (first M DCT-II modes).
    """
    M = len(coeffs)
    # Create zero-padded coefficient vector for full DCT
    full_coeffs = np.zeros(N)
    full_coeffs[:M] = coeffs
    
    # Inverse DCT-II to get h
    # scipy.fftpack.dct with type=2 uses the same normalization as the forward dct
    # We need to be careful with scaling. 
    # Forward: X[k] = 2 * sum_{n=0}^{N-1} x[n] cos(pi/N (n+0.5)k)
    # Inverse: x[n] = (1/N) [ X[0]/2 + sum_{k=1}^{N-1} X[k] cos(pi/N (n+0.5)k) ]
    # scipy dct type=2 returns X. idct type=2 returns x * N.
    # So h = idct(full_coeffs, type=2) / N
    
    h = dct(full_coeffs, type=2, axis=0) / N
    return h

def soft_max_value(h, temp=1.0):
    """
    Compute a smooth surrogate for the value using log-sum-exp for the max.
    value = abs(2*N * max(conv) / s^2)
    We approximate max(conv) with sum(exp(conv/temp)) * temp (shifted for stability).
    Note: We want to minimize the max. If max is negative, abs makes it positive.
    However, typically for these problems, we want the peak to be as small as possible.
    If we can make the max negative, the abs value is |max|.
    The log-sum-exp approximates the max.
    """
    s = np.sum(h)
    if abs(s) < 1e-12 or not np.all(np.isfinite(h)):
        return np.inf
    
    conv = np.convolve(h, h)
    
    # To handle the absolute value of the max, we can consider both the max and the min (negated).
    # But usually the positive peak is the one to worry about.
    # Let's use a soft max for the positive part and soft min for the negative part?
    # Actually, the objective is abs(max(conv)).
    # If max(conv) > 0, obj ~ 2*N*max(conv)/s^2.
    # If max(conv) < 0, obj ~ -2*N*max(conv)/s^2.
    
    # A simple smooth approximation for |max(x)| is sqrt(max(x)^2 + eps).
    # Or we can use log-sum-exp on x/temp and -x/temp?
    # Let's stick to the standard soft max for the upper bound, assuming we can't easily make the max negative 
    # without making the min very negative (which doesn't help the max).
    # Actually, if the signal is symmetric and zero-mean-ish, the max of conv is at lag 0.
    # We want to reduce the peak at lag 0 relative to the sum.
    
    # Let's use a smooth max.
    # Shift conv to avoid overflow in exp
    c = conv
    c_max = np.max(c)
    log_sum_exp = c_max + temp * np.log(np.sum(np.exp((c - c_max) / temp)))
    
    # This approximates max(c).
    # We want abs(log_sum_exp).
    # If log_sum_exp is close to 0, abs is fine.
    # To make it differentiable at 0, we can use sqrt(log_sum_exp^2 + eps).
    eps = 1e-12
    smooth_max = np.sqrt(log_sum_exp**2 + eps)
    
    return abs(2.0 * N * smooth_max / (s ** 2))

def objective_from_dct(coeffs):
    """Objective function operating on DCT coefficient space."""
    h = make_h_from_dct(coeffs)
    return soft_max_value(h, temp=1.0)

def construct():
    t_start = time.time()
    DEADLINE = t_start + 140.0  # safety margin before 150s kill

    def remaining():
        return DEADLINE - time.time()

    # Number of DCT modes to use
    # Using low frequencies only. 
    # Let's try M=20.
    M = 20

    # Initial best: seed
    best_h = np.ones(N)
    best_val = compute_value(best_h)  # 2.0

    # --- Phase 1: Global search with differential evolution ---
    # Bounds for DCT coefficients
    # The DC component (k=0) controls the mean. 
    # Other components control the shape.
    bounds = []
    for k in range(M):
        if k == 0:
            # DC component. For ones, the DC coeff is large.
            # h = ones. dct(ones) -> [2*N, 0, ...] roughly?
            # Let's check scaling.
            # If h=ones, sum=400. 
            # Let's just use a wide range.
            bounds.append((-5.0, 5.0))
        else:
            bounds.append((-2.0, 2.0))

    if remaining() > 50.0:
        try:
            res_de = differential_evolution(
                objective_from_dct,
                bounds,
                seed=42,
                maxiter=200,
                popsize=15,
                tol=1e-7,
                polish=False,
                init='sobol',
                updating='immediate',
                workers=1,
            )
            if res_de.fun < best_val:
                best_val = res_de.fun
                best_h = make_h_from_dct(res_de.x)
        except Exception:
            pass

    # --- Phase 2: Multi-restart L-BFGS-B on DCT coefficient space ---
    rng = np.random.default_rng(123)
    n_restarts = 10
    for i in range(n_restarts):
        if remaining() < 15.0:
            break

        # Generate random coefficients
        coeffs_rand = np.array([rng.uniform(b[0], b[1]) for b in bounds])

        try:
            res_lbfgs = minimize(
                objective_from_dct,
                coeffs_rand,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8},
            )
            if res_lbfgs.fun < best_val:
                best_val = res_lbfgs.fun
                best_h = make_h_from_dct(res_lbfgs.x)
        except Exception:
            pass

    # --- Phase 3: Coordinate descent polish directly on h ---
    if remaining() > 10.0:
        h = best_h.copy()
        for iteration in range(50):
            if remaining() < 3.0:
                break
            improved = False
            # Randomize coordinate order
            indices = rng.permutation(N)
            for i in indices:
                if remaining() < 3.0:
                    break
                base = h[i]
                # Try a few step sizes
                for step in [0.001, -0.001, 0.005, -0.005, 0.0001, -0.0001]:
                    h[i] = base + step
                    v = compute_value(h)
                    if v < best_val:
                        best_val = v
                        improved = True
                        break
                else:
                    h[i] = base
            if not improved:
                break
        best_h = h

    # Ensure validity
    if not np.all(np.isfinite(best_h)) or abs(np.sum(best_h)) < 1e-12:
        return np.ones(N)

    return best_h


if __name__ == "__main__":
    h = construct()
    print(f"Value: {compute_value(h)}")
    print(f"Score: {2.0 / compute_value(h)}")
