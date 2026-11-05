"""
Strategy: Frequency Domain Parameterization with Low-Dimensional Optimization.

Instead of optimizing 400 variables (h[0]...h[399]) directly, we represent the
step function h as a sum of a constant offset and a few low-frequency Fourier
modes.
    h[k] = offset + sum_{i=1}^{M} (a_i * cos(2*pi*i*k/N) + b_i * sin(2*pi*i*k/N))

This reduces the optimization space from 400 dimensions to 2*M + 1 dimensions.
We use FFT to compute the auto-correlation efficiently.
The objective is to minimize V = abs(2 * N * max(conv(h,h)) / sum(h)^2).

Since the problem is scale-invariant (scaling h by c scales conv by c^2 and sum^2 by c^2, so V is invariant),
we can fix the sum of h to a specific value (e.g., 1) or just let the optimizer handle it.
However, fixing the sum simplifies the objective to minimizing max(conv(h,h)) subject to sum(h)=1.
Actually, the formula is:
V = | 2 * N * max(conv) / sum(h)^2 |

If we normalize h such that sum(h) = 1, then V = | 2 * N * max(conv) |.
So we minimize max(conv) subject to sum(h) = 1.

We will use a parametric family:
h[k] = 1/N + sum_{i=1}^{M} c_i * phi_i(k)
where phi_i are orthogonal basis functions (cos/sin) and we ensure sum(h) is constant.
Actually, if we add zero-mean components, sum(h) remains N * (1/N) = 1.
So let h[k] = 1/N + sum_{i=1}^{M} (a_i cos(...) + b_i sin(...)).
Then sum(h) = 1.
Objective: minimize 2 * N * max(conv(h,h)).

We will use scipy.optimize.minimize with L-BFGS-B on the coefficients.
To handle the non-differentiability of max(), we can use a soft-max approximation
or simply rely on the fact that the maximum of a smooth function (conv is smooth in coeffs)
is often smooth enough for gradient-based methods if we are in the right basin.
Alternatively, we can use a penalty method or just evaluate the true max.
Since the dimension is low (e.g., 20), we can even use a global optimizer like Differential Evolution
to find a good basin, then refine with L-BFGS-B.
"""

N = 400
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

_DEADLINE = 140.0  # Wall-clock limit with safety margin

def _compute_h_from_coeffs(coeffs, M):
    """
    Reconstruct h from coefficients.
    coeffs: [a_1, b_1, ..., a_M, b_M]
    h[k] = 1/N + sum_{i=1}^{M} (a_i * cos(2*pi*i*k/N) + b_i * sin(2*pi*i*k/N))
    This ensures sum(h) = 1.
    """
    k = np.arange(N)
    h = np.full(N, 1.0 / N)
    for i in range(1, M + 1):
        a = coeffs[2 * (i - 1)]
        b = coeffs[2 * (i - 1) + 1]
        h += a * np.cos(2 * np.pi * i * k / N)
        h += b * np.sin(2 * np.pi * i * k / N)
    return h

def _evaluate_h(h):
    """Compute the exact task value."""
    s = np.sum(h)
    if s == 0 or not np.isfinite(s):
        return np.inf
    conv = np.convolve(h, h, mode='full')
    return abs(2.0 * N * np.max(conv) / (s * s))

def _objective(coeffs, M):
    """
    Objective function for optimization.
    We minimize the value V.
    To make it differentiable for L-BFGS-B, we use a soft-max approximation
    for the max operation, but since the dimension is low, we can also just
    use the hard max if we use a global optimizer first.
    For L-BFGS-B, soft-max is better.
    """
    h = _compute_h_from_coeffs(coeffs, M)
    s = np.sum(h)
    if s == 0 or not np.isfinite(s):
        return 1e10
    
    # Compute auto-correlation via FFT for speed
    # Note: np.convolve(h, h) is linear convolution.
    # FFT-based convolution:
    # We need zero-padding to length 2N-1.
    f_len = 2 * N - 1
    h_fft = np.fft.rfft(h, n=f_len)
    conv_fft = h_fft * h_fft
    conv = np.fft.irfft(conv_fft, n=f_len)
    
    # The objective is | 2 * N * max(conv) / s^2 |
    # Since s is fixed to 1 in our parametrization (sum of 1/N is 1, sum of zero-mean terms is ~0),
    # s^2 is 1.
    # We want to minimize max(conv).
    
    # Use soft-max to approximate max(conv)
    beta = 100.0
    m = np.max(conv)
    # Shift for stability
    conv_shifted = conv - m
    weights = np.exp(beta * conv_shifted)
    soft_max = m + np.log(np.sum(weights)) / beta
    
    val = abs(2.0 * N * soft_max / (s * s))
    return val

def _refine_local(coeffs_init, M, deadline):
    """Refine a set of coefficients using L-BFGS-B."""
    try:
        res = minimize(
            _objective,
            coeffs_init,
            args=(M,),
            method='L-BFGS-B',
            options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 500}
        )
        return res.x
    except Exception:
        return coeffs_init

def construct():
    """
    Construct step heights to minimize the autocorrelation inequality value.
    """
    t0 = time.time()
    deadline = t0 + _DEADLINE
    
    # Choose number of Fourier modes.
    # Low frequencies are usually where the "mass" of the correlation peak resides
    # for smooth-ish functions. High frequencies tend to cancel out in the sum.
    # Let's try M=10 (20 variables) and M=20 (40 variables).
    
    best_val = np.inf
    best_h = np.ones(N)
    
    # 1. Global Optimization with Differential Evolution on a small set of modes
    # This helps escape local minima.
    
    M_global = 5  # 10 variables
    bounds = [(-0.1, 0.1)] * (2 * M_global)
    
    if time.time() < deadline - 20:
        try:
            # Differential evolution is stochastic and can be slow, but for 10 vars it's fast.
            # We use a small population to save time.
            de_res = differential_evolution(
                _objective,
                bounds,
                args=(M_global,),
                maxiter=100,
                popsize=15,
                tol=1e-6,
                seed=42,
                polish=False  # We will polish manually
            )
            if de_res.success or de_res.fun < best_val:
                coeffs_best = de_res.x
                h_candidate = _compute_h_from_coeffs(coeffs_best, M_global)
                v_candidate = _evaluate_h(h_candidate)
                if v_candidate < best_val:
                    best_val = v_candidate
                    best_h = h_candidate
        except Exception:
            pass

    # 2. Local Refinement with L-BFGS-B on a larger set of modes
    # We can start from the global result (padded with zeros) or random inits.
    
    M_local = 15  # 30 variables
    
    # Initialize from the best found so far (if it has enough modes) or random
    if len(de_res.x) > 0:
        init_coeffs = np.zeros(2 * M_local)
        init_coeffs[:2 * M_global] = de_res.x
    else:
        init_coeffs = np.zeros(2 * M_local)
        
    # Try a few random initializations for local refinement
    rng = np.random.default_rng(42)
    inits = [init_coeffs]
    for _ in range(5):
        inits.append(rng.normal(0, 0.05, 2 * M_local))
        
    for init in inits:
        if time.time() > deadline - 10:
            break
        
        refined_coeffs = _refine_local(init, M_local, deadline - 5)
        h_candidate = _compute_h_from_coeffs(refined_coeffs, M_local)
        v_candidate = _evaluate_h(h_candidate)
        
        if v_candidate < best_val:
            best_val = v_candidate
            best_h = h_candidate

    # 3. Optional: Try even higher modes if time permits
    if time.time() < deadline - 30:
        M_high = 30
        # Use the best coefficients found so far as a starting point (padded)
        if len(refined_coeffs) > 0:
            init_high = np.zeros(2 * M_high)
            init_high[:min(2*M_local, 2*M_high)] = refined_coeffs[:min(2*M_local, 2*M_high)]
        else:
            init_high = np.zeros(2 * M_high)
            
        refined_high = _refine_local(init_high, M_high, deadline - 5)
        h_candidate = _compute_h_from_coeffs(refined_high, M_high)
        v_candidate = _evaluate_h(h_candidate)
        
        if v_candidate < best_val:
            best_val = v_candidate
            best_h = h_candidate

    # Final check
    if not np.all(np.isfinite(best_h)) or np.sum(best_h) == 0:
        return np.ones(N)
        
    return best_h
