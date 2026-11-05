"""
Strategy: DCT Spectral Shaping
We parameterize the step heights h using the Discrete Cosine Transform (DCT-II).
h = IDCT(c).
The objective is to minimize the peak autocorrelation value.
We use a smooth approximation of the max (log-sum-exp) on the convolution.
Optimizing in the DCT domain is advantageous because:
1. DCT is an orthogonal basis, so gradients in c-space are well-scaled.
2. Low-frequency DCT coefficients control the global shape (DC, slope), while high-frequency coefficients control local details.
3. We can use L-BFGS-B effectively in this space.
4. This avoids the local minima traps of direct time-domain coordinate descent by allowing global structural changes via low-frequency modes.

We start with a random DCT vector, optimize it, then map back to time domain for a final polish.
"""
import numpy as np
import time
import sys
from scipy.optimize import minimize

N = 400

# Precompute DCT/IDCT matrices for speed (N=400 is small enough for matrix ops, but FFT is faster)
# scipy.fft.dctn and idctn are efficient.

def _value(h):
    """Compute the exact evaluator value."""
    s = np.sum(h)
    if abs(s) < 1e-12:
        return np.inf
    conv = np.convolve(h, h)
    # The evaluator takes np.max(conv) (signed), then abs of the whole expression.
    # Note: conv[0] is sum(h^2), which is always positive and usually the largest.
    # However, the formula is abs(2*N*max(conv) / s^2).
    # If max(conv) is negative, the value is still positive.
    # Usually max(conv) is conv[0] > 0.
    return abs(2.0 * N * np.max(conv) / (s * s))

def _smooth_max(x, tau=0.05):
    """Soft max using log-sum-exp for differentiability."""
    x_max = np.max(x)
    # Shift to prevent overflow
    x_shifted = (x - x_max) / tau
    x_shifted = np.clip(x_shifted, -100, 100)
    return x_max + tau * np.log(np.sum(np.exp(x_shifted)))

def _surrogate_objective(c, tau=0.05):
    """
    Objective in DCT domain.
    c: DCT coefficients of h.
    """
    # Reconstruct h from DCT coefficients
    # scipy.fft.idctn assumes type-II DCT for dctn, and inverse is idctn
    # Note: scipy.fft.dctn normalizes by sqrt(2) factors? 
    # Actually, scipy.fft.dctn uses the convention:
    # Y[k] = sqrt(2/N) * sum_{n=0}^{N-1} x[n] cos(pi/N (n+0.5)k)
    # idctn is the exact inverse.
    h = scipy.fft.idctn(c, type=2, norm='ortho')
    
    s = np.sum(h)
    if abs(s) < 1e-4:
        return 1e10 + (1e-4 - abs(s)) * 1e6
    
    conv = np.convolve(h, h)
    sm = _smooth_max(conv, tau=tau)
    return 2.0 * N * sm / (s * s)

def _dct_obj_wrapper(c, tau):
    return _surrogate_objective(c, tau)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin

    # Initialize best
    best_h = np.ones(N)
    best_val = _value(best_h)
    
    # Phase 1: Optimize in DCT domain
    # We need an initial guess for DCT coefficients.
    # Start with a simple shape, e.g., a triangle or constant, or random.
    # A constant h has only DC component.
    # Let's try a few random initializations and pick the best start.
    
    rng = np.random.default_rng(42)
    
    # Candidate initial DCT vectors
    init_candidates = []
    
    # 1. DC only (constant h)
    c_dc = np.zeros(N)
    c_dc[0] = np.sqrt(N) # Norm ortho: IDCT of [sqrt(N), 0, ...] is ones? 
    # Check: idctn([sqrt(N), 0...]) -> 
    # Actually, let's just compute DCT of ones.
    h_ones = np.ones(N)
    c_ones = scipy.fft.dctn(h_ones, type=2, norm='ortho')
    init_candidates.append(c_ones.copy())
    
    # 2. Linear ramp (triangular wave in freq?)
    h_ramp = np.linspace(-1, 1, N)
    c_ramp = scipy.fft.dctn(h_ramp, type=2, norm='ortho')
    init_candidates.append(c_ramp.copy())
    
    # 3. Random low-pass
    c_rand = rng.normal(0, 1, N)
    # Zero out high frequencies to start smooth
    c_rand[N//4:] = 0
    init_candidates.append(c_rand)
    
    # 4. Random mid-pass
    c_rand2 = rng.normal(0, 1, N)
    c_rand2[:N//8] = 0
    c_rand2[N//2:] = 0
    init_candidates.append(c_rand2)

    best_c = init_candidates[0]
    best_c_val = _surrogate_objective(best_c, tau=0.1)
    
    for c_init in init_candidates:
        val = _surrogate_objective(c_init, tau=0.1)
        if val < best_c_val:
            best_c_val = val
            best_c = c_init.copy()
            
    # Refine best_c using L-BFGS-B
    if time.time() < deadline - 40.0:
        try:
            # Use a moderately smooth tau first
            result = minimize(
                _dct_obj_wrapper,
                best_c,
                args=(0.1,),
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8, 'maxls': 50}
            )
            if result.fun < best_c_val:
                best_c = result.x
                best_c_val = result.fun
                print(f"DCT Phase 1: {best_c_val:.6f}", file=sys.stderr)
        except Exception as e:
            print(f"DCT Phase 1 error: {e}", file=sys.stderr)

    # Convert best DCT to time domain
    h_dct = scipy.fft.idctn(best_c, type=2, norm='ortho')
    val_dct = _value(h_dct)
    if val_dct < best_val and np.all(np.isfinite(h_dct)) and abs(np.sum(h_dct)) > 1e-6:
        best_h = h_dct.copy()
        best_val = val_dct

    # Phase 2: Local refinement in time domain
    # The DCT optimization gives a good global shape. Now polish in time domain.
    if time.time() < deadline - 20.0:
        try:
            def time_obj(h, tau):
                s = np.sum(h)
                if abs(s) < 1e-4:
                    return 1e10 + (1e-4 - abs(s)) * 1e6
                conv = np.convolve(h, h)
                sm = _smooth_max(conv, tau=tau)
                return 2.0 * N * sm / (s * s)
            
            result = minimize(
                time_obj,
                best_h,
                args=(0.05,),
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 30}
            )
            
            h_opt = result.x
            val_opt = _value(h_opt)
            if val_opt < best_val and np.all(np.isfinite(h_opt)) and abs(np.sum(h_opt)) > 1e-6:
                best_h = h_opt.copy()
                best_val = val_opt
                print(f"Time Domain Polish 1: {best_val:.6f}", file=sys.stderr)
        except Exception as e:
            print(f"Time Domain Polish 1 error: {e}", file=sys.stderr)

    # Phase 3: Tighter smooth max polish
    if time.time() < deadline - 5.0 and best_val < 2.0:
        try:
            def time_obj_tight(h, tau):
                s = np.sum(h)
                if abs(s) < 1e-4:
                    return 1e10 + (1e-4 - abs(s)) * 1e6
                conv = np.convolve(h, h)
                sm = _smooth_max(conv, tau=tau)
                return 2.0 * N * sm / (s * s)
            
            result = minimize(
                time_obj_tight,
                best_h,
                args=(0.01,),
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 20}
            )
            
            h_opt = result.x
            val_opt = _value(h_opt)
            if val_opt < best_val and np.all(np.isfinite(h_opt)) and abs(np.sum(h_opt)) > 1e-6:
                best_h = h_opt.copy()
                best_val = val_opt
                print(f"Time Domain Polish 2: {best_val:.6f}", file=sys.stderr)
        except Exception as e:
            print(f"Time Domain Polish 2 error: {e}", file=sys.stderr)

    # Final check
    if time.time() > deadline - 2.0:
        return np.ones(N)

    if best_val < 2.0 and np.all(np.isfinite(best_h)) and abs(np.sum(best_h)) > 1e-6:
        return best_h
    else:
        return np.ones(N)

# Need to import scipy.fft
import scipy.fft
