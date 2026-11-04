"""
Structural change: Formulate the sequence as the impulse response of a minimum-phase 
linear prediction filter with optimized reflection coefficients. This ensures a 
strictly positive definite power spectrum that minimizes time-domain peak sidelobes. 
We optimize the short vector of reflection coefficients via L-BFGS-B to shape the 
spectral envelope, which inherently suppresses autoconvolution spikes by maximizing 
the entropy of the spectrum (flatness).
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 400

# EVOLVE-BLOCK-START

_FFT_SIZE = 1
while _FFT_SIZE < 2 * N - 1:
    _FFT_SIZE *= 2


def _autoconv_max_abs(x):
    fx = np.fft.rfft(x, _FFT_SIZE)
    conv = np.fft.irfft(fx * fx, _FFT_SIZE)
    return float(np.max(np.abs(conv[:2 * N - 1])))


def _integral(x):
    return float(np.sum(x))


def _metric(x):
    s = _integral(x)
    if abs(s) < 1e-12:
        return float('inf')
    return _autoconv_max_abs(x) / (s * s)


def _autoconv_values(x):
    fx = np.fft.rfft(x, _FFT_SIZE)
    conv = np.fft.irfft(fx * fx, _FFT_SIZE)
    return conv[:2 * N - 1]


def _soft_max_abs(conv, beta=500.0):
    abs_conv = np.abs(conv)
    max_val = np.max(abs_conv)
    shifted = abs_conv - max_val
    return max_val + np.log(np.sum(np.exp(beta * shifted))) / beta


def _reflection_to_impulse(refs, length):
    """
    Convert reflection coefficients to impulse response of a minimum-phase filter.
    refs: array of reflection coefficients (k1, k2, ..., kP)
    length: desired length of the impulse response
    
    The impulse response h satisfies:
    h[0] = 1
    For n >= 1:
      h[n] = k_n + sum_{m=1}^{n-1} (-1)^m k_{n-m} h[m]  (simplified Levinson-Durbin step)
      Actually, the standard Levinson-Durbin recursion is:
      a[n] = k_n + sum_{m=1}^{n-1} (-1)^m k_{n-m} a[m]  (for predictor coefficients)
      But we want the minimum phase filter h such that its autocorrelation matches the
      reflection coefficients.
      
      A simpler and more robust way: 
      The reflection coefficients define a unique minimum-phase sequence.
      We can construct it iteratively.
      
      Let's use the property that the minimum phase sequence h with reflection coeffs k
      satisfies:
      h[0] = 1
      h[n] = k_n + sum_{i=1}^{n-1} (-1)^i k_{n-i} h[i]  -- wait, this is for prediction error filter.
      
      Let's use the standard Levinson-Durbin algorithm in reverse or direct construction.
      Actually, the easiest way to get the minimum phase sequence from reflection coefficients
      is to use the fact that the sequence is the impulse response of the all-pole filter
      1 / (1 + sum_{i=1}^P a_i z^{-i}) where a_i are derived from k_i.
      
      Alternatively, we can just generate the sequence directly using the recursion for
      the minimum phase sequence:
      h[0] = 1
      For n = 1 to length-1:
        h[n] = k_n (if n <= P) else 0  -- No, that's not right.
        
      Let's use a more direct method:
      The reflection coefficients are related to the autocorrelation of the minimum phase sequence.
      But we can also just simulate the filter.
      
      Actually, the most straightforward implementation is:
      The minimum phase sequence h is the impulse response of the filter with
      transfer function H(z) = 1 / A(z), where A(z) is the prediction error filter
      derived from the reflection coefficients.
      
      However, a simpler approach for this optimization is to use the fact that
      the minimum phase sequence with given reflection coefficients can be generated
      by the Levinson-Durbin recursion.
      
      Let's implement the Levinson-Durbin recursion to get the predictor coefficients,
      then invert them to get the impulse response.
    """
    P = len(refs)
    # Extend refs with zeros if length > P
    k = np.zeros(length)
    k[:P] = refs
    
    # Levinson-Durbin to get prediction error filter coefficients a
    a = np.zeros(length)
    a[0] = 1.0
    e = 1.0  # prediction error
    
    for n in range(1, length):
        if n <= P:
            k_n = k[n]
        else:
            k_n = 0.0
            
        # Update coefficients
        a_new = a.copy()
        for m in range(n):
            a_new[n - m] = k_n * a[m] - a[n - m]
        
        # Update error
        e_new = e * (1 - k_n * k_n)
        
        a = a_new
        e = e_new
    
    # Now a is the prediction error filter coefficients.
    # The minimum phase sequence h is the impulse response of 1/A(z).
    # We can compute this by solving a linear system or using recursion.
    
    # h[0] = 1
    # sum_{m=0}^{n} a[m] h[n-m] = delta[n]
    # h[n] = -sum_{m=1}^{n} a[m] h[n-m]
    
    h = np.zeros(length)
    h[0] = 1.0
    for n in range(1, length):
        h[n] = -np.dot(a[1:n+1], h[n-1::-1])
    
    return h


def _impulse_to_reflection(h, P):
    """
    Convert impulse response h to reflection coefficients (first P).
    This is the inverse of the Levinson-Durbin process.
    """
    # Use Levinson-Durbin on the autocorrelation of h
    # But for optimization, we don't need this if we work directly with refs.
    pass


def _objective_refs(refs, beta=200.0):
    """
    Objective function for reflection coefficient optimization.
    refs: array of reflection coefficients
    """
    h = _reflection_to_impulse(refs, N)
    
    # Normalize to have integral 1
    s = np.sum(h)
    if abs(s) < 1e-12:
        return 1e10
    h = h / s
    
    conv = _autoconv_values(h)
    
    abs_conv = np.abs(conv)
    max_val = np.max(abs_conv)
    shifted = abs_conv - max_val
    soft_max = max_val + np.log(np.sum(np.exp(beta * shifted))) / beta
    
    # The metric is max_abs_conv / (integral^2)
    # Since we normalized h to have integral 1, the metric is just soft_max
    # But wait, the original metric is max_abs_conv / (integral^2)
    # If we normalize h to have integral 1, then integral^2 = 1, so metric = max_abs_conv
    # However, the normalization might affect the optimization landscape.
    # Let's keep the original formulation for consistency.
    
    return soft_max


def _gradient_refs(refs, beta=200.0):
    """
    Gradient of the objective with respect to reflection coefficients.
    We'll use numerical differentiation for simplicity, or derive analytically.
    For now, let's use a simple finite difference if analytic is too complex.
    But L-BFGS-B can work without gradients if we provide them numerically,
    but it's better to provide analytic gradients.
    
    Given the complexity of the reflection-to-impulse mapping, let's use
    scipy.optimize.approx_fprime or just let L-BFGS-B compute numerical gradients.
    """
    # For efficiency, we'll let L-BFGS-B compute numerical gradients
    # by not providing the jac argument.
    pass


def _optimize_reflections(x_init, deadline, rng):
    """
    Optimize reflection coefficients to shape the spectral envelope.
    """
    best_x = x_init.copy()
    best_val = _metric(best_x)
    
    # Choose a reasonable number of reflection coefficients
    # Typically, P ~ N/4 to N/2 is good for spectral shaping
    P = min(100, N // 4)
    
    # Initialize reflection coefficients
    # Small random values to avoid symmetry issues
    refs_init = rng.normal(0, 0.1, size=P)
    
    # We can also try to initialize from the current signal
    # by estimating its reflection coefficients, but that's complex.
    # Let's try multiple random initializations.
    
    num_runs = 5
    time_per_run = (deadline - time.time()) / num_runs
    
    for run_idx in range(num_runs):
        if time.time() > deadline - 10:
            break
        
        if run_idx == 0:
            refs_current = refs_init.copy()
        else:
            # Different random initialization
            scale = 0.1 * (run_idx + 1)
            refs_current = rng.normal(0, scale, size=P)
        
        start_time = time.time()
        try:
            # Use L-BFGS-B with numerical gradients
            result = minimize(
                _objective_refs,
                refs_current,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8, 'maxfun': 5000}
            )
            
            h_final = _reflection_to_impulse(result.x, N)
            s = np.sum(h_final)
            if abs(s) > 1e-12:
                h_final = h_final / s
            
            val_final = _metric(h_final)
            
            if val_final < best_val:
                best_val = val_final
                best_x = h_final.copy()
                
        except Exception:
            pass
        
        elapsed = time.time() - start_time
        if elapsed > time_per_run:
            pass

    return best_x, best_val


def _local_search_refs(x_init, deadline, rng, max_rounds=20000):
    """Local search in the reflection coefficient space."""
    best_x = x_init.copy()
    best_val = _metric(best_x)
    
    P = min(100, N // 4)
    
    # Initialize with small random values
    refs_current = rng.normal(0, 0.1, size=P)
    
    no_improve = 0
    step_scale = 0.05
    
    for round_idx in range(max_rounds):
        if time.time() > deadline:
            break
        
        if no_improve > 50:
            step_scale *= 0.9
            no_improve = 0
            if step_scale < 0.001:
                break
        
        # Pick a random reflection coefficient to perturb
        idx = int(rng.integers(0, P))
        delta = rng.normal(0, step_scale)
        
        refs_test = refs_current.copy()
        refs_test[idx] += delta
        
        h_test = _reflection_to_impulse(refs_test, N)
        s = np.sum(h_test)
        if abs(s) > 1e-12:
            h_test = h_test / s
        else:
            continue
            
        val_test = _metric(h_test)
        
        if val_test < best_val:
            best_val = val_test
            best_x = h_test.copy()
            refs_current = refs_test
            no_improve = 0
        else:
            no_improve += 1
    
    return best_x, best_val


def _final_polish(x, deadline, rng):
    """High-precision final polish with coordinate descent in time domain."""
    best_x = x.copy()
    best_val = _metric(best_x)
    
    for sweep in range(10):
        if time.time() > deadline:
            break
        
        improved = False
        for idx in range(N):
            if time.time() > deadline:
                break
            
            lo, hi = -0.01, 0.01
            best_delta = 0.0
            best_v = best_val
            
            for _ in range(5):
                if time.time() > deadline:
                    break
                m1 = lo + (hi - lo) / 3
                m2 = hi - (hi - lo) / 3
                
                x1 = best_x.copy()
                x1[idx] += m1
                v1 = _metric(x1)
                
                x2 = best_x.copy()
                x2[idx] += m2
                v2 = _metric(x2)
                
                if v1 < v2:
                    hi = m2
                else:
                    lo = m1
                
                if v1 < best_v:
                    best_v = v1
                    best_delta = m1
                if v2 < best_v:
                    best_v = v2
                    best_delta = m2
            
            if best_delta != 0.0 and best_v < best_val:
                best_x[idx] += best_delta
                best_val = best_v
                improved = True
        
        if not improved:
            break
    
    return best_x, best_val


def construct():
    deadline = time.time() + 145.0
    
    best_x = np.ones(N)
    best_val = _metric(best_x)
    
    rng = np.random.default_rng(42)
    
    # Phase 1: Reflection coefficient optimization
    if time.time() < deadline - 30:
        x_ref, val_ref = _optimize_reflections(best_x, deadline - 30, rng)
        if val_ref < best_val:
            best_val = val_ref
            best_x = x_ref
    
    # Phase 2: Local search in reflection space
    if time.time() < deadline - 10:
        x_ls, val_ls = _local_search_refs(best_x, deadline - 10, rng, max_rounds=15000)
        if val_ls < best_val:
            best_val = val_ls
            best_x = x_ls
    
    # Phase 3: Final polish in time domain
    if time.time() < deadline - 2:
        best_x, best_val = _final_polish(best_x, deadline - 2, rng)
    
    return best_x
# EVOLVE-BLOCK-END
