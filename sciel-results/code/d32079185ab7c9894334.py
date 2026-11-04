"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize

N = 100
DEADLINE = 150.0
SAFETY_MARGIN = 5.0

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

N = 100
DEADLINE = 150.0
SAFETY_MARGIN = 5.0

def _overlap_max(h):
    """Compute the maximum overlap of h with shifted 1-h.
    
    The overlap at shift k is sum_i h_i * (1 - h_{(i+k) % N})
    We want to minimize the maximum over all shifts k.
    """
    h = np.asarray(h, dtype=np.float64)
    one_minus_h = 1.0 - h
    # Circular cross-correlation: for each shift k, compute sum_i h[i] * one_minus_h[(i+k) % N]
    # This is the circular cross-correlation of h and one_minus_h
    # Using FFT: circular cross-correlation can be computed via FFT
    N = len(h)
    H = np.fft.rfft(h)
    O = np.fft.rfft(one_minus_h)
    # Cross-correlation: ifft( H * conj(O) ) gives sum_i h[i] * one_minus_h[(i-k) % N]
    # We want sum_i h[i] * one_minus_h[(i+k) % N] which is ifft(conj(H) * O)
    # Actually, let's be careful.
    # Circular cross-correlation C[k] = sum_i h[i] * g[(i+k) % N]
    # In frequency domain: C = ifft( H * conj(G) )? Let me think...
    # Actually, the circular cross-correlation of h and g is:
    # C[k] = sum_i h[i] * g[(i+k) % N]
    # This equals ifft( H * conj(G) ) where H = fft(h), G = fft(g)
    # Wait, let me verify with a simple example.
    # Actually, the standard formula is:
    # cross_corr = ifft( H * conj(G) ) gives sum_i h[i] * g[(i-k) % N]
    # So to get sum_i h[i] * g[(i+k) % N], we need ifft( conj(H) * G )
    
    # Let's just compute it directly for small N or use the FFT approach carefully.
    # For N=100, direct computation is O(N^2) = 10000 ops, which is fast enough.
    
    # Direct computation
    overlaps = np.zeros(N)
    for k in range(N):
        shifted = np.roll(h, k)
        overlaps[k] = np.dot(h, one_minus_h - shifted)  # Wait, this isn't right
    
    # Let me reconsider. The overlap at shift k should be:
    # sum_i h[i] * (1 - h[(i+k) % N])
    # = sum_i h[i] - sum_i h[i] * h[(i+k) % N]
    # = sum(h) - circular_autocorr(h, k)
    # where circular_autocorr(h, k) = sum_i h[i] * h[(i+k) % N]
    
    # Since sum(h) = N/2 (by constraint), we have:
    # overlap[k] = N/2 - autocorr[k]
    # To minimize max overlap, we want to maximize min autocorr[k]
    # Or equivalently, make autocorr[k] as constant as possible (close to N/4 for all k)
    
    # Let's compute autocorr directly
    autocorr = np.zeros(N)
    for k in range(N):
        shifted = np.roll(h, k)
        autocorr[k] = np.dot(h, shifted)
    
    overlaps = N / 2.0 - autocorr
    return np.max(overlaps)


def _objective(params):
    """Objective function: epigraph formulation for max overlap."""
    h = params[:N]
    t = params[N]
    # Compute overlaps
    one_minus_h = 1.0 - h
    autocorr = np.zeros(N)
    for k in range(N):
        shifted = np.roll(h, k)
        autocorr[k] = np.dot(h, shifted)
    overlaps = N / 2.0 - autocorr
    # Constraints: overlaps[k] <= t for all k
    # We want to minimize t
    # Penalty for violations
    violations = np.maximum(overlaps - t, 0)
    return t + 0.01 * np.sum(violations**2)


def _project(h):
    """Project h onto the feasible set: sum(h) = N/2, 0 <= h <= 1."""
    h = np.clip(h, 0.0, 1.0)
    # Adjust to satisfy sum constraint
    target_sum = N / 2.0
    current_sum = np.sum(h)
    if current_sum > 0:
        # Scale towards target
        h = h + (target_sum - current_sum) / N
        h = np.clip(h, 0.0, 1.0)
        # Iterate projection
        for _ in range(50):
            current_sum = np.sum(h)
            if abs(current_sum - target_sum) < 1e-10:
                break
            h = h + (target_sum - current_sum) / N
            h = np.clip(h, 0.0, 1.0)
    return h


def construct():
    start_time = time.time()
    
    best_h = np.full(N, 0.5)
    best_score = _overlap_max(best_h)
    
    # Try multiple initializations with gradient descent
    rng = np.random.default_rng(42)
    
    # Different initializations
    inits = []
    # Uniform
    inits.append(np.full(N, 0.5))
    # Random
    for _ in range(5):
        h_rand = rng.random(N) * 0.5 + 0.25
        h_rand = _project(h_rand)
        inits.append(h_rand)
    # Periodic patterns
    for period in [2, 4, 5, 10]:
        h_per = np.tile([0.5, 0.5] * (N // (2*period)) if period % 2 == 0 else [0.5]*N, 1)
        # Create alternating pattern
        h_alt = np.zeros(N)
        for i in range(N):
            if (i % period) < period // 2:
                h_alt[i] = 0.75
            else:
                h_alt[i] = 0.25
        h_alt = _project(h_alt)
        inits.append(h_alt)
    # Cosine patterns
    for freq in [1, 2, 3]:
        x = np.linspace(0, 2 * np.pi * freq, N, endpoint=False)
        h_cos = 0.5 + 0.25 * np.cos(x)
        h_cos = _project(h_cos)
        inits.append(h_cos)
    
    for h0 in inits:
        if time.time() - start_time > DEADLINE - SAFETY_MARGIN:
            break
        
        h0 = _project(h0)
        params0 = np.concatenate([h0, [_overlap_max(h0)]])
        
        try:
            res = minimize(
                _objective,
                params0,
                method='L-BFGS-B',
                options={'maxiter': 200, 'maxfun': 5000},
                timeout=30.0
            )
            
            h_opt = _project(res.x[:N])
            score_opt = _overlap_max(h_opt)
            
            if score_opt < best_score:
                best_score = score_opt
                best_h = h_opt
        except Exception:
            pass
    
    # Local refinement with direct search on best solution
    if time.time() - start_time < DEADLINE - SAFETY_MARGIN:
        # Try small perturbations
        for _ in range(100):
            if time.time() - start_time > DEADLINE - SAFETY_MARGIN:
                break
            h_pert = best_h + rng.normal(0, 0.01, N)
            h_pert = _project(h_pert)
            score_pert = _overlap_max(h_pert)
            if score_pert < best_score:
                best_score = score_pert
                best_h = h_pert
    
    return best_h
# EVOLVE-BLOCK-END
