import numpy as np
import time
from scipy.optimize import minimize

N = 100
TARGET_SUM = N / 2.0

def _compute_overlaps(h):
    """
    Compute C_k = sum_i h_i (1 - h_{i+k}) for k=1..N-1.
    """
    g = 1.0 - h
    c = np.correlate(h, g, mode='full')
    c_nonzero = np.concatenate([c[:N], c[N+1:]])
    return c_nonzero

def _compute_circular_overlaps(h):
    """
    Compute circular C_k = sum_i h_i (1 - h_{i+k mod N}) for k=1..N-1.
    """
    H = np.fft.fft(h)
    G = np.fft.fft(1.0 - h)
    # Circular correlation
    # R_k = sum h_i h_{i+k}
    # C_k = sum h_i - R_k = N/2 - R_k
    # R_k DFT is |H|^2
    R = np.fft.ifft(np.abs(H)**2).real
    C = N/2.0 - R
    # Exclude k=0
    return C[1:]

def _project_to_feasible(h):
    """
    Project h to [0,1] and sum 50.
    """
    h = h.copy()
    # Simple iterative projection
    for _ in range(100):
        h = np.clip(h, 0, 1)
        diff = TARGET_SUM - np.sum(h)
        if abs(diff) < 1e-8:
            break
        if diff > 0:
            mask = h < 1.0
        else:
            mask = h > 0.0
        if not np.any(mask):
            break
        h[mask] += diff / np.sum(mask)
    return h

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    best_h = None
    best_score = float('inf')
    
    # Initial candidates
    candidates = []
    
    # 1. Alternating
    h_alt = np.full(N, 0.5)
    candidates.append(h_alt)
    
    # 2. Sine waves
    t = np.arange(N)
    for freq in [1, 2, 3, 5, 10]:
        for amp in [0.3, 0.4, 0.49]:
            h_sin = 0.5 + amp * np.sin(2 * np.pi * freq * t / N)
            candidates.append(np.clip(h_sin, 0, 1))
            
    # 3. Random Binary
    rng = np.random.default_rng(42)
    for _ in range(5):
        h_bin = rng.choice([0, 1], size=N)
        s = np.sum(h_bin)
        if s > 50:
            idx_ones = np.where(h_bin == 1)[0]
            to_swap = s - 50
            swap_idx = rng.choice(idx_ones, size=to_swap, replace=False)
            h_bin[swap_idx] = 0
        elif s < 50:
            idx_zeros = np.where(h_bin == 0)[0]
            to_swap = 50 - s
            swap_idx = rng.choice(idx_zeros, size=to_swap, replace=False)
            h_bin[swap_idx] = 1
        candidates.append(h_bin.astype(float))
        
    # 4. Random Uniform
    for _ in range(5):
        h_rand = rng.uniform(0, 1, size=N)
        h_rand = h_rand - np.mean(h_rand) + 0.5
        h_rand = np.clip(h_rand, 0, 1)
        diff = 50 - np.sum(h_rand)
        if diff != 0:
            step = diff / N
            h_rand += step
            h_rand = np.clip(h_rand, 0, 1)
        candidates.append(h_rand)

    # Spectral Optimization
    # We optimize the DFT coefficients.
    # H[0] is fixed by sum constraint? No, we enforce sum in projection.
    # We optimize real and imaginary parts of H[1:N//2] and H[N//2+1:N-1] and H[N//2].
    # To simplify, we optimize h directly but add a spectral penalty to encourage low freq.
    # Or, we optimize H, compute h, project, and compute objective on projected h.
    # This is non-smooth due to projection.
    # Better: Optimize h with a smooth objective that approximates the spectral flatness.
    
    # Let's stick to optimizing h directly but using a better objective.
    # Objective: Max Circular Overlap + Penalty for High Freq Energy
    
    for i, h_init in enumerate(candidates):
        if time.time() > deadline - 10.0:
            break
            
        h_init = _project_to_feasible(h_init)
        
        def objective(x):
            h = x
            # Compute circular overlaps
            C_circ = _compute_circular_overlaps(h)
            max_circ = np.max(C_circ)
            
            # Spectral penalty: Encourage low frequency
            H = np.fft.fft(h)
            # Penalty on high frequency magnitudes
            # Mask for high freq
            m = np.arange(N)
            high_freq_mask = (m > 5) & (m < N - 5)
            penalty = np.sum(np.abs(H[high_freq_mask])**2) * 0.1
            
            return max_circ + penalty
        
        def grad_objective(x):
            h = x
            # Numerical gradient for simplicity and robustness in this step
            # N=100, numerical grad is 200 evals.
            # 50 candidates * 50 iters * 200 evals = 500k evals.
            # Each eval is FFT (N log N) + correlate.
            # Might be slow.
            # Let's use analytical gradient for circular part.
            # C_circ = N/2 - ifft(|H|^2).real
            # This is complex to differentiate.
            # Let's just use numerical gradient but limit iterations.
            
            eps = 1e-5
            grad = np.zeros(N)
            base_val = objective(h)
            for j in range(N):
                h_plus = h.copy()
                h_plus[j] += eps
                val_plus = objective(h_plus)
                h_minus = h.copy()
                h_minus[j] -= eps
                val_minus = objective(h_minus)
                grad[j] = (val_plus - val_minus) / (2 * eps)
            return grad

        try:
            res = minimize(objective, h_init, method='L-BFGS-B', jac=grad_objective,
                           bounds=[(0, 1)] * N,
                           options={'maxiter': 20, 'ftol': 1e-8})
            h_opt = _project_to_feasible(res.x)
        except Exception:
            h_opt = h_init

        # Evaluate true linear max overlap
        c_final = _compute_overlaps(h_opt)
        true_max = np.max(c_final)
        
        if true_max < best_score:
            best_score = true_max
            best_h = h_opt.copy()
            
    # If no valid h found, return a default
    if best_h is None:
        best_h = np.full(N, 0.5)
        
    # Final polish: Local search with small perturbations
    if time.time() < deadline - 2.0:
        rng_local = np.random.default_rng(123)
        h_best = best_h.copy()
        best_val = _max_overlap(h_best)
        
        for _ in range(50):
            if time.time() > deadline - 2.0: break
            idx = rng_local.choice(N, size=2, replace=False)
            delta = rng_local.uniform(-0.01, 0.01)
            h_try = h_best.copy()
            h_try[idx[0]] += delta
            h_try[idx[1]] -= delta
            h_try = np.clip(h_try, 0, 1)
            diff = 50 - np.sum(h_try)
            if abs(diff) > 1e-9:
                mask = h_try < 1.0 if diff > 0 else h_try > 0.0
                if np.any(mask):
                    h_try[mask] += diff / np.sum(mask)
                    h_try = np.clip(h_try, 0, 1)
            val = _max_overlap(h_try)
            if val < best_val:
                best_val = val
                h_best = h_try.copy()
                
        best_h = h_best
        
    return best_h

def _max_overlap(h):
    c = _compute_overlaps(h)
    return np.max(c)
