"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time

def _linear_overlap(h, k):
    """
    Compute overlap sum_i h[i]*(1-h[i+k]) for shift k (0 <= k < N).
    Uses linear (non-circular) shifts: h[i+k] is 0 when i+k >= N.
    """
    if k == 0:
        return np.sum(h * (1.0 - h))
    # h[i] * (1 - h[i+k]) for i = 0..N-1-k
    return np.sum(h[:N-k] * (1.0 - h[k:]))

def _max_overlap_linear(h):
    """
    Compute max over all shifts k of sum_i h[i]*(1-h[i+k]).
    Linear shift: h[i+k] = 0 if i+k >= N.
    """
    max_ov = 0.0
    for k in range(N):
        if k == 0:
            ov = np.sum(h * (1.0 - h))
        else:
            ov = np.sum(h[:N-k] * (1.0 - h[k:]))
        if ov > max_ov:
            max_ov = ov
    return max_ov

def _max_overlap_linear_fast(h):
    """
    Faster computation using FFT-based cross-correlation.
    overlap(k) = sum_i h[i]*(1-h[i+k]) = sum_i h[i] - sum_i h[i]*h[i+k]
    For linear shifts, sum_i h[i]*h[i+k] is the linear autocorrelation at lag k.
    """
    S = np.sum(h)
    # Linear autocorrelation via FFT
    n_fft = 2 * N
    H = np.fft.rfft(h, n=n_fft)
    ac = np.fft.irfft(H * np.conj(H), n=n_fft)[:N]
    # overlap(k) = S - ac[k] for k=0..N-1
    # But we need to be careful: for linear shift, the sum is only over valid indices
    # ac[k] = sum_{i=0}^{N-1-k} h[i]*h[i+k]
    # sum_i h[i] for i=0..N-1-k is not S, it's S - sum_{i=N-k}^{N-1} h[i]
    # So overlap(k) = sum_{i=0}^{N-1-k} h[i]*(1-h[i+k])
    #                = sum_{i=0}^{N-1-k} h[i] - sum_{i=0}^{N-1-k} h[i]*h[i+k]
    #                = (S - sum_{i=N-k}^{N-1} h[i]) - ac[k]
    max_ov = 0.0
    for k in range(N):
        if k == 0:
            ov = np.sum(h * (1.0 - h))
        else:
            partial_sum = S - np.sum(h[N-k:])
            ov = partial_sum - ac[k]
        if ov > max_ov:
            max_ov = ov
    return max_ov

def _project(x):
    """Project x onto the feasible set: x in [0,1]^N, sum(x) = N/2."""
    x = np.clip(x, 0.0, 1.0)
    s = x.sum()
    target = N / 2.0
    if abs(s - target) < 1e-14:
        return x
    for _ in range(200):
        s = x.sum()
        need = target - s
        if abs(need) < 1e-14:
            break
        if need > 0:
            mask = x < 1.0
            if not mask.any():
                break
            x[mask] += need / mask.sum()
        else:
            mask = x > 0.0
            if not mask.any():
                break
            x[mask] += need / mask.sum()
        x = np.clip(x, 0.0, 1.0)
    return x

def _sturmian_sequence(alpha, beta, N):
    """
    Generate a Sturmian-like sequence: h[i] = 1 if frac(i*alpha + beta) < alpha, else 0.
    This gives a sequence with sum approximately N*alpha.
    """
    seq = np.zeros(N)
    for i in range(N):
        val = (i * alpha + beta) % 1.0
        seq[i] = 1.0 if val < alpha else 0.0
    return seq

def _soft_min_overlap(h, temperature):
    """
    Differentiable approximation of max overlap using soft-min (actually soft-max of overlaps).
    We want to minimize max_k overlap(k). Use log-sum-exp as a smooth upper bound.
    """
    overlaps = np.array([_linear_overlap(h, k) for k in range(N)])
    # Use soft-max: log(sum(exp(overlap/t))) * t approximates max(overlap)
    max_ov = np.max(overlaps)
    shifted = (overlaps - max_ov) / temperature
    log_sum = max_ov / temperature + np.log(np.sum(np.exp(shifted)))
    return log_sum * temperature

def _grad_soft_max_overlap(h, temperature):
    """
    Gradient of the soft-max overlap w.r.t. h.
    """
    overlaps = np.array([_linear_overlap(h, k) for k in range(N)])
    max_ov = np.max(overlaps)
    shifted = (overlaps - max_ov) / temperature
    weights = np.exp(shifted) / np.sum(np.exp(shifted))
    
    grad = np.zeros(N)
    for k in range(N):
        w = weights[k]
        if w < 1e-10:
            continue
        if k == 0:
            # d/dh[i] of sum_i h[i]*(1-h[i]) = 1 - 2*h[i]
            grad += w * (1.0 - 2.0 * h)
        else:
            # d/dh[i] of sum_{j=0}^{N-1-k} h[j]*(1-h[j+k])
            # For h[i]: appears in term j=i as h[i]*(1-h[i+k]) -> derivative (1-h[i+k])
            #           appears in term j=i-k as h[i-k]*(1-h[i]) -> derivative -h[i-k]
            #           (if i-k >= 0 and i-k <= N-1-k, i.e., i <= N-1)
            # Also h[i] appears as h[j+k] when j+k=i, j=i-k: term h[i-k]*(1-h[i]), derivative -h[i-k]
            # And as h[j] when j=i: term h[i]*(1-h[i+k]), derivative (1-h[i+k])
            # But only if i < N-k (for h[i] as first factor) and i >= k (for h[i] as second factor)
            # h[i] as first factor: i in [0, N-1-k], derivative (1-h[i+k])
            # h[i] as second factor: i in [k, N-1], derivative -h[i-k]
            if i < N - k:
                pass  # handled below
            # Let's just compute directly
            # overlap(k) = sum_{j=0}^{N-1-k} h[j]*(1-h[j+k])
            # d/dh[i] = (1-h[i+k]) if i < N-k else 0
            #         + (-h[i-k]) if i >= k else 0
            if i < N - k:
                grad[i] += w * (1.0 - h[i+k])
            if i >= k:
                grad[i] += w * (-h[i-k])
    return grad

def _binary_swap_local_search(h, max_iters=2000):
    """
    Local search: try swapping pairs (i,j) where h[i]>0.5 and h[j]<0.5,
    or flipping individual bits, to reduce max overlap.
    """
    h = h.copy()
    best_val = _max_overlap_linear_fast(h)
    for it in range(max_iters):
        improved = False
        # Try flipping each position
        for i in range(N):
            h_new = h.copy()
            h_new[i] = 1.0 - h_new[i]
            h_new = _project(h_new)
            val = _max_overlap_linear_fast(h_new)
            if val < best_val - 1e-10:
                h = h_new
                best_val = val
                improved = True
                break
        if not improved:
            # Try pairwise swaps
            found = False
            for i in range(N):
                for j in range(i+1, N):
                    if abs(h[i] - h[j]) < 0.1:
                        continue
                    h_new = h.copy()
                    h_new[i], h_new[j] = h_new[j], h_new[i]
                    h_new = _project(h_new)
                    val = _max_overlap_linear_fast(h_new)
                    if val < best_val - 1e-10:
                        h = h_new
                        best_val = val
                        found = True
                        break
                if found:
                    break
            if not found:
                break
    return h, best_val

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s limit
    
    best_h = np.full(N, 0.5)
    best_val = _max_overlap_linear_fast(best_h)
    
    # Generate diverse initial candidates
    candidates = []
    
    # 1. Uniform
    candidates.append(np.full(N, 0.5))
    
    # 2. Sturmian sequences with various slopes
    rng = np.random.default_rng(42)
    for alpha in [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]:
        for beta in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
            seq = _sturmian_sequence(alpha, beta, N)
            seq = _project(seq)
            candidates.append(seq)
    
    # 3. Random
    for _ in range(20):
        h_rand = rng.uniform(0, 1, N)
        h_rand = _project(h_rand)
        candidates.append(h_rand)
    
    # 4. Sinusoidal patterns
    for freq in [1, 2, 3, 5, 8, 13]:
        for phase in [0.0, 0.25, 0.5, 0.75]:
            x = np.arange(N)
            h_sin = 0.5 + 0.5 * np.sin(2 * np.pi * freq * x / N + phase * 2 * np.pi)
            h_sin = _project(h_sin)
            candidates.append(h_sin)
    
    # Evaluate all candidates
    for cand in candidates:
        val = _max_overlap_linear_fast(cand)
        if val < best_val - 1e-10:
            best_val = val
            best_h = cand.copy()
    
    # Gradient-based optimization on the best candidate
    h = best_h.copy()
    temperature = 5.0
    
    # Run gradient descent on soft-max overlap
    for restart in range(5):
        if time.time() > deadline:
            break
        
        if restart == 0:
            h = best_h.copy()
        else:
            # Perturb
            h = best_h + rng.normal(0, 0.1, N)
            h = _project(h)
        
        lr = 0.005
        for it in range(300):
            if time.time() > deadline:
                break
            
            val = _max_overlap_linear_fast(h)
            if val < best_val - 1e-10:
                best_val = val
                best_h = h.copy()
            
            # Compute gradient of soft-max overlap
            grad = _grad_soft_max_overlap(h, temperature)
            
            # Projected gradient descent
            h_new = h - lr * grad
            h_new = _project(h_new)
            
            # Check improvement
            val_new = _max_overlap_linear_fast(h_new)
            if val_new < val + 1e-10:
                h = h_new
            else:
                lr *= 0.5
                if lr < 1e-8:
                    break
        
        # Local search refinement
        h_ls, val_ls = _binary_swap_local_search(h, max_iters=1000)
        if val_ls < best_val - 1e-10:
            best_val = val_ls
            best_h = h_ls.copy()
        
        # Reduce temperature
        temperature *= 0.7
    
    # Final local search on best
    if time.time() < deadline - 10:
        h_final, val_final = _binary_swap_local_search(best_h, max_iters=5000)
        if val_final < best_val - 1e-10:
            best_val = val_final
            best_h = h_final.copy()
    
    return best_h
# EVOLVE-BLOCK-END
