"""Combined approach: Bang-bang local search + continuous subgradient polish."""
import time
import numpy as np

N = 100
W = 2.0 / N
TARGET_SUM = N / 2.0

# EVOLVE-BLOCK-START
def _compute_overlaps(h):
    """
    Compute the maximum overlap of h with shifted 1-h for all shifts.
    Overlap for shift k is W * (sum_{j} h[j] - sum_{j} h[j]h[j-k])
    We use FFT for efficiency.
    """
    H = np.fft.fft(h)
    # Autocorrelation: sum h[j] h[j-k] for shift k
    # ifft(H * conj(H)) gives circular autocorrelation.
    # We need linear autocorrelation or handle the circular nature carefully.
    # Since h is defined on [0, 2] and we consider shifts, the domain is effectively periodic in the discrete sense?
    # The task says "shifted 1-h".
    # Let's stick to the direct definition used in inspiration 1 which was robust.
    # O[k] = W * (sum_{j: j-k in [0, N)} h[j] * (1 - h[j-k]) + sum_{j: j+k in [0, N)} ... ?)
    # Actually, the integral is over [0, 2].
    # Let's use the formula: Overlap(k) = \int_0^2 h(x) (1 - h(x-k)) dx (with h=0 outside [0,2]? or periodic?)
    # "step heights h on [0,2]". Usually implies h is 0 outside.
    # If h is 0 outside, then for shift k:
    # Overlap(k) = \int_{max(0, k)}^{min(2, 2+k)} h(x) (1 - h(x-k)) dx
    # Discretely:
    # Sum over j such that x_j in [0, 2] and x_j - k in [0, 2].
    
    # Let's use the vectorized approach from inspiration 1 which seemed to work well.
    # It computed O_pos and O_neg.
    
    # O_pos[k] = sum_{j=k}^{N-1} h[j] * (1 - h[j-k]) * W
    # O_neg[k] = sum_{j=0}^{N-1-k} h[j] * (1 - h[j+k]) * W
    # Note: h[j] * (1 - h[j-k]) = h[j] - h[j]h[j-k].
    # Sum h[j] for j=k..N-1 is S1[k].
    # Sum h[j]h[j-k] is dot(h[k:], h[:N-k]).
    
    S1 = np.cumsum(h[::-1])[::-1] # S1[k] = sum h[k:]
    S2 = np.array([np.dot(h[k:], h[:N - k]) for k in range(N)])
    O_pos = W * (S1 - S2)
    
    csum = np.cumsum(h)
    # S1n[k] = sum h[:N-k]
    S1n = csum[N - 1 - np.arange(N)]
    S2n = np.array([np.dot(h[:N - k], h[k:]) for k in range(N)])
    O_neg = W * (S1n - S2n)
    
    return np.concatenate([O_pos, O_neg])

def _max_overlap(h):
    return _compute_overlaps(h).max()

def _project(h):
    """Project h onto [0,1]^N with sum N/2."""
    lo = float(h.min()) - 1.0
    hi = float(h.max()) + 1.0
    for _ in range(64):
        mid = 0.5 * (lo + hi)
        s = np.clip(h - mid, 0.0, 1.0).sum()
        if s > TARGET_SUM:
            lo = mid
        else:
            hi = mid
    t = 0.5 * (lo + hi)
    return np.clip(h - t, 0.0, 1.0)

def _subgrad_polish(h, deadline):
    """
    Subgradient descent on the max overlap function.
    """
    h = _project(h.copy())
    best_h = h.copy()
    best_val = _max_overlap(h)
    
    # Estimate Lipschitz constant
    L = 10.0
    step = 1.0 / L
    t_fista = 1.0
    h_prev = h.copy()
    
    for i in range(2000):
        if time.time() > deadline - 5.0:
            break
            
        O = _compute_overlaps(h)
        Omax = O.max()
        
        # Subgradient of max is sum of gradients of active terms
        # Gradient of O[k] wrt h:
        # O_pos[k] = W * (S1[k] - S2[k])
        # d/dh_j O_pos[k] = W * (1_{j>=k} - h[j-k]1_{j>=k} - h[j]1_{j-k>=0, j-k < N})?
        # Let's derive d/dh_j [ h[j] - h[j]h[j-k] ]
        # Term 1: h[j] -> deriv 1
        # Term 2: - h[j]h[j-k] -> deriv -h[j-k] (if j-k valid) - h[j] (if j+k valid for the other term?)
        # This is getting messy. Let's use the explicit gradient from inspiration 1.
        
        # From inspiration 1:
        # Gpos = W * (ge_f - ge * h[idx_p] - lt * h[idx_m])
        # Where ge_f is 1 if I >= K, else 0.
        # idx_p = I - K (index of h[j-k])
        # idx_m = I + K (index of h[j+k] for the negative shift term? No, this was for O_neg?)
        
        # Let's just use a simple subgradient:
        # The active shifts are those with O[k] == Omax.
        # For each active shift k, add the gradient of O[k].
        
        # Gradient of O_pos[k] wrt h:
        # O_pos[k] = W * sum_{j=k}^{N-1} (h[j] - h[j]h[j-k])
        # d/dh_j = W * (1 - h[j-k]) for j >= k
        # d/dh_j = W * (-h[j+k]) for j < N-k (from term h[j+k]h[j] in sum? No, h[j+k] is the first term)
        # Wait, sum is over j. Term is h[j] - h[j]h[j-k].
        # If we vary h[j], it appears in:
        # 1. Term j: h[j] - h[j]h[j-k] -> deriv 1 - h[j-k]
        # 2. Term j+k: h[j+k] - h[j+k]h[j] -> deriv -h[j+k] (if j+k < N)
        # So grad_pos[k][j] = W * (1 - h[j-k]) * (j>=k) - W * h[j+k] * (j+k<N)
        
        # Gradient of O_neg[k] wrt h:
        # O_neg[k] = W * sum_{j=0}^{N-1-k} (h[j] - h[j]h[j+k])
        # Varying h[j]:
        # 1. Term j: h[j] - h[j]h[j+k] -> deriv 1 - h[j+k]
        # 2. Term j-k: h[j-k] - h[j-k]h[j] -> deriv -h[j-k] (if j-k >= 0)
        # So grad_neg[k][j] = W * (1 - h[j+k]) * (j+k<N) - W * h[j-k] * (j-k>=0)
        
        # We need to sum gradients for active k.
        # To be efficient, we can precompute matrices or just loop if N is small.
        # N=100, so 200 shifts. Looping 200 times is fine.
        
        grad = np.zeros(N)
        active = np.where(O > Omax - 1e-6)[0]
        
        for k_idx in active:
            if k_idx < N:
                k = k_idx
                # O_pos[k]
                # grad += W * (1 - h[j-k]) for j>=k
                # grad -= W * h[j+k] for j<N-k
                j_range = np.arange(N)
                mask1 = j_range >= k
                mask2 = j_range < N - k
                grad[mask1] += W * (1 - h[j_range[mask1] - k])
                grad[mask2] -= W * h[j_range[mask2] + k]
            else:
                k = k_idx - N
                # O_neg[k]
                # grad += W * (1 - h[j+k]) for j<N-k
                # grad -= W * h[j-k] for j>=k
                j_range = np.arange(N)
                mask1 = j_range < N - k
                mask2 = j_range >= k
                grad[mask1] += W * (1 - h[j_range[mask1] + k])
                grad[mask2] -= W * h[j_range[mask2] - k]
        
        # Normalize gradient?
        # FISTA step
        y = h - step * grad
        h_new = _project(y)
        
        # Check improvement
        val_new = _max_overlap(h_new)
        if val_new < best_val:
            best_h = h_new.copy()
            best_val = val_new
            
        # Update FISTA variables
        t_new = (1 + np.sqrt(1 + 4 * t_fista * t_fista)) / 2
        h_prev = h.copy()
        h = _project(h + ((t_fista - 1) / t_new) * (h - h_prev))
        t_fista = t_new
        
        # Adjust step size (simple heuristic)
        if val_new < Omax:
            step *= 1.1
        else:
            step *= 0.9
            step = max(step, 1e-4)
            
    return best_h, best_val

def _local_search_bb(h, deadline):
    """
    Convert to bang-bang and perform local search.
    """
    h_bb = (h >= 0.5).astype(float)
    # Ensure sum is N/2
    diff = TARGET_SUM - h_bb.sum()
    if diff > 0:
        zeros = np.where(h_bb == 0)[0]
        if len(zeros) >= int(diff):
            # Pick zeros that reduce overlap most?
            # For now, random
            rng = np.random.default_rng(42)
            idx = rng.choice(zeros, size=int(diff), replace=False)
            h_bb[idx] = 1.0
    elif diff < 0:
        ones = np.where(h_bb == 1)[0]
        if len(ones) >= int(-diff):
            rng = np.random.default_rng(42)
            idx = rng.choice(ones, size=int(-diff), replace=False)
            h_bb[idx] = 0.0
            
    best_val = _max_overlap(h_bb)
    best_h = h_bb.copy()
    
    rng = np.random.default_rng(0)
    for _ in range(5000):
        if time.time() > deadline - 5.0:
            break
        ones = np.where(h_bb == 1)[0]
        zeros = np.where(h_bb == 0)[0]
        if len(ones) == 0 or len(zeros) == 0:
            break
        i = rng.choice(ones)
        j = rng.choice(zeros)
        
        h_test = h_bb.copy()
        h_test[i] = 0.0
        h_test[j] = 1.0
        
        val = _max_overlap(h_test)
        if val < best_val - 1e-9:
            h_bb = h_test
            best_val = val
            best_h = h_bb.copy()
            
    return best_h, best_val

def construct():
    start = time.time()
    deadline = start + 140.0
    
    # Start with bang-bang patterns
    # Try a few different initial bang-bang configurations
    inits = []
    # 1. Alternating
    h1 = np.tile([1, 0], N // 2)
    inits.append(h1)
    # 2. Blocks
    h2 = np.zeros(N)
    h2[:N//2] = 1.0
    inits.append(h2)
    # 3. Random
    rng = np.random.default_rng(1)
    for _ in range(5):
        h_rand = rng.random(N)
        h_bb = (h_rand >= 0.5).astype(float)
        # Adjust sum
        diff = TARGET_SUM - h_bb.sum()
        if diff > 0:
            zeros = np.where(h_bb == 0)[0]
            h_bb[rng.choice(zeros, size=int(diff), replace=False)] = 1.0
        elif diff < 0:
            ones = np.where(h_bb == 1)[0]
            h_bb[rng.choice(ones, size=int(-diff), replace=False)] = 0.0
        inits.append(h_bb)
        
    best_h = None
    best_val = float('inf')
    
    for init in inits:
        if time.time() > deadline - 20.0:
            break
            
        # 1. Local search on bang-bang
        h_bb, val_bb = _local_search_bb(init, deadline)
        if val_bb < best_val:
            best_val = val_bb
            best_h = h_bb.copy()
            
        # 2. Subgradient polish from bang-bang
        h_pol, val_pol = _subgrad_polish(h_bb, deadline)
        if val_pol < best_val:
            best_val = val_pol
            best_h = h_pol.copy()
            
        # 3. Subgradient polish from random continuous
        if time.time() > deadline - 10.0:
            continue
        h_cont_init = rng.random(N)
        h_pol2, val_pol2 = _subgrad_polish(h_cont_init, deadline)
        if val_pol2 < best_val:
            best_val = val_pol2
            best_h = h_pol2.copy()
            
    if best_h is None:
        best_h = np.full(N, 0.5)
        
    return best_h
# EVOLVE-BLOCK-END
