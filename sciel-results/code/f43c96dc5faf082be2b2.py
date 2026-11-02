import time
import numpy as np

N = 100
SCALE = 2.0 / N

def _compute_O_orig(h):
    S1 = np.cumsum(h[::-1])[::-1]
    S2 = np.array([np.dot(h[k:], h[:N - k]) for k in range(N)])
    O_pos = SCALE * (S1 - S2)
    csum = np.cumsum(h)
    S1n = csum[N - 1 - np.arange(N)]
    S2n = np.array([np.dot(h[:N - k], h[k:]) for k in range(N)])
    O_neg = SCALE * (S1n - S2n)
    return np.concatenate([O_pos, O_neg])

def _project(z):
    lo = float(z.min()) - 1.0
    hi = float(z.max()) + 1.0
    target = N / 2.0
    for _ in range(64):
        mid = 0.5 * (lo + hi)
        s = np.clip(z - mid, 0.0, 1.0).sum()
        if s > target:
            lo = mid
        else:
            hi = mid
    t = 0.5 * (lo + hi)
    return np.clip(z - t, 0.0, 1.0)

def _solve_qp(h_init, deadline):
    """
    Solve min ||O(h) - mean(O(h))||^2 subject to 0<=h<=1, sum(h)=N/2.
    This minimizes variance, which helps flatten the max overlap.
    We use gradient descent with projection.
    """
    h = _project(h_init.copy())
    lr = 0.01
    best_h = h.copy()
    best_val = float(_compute_O_orig(h).max())
    
    # Precompute indices for gradient calculation
    # O[k] is linear in h for fixed other h? No, O is bilinear.
    # O_pos[k] = SCALE * (S1[k] - sum h[j]h[j-k])
    # d O_pos[k] / d h[i] = SCALE * (1 - sum_j (delta_{j,i}h[j-k] + h[j]delta_{j-k,i}))
    #                    = SCALE * (1 - h[i-k] - h[i+k]) (with bounds)
    # This is similar to the soft max gradient but with uniform weights.
    
    for i in range(500):
        if time.time() > deadline - 5.0:
            break
            
        O = _compute_O_orig(h)
        mean_O = O.mean()
        # Gradient of 0.5 * ||O - mean_O||^2
        # d/dh sum (O_k - mean)^2 = 2 * sum (O_k - mean) * dO_k/dh
        # dO_k/dh is the same structure as before
        
        K = np.arange(N)[:, None]
        I = np.arange(N)[None, :]
        ge = (I >= K)
        lt = (I < (N - K))
        idx_p = np.clip(I - K, 0, N - 1)
        idx_m = np.clip(I + K, 0, N - 1)
        
        # Weights are (O_pos[k] - mean) and (O_neg[k] - mean)
        w_pos = O[:N] - mean_O
        w_neg = O[N:] - mean_O
        
        Gpos = SCALE * (ge.astype(float) - ge * h[idx_p] - lt * h[idx_m])
        Gneg = SCALE * (lt.astype(float) - lt * h[idx_m] - ge * h[idx_p])
        
        grad = 2.0 * (w_pos @ Gpos + w_neg @ Gneg)
        
        h_new = h - lr * grad
        h_new = _project(h_new)
        
        val_new = float(_compute_O_orig(h_new).max())
        if val_new < best_val:
            best_val = val_new
            best_h = h_new.copy()
        
        h = h_new
        # Adaptive learning rate
        if i % 100 == 0 and i > 0:
            lr *= 0.8
            
    return best_h, best_val

def _local_search_bb(h, deadline):
    # Convert to bang-bang
    h_bb = (h >= 0.5).astype(float)
    # Ensure sum is N/2
    diff = N / 2.0 - h_bb.sum()
    if diff > 0:
        zeros = np.where(h_bb == 0)[0]
        if len(zeros) >= int(diff):
            h_bb[zeros[:int(diff)]] = 1.0
    elif diff < 0:
        ones = np.where(h_bb == 1)[0]
        if len(ones) >= int(-diff):
            h_bb[ones[:int(-diff)]] = 0.0
            
    # Local search
    best_val = _compute_O_orig(h_bb).max()
    for _ in range(2000):
        if time.time() > deadline - 2.0:
            break
        ones = np.where(h_bb == 1)[0]
        zeros = np.where(h_bb == 0)[0]
        if len(ones) == 0 or len(zeros) == 0:
            break
        i = np.random.choice(ones)
        j = np.random.choice(zeros)
        
        h_test = h_bb.copy()
        h_test[i] = 0.0
        h_test[j] = 1.0
        
        val = _compute_O_orig(h_test).max()
        if val < best_val - 1e-9:
            h_bb = h_test
            best_val = val
    return h_bb, best_val

def construct():
    start = time.time()
    deadline = start + 145.0
    
    seed = np.full(N, 0.5)
    best_h = seed.copy()
    best_val = float(_compute_O_orig(seed).max())
    
    idx = np.arange(N)
    inits = [seed.copy()]
    for f in (1, 2, 3, 5, 8):
        inits.append(0.5 + 0.30 * np.cos(2.0 * np.pi * f * idx / N))
        inits.append(0.5 + 0.40 * np.sign(np.cos(2.0 * np.pi * f * idx / N)))
    
    rng = np.random.default_rng(0)
    for _ in range(5):
        inits.append(rng.random(N))
    
    for init in inits:
        if time.time() > deadline - 15.0:
            break
        try:
            h_cont, val_cont = _solve_qp(init, deadline)
            if val_cont < best_val:
                best_h, best_val = h_cont, val_cont
            
            # Try bang-bang local search
            h_bb, val_bb = _local_search_bb(h_cont, deadline)
            if val_bb < best_val:
                best_h, best_val = h_bb, val_bb
                
        except Exception:
            continue
            
    return best_h
