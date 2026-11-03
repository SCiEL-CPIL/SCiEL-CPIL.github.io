import time
import numpy as np
from scipy.optimize import minimize

N = 100
SCALE = 2.0 / N

def _compute_O(h):
    """
    Compute the overlap vector O for all shifts k=0..N-1.
    O_pos[k] = SCALE * sum_{j=k}^{N-1} h[j]*(1-h[j-k])
    O_neg[k] = SCALE * sum_{j=0}^{N-1-k} h[j]*(1-h[j+k])
    """
    S1 = np.cumsum(h[::-1])[::-1]
    S2 = np.array([np.dot(h[k:], h[:N - k]) for k in range(N)])
    O_pos = SCALE * (S1 - S2)
    
    csum = np.cumsum(h)
    S1n = csum[N - 1 - np.arange(N)]
    S2n = np.array([np.dot(h[:N - k], h[k:]) for k in range(N)])
    O_neg = SCALE * (S1n - S2n)
    
    return np.concatenate([O_pos, O_neg])

def _project(z):
    """
    Project z onto {h in [0,1], sum h = N/2}
    """
    lo = float(z.min()) - 1.0
    hi = float(z.max()) + 1.0
    target = N / 2.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        s = np.clip(z - mid, 0.0, 1.0).sum()
        if s > target:
            lo = mid
        else:
            hi = mid
    t = 0.5 * (lo + hi)
    return np.clip(z - t, 0.0, 1.0)

def _sa_optimize(init, deadline, rng):
    """
    Simulated Annealing optimization.
    """
    h = _project(init.copy())
    current_val = float(_compute_O(h).max())
    best_h = h.copy()
    best_val = current_val
    
    T_start = 0.05
    T_end = 0.001
    steps = 5000
    alpha = (T_end / T_start) ** (1.0 / steps)
    
    T = T_start
    
    for i in range(steps):
        if time.time() > deadline - 2.0:
            break
            
        # Generate neighbor
        # Strategy: Random perturbation of a few elements
        # To maintain sum constraint roughly, we can pick 2 indices and swap/adjust
        # Or just add noise and project. Projecting is robust.
        
        # Add Gaussian noise to a random subset
        n_perturb = rng.integers(5, 20)
        idxs = rng.choice(N, n_perturb, replace=False)
        noise = rng.normal(0, 0.1, n_perturb)
        h_new = h.copy()
        h_new[idxs] += noise
        
        h_new = _project(h_new)
        
        new_val = float(_compute_O(h_new).max())
        
        delta = new_val - current_val
        
        if delta < 0:
            # Accept improvement
            h = h_new
            current_val = new_val
            if new_val < best_val:
                best_val = new_val
                best_h = h_new.copy()
        else:
            # Accept with probability exp(-delta / T)
            if rng.random() < np.exp(-delta / T):
                h = h_new
                current_val = new_val
        
        T *= alpha
        
    return best_h, best_val

def construct():
    start = time.time()
    deadline = start + 145.0  # Safety margin

    best_h = np.full(N, 0.5)
    best_val = float(_compute_O(best_h).max())
    
    rng = np.random.default_rng(42)
    
    # Generate diverse initializations
    inits = []
    
    # 1. Bang-bang random
    for _ in range(10):
        bb = np.zeros(N)
        bb[rng.choice(N, N // 2, replace=False)] = 1.0
        inits.append(bb)
        
    # 2. Structured continuous
    idx = np.arange(N)
    for f in (1, 2, 3, 5):
        inits.append(0.5 + 0.4 * np.cos(2.0 * np.pi * f * idx / N))
        
    # 3. Random continuous
    for _ in range(10):
        inits.append(rng.random(N))
        
    # 4. Periodic bang-bang
    for period in [3, 4, 5, 7, 11]:
        bb = np.zeros(N)
        ones = period // 2
        pattern = np.zeros(period)
        pattern[:ones] = 1.0
        bb = np.tile(pattern, N // period + 1)[:N]
        if bb.sum() > N/2:
             idxs = np.where(bb==1)[0]
             to_remove = rng.choice(idxs, int(bb.sum() - N/2), replace=False)
             bb[to_remove] = 0
        elif bb.sum() < N/2:
             idxs = np.where(bb==0)[0]
             to_add = rng.choice(idxs, int(N/2 - bb.sum()), replace=False)
             bb[to_add] = 1
        inits.append(bb)

    for init in inits:
        if time.time() > deadline - 10.0:
            break
        try:
            h, val = _sa_optimize(init, deadline, rng)
            if np.all(np.isfinite(h)) and val < best_val:
                best_h = h
                best_val = val
        except Exception:
            continue
            
    # Final polish: A few steps of gradient descent on the best SA result
    # to squeeze out a bit more if possible, using the existing gradient logic
    # but we removed it. Let's just do a fine-grained SA or local search.
    
    # Let's do a local search: try swapping pairs of values
    h = best_h.copy()
    val = best_val
    for _ in range(500):
        if time.time() > deadline - 1.0:
            break
        i, j = rng.choice(N, 2, replace=False)
        # Try moving mass from i to j
        delta = 0.05
        h_new = h.copy()
        h_new[i] -= delta
        h_new[j] += delta
        h_new = _project(h_new)
        new_val = float(_compute_O(h_new).max())
        if new_val < val:
            h = h_new
            val = new_val
            best_h = h
            best_val = val
        else:
            # Try reverse
            h_new2 = h.copy()
            h_new2[i] += delta
            h_new2[j] -= delta
            h_new2 = _project(h_new2)
            new_val2 = float(_compute_O(h_new2).max())
            if new_val2 < val:
                h = h_new2
                val = new_val2
                best_h = h
                best_val = val

    # Ensure exact sum and bounds
    best_h = _project(best_h)
    for _ in range(10):
        d = N / 2.0 - best_h.sum()
        if abs(d) < 1e-12:
            break
        best_h = np.clip(best_h + d / N, 0.0, 1.0)
        best_h = _project(best_h)
        
    return best_h
