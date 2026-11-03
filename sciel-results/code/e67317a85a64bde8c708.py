import time
import numpy as np

N = 100
SCALE = 2.0 / N

def _compute_O(h):
    """
    Compute the overlap vector O for all shifts k=0..N-1.
    Linear shifts.
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

def _compute_grad_O(h):
    """
    Compute the Jacobian of O w.r.t h.
    Returns a matrix G of shape (2*N, N) where G[k, i] = d O[k] / d h[i]
    """
    # d O_pos[k] / d h[i]
    # O_pos[k] = SCALE * sum_{j=k}^{N-1} (h[j] - h[j]h[j-k])
    # d/dh[i] is non-zero if i is in [k, N-1] (as h[j]) or i+k is in [k, N-1] (as h[j-k])
    # i in [k, N-1] => term is SCALE * (1 - h[i-k])
    # i+k in [k, N-1] => i in [0, N-1-k] => term is -SCALE * h[i+k]
    
    G_pos = np.zeros((N, N))
    G_neg = np.zeros((N, N))
    
    # Vectorized construction for G_pos
    # For each k, rows k..N-1 get 1-h[i-k], rows 0..N-1-k get -h[i+k]
    # This is a bit tricky to vectorize perfectly without loops, but N=100 is small.
    for k in range(N):
        # h[i] term
        idx1 = np.arange(k, N)
        G_pos[k, idx1] += SCALE * (1 - h[idx1 - k])
        # h[i-k] term (which is h[j] in the product h[j]h[j-k] where j=i+k)
        idx2 = np.arange(0, N - k)
        G_pos[k, idx2] -= SCALE * h[idx2 + k]
        
    for k in range(N):
        # O_neg[k] = SCALE * sum_{j=0}^{N-1-k} (h[j] - h[j]h[j+k])
        idx1 = np.arange(0, N - k)
        G_neg[k, idx1] += SCALE * (1 - h[idx1 + k])
        idx2 = np.arange(k, N)
        G_neg[k, idx2] -= SCALE * h[idx2 - k]
        
    return np.vstack([G_pos, G_neg])

def _soft_max_grad(h, tau):
    """
    Compute gradient of the soft-max of O.
    """
    O = _compute_O(h)
    Omax = O.max()
    z = tau * (O - Omax)
    ez = np.exp(z)
    w = ez / ez.sum()
    
    G = _compute_grad_O(h)
    grad = w @ G
    return Omax + np.log(ez.sum()) / tau, grad

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

def _fista(init, tau, deadline, max_iter=2000):
    h = _project(np.asarray(init, dtype=float).copy())
    h_prev = h.copy()
    t = 1.0
    L = 10.0 # Initial Lipschitz estimate guess
    best_h = h.copy()
    best_val = float(_compute_O(h).max())
    
    for k in range(max_iter):
        if time.time() > deadline - 2.0:
            return best_h, best_val
            
        # Gradient
        _, grad = _soft_max_grad(h, tau)
        
        # Backtracking line search
        y = h_prev + (t - 1) / (t + 1) * (h - h_prev)
        # Actually FISTA uses y for gradient evaluation
        # Let's restart if we want to be safe, or just use h
        # Standard FISTA:
        # 1. y_k = x_k + (t_{k-1}-1)/(t_k+1) (x_k - x_{k-1})
        # 2. grad at y_k
        # 3. x_{k+1} = Proj(y_k - 1/L grad(y_k))
        # 4. t_{k+1} = (1 + sqrt(1+4t_k^2))/2
        
        y = h + (t - 1) / (t + 1) * (h - h_prev)
        _, grad_y = _soft_max_grad(y, tau)
        
        h_new = _project(y - 1.0 / L * grad_y)
        
        # Check for decrease, else backtrack L
        val_new = float(_compute_O(h_new).max())
        # We want to decrease the soft-max objective, but we track hard max
        # For simplicity, we trust the gradient direction if L is large enough
        # A proper line search would check the soft-max value
        # Given N=100, we can afford a few extra O computations
        
        if val_new < best_val:
            best_h = h_new.copy()
            best_val = val_new
            
        h_prev = h
        h = h_new
        t = (1.0 + np.sqrt(1.0 + 4.0 * t * t)) / 2.0
        
        # Adaptive L: if we don't improve, increase L?
        # Or just keep it fixed if tau is small.
        
    return best_h, best_val

def _local_search_bb(h, deadline):
    """
    Try to improve by snapping to 0/1 and swapping.
    """
    h = _project(h)
    best_h = h.copy()
    best_val = float(_compute_O(h).max())
    
    # Snap to bang-bang
    # Sort by value, keep top N/2 as 1, rest as 0?
    # Or just project and see if it's close to 0/1
    
    # Simple local search: try flipping pairs
    idx = np.arange(N)
    rng = np.random.default_rng(123)
    
    for _ in range(5000):
        if time.time() > deadline - 2.0:
            break
            
        # Pick two random indices
        i, j = rng.choice(N, 2, replace=False)
        
        # Try swapping values? Or flipping 0->1 and 1->0?
        # Since sum is fixed, we must swap a 1 with a 0, or swap values.
        # Let's try swapping values h[i] and h[j]
        h_new = h.copy()
        h_new[i], h_new[j] = h[j], h[i]
        
        val_new = float(_compute_O(h_new).max())
        if val_new < best_val - 1e-9:
            best_h = h_new
            best_val = val_new
            h = h_new
            
        # Also try flipping if possible? No, sum constraint.
        
    return best_h, best_val

def construct():
    start = time.time()
    deadline = start + 145.0
    
    seed = np.full(N, 0.5)
    best_h = seed.copy()
    best_val = float(_compute_O(seed).max())
    
    idx = np.arange(N)
    inits = []
    
    # 1. Bang-bang random inits
    rng = np.random.default_rng(42)
    for _ in range(10):
        bb = np.zeros(N)
        bb[rng.choice(N, N // 2, replace=False)] = 1.0
        inits.append(bb)
        
    # 2. Structured continuous inits
    for f in (1, 2, 3, 5, 7):
        inits.append(0.5 + 0.4 * np.cos(2.0 * np.pi * f * idx / N))
        
    # 3. Random continuous
    for _ in range(10):
        inits.append(rng.random(N))
        
    # 4. "Sawtooth" or periodic bang-bang
    for period in [3, 4, 5, 7, 11, 13]:
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

    # Optimization loop
    for init in inits:
        if time.time() > deadline - 20.0:
            break
        try:
            # Use FISTA with moderate tau
            h, val = _fista(init, tau=10.0, deadline=deadline)
            if np.all(np.isfinite(h)) and val < best_val:
                best_h = h
                best_val = val
        except Exception:
            continue

    # Final polish with local search
    if time.time() < deadline - 5.0:
        h, val = _local_search_bb(best_h, deadline)
        if val < best_val:
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
