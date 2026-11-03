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

def _objective(x):
    """
    Objective for SLSQP: minimize max overlap with soft regularization.
    x is the variable vector of length N.
    """
    h = _project(x)
    O = _compute_O(h)
    # Use a soft max to make the objective smooth
    # tau = 50 for a good balance
    tau = 50.0
    Omax = float(O.max())
    z = tau * (O - Omax)
    ez = np.exp(z)
    return Omax + np.log(ez.sum()) / tau

def _gradient(x):
    """
    Gradient of the soft max objective.
    """
    h = _project(x)
    O = _compute_O(h)
    tau = 50.0
    Omax = float(O.max())
    z = tau * (O - Omax)
    ez = np.exp(z)
    w = ez / ez.sum()
    w_pos, w_neg = w[:N], w[N:]
    
    # Gradient of O w.r.t h
    K = np.arange(N)[:, None]
    I = np.arange(N)[None, :]
    
    ge = (I >= K)
    lt = (I < (N - K))
    
    idx_p = np.clip(I - K, 0, N - 1)
    idx_m = np.clip(I + K, 0, N - 1)
    
    Gpos = SCALE * (ge.astype(float) - ge * h[idx_p] - lt * h[idx_m])
    Gneg = SCALE * (lt.astype(float) - lt * h[idx_m] - ge * h[idx_p])
    
    grad_h = w_pos @ Gpos + w_neg @ Gneg
    
    # Chain rule: x -> h = project(x)
    # The projection is a soft thresholding, so the gradient is approximated by the active set
    # For simplicity, we return grad_h as the gradient w.r.t x, knowing that SLSQP will handle it
    # More accurately, we can use the fact that project is a proximal operator
    # d/dx (project(x)) = 1 if x_i is not at the boundary of the projection
    # But for SLSQP, it's often better to just optimize h directly with bounds and equality constraint
    
    return grad_h

def _optimize_slsqp(init, deadline):
    """
    Use SLSQP to optimize the soft max overlap.
    """
    h = _project(np.asarray(init, dtype=float).copy())
    
    # Define constraints
    constraints = [
        {'type': 'eq', 'fun': lambda x: np.sum(x) - N / 2.0, 'jac': lambda x: np.ones(N)}
    ]
    bounds = [(0.0, 1.0) for _ in range(N)]
    
    # Initial point
    x0 = h.copy()
    
    # Optimization options
    options = {
        'maxiter': 1000,
        'ftol': 1e-10,
        'disp': False
    }
    
    try:
        res = minimize(
            _objective,
            x0,
            method='SLSQP',
            jac=_gradient,
            bounds=bounds,
            constraints=constraints,
            options=options
        )
        if res.success and np.all(np.isfinite(res.x)):
            h_opt = _project(res.x)
            val = float(_compute_O(h_opt).max())
            return h_opt, val
    except Exception:
        pass
    
    return h, float(_compute_O(h).max())

def _optimize_gradient(init, deadline):
    """
    Gradient descent with projection and adaptive step size.
    """
    h = _project(np.asarray(init, dtype=float).copy())
    best_h = h.copy()
    best_val = float(_compute_O(h).max())
    
    tau = 100.0
    eta = 0.01
    
    for it in range(5000):
        if time.time() > deadline - 5.0:
            break
        
        # Compute gradient of soft max
        O = _compute_O(h)
        Omax = float(O.max())
        z = tau * (O - Omax)
        ez = np.exp(z)
        w = ez / ez.sum()
        w_pos, w_neg = w[:N], w[N:]
        
        K = np.arange(N)[:, None]
        I = np.arange(N)[None, :]
        
        ge = (I >= K)
        lt = (I < (N - K))
        
        idx_p = np.clip(I - K, 0, N - 1)
        idx_m = np.clip(I + K, 0, N - 1)
        
        Gpos = SCALE * (ge.astype(float) - ge * h[idx_p] - lt * h[idx_m])
        Gneg = SCALE * (lt.astype(float) - lt * h[idx_m] - ge * h[idx_p])
        
        grad = w_pos @ Gpos + w_neg @ Gneg
        
        # Update
        h_new = _project(h - eta * grad)
        val_new = float(_compute_O(h_new).max())
        
        if val_new < best_val:
            best_h = h_new.copy()
            best_val = val_new
            h = h_new
            # Increase eta slightly if we made progress
            eta = min(eta * 1.05, 0.1)
        else:
            h = best_h.copy()
            eta *= 0.9
            if eta < 1e-6:
                break
        
        # Increase tau gradually
        if it % 500 == 0 and it > 0:
            tau *= 2.0
    
    return best_h, best_val

def construct():
    start = time.time()
    deadline = start + 140.0  # Safety margin
    
    # Try multiple initialization strategies
    rng = np.random.default_rng(42)
    idx = np.arange(N)
    
    inits = []
    
    # 1. Uniform
    inits.append(np.full(N, 0.5))
    
    # 2. Bang-bang random
    for _ in range(5):
        bb = np.zeros(N)
        bb[rng.choice(N, N // 2, replace=False)] = 1.0
        inits.append(bb)
    
    # 3. Structured continuous
    for f in (1, 2, 3, 5):
        inits.append(0.5 + 0.4 * np.cos(2.0 * np.pi * f * idx / N))
    
    # 4. Periodic bang-bang
    for period in [2, 3, 4, 5, 7]:
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
    
    best_h = inits[0].copy()
    best_val = float(_compute_O(best_h).max())
    
    # Try SLSQP on a few promising inits
    for i, init in enumerate(inits[:6]):
        if time.time() > deadline - 30.0:
            break
        try:
            h, val = _optimize_slsqp(init, deadline)
            if val < best_val:
                best_h = h
                best_val = val
        except Exception:
            continue
    
    # Polish with gradient descent
    if time.time() < deadline - 10.0:
        h, val = _optimize_gradient(best_h, deadline)
        if val < best_val:
            best_h = h
            best_val = val
    
    # Final projection
    best_h = _project(best_h)
    for _ in range(10):
        d = N / 2.0 - best_h.sum()
        if abs(d) < 1e-12:
            break
        best_h = np.clip(best_h + d / N, 0.0, 1.0)
        best_h = _project(best_h)
    
    return best_h
