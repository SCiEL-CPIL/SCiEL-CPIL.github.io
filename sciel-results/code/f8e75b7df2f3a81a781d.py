"""
Packing 300 equal circles in a unit circle using Simulated Annealing + L-BFGS-B polish.
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 300
TIME_LIMIT = 115.0  # Safety margin for 120s/150s limits

def _hex_init(n=N):
    """Generate a hexagonal lattice initialization."""
    # Target radius ~0.053 -> spacing ~0.106. Start slightly tighter to force expansion.
    spacing = 0.105 
    k = int(1.1 / spacing) + 2
    xs, ys = [], []
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3.0) / 2.0
            xs.append(x)
            ys.append(y)
    pts = np.column_stack([xs, ys])
    r = np.hypot(pts[:, 0], pts[:, 1])
    inside = r <= 1.0
    pts = pts[inside]
    r = r[inside]
    if len(pts) < n:
        # Fallback if grid too small (unlikely for these params)
        spacing = 0.09
        k = int(1.1 / spacing) + 2
        xs, ys = [], []
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                xs.append(x)
                ys.append(y)
        pts = np.column_stack([xs, ys])
        r = np.hypot(pts[:, 0], pts[:, 1])
        inside = r <= 1.0
        pts = pts[inside]
        r = r[inside]
        
    order = np.argsort(r, kind="stable")
    return pts[order[:n]].copy()

def _score(pts):
    """Calculate the maximum valid radius for a given configuration."""
    n = len(pts)
    if n < 2:
        return 1.0 - np.max(np.hypot(pts[:, 0], pts[:, 1])) if n > 0 else 1.0
    
    # Pairwise distances
    diff = pts[:, None, :] - pts[None, :, :]
    d2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(d2, np.inf)
    min_pair = np.sqrt(np.min(d2)) / 2.0
    
    # Boundary distance
    min_bound = 1.0 - np.max(np.hypot(pts[:, 0], pts[:, 1]))
    
    return min(min_pair, min_bound)

def _sa_optimize(pts, start_time, time_limit):
    """Simulated Annealing to find a good local minimum."""
    n = len(pts)
    pts = pts.copy()
    # Ensure initial validity
    rad = np.hypot(pts[:, 0], pts[:, 1])
    out = rad > 1.0
    if np.any(out):
        pts[out] = pts[out] / rad[out, None]
        
    best_pts = pts.copy()
    best_score = _score(pts)
    
    # SA Parameters
    T = 0.01  # Initial temperature
    cooling_rate = 0.999
    min_T = 1e-6
    max_iter = 50000
    
    # Precompute neighbor pairs? No, random selection is better for global escape.
    # We will pick random pairs to move.
    
    current_score = best_score
    
    for i in range(max_iter):
        elapsed = time.time() - start_time
        if elapsed > time_limit:
            break
            
        if T < min_T:
            break
            
        # Pick a random circle
        idx = np.random.randint(n)
        
        # Generate a random perturbation
        # Scale perturbation with temperature
        step_size = 0.01 * (T / 0.01) + 0.001
        dx, dy = np.random.normal(0, step_size, 2)
        
        new_pts = pts.copy()
        new_pts[idx, 0] += dx
        new_pts[idx, 1] += dy
        
        # Keep inside container (project if needed)
        r = np.hypot(new_pts[idx, 0], new_pts[idx, 1])
        if r > 1.0:
            new_pts[idx] = new_pts[idx] / r
            
        # Calculate score
        new_score = _score(new_pts)
        
        # Accept if better, or with probability exp((new_score - current_score) / T)
        # Note: We are maximizing score.
        # Delta = new_score - current_score. If positive, accept.
        # If negative, accept with prob exp(Delta / T).
        
        delta = new_score - current_score
        if delta > 0 or np.random.rand() < np.exp(delta / T):
            pts = new_pts
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_pts = pts.copy()
        
        T *= cooling_rate
        
    return best_pts, best_score

def _polish(pts, start_time, time_limit):
    """Polish using L-BFGS-B with a penalty method."""
    n = len(pts)
    pts = pts.copy()
    rad = np.hypot(pts[:, 0], pts[:, 1])
    out = rad > 1.0
    if np.any(out):
        pts[out] = pts[out] / rad[out, None]
        
    best_pts = pts.copy()
    best_score = _score(pts)
    
    # Initial t guess
    t0 = best_score * 0.99
    
    def objective_and_grad(z):
        t = z[0]
        p = z[1:].reshape(n, 2)
        
        # Project to container
        r = np.hypot(p[:, 0], p[:, 1])
        mask = r > 1.0
        if np.any(mask):
            p[mask] = p[mask] / r[mask, None]
            r[mask] = 1.0
            
        diff = p[:, None, :] - p[None, :, :]
        d2 = np.sum(diff * diff, axis=2)
        np.fill_diagonal(d2, 1e10)
        d = np.sqrt(d2)
        
        b = 1.0 - r
        
        viol_pair = np.maximum(0.0, 2.0 * t - d)
        viol_bound = np.maximum(0.0, t - b)
        
        obj = -t + 0.5 * (np.sum(viol_pair * viol_pair) + np.sum(viol_bound * viol_bound))
        
        grad = np.zeros_like(z)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            unit_vec = diff / d[:, :, None]
        unit_vec = np.where(np.isfinite(unit_vec), unit_vec, 0.0)
        
        w_pair = 2.0 * viol_pair
        grad_p_pair = np.sum(w_pair[:, :, None] * unit_vec, axis=1)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            inward_vec = p / r[:, None]
        inward_vec = np.where(np.isfinite(inward_vec), inward_vec, 0.0)
        
        grad_p_bound = viol_bound[:, None] * inward_vec
        
        grad_p = -grad_p_pair + grad_p_bound
        
        grad[0] = -1.0 + 4.0 * np.sum(viol_pair) + np.sum(viol_bound)
        grad[1:] = grad_p.reshape(-1)
        
        return obj, grad

    def objective_only(z):
        return objective_and_grad(z)[0]
        
    def grad_only(z):
        return objective_and_grad(z)[1]

    z0 = np.concatenate(([t0], pts.flatten()))
    bounds = [(t0 * 0.5, 1.0)] + [(-1.0, 1.0)] * (2 * n)
    
    current_z = z0.copy()
    
    while True:
        elapsed = time.time() - start_time
        if elapsed > time_limit:
            break
            
        res = minimize(
            objective_only,
            current_z,
            jac=grad_only,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        
        if res.success:
            current_z = res.x
            new_pts = current_z[1:].reshape(n, 2)
            r = np.hypot(new_pts[:, 0], new_pts[:, 1])
            out = r > 1.0
            if np.any(out):
                new_pts[out] = new_pts[out] / r[out, None]
            
            new_score = _score(new_pts)
            
            if new_score > best_score:
                best_score = new_score
                best_pts = new_pts.copy()
            else:
                break
        else:
            current_z = res.x
            if time.time() - start_time > time_limit:
                break

    return best_pts, best_score

def construct():
    start_time = time.time()
    
    # 1. Initialize
    pts = _hex_init(N)
    rad = np.hypot(pts[:, 0], pts[:, 1])
    out = rad > 1.0
    if np.any(out):
        pts[out] = pts[out] / rad[out, None]
        
    best_pts = pts.copy()
    best_score = _score(pts)
    
    # 2. Simulated Annealing Phase (use ~60s)
    sa_time_limit = start_time + 60.0
    try:
        sa_pts, sa_score = _sa_optimize(pts, start_time, sa_time_limit)
        if sa_score > best_score:
            best_pts = sa_pts
            best_score = sa_score
    except Exception:
        pass
    
    # 3. Polish Phase (use remaining time, ~50s)
    polish_time_limit = start_time + TIME_LIMIT
    try:
        pol_pts, pol_score = _polish(best_pts, start_time, polish_time_limit)
        if pol_score > best_score:
            best_pts = pol_pts
            best_score = pol_score
    except Exception:
        pass
    
    # Final Safety Check
    best_pts = np.asarray(best_pts, dtype=float)
    if best_pts.shape != (N, 2):
        best_pts = _hex_init(N)
    if not np.all(np.isfinite(best_pts)):
        best_pts = _hex_init(N)
    rad = np.hypot(best_pts[:, 0], best_pts[:, 1])
    out = rad > 1.0
    if np.any(out):
        best_pts[out] = best_pts[out] / rad[out, None]
        
    return best_pts

if __name__ == "__main__":
    # Quick test
    t0 = time.time()
    res = construct()
    print(f"Time: {time.time() - t0:.2f}s")
    print(f"Score: {_score(res)}")
