import numpy as np
import time
from scipy.optimize import minimize

N = 50

def _violation_and_grad(pts, r):
    """
    Computes the violation and its gradient for a given radius r.
    pts: (N, 2) array of centers.
    r: target radius.
    Returns: (violation, grad)
    """
    n = len(pts)
    grad = np.zeros_like(pts)
    viol = 0.0
    
    margin = 0.5 - r
    
    # Boundary violations
    # Right: x > 0.5 - r
    viol_x_r = np.maximum(0, pts[:, 0] - margin)
    viol += np.sum(viol_x_r ** 2)
    grad[:, 0] += 2 * viol_x_r
    
    # Left: x < -0.5 + r
    viol_x_l = np.maximum(0, -pts[:, 0] - margin)
    viol += np.sum(viol_x_l ** 2)
    grad[:, 0] -= 2 * viol_x_l
    
    # Top: y > 0.5 - r
    viol_y_t = np.maximum(0, pts[:, 1] - margin)
    viol += np.sum(viol_y_t ** 2)
    grad[:, 1] += 2 * viol_y_t
    
    # Bottom: y < -0.5 + r
    viol_y_b = np.maximum(0, -pts[:, 1] - margin)
    viol += np.sum(viol_y_b ** 2)
    grad[:, 1] -= 2 * viol_y_b
    
    # Pairwise violations
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff ** 2, axis=2)
    dist = np.sqrt(dist_sq)
    
    dist_safe = np.where(dist < 1e-12, 1e-12, dist)
    
    overlap = np.maximum(0, 2 * r - dist)
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    
    G = -2.0 * overlap * diff / dist_safe[:, :, np.newaxis]
    G_masked = G * mask[:, :, np.newaxis]
    
    pair_grad = np.sum(G_masked, axis=1)
    grad += pair_grad
    
    viol += np.sum(overlap[mask] ** 2)
    
    return viol, grad

def _objective(pts, r):
    v, _ = _violation_and_grad(pts, r)
    return v

def _grad_objective(pts, r):
    _, g = _violation_and_grad(pts, r)
    return g

def _hex_initial():
    s = 0.15
    dx = s
    dy = s * np.sqrt(3.0) / 2.0
    xs = np.arange(-0.5, 0.5 + 1e-9, dx)
    pts = []
    row = 0
    y = -0.5
    while y <= 0.5 + 1e-9:
        offset = (row % 2) * (dx / 2.0)
        for x in xs:
            xx = x + offset
            if -0.5 <= xx <= 0.5 and -0.5 <= y <= 0.5:
                pts.append([xx, y])
        y += dy
        row += 1
    pts = np.array(pts, dtype=float)
    if len(pts) < N:
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        pts = np.vstack([pts, grid[len(pts):]])
    else:
        d2 = np.sum(pts * pts, axis=1)
        order = np.argsort(d2)
        pts = pts[order[:N]]
    return pts.copy()

def _solve_for_radius(r, initial_pts, deadline):
    x0 = initial_pts.flatten()
    
    def obj(x):
        return _objective(x.reshape(-1, 2), r)
        
    def grad(x):
        return _grad_objective(x.reshape(-1, 2), r)

    try:
        res = minimize(
            obj, x0, jac=grad, method='L-BFGS-B',
            bounds=[(-0.5, 0.5)] * N * 2,
            options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10, 'maxfun': 2000}
        )
    except Exception:
        return False, initial_pts, 1e6
    
    best_pts = res.x.reshape(-1, 2)
    best_viol = res.fun
    
    if time.time() > deadline:
        return best_viol < 1e-8, best_pts, best_viol
        
    return best_viol < 1e-8, best_pts, best_viol

def construct():
    start = time.time()
    deadline = start + 110.0
    
    pts_init = _hex_initial()
    
    r_low = 0.06
    r_high = 0.075
    
    best_pts = pts_init
    best_r = r_low
    
    while r_high - r_low > 1e-6 and time.time() < deadline:
        r_mid = (r_low + r_high) / 2.0
        
        success, pts_try, viol = _solve_for_radius(r_mid, best_pts, deadline)
        
        if success:
            r_low = r_mid
            best_pts = pts_try
            best_r = r_mid
        else:
            r_high = r_mid
            
    best_pts = np.clip(best_pts, -0.5, 0.5)
    
    return best_pts
