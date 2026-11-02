"""Circle packing in unit square for N=26 circles."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import Delaunay

N = 26

# EVOLVE-BLOCK-START

def _validate_and_score(circles):
    """Check validity and compute score. circles: (N,3) array [x,y,r]."""
    if circles.shape != (N, 3):
        return None
    x = circles[:, 0]
    y = circles[:, 1]
    r = circles[:, 2]
    if np.any(r <= 0):
        return None
    if np.any(x - r < -1e-9) or np.any(x + r > 1 + 1e-9) or np.any(y - r < -1e-9) or np.any(y + r > 1 + 1e-9):
        return None
    i_idx, j_idx = np.triu_indices(N, k=1)
    dx = x[i_idx] - x[j_idx]
    dy = y[i_idx] - y[j_idx]
    dist = np.sqrt(dx * dx + dy * dy)
    if np.any(dist < r[i_idx] + r[j_idx] - 1e-9):
        return None
    return float(np.sum(r))


def _grid_initial(n=N):
    """Create a grid-based initial configuration."""
    side = int(np.ceil(np.sqrt(n)))
    cells = []
    for i in range(side):
        for j in range(side):
            cx = (i + 0.5) / side
            cy = (j + 0.5) / side
            r = 0.5 / side - 1e-4
            cells.append((cx, cy, r))
    return np.array(cells[:n])


def _random_initial(n=N, seed=42):
    """Create a random initial configuration with non-overlapping circles."""
    rng = np.random.RandomState(seed)
    circles = np.zeros((n, 3))
    circles[:, 2] = 0.01 + 0.01 * rng.rand(n)
    circles[:, 0] = circles[:, 2] + (1 - 2 * circles[:, 2]) * rng.rand(n)
    circles[:, 1] = circles[:, 2] + (1 - 2 * circles[:, 2]) * rng.rand(n)
    return circles


def _structured_initial(n=N, seed=0):
    """Create a structured initial configuration based on hexagonal packing."""
    rng = np.random.RandomState(seed)
    circles = np.zeros((n, 3))
    
    # Hexagonal grid: 5 cols x 6 rows = 30, use 26
    cols = 5
    rows = 6
    dx = 1.0 / (cols + 0.5)
    dy = 1.0 / (rows + 0.5)
    r_base = min(dx, dy) * 0.48
    
    idx = 0
    for row in range(rows):
        for col in range(cols):
            if idx >= n:
                break
            x_offset = 0.5 * dx if row % 2 == 1 else 0
            cx = (col + 0.5 + x_offset) * dx
            cy = (row + 0.5) * dy
            cx += 0.01 * rng.randn()
            cy += 0.01 * rng.randn()
            cx = np.clip(cx, r_base, 1 - r_base)
            cy = np.clip(cy, r_base, 1 - r_base)
            circles[idx, 0] = cx
            circles[idx, 1] = cy
            circles[idx, 2] = r_base
            idx += 1
    
    return circles


def _resolve_overlaps(circles, max_iter=500):
    """Iteratively resolve overlaps by pushing circles apart."""
    x = circles[:, 0].copy()
    y = circles[:, 1].copy()
    r = circles[:, 2].copy()
    n = len(r)
    i_idx, j_idx = np.triu_indices(n, k=1)
    
    for it in range(max_iter):
        dx = x[i_idx] - x[j_idx]
        dy = y[i_idx] - y[j_idx]
        dist = np.sqrt(dx * dx + dy * dy)
        overlap = r[i_idx] + r[j_idx] - dist
        
        if not np.any(overlap > 1e-9):
            break
        
        for k in range(len(i_idx)):
            if overlap[k] > 1e-9:
                i = i_idx[k]
                j = j_idx[k]
                if dist[k] < 1e-12:
                    dx_k = 1e-6
                    dy_k = 0
                else:
                    dx_k = dx[k] / dist[k]
                    dy_k = dy[k] / dist[k]
                
                push = overlap[k] * 0.5
                x[i] -= push * dx_k
                y[i] -= push * dy_k
                x[j] += push * dx_k
                y[j] += push * dy_k
                
                shrink = overlap[k] * 0.1
                r[i] = max(r[i] - shrink, 0.001)
                r[j] = max(r[j] - shrink, 0.001)
        
        x = np.clip(x, 0, 1)
        y = np.clip(y, 0, 1)
        max_r = np.minimum(np.minimum(x, 1 - x), np.minimum(y, 1 - y))
        r = np.minimum(r, np.maximum(max_r, 0.001))
    
    return np.column_stack([x, y, r])


def _force_optimize(circles, time_limit=20, seed=0):
    """Physics-based force optimization with adaptive step size."""
    x = circles[:, 0].copy()
    y = circles[:, 1].copy()
    r = circles[:, 2].copy()
    n = len(r)
    
    start_time = time.time()
    best_score = _validate_and_score(np.column_stack([x, y, r]))
    best_circles = np.column_stack([x, y, r]).copy()
    if best_score is None:
        best_score = 0.0
    
    i_idx, j_idx = np.triu_indices(n, k=1)
    
    lr = 0.008
    decay = 0.99999
    momentum_x = np.zeros(n)
    momentum_y = np.zeros(n)
    momentum_r = np.zeros(n)
    beta = 0.90
    
    def _get_near_pairs(cx, cy):
        points = np.column_stack([cx, cy])
        try:
            tri = Delaunay(points)
            edges = set()
            for simplex in tri.simplices:
                for i in range(3):
                    for j in range(i+1, 3):
                        a, b = simplex[i], simplex[j]
                        if a > b:
                            a, b = b, a
                        edges.add((a, b))
            if not edges:
                return i_idx, j_idx
            edge_list = sorted(edges)
            return np.array([e[0] for e in edge_list]), np.array([e[1] for e in edge_list])
        except Exception:
            return i_idx, j_idx
    
    it = 0
    no_improve = 0
    max_no_improve = 500
    
    while True:
        if time.time() - start_time > time_limit:
            break
        
        if it % 30 == 0:
            d_i, d_j = _get_near_pairs(x, y)
        
        dx = x[d_i] - x[d_j]
        dy = y[d_i] - y[d_j]
        dist = np.sqrt(dx * dx + dy * dy)
        dist = np.maximum(dist, 1e-12)
        
        overlap = r[d_i] + r[d_j] - dist
        
        gx = np.zeros(n)
        gy = np.zeros(n)
        gr = np.zeros(n)
        
        mask = overlap > 0
        if np.any(mask):
            rep = 200.0 * overlap[mask] / dist[mask]
            gx[d_i[mask]] -= rep * dx[mask] / dist[mask]
            gy[d_i[mask]] -= rep * dy[mask] / dist[mask]
            gx[d_j[mask]] += rep * dx[mask] / dist[mask]
            gy[d_j[mask]] += rep * dy[mask] / dist[mask]
            gr[d_i[mask]] += 150.0
            gr[d_j[mask]] += 150.0
        
        gx += np.where(x < r, 350.0, 0.0) - np.where(1 - x < r, 350.0, 0.0)
        gy += np.where(y < r, 350.0, 0.0) - np.where(1 - y < r, 350.0, 0.0)
        
        growth_pressure = 3.0 + 2.0 * (it / max(1, int(time_limit * 50)))
        gr -= growth_pressure
        
        momentum_x = beta * momentum_x + (1 - beta) * gx
        momentum_y = beta * momentum_y + (1 - beta) * gy
        momentum_r = beta * momentum_r + (1 - beta) * gr
        
        x += lr * momentum_x
        y += lr * momentum_y
        r += lr * momentum_r
        
        x = np.clip(x, 0.001, 0.999)
        y = np.clip(y, 0.001, 0.999)
        r = np.maximum(r, 0.001)
        max_r = np.minimum(np.minimum(x, 1 - x), np.minimum(y, 1 - y))
        r = np.minimum(r, np.maximum(max_r, 0.001))
        
        lr *= decay
        it += 1
        
        if it % 10 == 0:
            current_circles = np.column_stack([x, y, r])
            score = _validate_and_score(current_circles)
            if score is not None and score > best_score:
                no_improve = 0
                best_score = score
                best_circles = current_circles.copy()
            else:
                no_improve += 1
            
            if no_improve >= max_no_improve:
                break
    
    final_circles = np.column_stack([x, y, r])
    final_score = _validate_and_score(final_circles)
    if final_score is not None and final_score > best_score:
        best_score = final_score
        best_circles = final_circles
    
    return best_circles, best_score


def _lbfgs_optimize(circles, time_limit=15, seed=0):
    """Use L-BFGS-B with smooth log-sum-exp penalty for robust local optimization."""
    x = circles[:, 0].copy()
    y = circles[:, 1].copy()
    r = circles[:, 2].copy()
    n = len(r)
    
    start_time = time.time()
    i_idx, j_idx = np.triu_indices(n, k=1)
    params = np.concatenate([x, y, r])
    
    pw = 100.0
    beta_lse = 80.0
    
    def _lse(v):
        if len(v) == 0:
            return 0.0
        v_max = np.max(v)
        if v_max < 1e-15:
            return 0.0
        return float(v_max + np.log(np.sum(np.exp(v - v_max)))) / beta_lse
    
    def objective(p):
        if time.time() - start_time > time_limit:
            return 1e10
        xx = p[:n]
        yy = p[n:2*n]
        rr = p[2*n:3*n]
        
        bv = np.maximum(0, rr - xx) + np.maximum(0, rr - (1 - xx)) + \
              np.maximum(0, rr - yy) + np.maximum(0, rr - (1 - yy))
        
        dx = xx[i_idx] - xx[j_idx]
        dy = yy[i_idx] - yy[j_idx]
        dist = np.sqrt(dx * dx + dy * dy)
        dist = np.maximum(dist, 1e-12)
        ov = np.maximum(0, rr[i_idx] + rr[j_idx] - dist)
        
        bv_pen = _lse(pw * bv) if np.max(bv) > 0 else 0.0
        ov_pen = _lse(pw * ov) if np.max(ov) > 0 else 0.0
        
        return -np.sum(rr) + bv_pen + ov_pen
    
    def gradient(p):
        if time.time() - start_time > time_limit:
            return np.zeros_like(p)
        xx = p[:n]
        yy = p[n:2*n]
        rr = p[2*n:3*n]
        
        gx = np.zeros(n)
        gy = np.zeros(n)
        gr = np.zeros(n)
        
        gx += np.where(xx < rr, pw, 0.0) - np.where(1 - xx < rr, pw, 0.0)
        gy += np.where(yy < rr, pw, 0.0) - np.where(1 - yy < rr, pw, 0.0)
        gr += np.where(xx < rr, pw, 0.0) + np.where(1 - xx < rr, pw, 0.0)
        gr += np.where(yy < rr, pw, 0.0) + np.where(1 - yy < rr, pw, 0.0)
        
        dx = xx[i_idx] - xx[j_idx]
        dy = yy[i_idx] - yy[j_idx]
        dist = np.sqrt(dx * dx + dy * dy)
        dist = np.maximum(dist, 1e-12)
        overlap = rr[i_idx] + rr[j_idx] - dist
        
        mask = overlap > 0
        if np.any(mask):
            ov_masked = overlap[mask]
            w = np.exp(pw * ov_masked) / np.sum(np.exp(pw * ov_masked))
            rep = pw * w / dist[mask]
            gx[i_idx[mask]] -= rep * dx[mask] / dist[mask]
            gy[i_idx[mask]] -= rep * dy[mask] / dist[mask]
            gx[j_idx[mask]] += rep * dx[mask] / dist[mask]
            gy[j_idx[mask]] += rep * dy[mask] / dist[mask]
            gr[i_idx[mask]] += pw * w
            gr[j_idx[mask]] += pw * w
        
        gr -= 1.0
        
        return np.concatenate([gx, gy, gr])
    
    bounds = [(0.001, 0.999)] * n + [(0.001, 0.999)] * n + [(0.001, 0.49)] * n
    
    try:
        result = minimize(objective, params, method='L-BFGS-B', jac=gradient,
                          bounds=bounds, options={'maxiter': 800, 'maxfun': 5000,
                                                   'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 50})
    except Exception:
        return circles.copy(), _validate_and_score(circles) or 0.0
    
    if time.time() - start_time > time_limit:
        return circles.copy(), _validate_and_score(circles) or 0.0
    
    new_params = result.x
    new_x = np.clip(new_params[:n], 0.001, 0.999)
    new_y = np.clip(new_params[n:2*n], 0.001, 0.999)
    new_r = np.maximum(new_params[2*n:3*n], 0.001)
    max_r = np.minimum(np.minimum(new_x, 1 - new_x), np.minimum(new_y, 1 - new_y))
    new_r = np.minimum(new_r, np.maximum(max_r, 0.001))
    
    new_circles = np.column_stack([new_x, new_y, new_r])
    new_score = _validate_and_score(new_circles)
    input_score = _validate_and_score(circles)
    if input_score is None:
        input_score = 0.0
    if new_score is None:
        new_score = 0.0
    
    if new_score >= input_score:
        return new_circles, new_score
    else:
        return circles.copy(), input_score


def _simulated_annealing(circles, time_limit=30, seed=0):
    """Simulated annealing with fast penalty-based scoring."""
    rng = np.random.RandomState(seed)
    x = circles[:, 0].copy()
    y = circles[:, 1].copy()
    r = circles[:, 2].copy()
    n = len(r)
    
    start_time = time.time()
    i_idx, j_idx = np.triu_indices(n, k=1)
    
    def _penalty_score(cx, cy, cr, pw=80.0):
        if np.any(cr <= 0):
            return -1e6
        bv = np.maximum(0, cr - cx) + np.maximum(0, cr - (1 - cx)) + \
              np.maximum(0, cr - cy) + np.maximum(0, cr - (1 - cy))
        dx = cx[i_idx] - cx[j_idx]
        dy = cy[i_idx] - cy[j_idx]
        dist = np.sqrt(dx * dx + dy * dy)
        ov = np.maximum(0, cr[i_idx] + cr[j_idx] - dist)
        return float(np.sum(cr) - pw * (np.sum(bv) + np.sum(ov)))
    
    def _strict_score(cx, cy, cr):
        if np.any(cr <= 0):
            return None
        if np.any(cx - cr < -1e-9) or np.any(cx + cr > 1 + 1e-9) or \
           np.any(cy - cr < -1e-9) or np.any(cy + cr > 1 + 1e-9):
            return None
        dx = cx[i_idx] - cx[j_idx]
        dy = cy[i_idx] - cy[j_idx]
        dist = np.sqrt(dx * dx + dy * dy)
        if np.any(dist < cr[i_idx] + cr[j_idx] - 1e-9):
            return None
        return float(np.sum(cr))
    
    best_strict_score = _strict_score(x, y, r)
    best_circles = np.column_stack([x, y, r]).copy()
    if best_strict_score is None:
        best_strict_score = 0.0
    
    current_ps = _penalty_score(x, y, r)
    
    t_init = 0.06
    t_min = 0.0005
    max_iter = 60000
    cooling = (t_min / t_init) ** (1.0 / max_iter)
    t = t_init
    
    for it in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        i = rng.randint(n)
        move_type = rng.rand()
        
        new_x = x.copy()
        new_y = y.copy()
        new_r = r.copy()
        
        if move_type < 0.40:
            delta = 0.015 * rng.randn(2)
            new_x[i] += delta[0]
            new_y[i] += delta[1]
        elif move_type < 0.70:
            new_r[i] *= (1 + 0.08 * rng.randn())
            new_r[i] = max(0.005, min(new_r[i], 0.5))
        elif move_type < 0.85:
            j = rng.randint(n)
            if j != i:
                new_r[i], new_r[j] = new_r[j], new_r[i]
        else:
            j = rng.randint(n)
            if j != i:
                delta = 0.005 * rng.randn()
                new_r[i] += delta
                new_r[j] -= delta
                new_r[i] = max(0.005, new_r[i])
                new_r[j] = max(0.005, new_r[j])
        
        new_ps = _penalty_score(new_x, new_y, new_r)
        delta = new_ps - current_ps
        if delta > 0 or rng.rand() < np.exp(delta / max(t, 1e-10)):
            x, y, r = new_x, new_y, new_r
            current_ps = new_ps
            
            if it % 40 == 0:
                s = _strict_score(x, y, r)
                if s is not None and s > best_strict_score:
                    best_strict_score = s
                    best_circles = np.column_stack([x, y, r]).copy()
        
        t *= cooling
    
    s = _strict_score(x, y, r)
    if s is not None and s > best_strict_score:
        best_strict_score = s
        best_circles = np.column_stack([x, y, r]).copy()
    
    return best_circles, best_strict_score


def _delaunay_polish(circles, time_limit=15, seed=0):
    """High-precision polish using Delaunay triangulation for contact pairs."""
    x = circles[:, 0].copy()
    y = circles[:, 1].copy()
    r = circles[:, 2].copy()
    n = len(r)
    
    start_time = time.time()
    best_score = _validate_and_score(np.column_stack([x, y, r]))
    best_circles = np.column_stack([x, y, r]).copy()
    if best_score is None:
        best_score = 0.0
    
    i_idx, j_idx = np.triu_indices(n, k=1)
    
    def _get_delaunay_pairs(cx, cy):
        points = np.column_stack([cx, cy])
        try:
            tri = Delaunay(points)
            edges = set()
            for simplex in tri.simplices:
                for i in range(3):
                    for j in range(i+1, 3):
                        a, b = simplex[i], simplex[j]
                        if a > b:
                            a, b = b, a
                        edges.add((a, b))
            if not edges:
                return i_idx, j_idx
            edge_list = sorted(edges)
            return np.array([e[0] for e in edge_list]), np.array([e[1] for e in edge_list])
        except Exception:
            return i_idx, j_idx
    
    lr = 0.003
    decay = 0.99999
    no_improve_count = 0
    max_no_improve = 600
    
    for it in range(25000):
        if time.time() - start_time > time_limit:
            break
        
        if it % 30 == 0:
            d_i, d_j = _get_delaunay_pairs(x, y)
        
        dx = x[d_i] - x[d_j]
        dy = y[d_i] - y[d_j]
        dist = np.sqrt(dx * dx + dy * dy)
        dist = np.maximum(dist, 1e-12)
        
        overlap = r[d_i] + r[d_j] - dist
        
        gx = np.zeros(n)
        gy = np.zeros(n)
        gr = np.zeros(n)
        
        mask = overlap > 0
        if np.any(mask):
            rep = 150.0 * overlap[mask] / dist[mask]
            gx[d_i[mask]] -= rep * dx[mask] / dist[mask]
            gy[d_i[mask]] -= rep * dy[mask] / dist[mask]
            gx[d_j[mask]] += rep * dx[mask] / dist[mask]
            gy[d_j[mask]] += rep * dy[mask] / dist[mask]
            gr[d_i[mask]] += 120.0
            gr[d_j[mask]] += 120.0
        
        gx += np.where(x < r, 300.0, 0.0) - np.where(1 - x < r, 300.0, 0.0)
        gy += np.where(y < r, 300.0, 0.0) - np.where(1 - y < r, 300.0, 0.0)
        
        gr -= 2.0
        
        x += lr * gx
        y += lr * gy
        r += lr * gr
        
        x = np.clip(x, 0.001, 0.999)
        y = np.clip(y, 0.001, 0.999)
        r = np.maximum(r, 0.001)
        max_r = np.minimum(np.minimum(x, 1 - x), np.minimum(y, 1 - y))
        r = np.minimum(r, np.maximum(max_r, 0.001))
        
        lr *= decay
        
        if it % 10 == 0:
            current_circles = np.column_stack([x, y, r])
            score = _validate_and_score(current_circles)
            if score is not None and score > best_score + 1e-14:
                no_improve_count = 0
                best_score = score
                best_circles = current_circles.copy()
            else:
                no_improve_count += 1
            
            if no_improve_count >= max_no_improve:
                break
    
    final_circles = np.column_stack([x, y, r])
    final_score = _validate_and_score(final_circles)
    if final_score is not None and final_score > best_score:
        best_score = final_score
        best_circles = final_circles
    
    return best_circles, best_score


def _refine_radii(circles, time_limit=10):
    """Try to grow radii while maintaining feasibility using iterative adjustment."""
    x = circles[:, 0].copy()
    y = circles[:, 1].copy()
    r = circles[:, 2].copy()
    n = len(r)
    
    start_time = time.time()
    i_idx, j_idx = np.triu_indices(n, k=1)
    
    best_score = _validate_and_score(np.column_stack([x, y, r]))
    best_circles = np.column_stack([x, y, r]).copy()
    if best_score is None:
        best_score = 0.0
    
    for it in range(8000):
        if time.time() - start_time > time_limit:
            break
        
        max_allowed = np.copy(r)
        
        boundary_max = np.minimum(np.minimum(x, 1 - x), np.minimum(y, 1 - y))
        max_allowed = np.minimum(max_allowed, boundary_max)
        
        for k in range(len(i_idx)):
            i = i_idx[k]
            j = j_idx[k]
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx * dx + dy * dy)
            max_allowed[i] = min(max_allowed[i], max(dist - r[j], 0.001))
            max_allowed[j] = min(max_allowed[j], max(dist - r[i], 0.001))
        
        headroom = max_allowed - r
        if np.max(headroom) < 1e-7:
            break
        
        n_grow = min(4, n)
        top_indices = np.argsort(headroom)[-n_grow:]
        grow_amount = 0.0015
        for idx in top_indices:
            r[idx] = min(r[idx] + grow_amount, max_allowed[idx])
        
        new_circles = np.column_stack([x, y, r])
        new_circles = _resolve_overlaps(new_circles, max_iter=150)
        
        score = _validate_and_score(new_circles)
        if score is not None and score > best_score:
            best_score = score
            best_circles = new_circles.copy()
            x = new_circles[:, 0].copy()
            y = new_circles[:, 1].copy()
            r = new_circles[:, 2].copy()
    
    return best_circles, best_score


def _high_precision_polish(circles, time_limit=20, seed=0):
    """High-precision polish with adaptive contact detection and radius growth."""
    x = circles[:, 0].copy()
    y = circles[:, 1].copy()
    r = circles[:, 2].copy()
    n = len(r)
    
    start_time = time.time()
    best_score = _validate_and_score(np.column_stack([x, y, r]))
    best_circles = np.column_stack([x, y, r]).copy()
    if best_score is None:
        best_score = 0.0
    
    i_idx, j_idx = np.triu_indices(n, k=1)
    
    def _get_contact_pairs(cx, cy, cr, tol=0.005):
        points = np.column_stack([cx, cy])
        try:
            tri = Delaunay(points)
            edges = set()
            for simplex in tri.simplices:
                for i in range(3):
                    for j in range(i+1, 3):
                        a, b = simplex[i], simplex[j]
                        if a > b:
                            a, b = b, a
                        edges.add((a, b))
            if not edges:
                return i_idx, j_idx
            edge_list = sorted(edges)
            ei = np.array([e[0] for e in edge_list])
            ej = np.array([e[1] for e in edge_list])
            dx = cx[ei] - cx[ej]
            dy = cy[ei] - cy[ej]
            dist = np.sqrt(dx*dx + dy*dy)
            gap = dist - (cr[ei] + cr[ej])
            mask = gap < tol
            return ei[mask], ej[mask]
        except Exception:
            return i_idx, j_idx
    
    def _compute_max_allowed(cx, cy, cr):
        ma = np.copy(cr)
        boundary_max = np.minimum(np.minimum(cx, 1 - cx), np.minimum(cy, 1 - cy))
        ma = np.minimum(ma, boundary_max)
        for k in range(len(i_idx)):
            i = i_idx[k]
            j = j_idx[k]
            dx = cx[i] - cx[j]
            dy = cy[i] - cy[j]
            dist = np.sqrt(dx*dx + dy*dy)
            ma[i] = min(ma[i], max(dist - cr[j], 0.0005))
            ma[j] = min(ma[j], max(dist - cr[i], 0.0005))
        return ma
    
    d_i, d_j = _get_contact_pairs(x, y, r)
    
    lr = 0.003
    decay = 0.99999
    no_improve = 0
    max_no_improve = 500
    
    it = 0
    while True:
        if time.time() - start_time > time_limit * 0.6:
            break
        
        if it % 20 == 0:
            d_i, d_j = _get_contact_pairs(x, y, r)
        
        if len(d_i) == 0:
            d_i, d_j = i_idx, j_idx
        
        dx = x[d_i] - x[d_j]
        dy = y[d_i] - y[d_j]
        dist = np.sqrt(dx * dx + dy * dy)
        dist = np.maximum(dist, 1e-12)
        
        overlap = r[d_i] + r[d_j] - dist
        
        gx = np.zeros(n)
        gy = np.zeros(n)
        gr = np.zeros(n)
        
        mask = overlap > -0.002
        if np.any(mask):
            smooth_ov = np.maximum(overlap[mask], 0.0) + 0.002
            rep = 250.0 * smooth_ov / (dist[mask] + 0.005)
            gx[d_i[mask]] -= rep * dx[mask] / dist[mask]
            gy[d_i[mask]] -= rep * dy[mask] / dist[mask]
            gx[d_j[mask]] += rep * dx[mask] / dist[mask]
            gy[d_j[mask]] += rep * dy[mask] / dist[mask]
            gr[d_i[mask]] += 160.0
            gr[d_j[mask]] += 160.0
        
        gx += np.where(x < r, 400.0, 0.0) - np.where(1 - x < r, 400.0, 0.0)
        gy += np.where(y < r, 400.0, 0.0) - np.where(1 - y < r, 400.0, 0.0)
        
        gr -= 3.0
        
        x += lr * gx
        y += lr * gy
        r += lr * gr
        
        x = np.clip(x, 0.0005, 0.9995)
        y = np.clip(y, 0.0005, 0.9995)
        r = np.maximum(r, 0.001)
        max_r = np.minimum(np.minimum(x, 1 - x), np.minimum(y, 1 - y))
        r = np.minimum(r, np.maximum(max_r, 0.001))
        
        lr *= decay
        it += 1
        
        if it % 6 == 0:
            current_circles = np.column_stack([x, y, r])
            score = _validate_and_score(current_circles)
            if score is not None and score > best_score + 1e-14:
                no_improve = 0
                best_score = score
                best_circles = current_circles.copy()
            else:
                no_improve += 1
            
            if no_improve >= max_no_improve:
                break
    
    remaining_time = time_limit - (time.time() - start_time)
    if remaining_time > 2:
        for polish_iter in range(300):
            if time.time() - start_time > time_limit:
                break
            
            max_allowed = _compute_max_allowed(x, y, r)
            headroom = max_allowed - r
            
            if np.max(headroom) < 1e-9:
                break
            
            n_grow = min(3, n)
            top_indices = np.argsort(headroom)[-n_grow:]
            grow_amount = 0.0008
            
            improved = False
            for idx in top_indices:
                if headroom[idx] > 1e-8:
                    new_r = r.copy()
                    new_r[idx] = min(r[idx] + grow_amount, max_allowed[idx])
                    new_circles = np.column_stack([x, y, new_r])
                    
                    sc = _validate_and_score(new_circles)
                    if sc is not None and sc > best_score + 1e-12:
                        r = new_r
                        best_score = sc
                        best_circles = np.column_stack([x, y, r]).copy()
                        improved = True
                        max_allowed = _compute_max_allowed(x, y, r)
                        headroom = max_allowed - r
            
            if not improved:
                d_i2, d_j2 = _get_contact_pairs(x, y, r, tol=0.001)
                if len(d_i2) > 0:
                    dx2 = x[d_i2] - x[d_j2]
                    dy2 = y[d_i2] - y[d_j2]
                    dist2 = np.sqrt(dx2*dx2 + dy2*dy2)
                    dist2 = np.maximum(dist2, 1e-12)
                    ov2 = r[d_i2] + r[d_j2] - dist2
                    
                    mask2 = ov2 > 1e-10
                    if np.any(mask2):
                        for k in np.where(mask2)[0]:
                            i = d_i2[k]
                            j = d_j2[k]
                            push = ov2[k] * 0.3
                            dx_k = dx2[k] / dist2[k]
                            dy_k = dy2[k] / dist2[k]
                            x[i] -= push * dx_k * 0.05
                            y[i] -= push * dy_k * 0.05
                            x[j] += push * dx_k * 0.05
                            y[j] += push * dy_k * 0.05
                        
                        x = np.clip(x, 0.0005, 0.9995)
                        y = np.clip(y, 0.0005, 0.9995)
    
    remaining_time = time_limit - (time.time() - start_time)
    if remaining_time > 3:
        try:
            lbfgs_c, lbfgs_s = _lbfgs_optimize(
                best_circles, time_limit=min(remaining_time * 0.7, 8), seed=seed)
            if lbfgs_s > best_score:
                best_score = lbfgs_s
                best_circles = lbfgs_c
        except Exception:
            pass
    
    return best_circles, best_score


def construct():
    """Construct optimal circle packing in unit square for N=26 circles."""
    start_time = time.time()
    time_budget = 140

    best_circles = None
    best_score = 0.0

    initializations = [
        ('structured1', _structured_initial(N, seed=1)),
        ('structured2', _structured_initial(N, seed=2)),
        ('structured3', _structured_initial(N, seed=7)),
        ('structured4', _structured_initial(N, seed=13)),
        ('grid', _grid_initial(N)),
        ('random1', _random_initial(N, seed=10)),
        ('random2', _random_initial(N, seed=20)),
        ('random3', _random_initial(N, seed=42)),
    ]

    for name, init_circles in initializations:
        elapsed = time.time() - start_time
        if elapsed > time_budget - 40:
            break

        resolved = _resolve_overlaps(init_circles, max_iter=300)

        remaining = time_budget - (time.time() - start_time) - 30
        if remaining > 12:
            loc_circles, loc_score = _force_optimize(
                resolved, time_limit=min(remaining * 0.30, 14), seed=hash(name) % 10000)
            if loc_score > best_score:
                best_score = loc_score
                best_circles = loc_circles

        remaining = time_budget - (time.time() - start_time) - 20
        if remaining > 12:
            sa_circles, sa_score = _simulated_annealing(
                best_circles if best_circles is not None else resolved,
                time_limit=min(remaining * 0.40, 18), seed=hash(name) % 10000)
            if sa_score > best_score:
                best_score = sa_score
                best_circles = sa_circles

    if best_circles is not None:
        remaining = time_budget - (time.time() - start_time) - 25
        if remaining > 10:
            lbfgs_circles, lbfgs_score = _lbfgs_optimize(
                best_circles, time_limit=min(remaining * 0.35, 12), seed=42)
            if lbfgs_score > best_score:
                best_score = lbfgs_score
                best_circles = lbfgs_circles

        remaining = time_budget - (time.time() - start_time) - 20
        if remaining > 10:
            hp_circles, hp_score = _high_precision_polish(
                best_circles, time_limit=min(remaining * 0.50, 18), seed=9999)
            if hp_score > best_score:
                best_score = hp_score
                best_circles = hp_circles

        remaining = time_budget - (time.time() - start_time) - 15
        if remaining > 8:
            del_circles, del_score = _delaunay_polish(
                best_circles, time_limit=min(remaining * 0.40, 10), seed=9999)
            if del_score > best_score:
                best_score = del_score
                best_circles = del_circles

        remaining = time_budget - (time.time() - start_time) - 10
        if remaining > 6:
            ref_circles, ref_score = _refine_radii(
                best_circles, time_limit=min(remaining * 0.35, 8))
            if ref_score > best_score:
                best_score = ref_score
                best_circles = ref_circles

        remaining = time_budget - (time.time() - start_time) - 8
        if remaining > 6:
            lbfgs2_circles, lbfgs2_score = _lbfgs_optimize(
                best_circles, time_limit=min(remaining * 0.40, 8), seed=12345)
            if lbfgs2_score > best_score:
                best_score = lbfgs2_score
                best_circles = lbfgs2_circles

        remaining = time_budget - (time.time() - start_time) - 5
        if remaining > 4:
            hp2_circles, hp2_score = _high_precision_polish(
                best_circles, time_limit=min(remaining * 0.75, 10), seed=11111)
            if hp2_score > best_score:
                best_score = hp2_score
                best_circles = hp2_circles

    if best_circles is None:
        return _grid_initial(N)

    final_score = _validate_and_score(best_circles)
    if final_score is None:
        return _grid_initial(N)

    return best_circles

# EVOLVE-BLOCK-END
