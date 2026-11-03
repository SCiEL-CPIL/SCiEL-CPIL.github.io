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


def _compact_hex_initial(n=N, seed=0):
    """
    Generate a compact hexagonal cluster.
    1. Generate a large hex grid.
    2. Sort points by distance to the center of the square (0.5, 0.5).
    3. Take the first n points.
    4. Scale and center them to fit comfortably in [0,1]^2.
    """
    rng = np.random.RandomState(seed)
    
    # Generate a sufficiently large hex grid
    # r0 is the "radius" of the grid spacing, not the circle radius
    r0 = 0.1
    centers = []
    for row in range(15):
        for col in range(15):
            x = col * 2 * r0
            y = row * np.sqrt(3) * r0
            if row % 2 == 1:
                x += r0
            centers.append((x, y))
    
    centers = np.array(centers)
    
    # Sort by distance to (0.5, 0.5)
    dists = np.sum((centers - 0.5)**2, axis=1)
    sorted_indices = np.argsort(dists)
    centers = centers[sorted_indices[:n]]
    
    # Calculate bounding box
    min_x, min_y = np.min(centers, axis=0)
    max_x, max_y = np.max(centers, axis=0)
    width = max_x - min_x
    height = max_y - min_y
    
    # Scale to fit in [0.1, 0.9] to leave room for radius growth
    # We want the centers to be well within the square so that 
    # initial radii can be substantial.
    scale = min(0.8 / width, 0.8 / height) if width > 0 and height > 0 else 1.0
    
    scaled_centers = (centers - np.array([min_x, min_y])) * scale + 0.1
    
    # Initial radii: slightly less than half the minimum distance to neighbors
    min_dist = np.inf
    for i in range(n):
        for j in range(i+1, n):
            d = np.sqrt((scaled_centers[i,0]-scaled_centers[j,0])**2 + (scaled_centers[i,1]-scaled_centers[j,1])**2)
            if d < min_dist:
                min_dist = d
    
    # Also check distance to walls
    min_boundary = np.inf
    for x, y in scaled_centers:
        min_boundary = min(min_boundary, x, 1-x, y, 1-y)
        
    r_init = min(min_dist / 2, min_boundary) * 0.9
    
    circles = np.zeros((n, 3))
    circles[:, 0] = scaled_centers[:, 0]
    circles[:, 1] = scaled_centers[:, 1]
    circles[:, 2] = r_init
    
    # Add tiny noise to break symmetry
    circles[:, 0] += 0.001 * rng.randn(n)
    circles[:, 1] += 0.001 * rng.randn(n)
    
    return circles


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


def _gauss_newton_contact(circles, time_limit=20, seed=0):
    """Gauss-Newton solver on the Delaunay contact graph."""
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
    
    def _get_contact_graph(cx, cy, cr):
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
                return i_idx, j_idx, np.zeros(n, dtype=bool), np.zeros(n, dtype=bool), \
                       np.zeros(n, dtype=bool), np.zeros(n, dtype=bool)
            edge_list = sorted(edges)
            ci = np.array([e[0] for e in edge_list])
            cj = np.array([e[1] for e in edge_list])
        except Exception:
            ci, cj = i_idx, j_idx
        
        tol = 0.01
        wall_left = cx <= cr + tol
        wall_right = (1 - cx) <= cr + tol
        wall_bottom = cy <= cr + tol
        wall_top = (1 - cy) <= cr + tol
        
        return ci, cj, wall_left, wall_right, wall_bottom, wall_top
    
    lr = 0.01
    no_improve = 0
    max_no_improve = 300
    
    for it in range(5000):
        if time.time() - start_time > time_limit:
            break
        
        ci, cj, wl, wr, wb, wt = _get_contact_graph(x, y, r)
        
        m = len(ci)
        
        dx = x[ci] - x[cj]
        dy = y[ci] - y[cj]
        dist = np.sqrt(dx*dx + dy*dy)
        dist = np.maximum(dist, 1e-12)
        
        res_edge = dist - (r[ci] + r[cj])
        
        JtJ = np.zeros((3*n, 3*n))
        Jt_res = np.zeros(3*n)
        
        for k in range(m):
            i = ci[k]
            j = cj[k]
            d = dist[k]
            r_k = res_edge[k]
            
            ux = dx[k] / d
            uy = dy[k] / d
            
            J_row = np.zeros(3*n)
            J_row[i] = ux
            J_row[n+i] = uy
            J_row[2*n+i] = -1.0
            J_row[j] = -ux
            J_row[n+j] = -uy
            J_row[2*n+j] = -1.0
            
            JtJ += np.outer(J_row, J_row)
            Jt_res += J_row * (-r_k)
        
        for i in np.where(wl)[0]:
            J_row = np.zeros(3*n)
            J_row[i] = 1.0
            J_row[2*n+i] = -1.0
            res_w = x[i] - r[i]
            JtJ += np.outer(J_row, J_row)
            Jt_res += J_row * (-res_w)
        
        for i in np.where(wr)[0]:
            J_row = np.zeros(3*n)
            J_row[i] = -1.0
            J_row[2*n+i] = -1.0
            res_w = 1 - x[i] - r[i]
            JtJ += np.outer(J_row, J_row)
            Jt_res += J_row * (-res_w)
        
        for i in np.where(wb)[0]:
            J_row = np.zeros(3*n)
            J_row[n+i] = 1.0
            J_row[2*n+i] = -1.0
            res_w = y[i] - r[i]
            JtJ += np.outer(J_row, J_row)
            Jt_res += J_row * (-res_w)
        
        for i in np.where(wt)[0]:
            J_row = np.zeros(3*n)
            J_row[n+i] = -1.0
            J_row[2*n+i] = -1.0
            res_w = 1 - y[i] - r[i]
            JtJ += np.outer(J_row, J_row)
            Jt_res += J_row * (-res_w)
        
        growth = np.zeros(3*n)
        growth[2*n:3*n] = 1.0
        
        lam = 0.01 + 0.1 * (it % 10) / 10.0
        JtJ_reg = JtJ + lam * np.eye(3*n)
        
        try:
            delta = np.linalg.solve(JtJ_reg, Jt_res + growth * 0.5)
        except np.linalg.LinAlgError:
            delta = np.linalg.lstsq(JtJ_reg, Jt_res + growth * 0.5, rcond=None)[0]
        
        alpha = lr
        success = False
        for _ in range(10):
            new_x = x + alpha * delta[:n]
            new_y = y + alpha * delta[n:2*n]
            new_r = r + alpha * delta[2*n:3*n]
            
            new_x = np.clip(new_x, 0.001, 0.999)
            new_y = np.clip(new_y, 0.001, 0.999)
            new_r = np.maximum(new_r, 0.001)
            max_r = np.minimum(np.minimum(new_x, 1 - new_x), np.minimum(new_y, 1 - new_y))
            new_r = np.minimum(new_r, np.maximum(max_r, 0.001))
            
            new_circles = np.column_stack([new_x, new_y, new_r])
            new_score = _validate_and_score(new_circles)
            
            if new_score is not None and new_score > best_score:
                x, y, r = new_x, new_y, new_r
                best_score = new_score
                best_circles = new_circles.copy()
                no_improve = 0
                success = True
                break
            elif new_score is not None:
                if it % 50 == 0 and new_score >= best_score * 0.99:
                    x, y, r = new_x, new_y, new_r
                    success = True
                    break
            
            alpha *= 0.5
        
        if not success:
            no_improve += 1
            if no_improve >= max_no_improve:
                break
        
        lr = max(0.001, lr * 0.999)
    
    final_circles = np.column_stack([x, y, r])
    final_score = _validate_and_score(final_circles)
    if final_score is not None and final_score > best_score:
        best_score = final_score
        best_circles = final_circles
    
    return best_circles, best_score


def _lbfgs_optimize(circles, time_limit=15, seed=0):
    """Use L-BFGS-B with smooth soft-min penalty."""
    x = circles[:, 0].copy()
    y = circles[:, 1].copy()
    r = circles[:, 2].copy()
    n = len(r)
    
    start_time = time.time()
    i_idx, j_idx = np.triu_indices(n, k=1)
    params = np.concatenate([x, y, r])
    
    def _softmax(v, temp=0.01):
        v_clipped = np.clip(v / temp, -50, 50)
        return temp * np.log1p(np.exp(v_clipped))
    
    def objective(p):
        if time.time() - start_time > time_limit:
            return 1e10
        xx = p[:n]
        yy = p[n:2*n]
        rr = p[2*n:3*n]
        
        bv = (_softmax(rr - xx) + _softmax(rr - (1 - xx)) + 
              _softmax(rr - yy) + _softmax(rr - (1 - yy)))
        
        dx = xx[i_idx] - xx[j_idx]
        dy = yy[i_idx] - yy[j_idx]
        dist = np.sqrt(dx * dx + dy * dy)
        dist = np.maximum(dist, 1e-12)
        ov = _softmax(rr[i_idx] + rr[j_idx] - dist)
        
        return -np.sum(rr) + 100.0 * (np.sum(bv) + np.sum(ov))
    
    def gradient(p):
        if time.time() - start_time > time_limit:
            return np.zeros_like(p)
        xx = p[:n]
        yy = p[n:2*n]
        rr = p[2*n:3*n]
        
        gx = np.zeros(n)
        gy = np.zeros(n)
        gr = np.zeros(n)
        
        temp = 0.01
        def _softplus_grad(v):
            v_clipped = np.clip(v / temp, -50, 50)
            return 1.0 / (1.0 + np.exp(-v_clipped))
        
        gx += 100.0 * (_softplus_grad(rr - xx) - _softplus_grad(rr - (1 - xx)))
        gy += 100.0 * (_softplus_grad(rr - yy) - _softplus_grad(rr - (1 - yy)))
        gr += 100.0 * (_softplus_grad(rr - xx) + _softplus_grad(rr - (1 - xx)) + 
                       _softplus_grad(rr - yy) + _softplus_grad(rr - (1 - yy)))
        
        dx = xx[i_idx] - xx[j_idx]
        dy = yy[i_idx] - yy[j_idx]
        dist = np.sqrt(dx * dx + dy * dy)
        dist = np.maximum(dist, 1e-12)
        overlap = rr[i_idx] + rr[j_idx] - dist
        
        mask = overlap > -0.01
        if np.any(mask):
            rep = 100.0 * _softplus_grad(overlap[mask]) / dist[mask]
            gx[i_idx[mask]] -= rep * dx[mask] / dist[mask]
            gy[i_idx[mask]] -= rep * dy[mask] / dist[mask]
            gx[j_idx[mask]] += rep * dx[mask] / dist[mask]
            gy[j_idx[mask]] += rep * dy[mask] / dist[mask]
            gr[i_idx[mask]] += 100.0 * _softplus_grad(overlap[mask])
            gr[j_idx[mask]] += 100.0 * _softplus_grad(overlap[mask])
        
        gr -= 1.0
        
        return np.concatenate([gx, gy, gr])
    
    bounds = [(0.0, 1.0)] * n + [(0.0, 1.0)] * n + [(0.001, 0.5)] * n
    
    try:
        result = minimize(objective, params, method='L-BFGS-B', jac=gradient,
                          bounds=bounds, options={'maxiter': 800, 'maxfun': 3000,
                                                   'ftol': 1e-14, 'gtol': 1e-12,
                                                   'maxls': 50})
    except Exception:
        return circles.copy(), _validate_and_score(circles) or 0.0
    
    if time.time() - start_time > time_limit:
        return circles.copy(), _validate_and_score(circles) or 0.0
    
    new_params = result.x
    new_x = np.clip(new_params[:n], 0, 1)
    new_y = np.clip(new_params[n:2*n], 0, 1)
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
    """Simulated annealing."""
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


def _high_precision_polish(circles, time_limit=20):
    """Final high-precision polish using very small steps."""
    x = circles[:, 0].copy()
    y = circles[:, 1].copy()
    r = circles[:, 2].copy()
    n = len(r)
    
    start_time = time.time()
    i_idx, j_idx = np.triu_indices(n, k=1)
    params = np.concatenate([x, y, r])
    
    temp = 0.005
    
    def _softplus(v):
        v_clipped = np.clip(v / temp, -50, 50)
        return temp * np.log1p(np.exp(v_clipped))
    
    def _softplus_grad(v):
        v_clipped = np.clip(v / temp, -50, 50)
        return 1.0 / (1.0 + np.exp(-v_clipped))
    
    def objective(p):
        if time.time() - start_time > time_limit:
            return 1e10
        xx = p[:n]
        yy = p[n:2*n]
        rr = p[2*n:3*n]
        
        bv = (_softplus(rr - xx) + _softplus(rr - (1 - xx)) + 
              _softplus(rr - yy) + _softplus(rr - (1 - yy)))
        
        dx = xx[i_idx] - xx[j_idx]
        dy = yy[i_idx] - yy[j_idx]
        dist = np.sqrt(dx * dx + dy * dy)
        dist = np.maximum(dist, 1e-12)
        ov = _softplus(rr[i_idx] + rr[j_idx] - dist)
        
        return -np.sum(rr) + 200.0 * (np.sum(bv) + np.sum(ov))
    
    def gradient(p):
        if time.time() - start_time > time_limit:
            return np.zeros_like(p)
        xx = p[:n]
        yy = p[n:2*n]
        rr = p[2*n:3*n]
        
        gx = np.zeros(n)
        gy = np.zeros(n)
        gr = np.zeros(n)
        
        gx += 200.0 * (_softplus_grad(rr - xx) - _softplus_grad(rr - (1 - xx)))
        gy += 200.0 * (_softplus_grad(rr - yy) - _softplus_grad(rr - (1 - yy)))
        gr += 200.0 * (_softplus_grad(rr - xx) + _softplus_grad(rr - (1 - xx)) + 
                       _softplus_grad(rr - yy) + _softplus_grad(rr - (1 - yy)))
        
        dx = xx[i_idx] - xx[j_idx]
        dy = yy[i_idx] - yy[j_idx]
        dist = np.sqrt(dx * dx + dy * dy)
        dist = np.maximum(dist, 1e-12)
        overlap = rr[i_idx] + rr[j_idx] - dist
        
        mask = overlap > -0.005
        if np.any(mask):
            rep = 200.0 * _softplus_grad(overlap[mask]) / dist[mask]
            gx[i_idx[mask]] -= rep * dx[mask] / dist[mask]
            gy[i_idx[mask]] -= rep * dy[mask] / dist[mask]
            gx[j_idx[mask]] += rep * dx[mask] / dist[mask]
            gy[j_idx[mask]] += rep * dy[mask] / dist[mask]
            gr[i_idx[mask]] += 200.0 * _softplus_grad(overlap[mask])
            gr[j_idx[mask]] += 200.0 * _softplus_grad(overlap[mask])
        
        gr -= 1.0
        
        return np.concatenate([gx, gy, gr])
    
    bounds = [(0.0, 1.0)] * n + [(0.0, 1.0)] * n + [(0.001, 0.5)] * n
    
    try:
        result = minimize(objective, params, method='L-BFGS-B', jac=gradient,
                          bounds=bounds, options={'maxiter': 2000, 'maxfun': 10000,
                                                   'ftol': 1e-16, 'gtol': 1e-14,
                                                   'maxls': 100})
    except Exception:
        return circles.copy(), _validate_and_score(circles) or 0.0
    
    if time.time() - start_time > time_limit:
        return circles.copy(), _validate_and_score(circles) or 0.0
    
    new_params = result.x
    new_x = np.clip(new_params[:n], 0, 1)
    new_y = np.clip(new_params[n:2*n], 0, 1)
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


def construct():
    """Construct optimal circle packing in unit square for N=26 circles."""
    start_time = time.time()
    time_budget = 135  # Keep 15s safety margin before 150s kill
    
    best_circles = None
    best_score = 0.0
    
    # Use the improved compact hexagonal initialization
    init_circles = _compact_hex_initial(N, seed=1)
    
    # Resolve initial overlaps to ensure validity
    resolved = _resolve_overlaps(init_circles, max_iter=200)
    
    # Primary optimization: Gauss-Newton
    remaining = time_budget - (time.time() - start_time) - 30
    if remaining > 15:
        gn_circles, gn_score = _gauss_newton_contact(resolved, time_limit=min(remaining * 0.5, 25), seed=123)
        if gn_score > best_score:
            best_score = gn_score
            best_circles = gn_circles
    
    # Secondary optimization: L-BFGS
    remaining = time_budget - (time.time() - start_time) - 20
    if remaining > 12 and best_circles is not None:
        lbfgs_circles, lbfgs_score = _lbfgs_optimize(best_circles, time_limit=min(remaining * 0.5, 15), seed=456)
        if lbfgs_score > best_score:
            best_score = lbfgs_score
            best_circles = lbfgs_circles
    
    # Tertiary optimization: Simulated Annealing
    remaining = time_budget - (time.time() - start_time) - 15
    if remaining > 15 and best_circles is not None:
        sa_circles, sa_score = _simulated_annealing(best_circles, time_limit=min(remaining * 0.6, 25), seed=789)
        if sa_score > best_score:
            best_score = sa_score
            best_circles = sa_circles
    
    # Final polish
    if best_circles is not None:
        # Another GN pass
        remaining = time_budget - (time.time() - start_time) - 15
        if remaining > 10:
            gn2_circles, gn2_score = _gauss_newton_contact(best_circles, time_limit=min(remaining * 0.5, 12), seed=321)
            if gn2_score > best_score:
                best_score = gn2_score
                best_circles = gn2_circles
        
        # High precision polish
        remaining = time_budget - (time.time() - start_time) - 5
        if remaining > 8:
            hp_circles, hp_score = _high_precision_polish(best_circles, time_limit=min(remaining * 0.8, 15))
            if hp_score > best_score:
                best_score = hp_score
                best_circles = hp_circles
    
    # Fallback
    if best_circles is None:
        return _grid_initial(N)
    
    final_score = _validate_and_score(best_circles)
    if final_score is None:
        return _grid_initial(N)
    
    return best_circles

# EVOLVE-BLOCK-END
