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


def _hex_dense_initial(n=N, seed=0):
    """Denser hexagonal initialization for better starting geometry."""
    rng = np.random.RandomState(seed)
    circles = np.zeros((n, 3))
    
    # Try a tighter hexagonal packing
    # Approximate radius for 26 circles in unit square
    # Area = 1, N*pi*r^2 ~ 0.85 (packing efficiency)
    # r ~ sqrt(0.85 / (26*pi)) ~ 0.1
    # Let's start with r=0.08 and place them densely
    
    r_start = 0.08
    # Generate hex points
    points = []
    dx = 2 * r_start * 1.1 # Slightly larger than diameter to avoid immediate overlap
    dy = dx * np.sqrt(3) / 2
    
    # Center the grid
    x_offset = 0.5 - (4 * dx) / 2.0
    y_offset = 0.5 - (5 * dy) / 2.0
    
    for row in range(6):
        for col in range(5):
            cx = x_offset + col * dx + (dx / 2 if row % 2 else 0)
            cy = y_offset + row * dy
            # Clamp to square
            cx = np.clip(cx, r_start, 1 - r_start)
            cy = np.clip(cy, r_start, 1 - r_start)
            points.append([cx, cy, r_start])
            
    points = np.array(points[:n])
    # Add small noise
    points[:, 0] += 0.005 * rng.randn(n)
    points[:, 1] += 0.005 * rng.randn(n)
    points[:, 0] = np.clip(points[:, 0], 0.01, 0.99)
    points[:, 1] = np.clip(points[:, 1], 0.01, 0.99)
    
    return points


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
    """Gauss-Newton solver on the Delaunay contact graph.
    
    Formulates the packing as a nonlinear least-squares problem where:
    - Contact edges enforce dist(i,j) = r_i + r_j (equality)
    - Wall contacts enforce x_i = r_i or x_i = 1 - r_i, etc.
    - Objective: maximize sum(r) subject to these constraints
    
    Uses iterative linearization of the distance constraints.
    """
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
        """Get Delaunay edges plus wall-contact indicators."""
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
        
        # Wall contacts (within tolerance)
        tol = 0.01
        wall_left = cx <= cr + tol
        wall_right = (1 - cx) <= cr + tol
        wall_bottom = cy <= cr + tol
        wall_top = (1 - cy) <= cr + tol
        
        return ci, cj, wall_left, wall_right, wall_bottom, wall_top
    
    # Gauss-Newton iteration
    lr = 0.01
    damping = 0.5
    no_improve = 0
    max_no_improve = 300
    
    for it in range(5000):
        if time.time() - start_time > time_limit:
            break
        
        # Get current contact graph
        ci, cj, wl, wr, wb, wt = _get_contact_graph(x, y, r)
        
        # Build the Jacobian and residual for the contact constraints
        # For each contact edge (i,j): dist(i,j) - (r_i + r_j) = 0
        # For wall contacts: appropriate constraint
        
        m = len(ci)  # number of contact edges
        
        # Compute residuals and Jacobian columns
        dx = x[ci] - x[cj]
        dy = y[ci] - y[cj]
        dist = np.sqrt(dx*dx + dy*dy)
        dist = np.maximum(dist, 1e-12)
        
        # Residual: dist - (r_i + r_j), should be ~0 at contact
        res_edge = dist - (r[ci] + r[cj])
        
        # Jacobian for edge constraints: d(dist)/d(params)
        # d(dist)/dx_i = dx/dist, d(dist)/dy_i = dy/dist
        # d(dist)/dx_j = -dx/dist, d(dist)/dy_j = -dy/dist
        # d(dist)/dr_i = -1, d(dist)/dr_j = -1
        
        # We want to solve: J * delta = -res
        # But we also want to grow radii. So we add a "growth" objective.
        
        # Build sparse-like system using dense for simplicity (n=26 is small)
        # Parameters: [x(0..n-1), y(0..n-1), r(0..n-1)]
        # Total params: 3n
        
        # For each edge k with circles (i, j):
        # d(res_k)/dx_i = dx_k/dist_k
        # d(res_k)/dy_i = dy_k/dist_k  
        # d(res_k)/dx_j = -dx_k/dist_k
        # d(res_k)/dy_j = -dy_k/dist_k
        # d(res_k)/dr_i = -1
        # d(res_k)/dr_j = -1
        
        # We'll use a regularized Gauss-Newton:
        # (J^T J + lambda*I) * delta = J^T * (-res) + growth_gradient
        
        # Compute J^T J and J^T * res
        JtJ = np.zeros((3*n, 3*n))
        Jt_res = np.zeros(3*n)
        
        for k in range(m):
            i = ci[k]
            j = cj[k]
            d = dist[k]
            r_k = res_edge[k]
            
            ux = dx[k] / d
            uy = dy[k] / d
            
            # J row for this edge (in order: x_i, y_i, r_i, x_j, y_j, r_j)
            J_row = np.zeros(3*n)
            J_row[i] = ux
            J_row[n+i] = uy
            J_row[2*n+i] = -1.0
            J_row[j] = -ux
            J_row[n+j] = -uy
            J_row[2*n+j] = -1.0
            
            JtJ += np.outer(J_row, J_row)
            Jt_res += J_row * (-r_k)
        
        # Add wall contact constraints
        # For wall_left[i]: x_i - r_i = 0
        # For wall_right[i]: (1-x_i) - r_i = 0 => -x_i - r_i + 1 = 0
        # etc.
        
        # Wall left: x_i - r_i = 0
        for i in np.where(wl)[0]:
            J_row = np.zeros(3*n)
            J_row[i] = 1.0
            J_row[2*n+i] = -1.0
            res_w = x[i] - r[i]
            JtJ += np.outer(J_row, J_row)
            Jt_res += J_row * (-res_w)
        
        # Wall right: 1 - x_i - r_i = 0
        for i in np.where(wr)[0]:
            J_row = np.zeros(3*n)
            J_row[i] = -1.0
            J_row[2*n+i] = -1.0
            res_w = 1 - x[i] - r[i]
            JtJ += np.outer(J_row, J_row)
            Jt_res += J_row * (-res_w)
        
        # Wall bottom: y_i - r_i = 0
        for i in np.where(wb)[0]:
            J_row = np.zeros(3*n)
            J_row[n+i] = 1.0
            J_row[2*n+i] = -1.0
            res_w = y[i] - r[i]
            JtJ += np.outer(J_row, J_row)
            Jt_res += J_row * (-res_w)
        
        # Wall top: 1 - y_i - r_i = 0
        for i in np.where(wt)[0]:
            J_row = np.zeros(3*n)
            J_row[n+i] = -1.0
            J_row[2*n+i] = -1.0
            res_w = 1 - y[i] - r[i]
            JtJ += np.outer(J_row, J_row)
            Jt_res += J_row * (-res_w)
        
        # Add growth gradient: we want to increase sum(r)
        # This is equivalent to adding a term that pushes r upward
        # In Gauss-Newton, we can add it to the RHS
        growth = np.zeros(3*n)
        growth[2*n:3*n] = 1.0  # gradient of sum(r) w.r.t. r
        
        # Regularize
        lam = 0.01 + 0.1 * (it % 10) / 10.0
        JtJ_reg = JtJ + lam * np.eye(3*n)
        
        # Solve
        try:
            delta = np.linalg.solve(JtJ_reg, Jt_res + growth * 0.5)
        except np.linalg.LinAlgError:
            delta = np.linalg.lstsq(JtJ_reg, Jt_res + growth * 0.5, rcond=None)[0]
        
        # Line search with backtracking
        alpha = lr
        success = False
        for _ in range(10):
            new_x = x + alpha * delta[:n]
            new_y = y + alpha * delta[n:2*n]
            new_r = r + alpha * delta[2*n:3*n]
            
            # Clamp
            new_x = np.clip(new_x, 0.001, 0.999)
            new_y = np.clip(new_y, 0.001, 0.999)
            new_r = np.maximum(new_r, 0.001)
            max_r = np.minimum(np.minimum(new_x, 1 - new_x), np.minimum(new_y, 1 - new_y))
            new_r = np.minimum(new_r, np.maximum(max_r, 0.001))
            
            # Check if this is valid and improves
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
                # Accept if it's valid, even if not better (for exploration)
                if it % 50 == 0 and new_score >= best_score * 0.99:
                    x, y, r = new_x, new_y, new_r
                    success = True
                    break
            
            alpha *= 0.5
        
        if not success:
            no_improve += 1
            if no_improve >= max_no_improve:
                break
        
        # Decay learning rate slowly
        lr = max(0.001, lr * 0.999)
    
    # Final validation
    final_circles = np.column_stack([x, y, r])
    final_score = _validate_and_score(final_circles)
    if final_score is not None and final_score > best_score:
        best_score = final_score
        best_circles = final_circles
    
    return best_circles, best_score


def _lbfgs_optimize(circles, time_limit=15, seed=0):
    """Use L-BFGS-B with smooth soft-min penalty for robust local optimization."""
    x = circles[:, 0].copy()
    y = circles[:, 1].copy()
    r = circles[:, 2].copy()
    n = len(r)
    
    start_time = time.time()
    i_idx, j_idx = np.triu_indices(n, k=1)
    params = np.concatenate([x, y, r])
    
    def _softmax(v, temp=0.01):
        """Smooth approximation of max(0, v) using log-sum-exp trick."""
        # Use softplus-like: temp * log(1 + exp(v/temp))
        v_clipped = np.clip(v / temp, -50, 50)
        return temp * np.log1p(np.exp(v_clipped))
    
    def objective(p):
        if time.time() - start_time > time_limit:
            return 1e10
        xx = p[:n]
        yy = p[n:2*n]
        rr = p[2*n:3*n]
        
        # Soft boundary violations
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
        
        # Soft gradient for boundary violations
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
        
        mask = overlap > -0.01  # Include near-boundary pairs
        if np.any(mask):
            # Smooth repulsion
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
            # Move position
            delta = 0.015 * rng.randn(2)
            new_x[i] += delta[0]
            new_y[i] += delta[1]
        elif move_type < 0.70:
            # Change radius
            new_r[i] *= (1 + 0.08 * rng.randn())
            new_r[i] = max(0.005, min(new_r[i], 0.5))
        elif move_type < 0.85:
            # Swap two radii
            j = rng.randint(n)
            if j != i:
                new_r[i], new_r[j] = new_r[j], new_r[i]
        else:
            # Nudge pair
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
    """High-precision polish using Delaunay triangulation for contact pairs.
    
    Uses a two-phase approach: first resolve overlaps, then grow radii.
    """
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
    decay = 0.99997
    no_improve_count = 0
    max_no_improve = 1000
    
    for it in range(30000):
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
        
        # Strong repulsion for overlaps
        mask = overlap > 0
        if np.any(mask):
            rep = 200.0 * overlap[mask] / dist[mask] + 100.0 * overlap[mask]
            gx[d_i[mask]] -= rep * dx[mask] / dist[mask]
            gy[d_i[mask]] -= rep * dy[mask] / dist[mask]
            gx[d_j[mask]] += rep * dx[mask] / dist[mask]
            gy[d_j[mask]] += rep * dy[mask] / dist[mask]
            gr[d_i[mask]] -= 100.0 * np.minimum(overlap[mask], 0.03)
            gr[d_j[mask]] -= 100.0 * np.minimum(overlap[mask], 0.03)
        
        # Boundary forces
        gx += np.where(x < r, 500.0, 0.0) - np.where(1 - x < r, 500.0, 0.0)
        gy += np.where(y < r, 500.0, 0.0) - np.where(1 - y < r, 500.0, 0.0)
        bx = np.maximum(0, r - x) + np.maximum(0, r - (1 - x))
        by = np.maximum(0, r - y) + np.maximum(0, r - (1 - y))
        gr -= 80.0 * (bx + by)
        
        # Growth pressure - stronger when feasible
        current_score = _validate_and_score(np.column_stack([x, y, r]))
        if current_score is not None:
            gr -= 2.5 + 0.5 * (current_score / max(best_score, 1e-6))
        else:
            gr -= 1.0
        
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
            if score is not None and score > best_score + 1e-10:
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


def _high_precision_polish(circles, time_limit=20):
    """Final high-precision polish using very small steps and tight tolerances.
    
    Uses L-BFGS-B with extremely tight tolerances and a smooth softplus penalty
    to squeeze out the last bits of radius growth.
    """
    x = circles[:, 0].copy()
    y = circles[:, 1].copy()
    r = circles[:, 2].copy()
    n = len(r)
    
    start_time = time.time()
    i_idx, j_idx = np.triu_indices(n, k=1)
    params = np.concatenate([x, y, r])
    
    temp = 0.005  # Smaller temperature for sharper penalty
    
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
    
    for it in range(5000):
        if time.time() - start_time > time_limit:
            break
        
        # Find the circle that can grow the most
        max_allowed = np.copy(r)
        
        # Boundary constraints
        boundary_max = np.minimum(np.minimum(x, 1 - x), np.minimum(y, 1 - y))
        max_allowed = np.minimum(max_allowed, boundary_max)
        
        # Pair constraints: for circle i, r_i <= dist(i,j) - r_j for all j
        for k in range(len(i_idx)):
            i = i_idx[k]
            j = j_idx[k]
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx * dx + dy * dy)
            max_allowed[i] = min(max_allowed[i], max(dist - r[j], 0.001))
            max_allowed[j] = min(max_allowed[j], max(dist - r[i], 0.001))
        
        # Try to grow the circle with most headroom
        headroom = max_allowed - r
        if np.max(headroom) < 1e-6:
            break
        
        # Grow a few circles with most headroom
        n_grow = min(3, n)
        top_indices = np.argsort(headroom)[-n_grow:]
        grow_amount = 0.002
        for idx in top_indices:
            r[idx] = min(r[idx] + grow_amount, max_allowed[idx])
        
        # If this creates overlaps, resolve them
        new_circles = np.column_stack([x, y, r])
        new_circles = _resolve_overlaps(new_circles, max_iter=100)
        
        score = _validate_and_score(new_circles)
        if score is not None and score > best_score:
            best_score = score
            best_circles = new_circles.copy()
            x = new_circles[:, 0].copy()
            y = new_circles[:, 1].copy()
            r = new_circles[:, 2].copy()
    
    return best_circles, best_score


def construct():
    """Construct optimal circle packing in unit square for N=26 circles."""
    start_time = time.time()
    time_budget = 135  # Keep 15s safety margin before 150s kill
    
    best_circles = None
    best_score = 0.0
    
    # Try multiple initializations with focused optimization
    initializations = [
        ('hex_dense1', _hex_dense_initial(N, seed=1)),
        ('hex_dense2', _hex_dense_initial(N, seed=5)),
        ('structured1', _structured_initial(N, seed=3)),
        ('structured2', _structured_initial(N, seed=8)),
        ('grid', _grid_initial(N)),
        ('random1', _random_initial(N, seed=15)),
    ]
    
    for name, init_circles in initializations:
        elapsed = time.time() - start_time
        if elapsed > time_budget - 30:
            break
        
        # Resolve initial overlaps
        resolved = _resolve_overlaps(init_circles, max_iter=200)
        
        # Gauss-Newton contact graph optimization - primary optimizer
        remaining = time_budget - (time.time() - start_time) - 25
        if remaining > 15:
            gn_circles, gn_score = _gauss_newton_contact(resolved, time_limit=min(remaining * 0.40, 20), seed=hash(name) % 10000)
            if gn_score > best_score:
                best_score = gn_score
                best_circles = gn_circles
        
        # L-BFGS optimization with smooth penalty
        remaining = time_budget - (time.time() - start_time) - 20
        if remaining > 12:
            lbfgs_circles, lbfgs_score = _lbfgs_optimize(
                best_circles if best_circles is not None else resolved,
                time_limit=min(remaining * 0.30, 15), seed=hash(name) % 10000)
            if lbfgs_score > best_score:
                best_score = lbfgs_score
                best_circles = lbfgs_circles
        
        # Simulated annealing for escape from local optima
        remaining = time_budget - (time.time() - start_time) - 15
        if remaining > 15:
            sa_circles, sa_score = _simulated_annealing(
                best_circles if best_circles is not None else resolved,
                time_limit=min(remaining * 0.40, 25), seed=hash(name) % 10000)
            if sa_score > best_score:
                best_score = sa_score
                best_circles = sa_circles
    
    # Final polish phase with multiple passes
    if best_circles is not None:
        # Gauss-Newton contact polish
        remaining = time_budget - (time.time() - start_time) - 20
        if remaining > 12:
            gn_polish_circles, gn_polish_score = _gauss_newton_contact(best_circles, time_limit=min(remaining * 0.5, 18), seed=9999)
            if gn_polish_score > best_score:
                best_score = gn_polish_score
                best_circles = gn_polish_circles
        
        # L-BFGS refinement
        remaining = time_budget - (time.time() - start_time) - 15
        if remaining > 10:
            lbfgs2_circles, lbfgs2_score = _lbfgs_optimize(best_circles, time_limit=min(remaining * 0.4, 12), seed=7777)
            if lbfgs2_score > best_score:
                best_score = lbfgs2_score
                best_circles = lbfgs2_circles
        
        # Radius refinement
        remaining = time_budget - (time.time() - start_time) - 12
        if remaining > 8:
            ref_circles, ref_score = _refine_radii(best_circles, time_limit=min(remaining * 0.4, 10))
            if ref_score > best_score:
                best_score = ref_score
                best_circles = ref_circles
        
        # Second Gauss-Newton pass
        remaining = time_budget - (time.time() - start_time) - 10
        if remaining > 8:
            gn2_circles, gn2_score = _gauss_newton_contact(best_circles, time_limit=min(remaining * 0.5, 12), seed=12345)
            if gn2_score > best_score:
                best_score = gn2_score
                best_circles = gn2_circles
        
        # Final Delaunay polish
        remaining = time_budget - (time.time() - start_time) - 5
        if remaining > 6:
            del2_circles, del2_score = _delaunay_polish(best_circles, time_limit=min(remaining * 0.5, 8), seed=11111)
            if del2_score > best_score:
                best_score = del2_score
                best_circles = del2_circles
        
        # High-precision final polish
        remaining = time_budget - (time.time() - start_time) - 3
        if remaining > 8:
            hp_circles, hp_score = _high_precision_polish(best_circles, time_limit=min(remaining * 0.8, 18))
            if hp_score > best_score:
                best_score = hp_score
                best_circles = hp_circles
    
    # Final validation
    if best_circles is None:
        return _grid_initial(N)
    
    final_score = _validate_and_score(best_circles)
    if final_score is None:
        return _grid_initial(N)
    
    return best_circles

# EVOLVE-BLOCK-END
