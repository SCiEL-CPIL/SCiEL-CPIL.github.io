import numpy as np
from scipy.optimize import minimize, linprog
from scipy.spatial import Delaunay
import time

N = 21

def _softplus(x, beta=10.0):
    """Smooth approximation of max(0, x) using softplus."""
    return np.log1p(np.exp(-beta * x)) / beta

def _softplus_grad(x, beta=10.0):
    """Derivative of softplus: sigmoid(-beta*x)."""
    return 1.0 / (1.0 + np.exp(beta * x))

def _delaunay_edges(points):
    """Get Delaunay triangle edges as a set of (i,j) pairs."""
    try:
        tri = Delaunay(points)
        edges = set()
        for simplex in tri.simplices:
            for i in range(3):
                for j in range(i+1, 3):
                    a, b = int(simplex[i]), int(simplex[j])
                    if a < b:
                        edges.add((a, b))
                    else:
                        edges.add((b, a))
        return edges
    except Exception:
        return set()

def _solve_lp_radii(xs, ys, w, h, edges):
    """
    Solve LP to maximize sum of radii given fixed centers and contact graph.
    Constraints:
      - r_i >= 0
      - r_i <= x_i, r_i <= w - x_i, r_i <= y_i, r_i <= h - y_i  (boundary)
      - r_i + r_j <= dist(i,j) for all edge pairs (non-overlap on contact graph)
    """
    n = N
    # Variables: r_0, ..., r_{n-1}
    c = -np.ones(n)  # minimize -sum(r) = maximize sum(r)
    
    A_ub = []
    b_ub = []
    
    # Boundary constraints: r_i <= min(x_i, w-x_i, y_i, h-y_i)
    bounds_list = []
    max_r = np.minimum(np.minimum(xs, w - xs), np.minimum(ys, h - ys))
    max_r = np.maximum(max_r, 0.0)
    for i in range(n):
        bounds_list.append((0.0, max_r[i]))
    
    # Pairwise constraints for edges: r_i + r_j <= dist_ij
    for (i, j) in edges:
        dx = xs[i] - xs[j]
        dy = ys[i] - ys[j]
        dist = np.sqrt(dx*dx + dy*dy)
        row = np.zeros(n)
        row[i] = 1.0
        row[j] = 1.0
        A_ub.append(row)
        b_ub.append(dist)
    
    if A_ub:
        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)
    else:
        A_ub = None
        b_ub = None
    
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds_list, method='highs')
        if res.success:
            return res.x
        else:
            # Fallback: use min of boundary and half-distances
            r = np.copy(max_r)
            for (i, j) in edges:
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                dist = np.sqrt(dx*dx + dy*dy)
                r[i] = min(r[i], dist / 2.0)
                r[j] = min(r[j], dist / 2.0)
            return np.maximum(r, 0.0)
    except Exception:
        r = np.copy(max_r)
        for (i, j) in edges:
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist = np.sqrt(dx*dx + dy*dy)
            r[i] = min(r[i], dist / 2.0)
            r[j] = min(r[j], dist / 2.0)
        return np.maximum(r, 0.0)

def _is_valid(params, w, h, tol=1e-6):
    """Check if the packing is valid."""
    xs = params[0::3]
    ys = params[1::3]
    rs = params[2::3]
    
    if np.any(rs < -tol): return False
    if np.any(xs < -tol) or np.any(w - xs < -tol): return False
    if np.any(ys < -tol) or np.any(h - ys < -tol): return False
    if np.any(rs - xs < -tol) or np.any(rs - (w - xs) < -tol): return False
    if np.any(rs - ys < -tol) or np.any(rs - (h - ys) < -tol): return False
    
    # Check all pairs for overlap
    for i in range(N):
        for j in range(i+1, N):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist = np.sqrt(dx*dx + dy*dy)
            if dist < rs[i] + rs[j] - tol:
                return False
    return True

def construct():
    start_time = time.time()
    time_limit = 145.0  # Safety margin before 150s kill
    
    def get_wh(a):
        h = 2.0 / (1.0 + a)
        w = 2.0 * a / (1.0 + a)
        return w, h

    def position_objective_and_grad(params, w, h, beta=8.0):
        """
        Optimize positions only (with fixed small radii for repulsion).
        Objective: -sum of radii proxy + softplus penalties for overlaps and boundaries.
        We use a fixed radius for repulsion to guide positions, then solve LP for radii.
        """
        n = N
        xs = params[0:n]
        ys = params[n:2*n]
        
        # Use a target radius for repulsion
        r_target = 0.15  # approximate average radius
        
        f = 0.0
        g = np.zeros(2*n)
        
        # Boundary penalties
        viol_x0 = r_target - xs
        sp = _softplus(viol_x0, beta)
        f += sp.sum()
        g[0:n] += _softplus_grad(viol_x0, beta)
        
        viol_x1 = r_target - (w - xs)
        sp = _softplus(viol_x1, beta)
        f += sp.sum()
        g[0:n] -= _softplus_grad(viol_x1, beta)
        
        viol_y0 = r_target - ys
        sp = _softplus(viol_y0, beta)
        f += sp.sum()
        g[n:2*n] += _softplus_grad(viol_y0, beta)
        
        viol_y1 = r_target - (h - ys)
        sp = _softplus(viol_y1, beta)
        f += sp.sum()
        g[n:2*n] -= _softplus_grad(viol_y1, beta)
        
        # Overlap penalties using Delaunay-identified neighbors + all pairs as backup
        # For speed, use all pairs but vectorize
        dx = xs[:, None] - xs[None, :]
        dy = ys[:, None] - ys[None, :]
        dist_sq = dx*dx + dy*dy
        np.fill_diagonal(dist_sq, np.inf)
        dist = np.sqrt(dist_sq + 1e-12)
        
        # Only penalize pairs that are close (within 2*r_target)
        mask = dist < 2.5 * r_target
        # Exclude diagonal
        mask = mask & (np.arange(n)[:, None] != np.arange(n)[None, :])
        
        viol = 2.0 * r_target - dist
        sp_vals = _softplus(viol, beta)
        sp_vals[~mask] = 0.0
        f += sp_vals.sum()
        
        grad_sp = _softplus_grad(viol, beta)
        grad_sp[~mask] = 0.0
        
        # Gradient with respect to xs
        # d(viol_ij)/d(xs_i) = -dx_ij / dist_ij
        g_x = np.zeros(n)
        for i in range(n):
            for j in range(n):
                if mask[i, j] and i != j:
                    g_x[i] += grad_sp[i, j] * (-dx[i, j] / dist[i, j])
        
        # Gradient with respect to ys
        g_y = np.zeros(n)
        for i in range(n):
            for j in range(n):
                if mask[i, j] and i != j:
                    g_y[i] += grad_sp[i, j] * (-dy[i, j] / dist[i, j])
        
        g[0:n] += g_x
        g[n:2*n] += g_y
        
        return f, g

    def make_grid_init(w, h, rng, noise_scale=0.02):
        """Create a grid-based initialization of positions."""
        best_init = None
        best_sum = -1
        
        for rows in range(3, 8):
            cols = int(np.ceil(N / rows))
            if cols < 2:
                continue
            
            xs_init = np.zeros(N)
            ys_init = np.zeros(N)
            
            for i in range(N):
                row = i // cols
                col = i % cols
                xs_init[i] = (col + 0.5) * (w / cols)
                ys_init[i] = (row + 0.5) * (h / rows)
            
            xs_init += rng.normal(0, noise_scale, N)
            ys_init += rng.normal(0, noise_scale, N)
            
            xs_init = np.clip(xs_init, 0.05, w - 0.05)
            ys_init = np.clip(ys_init, 0.05, h - 0.05)
            
            params = np.concatenate([xs_init, ys_init])
            
            # Quick check
            s = np.sum(np.minimum(np.minimum(xs_init, w - xs_init), 
                                  np.minimum(ys_init, h - ys_init)))
            if s > best_sum:
                best_sum = s
                best_init = params
        
        return best_init

    def make_hex_init(w, h, rng):
        """Create a hexagonal packing initialization of positions."""
        patterns = [
            [5, 4, 5, 4, 3],
            [4, 5, 4, 5, 3],
            [5, 5, 5, 4, 2],
            [3, 4, 5, 4, 5],
            [4, 4, 5, 4, 4],
            [7, 7, 7],
            [4, 4, 4, 4, 5],
        ]
        
        best_init = None
        best_sum = -1
        
        for pattern in patterns:
            if sum(pattern) != N:
                continue
            
            xs_init = np.zeros(N)
            ys_init = np.zeros(N)
            
            idx = 0
            n_rows = len(pattern)
            row_h = h / n_rows
            
            for row_idx, n_cols in enumerate(pattern):
                col_w = w / n_cols
                offset = 0.0 if row_idx % 2 == 0 else col_w / 2
                for col_idx in range(n_cols):
                    if idx >= N:
                        break
                    x = (col_idx + 0.5) * col_w + offset
                    y = (row_idx + 0.5) * row_h
                    xs_init[idx] = x
                    ys_init[idx] = y
                    idx += 1
            
            if idx != N:
                continue
            
            xs_init += rng.normal(0, 0.01, N)
            ys_init += rng.normal(0, 0.01, N)
            
            xs_init = np.clip(xs_init, 0.05, w - 0.05)
            ys_init = np.clip(ys_init, 0.05, h - 0.05)
            
            params = np.concatenate([xs_init, ys_init])
            s = np.sum(np.minimum(np.minimum(xs_init, w - xs_init), 
                                  np.minimum(ys_init, h - ys_init)))
            if s > best_sum:
                best_sum = s
                best_init = params
        
        return best_init

    def make_random_init(w, h, rng):
        """Random initialization."""
        xs_init = rng.uniform(0.1, w - 0.1, N)
        ys_init = rng.uniform(0.1, h - 0.1, N)
        return np.concatenate([xs_init, ys_init])

    def evaluate_positions(xs, ys, w, h):
        """Given positions, compute optimal radii via LP and return sum of radii."""
        points = np.column_stack([xs, ys])
        edges = _delaunay_edges(points)
        
        # Also add edges for close pairs that Delaunay might miss
        for i in range(N):
            for j in range(i+1, N):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < 0.6:  # close enough to potentially touch
                    edges.add((min(i,j), max(i,j)))
        
        rs = _solve_lp_radii(xs, ys, w, h, edges)
        
        # Verify validity
        if not _is_valid(np.concatenate([xs, ys, rs]), w, h):
            # Shrink radii slightly
            rs *= 0.999
            if not _is_valid(np.concatenate([xs, ys, rs]), w, h):
                return 0.0, None
        
        return np.sum(rs), rs

    best_val = 0.0
    best_params = None
    best_a = 1.0
    
    rng = np.random.default_rng(42)
    
    # Aspect ratio search
    a_values = np.linspace(0.85, 1.20, 14)
    
    for a in a_values:
        if time.time() - start_time > time_limit - 30:
            break
            
        w, h = get_wh(a)
        
        # Multiple restarts for this aspect ratio
        n_restarts = 12
        for k in range(n_restarts):
            if time.time() - start_time > time_limit - 20:
                break
            
            # Choose initialization type
            if k % 4 == 0:
                pos0 = make_grid_init(w, h, rng, noise_scale=0.02)
            elif k % 4 == 1:
                pos0 = make_hex_init(w, h, rng)
                if pos0 is None:
                    pos0 = make_grid_init(w, h, rng, noise_scale=0.03)
            elif k % 4 == 2:
                pos0 = make_grid_init(w, h, rng, noise_scale=0.05)
            else:
                pos0 = make_random_init(w, h, rng)
            
            if pos0 is None:
                continue
            
            # Optimize positions with L-BFGS-B
            try:
                res = minimize(position_objective_and_grad, pos0, args=(w, h), 
                               jac=True, method='L-BFGS-B',
                               options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8, 'maxls': 50})
                
                xs_opt = res.x[0:N]
                ys_opt = res.x[N:2*N]
                
                # Clip to valid range
                xs_opt = np.clip(xs_opt, 0.01, w - 0.01)
                ys_opt = np.clip(ys_opt, 0.01, h - 0.01)
                
                # Evaluate with LP radii
                sum_r, rs = evaluate_positions(xs_opt, ys_opt, w, h)
                
                if rs is not None and sum_r > best_val:
                    best_val = sum_r
                    best_params = np.concatenate([xs_opt, ys_opt, rs])
                    best_a = a
            except Exception:
                pass
    
    # Final polish: re-optimize positions starting from best, with tighter tolerance
    if best_params is not None:
        if time.time() - start_time < time_limit - 10:
            w, h = get_wh(best_a)
            xs_best = best_params[0:N]
            ys_best = best_params[N:2*N]
            pos_best = np.concatenate([xs_best, ys_best])
            
            try:
                res = minimize(position_objective_and_grad, pos_best, args=(w, h), 
                               jac=True, method='L-BFGS-B',
                               options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 100})
                
                xs_pol = np.clip(res.x[0:N], 0.01, w - 0.01)
                ys_pol = np.clip(res.x[N:2*N], 0.01, h - 0.01)
                
                sum_r, rs = evaluate_positions(xs_pol, ys_pol, w, h)
                
                if rs is not None and sum_r > best_val:
                    best_val = sum_r
                    best_params = np.concatenate([xs_pol, ys_pol, rs])
            except Exception:
                pass
    
    # Format output: array of shape (n, 3)
    if best_params is not None:
        out = best_params.reshape(N, 3)
        out[:, 2] = np.maximum(0, out[:, 2])
        return out
    else:
        # Fallback: simple grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
