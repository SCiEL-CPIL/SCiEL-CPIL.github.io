import numpy as np
from scipy.optimize import minimize, minimize_scalar
import time

N = 21

def _softplus(x, beta=10.0):
    """Smooth approximation of max(0, x) using softplus."""
    return np.log1p(np.exp(-beta * x)) / beta

def _softplus_grad(x, beta=10.0):
    """Derivative of softplus: sigmoid(-beta*x)."""
    return 1.0 / (1.0 + np.exp(beta * x))

def construct():
    start_time = time.time()
    time_limit = 145.0  # Safety margin before 150s kill
    
    def get_wh(a):
        h = 2.0 / (1.0 + a)
        w = 2.0 * a / (1.0 + a)
        return w, h

    def objective_and_grad(params, a, beta=10.0):
        """
        Objective: -sum(r) + lambda * (softplus penalties)
        Returns (f, g) for L-BFGS-B.
        """
        w, h = get_wh(a)
        pos = params.reshape(N, 3)
        xs = pos[:, 0]
        ys = pos[:, 1]
        rs = pos[:, 2]
        
        f = -np.sum(rs)
        g = np.zeros_like(params)
        
        # Negative radii penalty
        neg_r = -rs
        sp = _softplus(neg_r, beta)
        f += sp.sum()
        g[2::3] -= _softplus_grad(neg_r, beta)
        
        # Boundary penalties: x - r >= 0, w - x - r >= 0, y - r >= 0, h - y - r >= 0
        # Violation: r - x < 0 means penalty on (r - x)
        viol_x0 = rs - xs
        sp = _softplus(viol_x0, beta)
        f += sp.sum()
        g[0::3] += _softplus_grad(viol_x0, beta)
        g[2::3] -= _softplus_grad(viol_x0, beta)
        
        viol_x1 = rs - (w - xs)
        sp = _softplus(viol_x1, beta)
        f += sp.sum()
        g[0::3] -= _softplus_grad(viol_x1, beta)
        g[2::3] -= _softplus_grad(viol_x1, beta)
        
        viol_y0 = rs - ys
        sp = _softplus(viol_y0, beta)
        f += sp.sum()
        g[1::3] += _softplus_grad(viol_y0, beta)
        g[2::3] -= _softplus_grad(viol_y0, beta)
        
        viol_y1 = rs - (h - ys)
        sp = _softplus(viol_y1, beta)
        f += sp.sum()
        g[1::3] -= _softplus_grad(viol_y1, beta)
        g[2::3] -= _softplus_grad(viol_y1, beta)
        
        # Overlap penalties: for each pair, violation = (ri + rj - dist)
        # Use softplus on (ri + rj - dist)
        lam_overlap = 1.0
        for i in range(N):
            for j in range(i + 1, N):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                dist_sq = dx * dx + dy * dy
                dist = np.sqrt(dist_sq + 1e-12)
                rsum = rs[i] + rs[j]
                viol = rsum - dist
                sp_val = _softplus(viol, beta)
                f += lam_overlap * sp_val
                grad_sp = _softplus_grad(viol, beta)
                
                # d(viol)/d(xs[i]) = -dx/dist, d(viol)/d(ys[i]) = -dy/dist
                # d(viol)/d(xs[j]) = dx/dist, d(viol)/d(ys[j]) = dy/dist
                # d(viol)/d(rs[i]) = 1, d(viol)/d(rs[j]) = 1
                g[3*i] += lam_overlap * grad_sp * (-dx / dist)
                g[3*i+1] += lam_overlap * grad_sp * (-dy / dist)
                g[3*i+2] += lam_overlap * grad_sp
                g[3*j] += lam_overlap * grad_sp * (dx / dist)
                g[3*j+1] += lam_overlap * grad_sp * (dy / dist)
                g[3*j+2] += lam_overlap * grad_sp
        
        return f, g

    def is_valid(params, a, tol=1e-6):
        w, h = get_wh(a)
        pos = params.reshape(N, 3)
        xs = pos[:, 0]
        ys = pos[:, 1]
        rs = pos[:, 2]
        
        if np.any(rs < -tol): return False
        if np.any(xs < -tol) or np.any(w - xs < -tol): return False
        if np.any(ys < -tol) or np.any(h - ys < -tol): return False
        if np.any(rs - xs < -tol) or np.any(rs - (w - xs) < -tol): return False
        if np.any(rs - ys < -tol) or np.any(rs - (h - ys) < -tol): return False
        
        for i in range(N):
            for j in range(i+1, N):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < rs[i] + rs[j] - tol:
                    return False
        return True

    def make_grid_init(w, h, rng, noise_scale=0.02):
        """Create a grid-based initialization."""
        # Try different grid configurations
        best_init = None
        best_sum = -1
        
        for rows in range(3, 8):
            cols = int(np.ceil(N / rows))
            if cols < 2:
                continue
            
            xs_init = np.zeros(N)
            ys_init = np.zeros(N)
            rs_init = np.zeros(N)
            
            for i in range(N):
                row = i // cols
                col = i % cols
                xs_init[i] = (col + 0.5) * (w / cols)
                ys_init[i] = (row + 0.5) * (h / rows)
                cell_w = w / cols
                cell_h = h / rows
                rs_init[i] = 0.45 * min(cell_w, cell_h)
            
            # Add noise
            xs_init += rng.normal(0, noise_scale, N)
            ys_init += rng.normal(0, noise_scale, N)
            rs_init += rng.normal(0, noise_scale * 0.5, N)
            
            # Clip
            xs_init = np.clip(xs_init, 0.05, w - 0.05)
            ys_init = np.clip(ys_init, 0.05, h - 0.05)
            rs_init = np.clip(rs_init, 0.01, min(w, h) / 2)
            
            params = np.concatenate([xs_init, ys_init, rs_init])
            
            # Quick check: estimate sum of radii
            s = np.sum(rs_init)
            if s > best_sum:
                best_sum = s
                best_init = params
        
        return best_init

    def make_hex_init(w, h, rng):
        """Create a hexagonal packing initialization."""
        # Hexagonal packing: rows offset
        # For 21 circles, try 5-4-5-4-3 or similar patterns
        patterns = [
            [5, 4, 5, 4, 3],
            [4, 5, 4, 5, 3],
            [5, 5, 5, 4, 2],
            [3, 4, 5, 4, 5],
            [4, 4, 5, 4, 4],
        ]
        
        best_init = None
        best_sum = -1
        
        for pattern in patterns:
            if sum(pattern) != N:
                continue
            
            xs_init = np.zeros(N)
            ys_init = np.zeros(N)
            rs_init = np.zeros(N)
            
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
                    r = 0.45 * min(col_w, row_h)
                    xs_init[idx] = x
                    ys_init[idx] = y
                    rs_init[idx] = r
                    idx += 1
            
            if idx != N:
                continue
            
            # Add noise
            xs_init += rng.normal(0, 0.01, N)
            ys_init += rng.normal(0, 0.01, N)
            rs_init += rng.normal(0, 0.005, N)
            
            # Clip
            xs_init = np.clip(xs_init, 0.05, w - 0.05)
            ys_init = np.clip(ys_init, 0.05, h - 0.05)
            rs_init = np.clip(rs_init, 0.01, min(w, h) / 2)
            
            params = np.concatenate([xs_init, ys_init, rs_init])
            s = np.sum(rs_init)
            if s > best_sum:
                best_sum = s
                best_init = params
        
        return best_init

    best_val = 0.0
    best_params = None
    best_a = 1.0
    
    rng = np.random.default_rng(42)
    
    # Aspect ratio search
    a_values = np.linspace(0.85, 1.20, 12)
    
    for a in a_values:
        if time.time() - start_time > time_limit - 30:
            break
            
        w, h = get_wh(a)
        
        # Multiple restarts for this aspect ratio
        n_restarts = 10
        for k in range(n_restarts):
            if time.time() - start_time > time_limit - 25:
                break
            
            # Choose initialization type
            if k % 3 == 0:
                params0 = make_grid_init(w, h, rng, noise_scale=0.02)
            elif k % 3 == 1:
                params0 = make_hex_init(w, h, rng)
                if params0 is None:
                    params0 = make_grid_init(w, h, rng, noise_scale=0.03)
            else:
                params0 = make_grid_init(w, h, rng, noise_scale=0.05)
            
            if params0 is None:
                continue
            
            # Optimize with L-BFGS-B using analytical gradient
            try:
                res = minimize(objective_and_grad, params0, args=(a,), 
                               jac=True, method='L-BFGS-B',
                               options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-8, 'maxls': 50})
                
                if is_valid(res.x, a):
                    sum_r = np.sum(res.x[2::3])
                    if sum_r > best_val:
                        best_val = sum_r
                        best_params = res.x.copy()
                        best_a = a
            except Exception:
                pass
    
    # Final polish with tighter tolerance
    if best_params is not None:
        if time.time() - start_time < time_limit - 10:
            try:
                res = minimize(objective_and_grad, best_params, args=(best_a,), 
                               jac=True, method='L-BFGS-B',
                               options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 100})
                if is_valid(res.x, best_a):
                    sum_r = np.sum(res.x[2::3])
                    if sum_r > best_val:
                        best_val = sum_r
                        best_params = res.x.copy()
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
