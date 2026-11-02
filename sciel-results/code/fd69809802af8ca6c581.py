import numpy as np
from scipy.optimize import minimize
import time

N = 21

def _softplus(x, beta=50.0):
    """Smooth approximation of max(0, x) using softplus.
    beta controls the sharpness. Higher beta is sharper.
    """
    # To avoid overflow in exp, we can use np.log1p(np.exp(-beta * x)) / beta for x < 0
    # and -x for x >= 0. But for simplicity and stability in optimization, 
    # let's use a safe version.
    # Actually, standard softplus is (1/beta) * log(1 + exp(beta*x)).
    # For large positive beta*x, it approximates x.
    # For large negative beta*x, it approximates 0.
    # We want to penalize violations where violation > 0.
    # So we want softplus(violation).
    
    # Safe softplus to avoid overflow
    # max(0, x) approx
    # If x is large, exp(beta*x) overflows.
    # We can use: if x > 0, return x (approx). If x <= 0, return log(1+exp(beta*x))/beta.
    # But for differentiability, we stick to the formula but clip the input to exp.
    
    # A simpler stable softplus for penalties:
    # f(x) = log(1 + exp(beta*x)) / beta
    # For x > 30/beta, exp is huge. 
    # Let's just use a hard clip in the exp argument.
    arg = beta * x
    # np.clip(arg, -50, 50)
    # Actually, scipy's minimize handles gradients. 
    # Let's use a slightly different formulation that is stable.
    # f(x) = x * sigmoid(beta*x) + log(1+exp(-beta*x))/beta ? No.
    
    # Let's use the standard:
    # f(x) = (1/beta) * log(1 + exp(beta*x))
    # To prevent overflow:
    # if beta*x > 50: return x
    # if beta*x < -50: return 0
    # else: return (1/beta)*np.log(1+np.exp(beta*x))
    
    # Vectorized:
    out = np.zeros_like(x)
    large = beta * x > 50
    small = beta * x < -50
    mid = ~(large | small)
    
    out[large] = x[large]
    # out[small] is 0
    out[mid] = (1.0 / beta) * np.log1p(np.exp(beta * x[mid]))
    return out

def _softplus_grad(x, beta=50.0):
    """Gradient of the stable softplus above."""
    # Derivative of (1/beta) log(1+exp(beta*x)) is sigmoid(beta*x) = 1/(1+exp(-beta*x))
    # For large x, sigmoid -> 1.
    # For small x, sigmoid -> 0.
    out = np.zeros_like(x)
    large = beta * x > 50
    small = beta * x < -50
    mid = ~(large | small)
    
    out[large] = 1.0
    # out[small] is 0
    out[mid] = 1.0 / (1.0 + np.exp(-beta * x[mid]))
    return out

def construct():
    start_time = time.time()
    time_limit = 145.0  # Safety margin before 150s kill

    def get_wh(w):
        h = 2.0 - w
        return w, h

    def objective_and_grad(params, w, h, beta=50.0, lambda_pen=100.0):
        """
        params: [x_0, ..., x_{N-1}, y_0, ..., y_{N-1}, r_0, ..., r_{N-1}]
        Objective: -sum(r) + penalties for overlaps and boundary violations.
        """
        n = N
        xs = params[0:n]
        ys = params[n:2*n]
        rs = params[2*n:3*n]
        
        f = -np.sum(rs)
        g = np.zeros(3*n)
        g[2*n:3*n] = -1.0  # Gradient of -sum(r) w.r.t r is -1
        
        # 1. Boundary Penalties
        # Violations:
        # v_x0 = r_i - x_i
        # v_x1 = r_i - (w - x_i)
        # v_y0 = r_i - y_i
        # v_y1 = r_i - (h - y_i)
        
        v_x0 = rs - xs
        v_x1 = rs - (w - xs)
        v_y0 = rs - ys
        v_y1 = rs - (h - ys)
        
        pen_x0 = _softplus(v_x0, beta)
        pen_x1 = _softplus(v_x1, beta)
        pen_y0 = _softplus(v_y0, beta)
        pen_y1 = _softplus(v_y1, beta)
        
        f += lambda_pen * (np.sum(pen_x0) + np.sum(pen_x1) + np.sum(pen_y0) + np.sum(pen_y1))
        
        # Gradients for boundary
        g_x0 = _softplus_grad(v_x0, beta)
        g_x1 = _softplus_grad(v_x1, beta)
        g_y0 = _softplus_grad(v_y0, beta)
        g_y1 = _softplus_grad(v_y1, beta)
        
        # d/dx_i of pen_x0 = -g_x0_i
        g[0:n] -= lambda_pen * g_x0
        # d/dx_i of pen_x1 = g_x1_i (since w-x_i)
        g[0:n] += lambda_pen * g_x1
        # d/dy_i of pen_y0 = -g_y0_i
        g[n:2*n] -= lambda_pen * g_y0
        # d/dy_i of pen_y1 = g_y1_i
        g[n:2*n] += lambda_pen * g_y1
        # d/dr_i of boundary pens
        g[2*n:3*n] += lambda_pen * (g_x0 + g_x1 + g_y0 + g_y1)
        
        # 2. Overlap Penalties
        # We need to compute pairwise distances.
        # For N=21, N(N-1)/2 = 210 pairs. This is small enough to compute all pairs.
        dx = xs[:, None] - xs[None, :]
        dy = ys[:, None] - ys[None, :]
        dist_sq = dx*dx + dy*dy
        np.fill_diagonal(dist_sq, 1e-12) # Avoid div by zero
        dist = np.sqrt(dist_sq)
        
        # Violation: r_i + r_j - dist_ij
        # We want to penalize only if positive.
        # sum over i < j
        
        # Vectorized calculation of violations
        # V_ij = rs_i + rs_j - dist_ij
        # We can't easily vectorize the sum over i<j with the softplus in a single line 
        # without creating a full NxN matrix, which is fine for N=21.
        
        r_sum = rs[:, None] + rs[None, :]
        viol = r_sum - dist
        
        # Set diagonal to 0 (no self-overlap)
        np.fill_diagonal(viol, 0.0)
        
        # Only consider upper triangle to avoid double counting
        iu = np.triu_indices(n, 1)
        viol_pairs = viol[iu]
        
        pen_overlap = _softplus(viol_pairs, beta)
        f += lambda_pen * np.sum(pen_overlap)
        
        # Gradients for overlap
        # g_sp = d(softplus)/d(viol)
        g_sp = _softplus_grad(viol_pairs, beta)
        
        # We need to map these back to the full gradient vectors.
        # For each pair (i, j) with i < j:
        # d(viol_ij)/dx_i = -dx_ij / dist_ij
        # d(viol_ij)/dx_j = dx_ij / dist_ij
        # d(viol_ij)/dy_i = -dy_ij / dist_ij
        # d(viol_ij)/dy_j = dy_ij / dist_ij
        # d(viol_ij)/dr_i = 1
        # d(viol_ij)/dr_j = 1
        
        # Extract i, j indices
        i_idx, j_idx = iu
        
        # Distances for these pairs
        dist_pairs = dist[iu]
        dx_pairs = dx[iu]
        dy_pairs = dy[iu]
        
        # Weights
        w_x = g_sp * (-dx_pairs / dist_pairs)
        w_y = g_sp * (-dy_pairs / dist_pairs)
        
        # Accumulate gradients using np.add.at for efficiency
        # d/dx_i
        np.add.at(g[0:n], i_idx, lambda_pen * w_x)
        np.add.at(g[0:n], j_idx, -lambda_pen * w_x) # Note: dx_ij = x_i - x_j, so d/dx_j is +dx_ij/dist
        
        # Wait, let's re-verify signs.
        # viol = r_i + r_j - dist
        # dist = sqrt((x_i - x_j)^2 + ...)
        # d(dist)/dx_i = (x_i - x_j)/dist
        # d(viol)/dx_i = - (x_i - x_j)/dist
        # So term for x_i is -w_x.
        # Term for x_j: d(dist)/dx_j = -(x_i - x_j)/dist = (x_j - x_i)/dist
        # d(viol)/dx_j = - (x_j - x_i)/dist = (x_i - x_j)/dist = w_x
        # So:
        # g[x_i] += lambda * g_sp * (-(x_i - x_j)/dist)
        # g[x_j] += lambda * g_sp * ((x_i - x_j)/dist)
        
        # My w_x was defined as g_sp * (-dx_pairs / dist_pairs) where dx_pairs = x_i - x_j.
        # So w_x = g_sp * (-(x_i - x_j)/dist).
        # So g[x_i] += lambda * w_x.
        # And g[x_j] += lambda * (-w_x).
        
        # Let's correct the np.add.at calls:
        g[0:n][i_idx] += 0 # Reset to be safe if we were doing it wrong, but we are adding.
        # Actually, np.add.at handles the accumulation.
        
        # Let's re-run the accumulation logic carefully.
        # We calculated w_x = g_sp * (-dx_pairs / dist_pairs).
        # This is the contribution to the gradient of x_i.
        # The contribution to x_j is -w_x.
        
        # I already added to g in the previous block? No, I just calculated f.
        # Now I add to g.
        
        # Correction: I didn't add to g in the previous lines, I just calculated w_x.
        # So:
        np.add.at(g[0:n], i_idx, lambda_pen * w_x)
        np.add.at(g[0:n], j_idx, -lambda_pen * w_x)
        
        np.add.at(g[n:2*n], i_idx, lambda_pen * w_y)
        np.add.at(g[n:2*n], j_idx, -lambda_pen * w_y)
        
        # Radii gradients
        # d(viol)/dr_i = 1
        # d(viol)/dr_j = 1
        np.add.at(g[2*n:3*n], i_idx, lambda_pen * g_sp)
        np.add.at(g[2*n:3*n], j_idx, lambda_pen * g_sp)
        
        return f, g

    def make_grid_init(w, h, rng, noise_scale=0.01):
        best_init = None
        best_sum = -1
        for rows in range(3, 8):
            cols = int(np.ceil(N / rows))
            if cols < 2: continue
            
            xs = np.zeros(N)
            ys = np.zeros(N)
            rs = np.zeros(N)
            
            for i in range(N):
                row = i // cols
                col = i % cols
                xs[i] = (col + 0.5) * (w / cols)
                ys[i] = (row + 0.5) * (h / rows)
                # Initial radius estimate: min distance to boundary / 2
                r_est = 0.5 * min(xs[i], w - xs[i], ys[i], h - ys[i])
                rs[i] = r_est
            
            xs += rng.normal(0, noise_scale, N)
            ys += rng.normal(0, noise_scale, N)
            rs = np.clip(rs, 0.01, 0.5)
            
            params = np.concatenate([xs, ys, rs])
            # Simple heuristic score
            s = np.sum(rs)
            if s > best_sum:
                best_sum = s
                best_init = params
        return best_init

    def make_hex_init(w, h, rng):
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
            if sum(pattern) != N: continue
            
            xs = np.zeros(N)
            ys = np.zeros(N)
            rs = np.zeros(N)
            
            idx = 0
            n_rows = len(pattern)
            row_h = h / n_rows
            
            for row_idx, n_cols in enumerate(pattern):
                col_w = w / n_cols
                offset = 0.0 if row_idx % 2 == 0 else col_w / 2
                for col_idx in range(n_cols):
                    if idx >= N: break
                    x = (col_idx + 0.5) * col_w + offset
                    y = (row_idx + 0.5) * row_h
                    xs[idx] = x
                    ys[idx] = y
                    r_est = 0.5 * min(x, w - x, y, h - y)
                    rs[idx] = r_est
                    idx += 1
            
            if idx != N: continue
            
            xs += rng.normal(0, 0.01, N)
            ys += rng.normal(0, 0.01, N)
            rs = np.clip(rs, 0.01, 0.5)
            
            params = np.concatenate([xs, ys, rs])
            s = np.sum(rs)
            if s > best_sum:
                best_sum = s
                best_init = params
        return best_init

    def make_random_init(w, h, rng):
        xs = rng.uniform(0.1, w - 0.1, N)
        ys = rng.uniform(0.1, h - 0.1, N)
        rs = rng.uniform(0.05, 0.2, N)
        return np.concatenate([xs, ys, rs])

    def is_valid(params, w, h, tol=1e-5):
        xs = params[0:N]
        ys = params[N:2*N]
        rs = params[2*N:3*N]
        
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

    best_val = 0.0
    best_params = None
    best_w = 1.0
    
    rng = np.random.default_rng(42)
    
    # Search over aspect ratio (width w, height h=2-w)
    # w in [0.5, 1.5] since h=2-w and w,h > 0.
    # Symmetry: w and h are interchangeable, so we can restrict w to [0.5, 1.0] 
    # and just swap w,h if needed, or just search [0.5, 1.5] and let the optimizer handle it.
    # However, the objective function treats x and y differently if we don't enforce symmetry.
    # But since we optimize x,y,r, the shape matters.
    # Let's search w from 0.6 to 1.4.
    
    w_values = np.linspace(0.6, 1.4, 15)
    
    for w in w_values:
        if time.time() - start_time > time_limit - 20:
            break
            
        h = 2.0 - w
        
        # Multiple restarts
        n_restarts = 10
        for k in range(n_restarts):
            if time.time() - start_time > time_limit - 10:
                break
                
            if k % 3 == 0:
                params0 = make_grid_init(w, h, rng, noise_scale=0.02)
            elif k % 3 == 1:
                params0 = make_hex_init(w, h, rng)
                if params0 is None:
                    params0 = make_grid_init(w, h, rng, noise_scale=0.03)
            else:
                params0 = make_random_init(w, h, rng)
                
            if params0 is None:
                continue
                
            try:
                # Use L-BFGS-B
                # Bounds:
                # x in [0, w], y in [0, h], r in [0, min(w,h)/2]
                max_r = min(w, h) / 2.0
                bounds = []
                for i in range(N):
                    bounds.append((0.0, w))
                for i in range(N):
                    bounds.append((0.0, h))
                for i in range(N):
                    bounds.append((0.0, max_r))
                    
                res = minimize(objective_and_grad, params0, args=(w, h), 
                               jac=True, method='L-BFGS-B',
                               bounds=bounds,
                               options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50})
                
                if res.success:
                    # Check validity
                    if is_valid(res.x, w, h):
                        val = np.sum(res.x[2*N:3*N])
                        if val > best_val:
                            best_val = val
                            best_params = res.x.copy()
                            best_w = w
                    else:
                        # If invalid, try to fix by shrinking radii slightly
                        # This is a fallback, ideally the penalty should have prevented this
                        pass
            except Exception:
                pass

    # Final Polish
    if best_params is not None and time.time() - start_time < time_limit - 5:
        w = best_w
        h = 2.0 - w
        max_r = min(w, h) / 2.0
        bounds = []
        for i in range(N):
            bounds.append((0.0, w))
        for i in range(N):
            bounds.append((0.0, h))
        for i in range(N):
            bounds.append((0.0, max_r))
            
        try:
            # Higher precision polish
            res = minimize(objective_and_grad, best_params, args=(w, h), 
                           jac=True, method='L-BFGS-B',
                           bounds=bounds,
                           options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-10, 'maxls': 100})
            
            if res.success and is_valid(res.x, w, h):
                val = np.sum(res.x[2*N:3*N])
                if val > best_val:
                    best_val = val
                    best_params = res.x.copy()
        except Exception:
            pass

    if best_params is not None:
        out = best_params.reshape(N, 3)
        out[:, 2] = np.maximum(0, out[:, 2])
        return out
    else:
        # Fallback: simple grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
