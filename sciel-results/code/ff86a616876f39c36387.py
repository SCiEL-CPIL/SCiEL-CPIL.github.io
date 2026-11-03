import numpy as np
import time
from scipy.spatial import cKDTree
from scipy.optimize import minimize

N = 300

def construct():
    start_time = time.time()
    time_limit = 145.0  # Safety margin before 150s kill

    def compute_score(pts):
        """Compute the packing radius score."""
        tree = cKDTree(pts)
        k = min(20, N - 1)
        dists_k, idx_k = tree.query(pts, k=k + 1)
        rad = np.linalg.norm(pts, axis=1)
        bound_dists = 1.0 - rad
        
        min_pair_dist = np.inf
        for i in range(N):
            for j in range(1, k + 1):
                d = dists_k[i, j]
                if d > 1e-10:
                    min_pair_dist = min(min_pair_dist, d)
        
        min_bound = np.min(bound_dists)
        return min(min_pair_dist / 2.0, min_bound)

    def soft_min_obj(x, k):
        """
        Objective: -soft_min of all constraints (pair distances / 2 and boundary distances).
        We want to maximize the minimum, so we minimize the negative soft-min.
        x is a flat array of shape (2N,).
        """
        pts = x.reshape(N, 2)
        tree = cKDTree(pts)
        k_neighbors = min(20, N - 1)
        dists_k, idx_k = tree.query(pts, k=k_neighbors + 1)
        
        rad = np.linalg.norm(pts, axis=1)
        bound_dists = 1.0 - rad
        
        # Collect pair distances (avoid double counting)
        pair_vals = []
        for i in range(N):
            for j in range(1, k_neighbors + 1):
                d = dists_k[i, j]
                j_idx = idx_k[i, j]
                if i < j_idx and d > 1e-10:
                    pair_vals.append(d / 2.0)
        
        pair_vals = np.array(pair_vals)
        all_vals = np.concatenate([pair_vals, bound_dists])
        
        # Soft min: -log(sum(exp(-k * vals))) / k
        # To avoid overflow, shift by min
        m = np.min(all_vals)
        shifted = all_vals - m
        # exp(-k * shifted) is safe since shifted >= 0
        log_sum = np.log(np.sum(np.exp(-k * shifted)))
        soft_min = m - log_sum / k
        
        return -soft_min

    def soft_min_grad(x, k):
        """
        Gradient of -soft_min.
        d(-soft_min)/dx = - d(soft_min)/dx
        soft_min = m - (1/k) log(sum exp(-k * v_i))
        d(soft_min)/dx = - (1/k) * (sum (-k * exp(-k*v_i) * dv_i/dx) / sum(exp(-k*v_i)))
                       = sum(exp(-k*v_i) * dv_i/dx / sum(exp(-k*v_i)))
        So d(-soft_min)/dx = - sum(w_i * dv_i/dx) where w_i = exp(-k*v_i) / sum(exp(-k*v_j))
        """
        pts = x.reshape(N, 2)
        tree = cKDTree(pts)
        k_neighbors = min(20, N - 1)
        dists_k, idx_k = tree.query(pts, k=k_neighbors + 1)
        
        rad = np.linalg.norm(pts, axis=1)
        bound_dists = 1.0 - rad
        
        pair_vals = []
        pair_i = []
        pair_j = []
        for i in range(N):
            for j in range(1, k_neighbors + 1):
                d = dists_k[i, j]
                j_idx = idx_k[i, j]
                if i < j_idx and d > 1e-10:
                    pair_vals.append(d / 2.0)
                    pair_i.append(i)
                    pair_j.append(j_idx)
        
        pair_vals = np.array(pair_vals)
        pair_i = np.array(pair_i)
        pair_j = np.array(pair_j)
        all_vals = np.concatenate([pair_vals, bound_dists])
        
        m = np.min(all_vals)
        shifted = all_vals - m
        exp_terms = np.exp(-k * shifted)
        sum_exp = np.sum(exp_terms)
        if sum_exp < 1e-300:
            weights = np.ones_like(all_vals) / len(all_vals)
        else:
            weights = exp_terms / sum_exp
        
        # Gradient of soft_min w.r.t. x
        # d(soft_min)/dx = sum(w_i * dv_i/dx)
        # We want d(-soft_min)/dx = - sum(w_i * dv_i/dx)
        
        grad = np.zeros_like(x)
        
        # Pair contributions
        if len(pair_vals) > 0:
            pair_weights = weights[:len(pair_vals)]
            for idx in range(len(pair_vals)):
                i = pair_i[idx]
                j = pair_j[idx]
                w = pair_weights[idx]
                d_val = dists_k[i, np.where(idx_k[i] == j)[0][0]]
                if d_val > 1e-10:
                    vec = pts[i] - pts[j]
                    vec_norm = vec / d_val
                    # v_i = d/2, dv_i/dx_i = vec/d, dv_i/dx_j = -vec/d
                    # d(soft_min)/dx_i += w * vec/d
                    # d(-soft_min)/dx_i -= w * vec/d
                    grad[i * 2: i * 2 + 2] -= w * vec_norm
                    grad[j * 2: j * 2 + 2] += w * vec_norm
        
        # Boundary contributions
        bound_weights = weights[len(pair_vals):]
        rad_safe = np.where(rad > 1e-10, rad, 1.0)
        # v_i = 1 - |x_i|, dv_i/dx_i = -x_i/|x_i|
        # d(soft_min)/dx_i += w_i * (-x_i/|x_i|)
        # d(-soft_min)/dx_i -= w_i * (-x_i/|x_i|) = w_i * x_i/|x_i|
        bound_grads = (pts / rad_safe[:, np.newaxis]) * bound_weights[:, np.newaxis]
        grad += bound_grads
        
        return grad

    def hex_lattice_init(spacing):
        """Generate hexagonal lattice points inside unit circle."""
        y_step = spacing * np.sqrt(3) / 2
        max_j = int(1.0 / y_step) + 2
        max_i = int(1.0 / spacing) + 2
        
        pts = []
        for j in range(-max_j, max_j + 1):
            y = j * y_step
            offset = 0 if j % 2 == 0 else spacing / 2
            for i in range(-max_i, max_i + 1):
                x = i * spacing + offset
                if np.hypot(x, y) < 0.98:
                    pts.append([x, y])
        
        pts = np.array(pts)
        
        if len(pts) >= N:
            # Select N points using farthest-point sampling
            selected = [np.random.choice(len(pts))]
            while len(selected) < N:
                dists = np.min([np.linalg.norm(pts - pts[s], axis=1) for s in selected], axis=0)
                next_idx = np.argmax(dists)
                selected.append(next_idx)
            pts = pts[selected]
        elif len(pts) < N:
            while len(pts) < N:
                p = np.random.uniform(-0.95, 0.95, 2)
                if np.linalg.norm(p) < 0.95:
                    pts = np.vstack([pts, p])
            pts = pts[:N]
        
        return pts

    def random_init():
        """Random initialization."""
        pts = []
        while len(pts) < N:
            p = np.random.uniform(-0.98, 0.98, 2)
            if np.linalg.norm(p) < 0.98:
                pts.append(p)
        return np.array(pts)

    # Generate multiple initializations
    best_pts = None
    best_score = 0.0

    # Phase 1: Hexagonal lattice initializations
    base_spacing = np.sqrt(2 * np.pi / (np.sqrt(3) * N))
    spacing_multipliers = [0.90, 0.95, 1.00, 1.05]
    
    for mult in spacing_multipliers:
        if time.time() - start_time > time_limit * 0.50:
            break
        
        spacing = base_spacing * mult
        pts = hex_lattice_init(spacing)
        x0 = pts.flatten()
        
        # Run L-BFGS-B with soft-min objective
        # Start with small k (smooth), increase over iterations
        k_start = 10.0
        k_end = 500.0
        
        # Use a callback to increase k over time
        state = {'k': k_start, 'iter': 0}
        
        def callback(xk):
            state['iter'] += 1
            # Increase k every 20 iterations
            if state['iter'] % 20 == 0:
                state['k'] = min(state['k'] * 1.3, k_end)
            return False
        
        # Time budget per optimization
        time_budget = min(10.0, time_limit * 0.10)
        end_time = start_time + time_budget
        
        # We need to stop L-BFGS-B based on time. Use a wrapper.
        # Since scipy.optimize.minimize doesn't have a time-based callback easily,
        # we'll use maxiter and estimate.
        # Alternatively, run in a loop and break if time exceeds.
        
        # Let's do a manual loop with L-BFGS-B steps? No, L-BFGS-B is a black box.
        # We'll just set maxiter high and rely on the time check after each call?
        # Actually, minimize() runs to completion. We need to interrupt it.
        # One way: use a custom callback that raises an exception or returns True to stop.
        # But we also want to update k.
        
        # Let's try: run minimize with a callback that checks time and returns True to stop.
        # Also update k in the callback.
        
        def time_callback(xk):
            state['iter'] += 1
            if time.time() > end_time:
                return True  # Stop optimization
            if state['iter'] % 20 == 0:
                state['k'] = min(state['k'] * 1.3, k_end)
            return False
        
        try:
            result = minimize(
                soft_min_obj,
                x0,
                jac=soft_min_grad,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8, 'maxfun': 10000},
                callback=time_callback
            )
        except Exception:
            result = None
        
        if result is not None and result.success:
            pts_opt = result.x.reshape(N, 2)
            # Project back inside circle
            rad = np.linalg.norm(pts_opt, axis=1)
            mask_out = rad > 1.0
            if np.any(mask_out):
                pts_opt[mask_out] /= rad[mask_out, np.newaxis]
            
            score = compute_score(pts_opt)
            if score > best_score:
                best_score = score
                best_pts = pts_opt.copy()
        
        # Also try the initial point
        score_init = compute_score(pts)
        if score_init > best_score:
            best_score = score_init
            best_pts = pts.copy()

    # Phase 2: Random initializations
    if time.time() - start_time < time_limit * 0.70:
        for _ in range(2):
            if time.time() - start_time > time_limit * 0.70:
                break
            
            pts = random_init()
            x0 = pts.flatten()
            
            k_start = 5.0
            k_end = 300.0
            state = {'k': k_start, 'iter': 0}
            
            time_budget = min(8.0, time_limit * 0.08)
            end_time = start_time + time_budget
            
            def time_callback(xk):
                state['iter'] += 1
                if time.time() > end_time:
                    return True
                if state['iter'] % 20 == 0:
                    state['k'] = min(state['k'] * 1.3, k_end)
                return False
            
            try:
                result = minimize(
                    soft_min_obj,
                    x0,
                    jac=soft_min_grad,
                    method='L-BFGS-B',
                    options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8, 'maxfun': 10000},
                    callback=time_callback
                )
            except Exception:
                result = None
            
            if result is not None:
                pts_opt = result.x.reshape(N, 2)
                rad = np.linalg.norm(pts_opt, axis=1)
                mask_out = rad > 1.0
                if np.any(mask_out):
                    pts_opt[mask_out] /= rad[mask_out, np.newaxis]
                
                score = compute_score(pts_opt)
                if score > best_score:
                    best_score = score
                    best_pts = pts_opt.copy()
            
            score_init = compute_score(pts)
            if score_init > best_score:
                best_score = score_init
                best_pts = pts.copy()

    # Phase 3: Polish with higher k
    if best_pts is not None and time.time() - start_time < time_limit * 0.85:
        remaining_time = time_limit * 0.95 - time.time()
        if remaining_time > 3.0:
            x0 = best_pts.flatten()
            k_start = 500.0
            k_end = 2000.0
            state = {'k': k_start, 'iter': 0}
            
            time_budget = min(remaining_time, 15.0)
            end_time = start_time + time_budget
            
            def time_callback(xk):
                state['iter'] += 1
                if time.time() > end_time:
                    return True
                if state['iter'] % 10 == 0:
                    state['k'] = min(state['k'] * 1.2, k_end)
                return False
            
            try:
                result = minimize(
                    soft_min_obj,
                    x0,
                    jac=soft_min_grad,
                    method='L-BFGS-B',
                    options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 10000},
                    callback=time_callback
                )
            except Exception:
                result = None
            
            if result is not None:
                pts_opt = result.x.reshape(N, 2)
                rad = np.linalg.norm(pts_opt, axis=1)
                mask_out = rad > 1.0
                if np.any(mask_out):
                    pts_opt[mask_out] /= rad[mask_out, np.newaxis]
                
                score = compute_score(pts_opt)
                if score > best_score:
                    best_score = score
                    best_pts = pts_opt.copy()

    # Final verification
    if best_pts is None:
        best_pts = hex_lattice_init(base_spacing)
    
    # Ensure all points are inside the circle
    rad = np.linalg.norm(best_pts, axis=1)
    mask_out = rad > 1.0
    if np.any(mask_out):
        best_pts[mask_out] /= rad[mask_out, np.newaxis]
    
    return best_pts
