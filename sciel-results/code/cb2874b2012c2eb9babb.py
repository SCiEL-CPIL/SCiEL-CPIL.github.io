# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def moving_sofa_2d() -> np.ndarray:
    """
    Finds a path for moving a sofa through an L-shaped corridor.
    Returns a float array of shape (n, 3) where each row is (x, y, angle_degrees).
    """
    start_time = time.time()
    deadline = start_time + 580.0  # Safety margin

    # Grid points for sofa area calculation
    u = np.arange(-3.0, 1.5, 0.01)
    v = np.arange(-1.5, 1.5, 0.01)
    gu, gv = np.meshgrid(u, v, indexing='ij')
    grid_points = np.column_stack((gu.ravel(), gv.ravel()))
    N = grid_points.shape[0]
    gx = grid_points[:, 0]
    gy = grid_points[:, 1]

    def densify_path(path):
        """
        Prepend (0,0,0), append (-2,0,90), then insert linear interpolations
        between consecutive rows so that each segment spans <= 1 degree of rotation.
        """
        start = np.array([0.0, 0.0, 0.0])
        end = np.array([-2.0, 0.0, 90.0])
        full = np.vstack(([start], path, [end]))
        
        dense_points = [full[0]]
        for i in range(len(full) - 1):
            p1 = full[i]
            p2 = full[i + 1]
            da = p2[2] - p1[2]
            m = max(1, int(np.ceil(abs(da))))
            
            if m > 1:
                ts = np.linspace(1.0 / m, 1.0 - 1.0 / m, m - 1)
                new_pts = p1 + ts[:, np.newaxis] * (p2 - p1)
                dense_points.extend(new_pts)
            
            dense_points.append(p2)
        
        return np.array(dense_points)

    def score_path_hard(path):
        """
        Compute the raw score (area) using the strict definition:
        Points must be inside ALL placements.
        """
        dense = densify_path(path)
        n_dense = len(dense)
        
        a_rad = np.deg2rad(dense[:, 2])
        cos_a = np.cos(a_rad)
        sin_a = np.sin(a_rad)
        
        inside_all = np.ones(N, dtype=bool)
        
        batch_size = 50
        
        for batch_start in range(0, n_dense, batch_size):
            batch_end = min(batch_start + batch_size, n_dense)
            batch_cos = cos_a[batch_start:batch_end]
            batch_sin = sin_a[batch_start:batch_end]
            batch_x = dense[batch_start:batch_end, 0]
            batch_y = dense[batch_start:batch_end, 1]
            
            wx = gx[np.newaxis, :] - batch_x[:, np.newaxis]
            wy = gy[np.newaxis, :] - batch_y[:, np.newaxis]
            
            ul = batch_cos[:, np.newaxis] * wx + batch_sin[:, np.newaxis] * wy
            vl = -batch_sin[:, np.newaxis] * wx + batch_cos[:, np.newaxis] * wy
            
            in_r1 = (ul >= 0.0) & (ul <= 1.0) & (vl >= -3.0) & (vl <= 1.0)
            in_r2 = (ul >= -3.0) & (ul <= 1.0) & (vl >= 0.0) & (vl <= 1.0)
            in_base = in_r1 | in_r2
            
            inside_all &= in_base.all(axis=0)
            
            if not np.any(inside_all):
                break
        
        return np.sum(inside_all) * 0.0001

    def soft_min_score(path, tau=0.02):
        """
        Compute a smooth surrogate score using log-sum-exp soft-min over placements.
        This is differentiable and allows gradient-based optimization to work effectively.
        Returns the score (higher is better).
        """
        dense = densify_path(path)
        n_dense = len(dense)
        
        a_rad = np.deg2rad(dense[:, 2])
        cos_a = np.cos(a_rad)
        sin_a = np.sin(a_rad)
        
        # We want to maximize the sum over grid points of the "soft min" of margins.
        # Margin > 0 means inside.
        # soft_min(margins) approx min(margins).
        # We want to maximize sum(soft_min).
        
        # To avoid memory explosion, we accumulate the log-sum-exp terms.
        # logsumexp(-m/tau) = max(-m/tau) + log(sum(exp(-m/tau - max(-m/tau))))
        # soft_min = -tau * logsumexp(-m/tau)
        
        # We process in batches of placements.
        # For each grid point, we need the global logsumexp over all placements.
        # This requires storing the max and the sum of exps for each grid point.
        
        max_neg_m_div_tau = np.full(N, -np.inf)
        sum_exp_neg_m_div_tau = np.zeros(N)
        
        batch_size = 50
        
        for batch_start in range(0, n_dense, batch_size):
            batch_end = min(batch_start + batch_size, n_dense)
            batch_cos = cos_a[batch_start:batch_end]
            batch_sin = sin_a[batch_start:batch_end]
            batch_x = dense[batch_start:batch_end, 0]
            batch_y = dense[batch_start:batch_end, 1]
            
            wx = gx[np.newaxis, :] - batch_x[:, np.newaxis]
            wy = gy[np.newaxis, :] - batch_y[:, np.newaxis]
            
            ul = batch_cos[:, np.newaxis] * wx + batch_sin[:, np.newaxis] * wy
            vl = -batch_sin[:, np.newaxis] * wx + batch_cos[:, np.newaxis] * wy
            
            # Margins for rectangle 1
            margin_r1 = np.minimum(
                np.minimum(ul, 1.0 - ul),
                np.minimum(vl + 3.0, 1.0 - vl)
            )
            # Margins for rectangle 2
            margin_r2 = np.minimum(
                np.minimum(ul + 3.0, 1.0 - ul),
                np.minimum(vl, 1.0 - vl)
            )
            
            # Point is inside if max(margin_r1, margin_r2) > 0
            # The "effective margin" for being inside the L-shape is max(margin_r1, margin_r2)
            # Wait, the logic "inside if in_r1 OR in_r2" means the margin is indeed the max of the two distances.
            margin = np.maximum(margin_r1, margin_r2)
            
            # We need -margin / tau
            neg_m_div_tau = -margin / tau
            
            # Update max and sum_exp for each grid point
            # max_neg_m_div_tau[j] = max(max_neg_m_div_tau[j], neg_m_div_tau[i, j])
            # This is tricky with batched updates because we need to combine with previous batches.
            
            # Let's do it carefully.
            # Current batch max:
            batch_max = np.max(neg_m_div_tau, axis=0)
            
            # Global max so far vs batch max
            new_max = np.maximum(max_neg_m_div_tau, batch_max)
            
            # We need to rescale the sum_exp if the max increases.
            # sum_exp_new = sum_exp_old * exp(old_max - new_max) + sum(batch_exp * exp(batch_max - new_max))
            
            # old_max is max_neg_m_div_tau
            # batch_max is batch_max
            # new_max is new_max
            
            # Rescale old sum
            old_scale = np.exp(max_neg_m_div_tau - new_max)
            # Handle -inf
            old_scale = np.where(np.isfinite(max_neg_m_div_tau), old_scale, 0.0)
            current_sum_exp = sum_exp_neg_m_div_tau * old_scale
            
            # Calculate batch sum exp relative to new_max
            # exp(neg_m_div_tau - new_max)
            batch_exp_terms = np.exp(neg_m_div_tau - new_max[np.newaxis, :])
            batch_sum_exp = np.sum(batch_exp_terms, axis=0)
            
            sum_exp_neg_m_div_tau = current_sum_exp + batch_sum_exp
            max_neg_m_div_tau = new_max
        
        # Final soft_min calculation
        # logsumexp = max + log(sum_exp)
        # Handle cases where sum_exp is 0 (shouldn't happen if at least one placement)
        logsumexp = max_neg_m_div_tau + np.log(sum_exp_neg_m_div_tau)
        soft_min_vals = -tau * logsumexp
        
        score = np.sum(soft_min_vals) * 0.0001
        return score

    def objective(params):
        """
        Objective for minimization: minimize negative soft_min_score.
        """
        n_path = len(params) // 3
        path = params.reshape(n_path, 3)
        # Clip angles to valid range to help optimizer
        path[:, 2] = np.clip(path[:, 2], 0, 90)
        return -soft_min_score(path)

    # Initial paths
    initial_paths = [
        np.array([
            [-0.1, 0.1, 15.0],
            [-0.2, 0.2, 25.0],
            [-0.3, 0.3, 35.0],
            [-0.4, 0.4, 45.0],
            [-0.5, 0.5, 55.0],
            [-0.6, 0.6, 65.0],
            [-0.7, 0.7, 75.0],
            [-0.8, 0.8, 85.0]
        ]),
        np.array([
            [-0.1, 0.1, 15.0],
            [-0.2, 0.2, 20.0],
            [-0.3, 0.3, 34.0],
            [-0.3, 0.3, 40.0],
            [-0.4, 0.3, 50.0],
            [-0.5, 0.2, 55.0],
            [-0.6, 0.2, 75.0],
            [-0.7, 0.3, 86.0]
        ]),
        np.array([
            [0.0, 0.0, 5.0],
            [-0.05, 0.05, 15.0],
            [-0.1, 0.1, 25.0],
            [-0.15, 0.15, 35.0],
            [-0.2, 0.2, 45.0],
            [-0.3, 0.3, 55.0],
            [-0.4, 0.4, 65.0],
            [-0.5, 0.5, 75.0],
            [-0.6, 0.6, 85.0]
        ])
    ]
    
    best_path = initial_paths[0]
    best_hard_score = score_path_hard(best_path)
    
    # Optimization loop
    for init_path in initial_paths:
        if time.time() > deadline - 60.0:
            break
            
        flat_path = init_path.flatten()
        
        try:
            res = minimize(objective, flat_path, method='L-BFGS-B', 
                           options={'maxiter': 100, 'gtol': 1e-5, 'ftol': 1e-7})
            if time.time() > deadline - 30.0:
                continue
                
            candidate_path = res.x.reshape(-1, 3)
            candidate_path[:, 2] = np.clip(candidate_path[:, 2], 0, 90)
            
            # Evaluate hard score
            candidate_score = score_path_hard(candidate_path)
            
            if candidate_score > best_hard_score:
                best_path = candidate_path
                best_hard_score = candidate_score
                
        except Exception:
            pass

    # Local polish with coordinate ascent on the hard score
    if time.time() < deadline - 20.0:
        current_path = best_path.copy()
        current_score = best_hard_score
        n_points = current_path.shape[0]
        
        step_size = 0.05
        improvement_found = True
        max_polish_iter = 30
        polish_iter = 0
        
        while improvement_found and polish_iter < max_polish_iter:
            if time.time() > deadline - 10.0:
                break
            improvement_found = False
            polish_iter += 1
            
            for i in range(n_points):
                # Try x perturbation
                for dx in [-step_size, step_size]:
                    new_path = current_path.copy()
                    new_path[i, 0] += dx
                    s = score_path_hard(new_path)
                    if s > current_score:
                        current_path = new_path
                        current_score = s
                        improvement_found = True
                # Try y perturbation
                for dy in [-step_size, step_size]:
                    new_path = current_path.copy()
                    new_path[i, 1] += dy
                    s = score_path_hard(new_path)
                    if s > current_score:
                        current_path = new_path
                        current_score = s
                        improvement_found = True
                # Try angle perturbation
                for da in [-step_size*5, step_size*5]:
                    new_path = current_path.copy()
                    new_a = current_path[i, 2] + da
                    if 0 <= new_a <= 90:
                        new_path[i, 2] = new_a
                        s = score_path_hard(new_path)
                        if s > current_score:
                            current_path = new_path
                            current_score = s
                            improvement_found = True
            
            # If no improvement, reduce step size
            if not improvement_found:
                step_size *= 0.7
                if step_size < 1e-4:
                    break
            
        if current_score > best_hard_score:
            best_path = current_path
            best_hard_score = current_score

    return best_path

# EVOLVE-BLOCK-END
