    import numpy as np
    from scipy.optimize import minimize
    import time

    N = 32
    TIME_LIMIT = 140.0

    def _elapsed(start):
        return time.time() - start

    def _is_valid(x, y, r, eps=1e-8):
        if np.any(x < -eps) or np.any(x > 1.0 + eps): return False
        if np.any(y < -eps) or np.any(y > 1.0 + eps): return False
        if np.any(r < -eps): return False
        if np.any(x < r - eps) or np.any(x > 1.0 - r + eps): return False
        if np.any(y < r - eps) or np.any(y > 1.0 - r + eps): return False
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist = np.sqrt(dx*dx + dy*dy)
        need = r[:, None] + r[None, :]
        if np.any(dist < need - eps): return False
        return True

    def _calc_max_radii(x, y, iters=50):
        r = np.minimum(np.minimum(x, 1.0 - x), np.minimum(y, 1.0 - y))
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist = np.sqrt(dx*dx + dy*dy)
        np.fill_diagonal(dist, np.inf)
        for _ in range(iters):
            limits = dist - r[None, :]
            min_limits = np.min(limits, axis=1)
            new_r = np.minimum(r, min_limits)
            new_r = np.maximum(new_r, 0.0)
            if np.allclose(new_r, r, rtol=1e-8, atol=1e-10):
                r = new_r
                break
            r = new_r
        return r

    def _smooth_objective(params, mu=100.0):
        """
        Minimizes -sum(r) + penalty for violations.
        params: [x, y, r] of length 3N
        """
        x = params[:N]
        y = params[N:2*N]
        r = params[2*N:3*N]
        
        # Soft penalty for violations
        # 1. Bounds
        # x - r >= 0 -> viol = max(0, r - x)
        # x + r <= 1 -> viol = max(0, x + r - 1)
        # y - r >= 0 -> viol = max(0, r - y)
        # y + r <= 1 -> viol = max(0, y + r - 1)
        
        # Using smooth approximation for max(0, z) is z/2 + sqrt(z^2 + eps)/2
        # But for L-BFGS-B, standard max(0, z) is C1, which is often enough.
        # Let's use standard hinge loss for simplicity and speed.
        
        viol_x_min = np.maximum(0, r - x)
        viol_x_max = np.maximum(0, x + r - 1)
        viol_y_min = np.maximum(0, r - y)
        viol_y_max = np.maximum(0, y + r - 1)
        
        # Pairwise
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist = np.sqrt(dx*dx + dy*dy)
        need = r[:, None] + r[None, :]
        # dist >= need -> viol = max(0, need - dist)
        viol_pair = np.maximum(0, need - dist)
        
        # Sum of squared violations
        penalty = np.sum(viol_x_min**2) + np.sum(viol_x_max**2) + \
                  np.sum(viol_y_min**2) + np.sum(viol_y_max**2) + \
                  np.sum(viol_pair**2)
        
        return -np.sum(r) + mu * penalty

    def _optimize(x, y, start):
        best_x = x.copy()
        best_y = y.copy()
        best_r = _calc_max_radii(x, y)
        best_score = float(np.sum(best_r))
        
        # Initialize r for joint optimization
        r_init = best_r.copy()
        
        if _elapsed(start) > TIME_LIMIT - 5:
            return best_x, best_y, best_r
            
        try:
            # L-BFGS-B on [x, y, r]
            # Bounds for x, y: [0, 1]
            # Bounds for r: [0, 0.5]
            bounds = []
            for i in range(N):
                bounds.append((0.0, 1.0)) # x
                bounds.append((0.0, 1.0)) # y
                bounds.append((0.0, 0.5)) # r
                
            x0 = np.concatenate([x, y, r_init])
            
            res = minimize(
                _smooth_objective,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-8, 'maxls': 20}
            )
            
            if res.success or res.fun < 0: # Check if it improved
                vx = res.x[:N]
                vy = res.x[N:2*N]
                vr = res.x[2*N:3*N]
                
                # Clean up: ensure validity and recalc max radii
                # The optimizer might leave circles slightly overlapping or out of bounds due to penalty trade-off
                # We enforce strict validity by shrinking radii or adjusting positions
                # Simplest: use the centers from optimizer, recalc max radii
                # But the optimizer also moved radii. 
                # If we just take centers, we lose the radius optimization benefit.
                # However, _calc_max_radii is the ground truth for "max possible radius".
                # So, take centers from optimizer, recalc radii.
                
                # But wait, the joint optimization might find a configuration where radii are not at their max for those centers?
                # No, if it's optimal, it should be.
                # But to be safe, let's take the centers from the optimizer, and recalculate the max radii.
                # This ensures we get the best possible radii for those centers.
                
                vx_clean = np.clip(vx, 0.0, 1.0)
                vy_clean = np.clip(vy, 0.0, 1.0)
                vr_clean = _calc_max_radii(vx_clean, vy_clean)
                
                if _is_valid(vx_clean, vy_clean, vr_clean):
                    s = float(np.sum(vr_clean))
                    if s > best_score:
                        best_score = s
                        best_x = vx_clean
                        best_y = vy_clean
                        best_r = vr_clean
        except Exception:
            pass
            
        return best_x, best_y, best_r

    # ... rest of the file ...
    