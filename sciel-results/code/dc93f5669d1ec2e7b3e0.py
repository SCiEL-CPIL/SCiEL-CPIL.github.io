import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin before 150s kill
    n = 8
    
    # Precompute mask for off-diagonal elements
    mask = ~np.eye(n, dtype=bool)
    
    def compute_score_log(points, beta):
        """
        Computes the log of the score: sum(log(dist)) - 8 * log(diameter).
        Uses soft-max for diameter if beta < 1e6.
        """
        # points: (8, 2)
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        dists = np.sqrt(dists_sq)
        
        off_diag_dists = dists[mask]
        
        if len(off_diag_dists) == 0:
            return -np.inf

        # Compute diameter
        if beta < 1e6:
            # Smooth soft-max approximation of max
            # log(sum(exp(dists/beta))) * beta
            # To avoid overflow: log(sum(exp(dists - max(dists)))) + max(dists)
            m = np.max(off_diag_dists)
            if m < 1e-10:
                return -np.inf
            log_sum_exp = m + np.log(np.sum(np.exp((off_diag_dists - m) / beta)))
            diameter = beta * (log_sum_exp / beta) # This is just log_sum_exp * beta? No.
            # Formula: max(x) approx (1/beta) * log(sum(exp(beta * x)))
            # Let y = x * beta. max(x) = (1/beta) * log(sum(exp(y)))
            # So diameter = (1/beta) * (m*beta + log(sum(exp(off_diag_dists - m))))
            # diameter = m + (1/beta) * log(sum(exp((off_diag_dists - m))))
            diameter = m + (1.0 / beta) * np.log(np.sum(np.exp((off_diag_dists - m))))
        else:
            diameter = np.max(off_diag_dists)
            
        if diameter < 1e-10:
            return -np.inf

        # Scale distances to diameter 2
        # d_scaled = d * (2 / diameter)
        # log(d_scaled) = log(d) + log(2) - log(diameter)
        # Sum over 56 pairs:
        # sum(log(d)) + 56*log(2) - 56*log(diameter)
        
        # The score definition in prompt: Delta / n^n
        # Delta = product of ordered pairs.
        # log(Delta) = sum(log(d_ij)) for i!=j.
        # We want to maximize Delta_scaled / 8^8 ?
        # Prompt: "score is Delta/n^n ... after scaling set to diameter 2"
        # So Score = (Product of scaled dists) / 8^8.
        # log(Score) = sum(log(d_scaled)) - 8*log(8)
        # sum(log(d_scaled)) = sum(log(d) * 2/D) = sum(log(d)) + 56*log(2) - 56*log(D)
        
        # Wait, the prompt says "regular octagon scores exactly 1".
        # Let's check my previous derivation.
        # Regular Octagon: Product of ordered dists = 8^8.
        # Diameter = 2.
        # Scaled dists are the same.
        # Score = 8^8 / 8^8 = 1.
        # So we just need to maximize sum(log(d_scaled)).
        # sum(log(d_scaled)) = sum(log(d)) - 56 * log(D/2)
        
        log_dists_sum = np.sum(np.log(off_diag_dists))
        # log(D/2) = log(D) - log(2)
        log_score_unnormalized = log_dists_sum - 56.0 * (np.log(diameter) - np.log(2.0))
        
        # The actual score is exp(log_score_unnormalized) / 8^8
        # We want to maximize this, so maximize log_score_unnormalized - 8*log(8)
        # Since 8*log(8) is constant, we just maximize log_score_unnormalized.
        
        return log_score_unnormalized

    def objective(x, beta):
        points = x.reshape(n, 2)
        val = compute_score_log(points, beta)
        if np.isnan(val) or np.isinf(val):
            return 1e100
        return -val

    # Initializations
    starts = []
    
    # 1. Regular Octagon
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    p_oct = np.column_stack((np.cos(angles), np.sin(angles)))
    starts.append(p_oct.flatten())
    
    # 2. Two Squares with various rotations
    # Square 1: (1,0), (0,1), (-1,0), (0,-1)
    # Square 2: rotated by theta
    for theta in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        a1 = np.array([0, np.pi/2, np.pi, 3*np.pi/2])
        a2 = a1 + theta
        sq1 = np.column_stack((np.cos(a1), np.sin(a1)))
        sq2 = np.column_stack((np.cos(a2), np.sin(a2)))
        p_two_sq = np.vstack([sq1, sq2])
        starts.append(p_two_sq.flatten())
        
    # 3. Antipodal Pairs
    # 4 pairs of antipodal points.
    # (cos(a), sin(a)), (-cos(a), -sin(a))
    for seed in [1, 2, 3]:
        np.random.seed(seed)
        base_angles = np.sort(np.random.uniform(0, np.pi, 4))
        all_angles = np.concatenate([base_angles, base_angles + np.pi])
        pts = np.column_stack((np.cos(all_angles), np.sin(all_angles)))
        starts.append(pts.flatten())
        
    # 4. Random Points
    np.random.seed(42)
    for _ in range(10):
        pts = np.random.uniform(-1, 1, (n, 2))
        starts.append(pts.flatten())
        
    # 5. Perturbed Octagon
    for _ in range(5):
        pts = p_oct + np.random.normal(0, 0.1, p_oct.shape)
        starts.append(pts.flatten())

    best_score = -np.inf
    best_points = p_oct
    
    # Phase 1: Coarse Optimization (Low Beta)
    beta_coarse = 20.0
    for s in starts:
        if time.time() > deadline - 40:
            break
        try:
            res = minimize(
                lambda x: objective(x, beta_coarse),
                s,
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-6}
            )
            if res.fun < 1e90:
                score = -res.fun
                if score > best_score:
                    best_score = score
                    best_points = res.x.reshape(n, 2)
        except Exception:
            pass

    # Phase 2: Refinement (Medium Beta)
    if time.time() < deadline - 20:
        beta_med = 100.0
        try:
            res = minimize(
                lambda x: objective(x, beta_med),
                best_points.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            if res.fun < 1e90:
                score = -res.fun
                if score > best_score:
                    best_score = score
                    best_points = res.x.reshape(n, 2)
        except Exception:
            pass

    # Phase 3: High Precision (High Beta)
    if time.time() < deadline - 10:
        beta_high = 500.0
        try:
            res = minimize(
                lambda x: objective(x, beta_high),
                best_points.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            if res.fun < 1e90:
                score = -res.fun
                if score > best_score:
                    best_score = score
                    best_points = res.x.reshape(n, 2)
        except Exception:
            pass
            
    # Phase 4: Final Polish (Exact Max)
    if time.time() < deadline - 2:
        # Use Nelder-Mead for robustness on the non-smooth exact objective
        try:
            res = minimize(
                lambda x: objective(x, 1e8),
                best_points.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-10}
            )
            if res.fun < 1e90:
                score = -res.fun
                if score > best_score:
                    best_score = score
                    best_points = res.x.reshape(n, 2)
        except Exception:
            pass

    # Validation
    dists = np.sqrt(np.sum((best_points[:, np.newaxis, :] - best_points[np.newaxis, :, :])**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    if np.min(dists) < 1e-10:
        best_points = p_oct

    # Scale to ensure coordinates are reasonable
    max_coord = np.max(np.abs(best_points))
    if max_coord > 1e6:
        best_points = best_points / max_coord * 1e5

    return [[float(p[0]), float(p[1])] for p in best_points]
