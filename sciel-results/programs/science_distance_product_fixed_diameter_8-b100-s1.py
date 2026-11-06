import numpy as np
from scipy.optimize import minimize
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    n = 8
    best_score = -np.inf
    best_pts = None
    
    # Objective: Minimize -sum(log(d_ij)) subject to d_ij <= 1
    # We will scale the result to diameter 2 at the end.
    
    def objective(x):
        pts = x.reshape(n, 2)
        # Calculate pairwise distances squared
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        # Mask diagonal
        mask = ~np.eye(n, dtype=bool)
        d_sq = dists_sq[mask]
        
        # Avoid log(0)
        if np.any(d_sq < 1e-12):
            return 1e10
        
        # Sum of log distances
        # log(d) = 0.5 * log(d^2)
        log_sum = 0.5 * np.sum(np.log(d_sq))
        return -log_sum

    def constraint(x):
        pts = x.reshape(n, 2)
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        mask = ~np.eye(n, dtype=bool)
        d_sq = dists_sq[mask]
        # Constraint: 1 - d^2 >= 0
        return 1 - d_sq

    # Generate initial guesses
    guesses = []
    
    # 1. Regular octagon (scaled to diameter 1)
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    p_reg = np.column_stack([np.cos(angles), np.sin(angles)])
    # Diameter of unit circle octagon is 2. Scale to 1.
    p_reg = p_reg * 0.5
    guesses.append(p_reg.flatten())
    
    # 2. Perturbed regular octagon
    p_pert = p_reg + 0.05 * np.random.randn(n, 2)
    guesses.append(p_pert.flatten())
    
    # 3. Random points in unit disk
    for _ in range(5):
        r = np.random.uniform(0, 1, n)
        theta = np.random.uniform(0, 2*np.pi, n)
        p_rand = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
        guesses.append(p_rand.flatten())
        
    # 4. Two squares (4 points on outer square, 4 on inner?)
    # Just a heuristic
    p_sq1 = np.array([[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5], 
                      [-0.2, -0.2], [0.2, -0.2], [0.2, 0.2], [-0.2, 0.2]])
    guesses.append(p_sq1.flatten())

    constraints = [{'type': 'ineq', 'fun': constraint}]
    
    for x0 in guesses:
        if time.time() > deadline:
            break
            
        try:
            res = minimize(objective, x0, method='SLSQP', constraints=constraints,
                           options={'maxiter': 1000, 'ftol': 1e-12})
            
            if res.success or res.fun < best_score:
                # Calculate score for this result
                pts = res.x.reshape(n, 2)
                # Check validity
                diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
                dists = np.sqrt(np.sum(diff**2, axis=2))
                mask = ~np.eye(n, dtype=bool)
                d = dists[mask]
                
                if np.any(d < 1e-9):
                    continue
                    
                diam = np.max(d)
                if diam < 1e-9:
                    continue
                
                # Scale to diameter 2
                pts_scaled = pts * (2.0 / diam)
                
                # Calculate final score
                diff_s = pts_scaled[:, np.newaxis, :] - pts_scaled[np.newaxis, :, :]
                dists_s = np.sqrt(np.sum(diff_s**2, axis=2))
                d_s = dists_s[mask]
                
                # Product of ordered pairs
                # log(Delta) = sum(log(d_ij)) for i != j
                # sum over all i != j is 2 * sum over i < j
                log_delta = np.sum(np.log(d_s))
                delta = np.exp(log_delta)
                score = delta / (n ** n)
                
                if score > best_score:
                    best_score = score
                    best_pts = pts_scaled
        except Exception:
            pass

    if best_pts is None:
        # Fallback
        angles = np.linspace(0, 2*np.pi, n, endpoint=False)
        p_reg = np.column_stack([np.cos(angles), np.sin(angles)])
        return p_reg.tolist()
        
    return best_pts.tolist()
