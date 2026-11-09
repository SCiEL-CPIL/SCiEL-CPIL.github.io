import numpy as np
from scipy.optimize import minimize

def construct():
    n = 8
    
    def angle_objective(angles):
        # angles: array of n floats
        # Calculate sum of log distances
        log_sum = 0.0
        for i in range(n):
            for j in range(i+1, n):
                diff = angles[i] - angles[j]
                # Distance on unit circle: 2 * |sin(diff/2)|
                # If diff is close to 0 or 2*pi, sin is 0, log is -inf.
                # We add a small epsilon to avoid -inf, which acts as a penalty for coincident points.
                val = np.sin(diff / 2.0)
                # Ensure positive for log
                if abs(val) < 1e-12:
                    return 1e10 # Penalty
                log_sum += np.log(abs(val))
        return -log_sum # Minimize negative log sum

    best_score = -np.inf
    best_angles = None
    
    # Multiple random restarts
    np.random.seed(42)
    for restart in range(20):
        # Start near regular octagon but with random perturbation to break symmetry
        base_angles = np.linspace(0, 2*np.pi, n, endpoint=False)
        noise = np.random.normal(0, 0.5, n)
        angles0 = base_angles + noise
        
        try:
            res = minimize(
                angle_objective, 
                angles0, 
                method='Nelder-Mead',
                options={'maxiter': 10000, 'xatol': 1e-10, 'fatol': 1e-10}
            )
            if res.fun < 1e9:
                # Calculate the actual score (product of distances)
                # The objective was -sum(log(sin(diff/2))).
                # Score is proportional to exp(sum(log(sin))) = prod(sin).
                # We need to compare this to the regular octagon score.
                # Regular octagon score is 1.0 in the problem's normalized scale.
                # Let's calculate the raw product of distances for the found angles.
                pts = np.column_stack((np.cos(res.x), np.sin(res.x)))
                # Calculate pairwise distances
                diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
                dists = np.linalg.norm(diff, axis=2)
                mask = np.triu(np.ones((n, n), dtype=bool), k=1)
                valid_dists = dists[mask]
                # Product of distances
                prod_dists = np.prod(valid_dists)
                
                # The problem score is Delta / n^n after scaling to diameter 2.
                # Here, diameter is 2 (points on unit circle).
                # So the value is just prod_dists / n^n ?
                # Wait, the problem says "regular octagon scores exactly 1".
                # Let's check the regular octagon value.
                # Distances in regular octagon (diam 2):
                # Side: 2*sin(pi/8)
                # Diag1: 2*sin(2pi/8)
                # Diag2: 2*sin(3pi/8)
                # Diag3: 2 (diameter)
                # Counts:
                # 8 sides, 8 diag1, 8 diag2, 4 diameters.
                # Product = (2*sin(pi/8))^8 * (2*sin(pi/4))^8 * (2*sin(3pi/8))^8 * (2)^4
                # Let's just compute it numerically to be sure.
                
                # We want to maximize the score.
                # Score = prod_dists / (n**n) ?
                # The problem statement says "Delta/n^n".
                # Let's assume the evaluator computes this.
                # We just need to maximize the product of distances.
                
                if prod_dists > best_score:
                    best_score = prod_dists
                    best_angles = res.x
        except Exception:
            pass

    if best_angles is None:
        # Fallback to regular octagon
        best_angles = np.linspace(0, 2*np.pi, n, endpoint=False)

    # Convert to points
    final_pts = np.column_stack((np.cos(best_angles), np.sin(best_angles)))
    
    # Scale to diameter 2 (already on unit circle, so diameter is 2)
    # Just in case, normalize
    diff = final_pts[:, np.newaxis, :] - final_pts[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    valid_dists = dists[mask]
    current_diam = np.max(valid_dists)
    if current_diam > 1e-12:
        final_pts = final_pts * (2.0 / current_diam)

    return final_pts.tolist()
