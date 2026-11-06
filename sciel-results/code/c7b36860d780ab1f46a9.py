import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin

    def get_points(r1, r2, theta):
        # Square 1: angles 0, pi/2, pi, 3pi/2
        angles1 = np.array([0, np.pi/2, np.pi, 3*np.pi/2])
        pts1 = np.column_stack([r1 * np.cos(angles1), r1 * np.sin(angles1)])
        
        # Square 2: angles theta, theta+pi/2, theta+pi, theta+3pi/2
        angles2 = angles1 + theta
        pts2 = np.column_stack([r2 * np.cos(angles2), r2 * np.sin(angles2)])
        
        return np.vstack([pts1, pts2])

    def compute_score(pts):
        # pts is 8x2
        # Calculate pairwise distances
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        
        # Avoid log(0) for self-distances, though we only care about i != j
        # We need product over ordered pairs? No, "product of distances". 
        # Usually this means unordered pairs i<j. 
        # The task says "product over ordered pairs of distances". 
        # Wait, "ordered pairs" usually implies i!=j. 
        # Let's check the score definition: Delta/n^n.
        # For regular octagon, score is 1.
        # If we take product over i<j, there are 28 pairs.
        # If we take product over i!=j, there are 56 pairs.
        # Let's assume standard "Vandermonde-like" product over i<j.
        # Let's verify with regular octagon.
        # Regular octagon side length s, diameter 2.
        # R=1. s = 2*sin(pi/8).
        # Distances are s, s*sqrt(2), 2.
        # Let's just compute the score and see if it's 1.
        
        # Upper triangle indices
        i, j = np.triu_indices(8, k=1)
        d = dists[i, j]
        
        # Log product
        log_prod = np.sum(np.log(d))
        
        # Diameter
        D = np.max(dists)
        
        # Scale factor to make diameter 2
        if D == 0:
            return -np.inf
        scale = 2.0 / D
        
        # The product scales by scale^28
        log_scaled_prod = log_prod + 28 * np.log(scale)
        
        # Score = exp(log_scaled_prod) / 8^8
        # We want to maximize this.
        # log_score = log_scaled_prod - 8*log(8)
        log_score = log_scaled_prod - 8 * np.log(8)
        
        return np.exp(log_score)

    def objective(params):
        r1, r2, theta = params
        if r1 <= 0 or r2 <= 0:
            return 0.0
        pts = get_points(r1, r2, theta)
        s = compute_score(pts)
        return -np.log(s) if s > 0 else 1e10

    # Grid search
    best_score = 0.0
    best_params = (1.0, 1.0, np.pi/4)
    
    # Ranges
    r_range = np.linspace(0.5, 1.5, 20)
    theta_range = np.linspace(0.0, np.pi/4, 20)
    
    for r1 in r_range:
        for r2 in r_range:
            for theta in theta_range:
                if time.time() > deadline:
                    break
                pts = get_points(r1, r2, theta)
                s = compute_score(pts)
                if s > best_score:
                    best_score = s
                    best_params = (r1, r2, theta)
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break

    # Local refinement
    if time.time() < deadline:
        # Try refining from the best grid point
        try:
            res = minimize(objective, best_params, method='Nelder-Mead', 
                           options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8})
            if res.fun < -np.log(best_score): # res.fun is -log(score)
                best_params = res.x
                pts = get_points(*best_params)
                best_score = compute_score(pts)
        except Exception:
            pass
        
        # Try a few random starts for robustness
        np.random.seed(42)
        for _ in range(5):
            if time.time() > deadline:
                break
            r1 = np.random.uniform(0.5, 1.5)
            r2 = np.random.uniform(0.5, 1.5)
            theta = np.random.uniform(0.0, np.pi/4)
            try:
                res = minimize(objective, [r1, r2, theta], method='Nelder-Mead', 
                               options={'maxiter': 100, 'xatol': 1e-6, 'fatol': 1e-6})
                if res.fun < -np.log(best_score):
                    best_params = res.x
                    pts = get_points(*best_params)
                    best_score = compute_score(pts)
            except Exception:
                pass

    # Final result
    r1, r2, theta = best_params
    pts = get_points(r1, r2, theta)
    
    # Scale to diameter 2
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    D = np.max(dists)
    if D > 0:
        pts = pts * (2.0 / D)
        
    return [[float(p[0]), float(p[1])] for p in pts]

if __name__ == "__main__":
    # Quick test
    pts = construct()
    # print(pts)
    pass
