import numpy as np
from scipy.optimize import minimize
import time

def construct():
    start_time = time.time()
    deadline = 120.0  # 120 seconds
    
    # Initial guess: Regular Octagon
    # Scale to diameter 2
    angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    points = np.vstack([np.cos(angles), np.sin(angles)]).T
    # Diameter of unit circle is 2. So this is already diameter 2.
    
    def neg_log_product(pts):
        # pts is shape (8, 2)
        # Calculate all pairwise distances
        # Use broadcasting
        diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.linalg.norm(diffs, axis=2)
        
        # Avoid log(0)
        dists = np.maximum(dists, 1e-10)
        
        # Sum of logs of all pairs i < j
        # We can sum all and divide by 2, or use triu_indices
        iu = np.triu_indices(8, k=1)
        log_prod = np.sum(np.log(dists[iu]))
        
        # Diameter
        max_dist = np.max(dists)
        
        # Score is proportional to log_prod - 28 * log(max_dist / 2)
        # We want to maximize this, so minimize the negative
        if max_dist < 1e-10:
            return 1e10
        return -(log_prod - 28 * np.log(max_dist / 2.0))

    # We need to optimize the coordinates.
    # To reduce symmetry issues, we can fix p0 = (-1, 0) and p1 = (1, 0) ?
    # No, the diameter might not be horizontal. But we can rotate the final result.
    # However, fixing two points might prevent finding the global optimum if the optimal diameter pair is not fixed in a specific way, or if the optimal configuration doesn't have a pair exactly at (-1,0) and (1,0).
    # But since we can rotate, we can fix p0 = (0,0) and p1 = (d, 0) where d is diameter?
    # Actually, fixing the diameter to 2 and optimizing all 16 coordinates is safe.
    # The objective function is scale invariant, so the optimizer might drift in scale?
    # No, the term -28 log(D/2) penalizes large D. If D increases, log_prod increases linearly with log D (8*7/2 * log D), and penalty is 28 log D.
    # Wait. Product of 28 distances scales as $L^{28}$. Log product scales as $28 \log L$.
    # Diameter scales as $L$. Log diameter scales as $\log L$.
    # Objective: $28 \log L - 28 \log L = 0$.
    # So the objective is indeed scale invariant. The optimizer might struggle with scale drift.
    # To fix this, we can normalize the points to diameter 2 inside the objective function?
    # No, that makes the function non-smooth if normalization is involved.
    # Better: Fix one point to origin, and another to (1,0)? No, diameter is 2.
    # Let's fix p[0] = (-1, 0) and p[1] = (1, 0). This fixes translation and rotation (mostly) and scale.
    # Is it guaranteed that the optimal configuration has a pair of points at distance 2? Yes, by definition of diameter.
    # Is it guaranteed that we can rotate/translate so this pair is at (-1,0) and (1,0)? Yes.
    # So we can fix p[0] and p[1] and optimize the other 6 points (12 variables).
    
    def neg_log_product_fixed(x):
        # x contains coordinates for points 2..7 (12 values)
        pts = np.zeros((8, 2))
        pts[0] = [-1.0, 0.0]
        pts[1] = [1.0, 0.0]
        pts[2:, 0] = x[0::2]
        pts[2:, 1] = x[1::2]
        
        diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.linalg.norm(diffs, axis=2)
        dists = np.maximum(dists, 1e-10)
        
        iu = np.triu_indices(8, k=1)
        log_prod = np.sum(np.log(dists[iu]))
        
        # Diameter is at least 2 (distance between p0 and p1).
        # If other points are farther, diameter > 2.
        # The score formula normalizes by actual diameter.
        # So we must compute actual diameter.
        max_dist = np.max(dists)
        
        if max_dist < 1e-10:
            return 1e10
        
        # We maximize log_prod - 28 log(max_dist/2)
        return -(log_prod - 28 * np.log(max_dist / 2.0))

    # Initial guess for the other 6 points from regular octagon
    # Angles: 0, pi/4, pi/2, 3pi/4, pi, 5pi/4, 3pi/2, 7pi/4
    # p0 is at pi (angle pi). p1 is at 0 (angle 0).
    # Wait, my fixed points are (-1,0) (angle pi) and (1,0) (angle 0).
    # Regular octagon angles:
    # k=0: 0 -> (1,0) -> p1
    # k=1: pi/4 -> (cos pi/4, sin pi/4)
    # k=2: pi/2 -> (0,1)
    # k=3: 3pi/4 -> (-cos pi/4, sin pi/4)
    # k=4: pi -> (-1,0) -> p0
    # k=5: 5pi/4 -> (-cos pi/4, -sin pi/4)
    # k=6: 3pi/2 -> (0,-1)
    # k=7: 7pi/4 -> (cos pi/4, -sin pi/4)
    
    x0 = np.array([
        np.cos(np.pi/4), np.sin(np.pi/4),   # p2
        0, 1,                                 # p3
        -np.cos(np.pi/4), np.sin(np.pi/4),   # p4
        -np.cos(np.pi/4), -np.sin(np.pi/4),  # p5
        0, -1                                 # p6
        # Wait, I need 6 points. p2, p3, p4, p5, p6, p7.
        # My list above has 6 points.
    ])
    # Let's re-verify the mapping.
    # pts[0] = p0 (fixed)
    # pts[1] = p1 (fixed)
    # pts[2] = p2
    # pts[3] = p3
    # pts[4] = p4
    # pts[5] = p5
    # pts[6] = p6
    # pts[7] = p7
    
    # Regular octagon points:
    # p0: (-1, 0)
    # p1: (1, 0)
    # p2: (cos(pi/4), sin(pi/4))
    # p3: (0, 1)
    # p4: (-cos(pi/4), sin(pi/4))
    # p5: (-cos(pi/4), -sin(pi/4))
    # p6: (0, -1)
    # p7: (cos(pi/4), -sin(pi/4))
    
    x0 = np.array([
        np.cos(np.pi/4), np.sin(np.pi/4),
        0, 1,
        -np.cos(np.pi/4), np.sin(np.pi/4),
        -np.cos(np.pi/4), -np.sin(np.pi/4),
        0, -1,
        np.cos(np.pi/4), -np.sin(np.pi/4)
    ])
    
    best_score = -1e10
    best_x = x0.copy()
    
    # Try a few random restarts or just one good optimization
    # Since we have time, let's do a few.
    
    for i in range(10):
        if time.time() > deadline - 10:
            break
            
        if i == 0:
            x_start = x0
        else:
            # Add noise to x0
            x_start = x0 + np.random.normal(0, 0.1, size=12)
            
        try:
            res = minimize(neg_log_product_fixed, x_start, method='L-BFGS-B', options={'maxiter': 2000, 'ftol': 1e-12})
            if res.success or res.fun < best_score:
                # Check if this is better
                if res.fun < best_score:
                    best_score = res.fun
                    best_x = res.x
        except Exception as e:
            pass
            
    # Reconstruct points
    pts = np.zeros((8, 2))
    pts[0] = [-1.0, 0.0]
    pts[1] = [1.0, 0.0]
    pts[2:, 0] = best_x[0::2]
    pts[2:, 1] = best_x[1::2]
    
    # Final sanity check: ensure distinct points and coordinates < 1e6
    # The optimization should keep them bounded.
    
    # Return as list of lists
    return pts.tolist()
