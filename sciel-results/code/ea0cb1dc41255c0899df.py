import time
import numpy as np
from scipy.optimize import differential_evolution, minimize

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s kill

    def score(pts):
        # pts: (8, 2) array
        # Compute all pairwise distances
        n = len(pts)
        d = np.linalg.norm(pts[:, np.newaxis, :] - pts[np.newaxis, :, :], axis=2)
        # Diameter is max distance
        diam = np.max(d)
        if diam == 0:
            return 0.0
        # Scale to diameter 2
        # Product of distances over ordered pairs (i != j)
        # The score is Delta / n^n where Delta is product of distances after scaling to diameter 2
        # Note: "ordered pairs" usually means i!=j, so n*(n-1) terms.
        # Regular octagon scores 1. Let's verify the normalization.
        # If diam=2, regular octagon side s = 2*sin(pi/8).
        # Distances in regular octagon: 8 sides (s), 8 diagonals (2s), 4 diameters (2).
        # Product P = s^8 * (2s)^8 * 2^4 = 2^8 * s^16 * 16 = 2^12 * s^16.
        # Score = P / 8^8.
        # s = 2 sin(pi/8). s^16 = 2^16 sin^16(pi/8).
        # P = 2^12 * 2^16 * sin^16(pi/8) = 2^28 * sin^16(pi/8).
        # 8^8 = 2^24.
        # Score = 2^4 * sin^16(pi/8) = 16 * sin^16(pi/8).
        # sin(pi/8) = sqrt(2-sqrt(2))/2. sin^2 = (2-sqrt(2))/4.
        # sin^16 = ((2-sqrt(2))/4)^8.
        # 16 * ((2-sqrt(2))/4)^8 = 16 * (2-sqrt(2))^8 / 4^8 = (2-sqrt(2))^8 / 16.
        # (2-sqrt(2))^2 = 6-4sqrt(2). (2-sqrt(2))^4 = (6-4sqrt(2))^2 = 36 - 48sqrt(2) + 32 = 68 - 48sqrt(2).
        # (2-sqrt(2))^8 = (68-48sqrt(2))^2 = 4624 - 6528sqrt(2) + 4608 = 9232 - 6528sqrt(2).
        # Score = (9232 - 6528sqrt(2)) / 16 = 577 - 408sqrt(2) ≈ 577 - 577.00... = 1.
        # Yes, it works out to exactly 1.
        
        # So we just need to maximize: (Product of all pairwise distances) / (Diameter^112)
        # Because there are 8*7 = 56 ordered pairs. Wait.
        # "product over ordered pairs of distances".
        # Number of ordered pairs i!=j is 8*7 = 56.
        # If we scale points by 1/diam, distances scale by 1/diam.
        # Product scales by (1/diam)^56.
        # So Delta_scaled = Delta_raw / diam^56.
        # Score = Delta_scaled / 8^8.
        
        # To avoid underflow/overflow, work in log space.
        # log(Score) = sum(log(d_ij)) - 56*log(diam) - 8*log(8)
        
        mask = ~np.eye(n, dtype=bool)
        dists = d[mask]
        log_prod = np.sum(np.log(dists))
        log_diam = np.log(diam)
        log_score = log_prod - 56 * log_diam - 8 * np.log(8)
        return np.exp(log_score)

    def neg_log_score(angles):
        # angles: (8,)
        if time.time() > deadline:
            return 0.0
        pts = np.column_stack([np.cos(angles), np.sin(angles)])
        # Centering: The optimal set is likely centered. 
        # Actually, for this specific objective (product of distances / diameter^k), 
        # translation invariance is broken by the diameter scaling? No, diameter is translation invariant.
        # So we can fix the center of mass at (0,0) or just fix one point to break rotational symmetry.
        # Fixing angles on unit circle:
        # We can fix angles[0] = 0 to break rotational symmetry.
        # Then optimize angles[1:].
        a = np.zeros(8)
        a[0] = 0
        a[1:] = angles
        # Sort angles to ensure consistent ordering? Not strictly necessary for product, but good for DE.
        a = np.sort(a)
        pts = np.column_stack([np.cos(a), np.sin(a)])
        
        n = len(pts)
        d = np.linalg.norm(pts[:, np.newaxis, :] - pts[np.newaxis, :, :], axis=2)
        diam = np.max(d)
        if diam == 0:
            return 0.0
        mask = ~np.eye(n, dtype=bool)
        dists = d[mask]
        # Add small epsilon to avoid log(0) if points coincide
        dists = np.maximum(dists, 1e-12)
        log_prod = np.sum(np.log(dists))
        log_diam = np.log(diam)
        log_score = log_prod - 56 * log_diam - 8 * np.log(8)
        return -log_score

    # Initial guess: Regular octagon
    # Angles: 0, pi/4, pi/2, 3pi/4, pi, 5pi/4, 3pi/2, 7pi/4
    # Fixed a[0]=0, so angles to optimize are the other 7.
    x0 = np.array([np.pi/4, np.pi/2, 3*np.pi/4, np.pi, 5*np.pi/4, 3*np.pi/2, 7*np.pi/4])
    
    # Bounds for angles: (0, 2*pi) roughly, but since we sort and fix one, 
    # it's better to think of them as relative positions.
    # Let's just optimize 7 angles in [0, 2*pi].
    bounds = [(0, 2*np.pi) for _ in range(7)]
    
    best_val = float('inf')
    best_angles = x0
    
    # Use Differential Evolution for global search first
    # Then Local Refinement
    
    try:
        # DE is stochastic and can be slow, but robust.
        # Use small population for speed.
        res = differential_evolution(
            neg_log_score, 
            bounds, 
            seed=42, 
            maxiter=50, 
            popsize=5, 
            tol=1e-8,
            polish=True
        )
        if res.fun < best_val:
            best_val = res.fun
            best_angles = res.x
    except Exception:
        pass
    
    if time.time() > deadline:
        # Return best found so far
        a = np.zeros(8)
        a[0] = 0
        a[1:] = best_angles
        a = np.sort(a)
        pts = np.column_stack([np.cos(a), np.sin(a)])
        return pts.tolist()

    # Local refinement using L-BFGS-B or Nelder-Mead on the best DE result
    try:
        res2 = minimize(
            neg_log_score, 
            best_angles, 
            method='L-BFGS-B', 
            bounds=bounds,
            options={'maxiter': 1000}
        )
        if res2.fun < best_val:
            best_val = res2.fun
            best_angles = res2.x
    except Exception:
        pass
    
    # Final check and return
    a = np.zeros(8)
    a[0] = 0
    a[1:] = best_angles
    a = np.sort(a)
    pts = np.column_stack([np.cos(a), np.sin(a)])
    
    # Verify score
    # s = score(pts)
    # print(s)
    
    return pts.tolist()

# For testing
if __name__ == "__main__":
    pts = construct()
    print(pts)
