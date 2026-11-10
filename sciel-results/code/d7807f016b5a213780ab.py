# EVOLVE-BLOCK-START
import numpy as np

def erdos_szekeres_problem7() -> np.ndarray:
    """
    Returns a (33, 2) array of points in the unit square minimizing the number of convex 7-gons.
    Uses a cup-cap construction with deterministic jitter to ensure general position.
    """
    n = 33
    n_half = 16
    
    # Cup points: strictly convex curve in bottom-left
    x_cup = np.linspace(0.05, 0.45, n_half)
    y_cup = (x_cup - 0.05)**2 * 0.5
    
    # Cap points: strictly concave curve in top-right
    x_cap = np.linspace(0.55, 0.95, n_half)
    y_cap = 1 - (x_cap - 0.95)**2 * 0.5
    
    # Middle point
    x_mid = np.array([0.5])
    y_mid = np.array([0.5])
    
    # Combine
    points = np.vstack([
        np.column_stack([x_cup, y_cup]),
        np.column_stack([x_cap, y_cap]),
        np.column_stack([x_mid, y_mid])
    ])
    
    # Deterministic jitter to avoid collinearity and identical points
    # Using irrational-like frequencies to ensure no simple periodicity
    indices = np.arange(n)
    jitter_x = 0.001 * np.sin(indices * 12.9898 + 1.0)
    jitter_y = 0.001 * np.cos(indices * 78.233 + 2.0)
    
    points[:, 0] += jitter_x
    points[:, 1] += jitter_y
    
    # Ensure points are strictly inside [0, 1]
    points = np.clip(points, 0.01, 0.99)
    
    # Verify validity and fix if necessary (safety net)
    # Check for duplicate points (dist^2 < 1e-12)
    # Check for collinearity (det < 1e-9)
    # If invalid, apply stronger jitter
    for _ in range(10):
        valid = True
        # Distance check
        dists = np.linalg.norm(points[:, np.newaxis] - points[np.newaxis, :], axis=-1)
        np.fill_diagonal(dists, np.inf)
        if np.min(dists) < 1e-6: # 1e-12 sq dist is 1e-6 dist
            valid = False
            
        # Collinearity check (sampled to be fast, but n=33 is small so full check is fine)
        # Full check O(n^3) = 33^3 ~ 35k ops, very fast
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    x1, y1 = points[i]
                    x2, y2 = points[j]
                    x3, y3 = points[k]
                    det = x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2)
                    if abs(det) < 1e-9:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
                
        if valid:
            break
        
        # If invalid, apply more jitter
        points[:, 0] += 0.0001 * np.sin(indices * 1.1 + _ * 0.1)
        points[:, 1] += 0.0001 * np.cos(indices * 2.2 + _ * 0.1)
        points = np.clip(points, 0.01, 0.99)

    return points

# EVOLVE-BLOCK-END
