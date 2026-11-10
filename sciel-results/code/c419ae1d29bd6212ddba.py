import numpy as np
from scipy.optimize import minimize

def equidistant_points_polygon20() -> np.ndarray:
    n = 20
    
    # Initial guess: regular polygon
    # Turning angles for regular n-gon
    alpha_init = np.full(n - 1, 2.0 * np.pi / n)
    
    def get_polygon_from_angles(alpha):
        # alpha is array of n-1 turning angles
        # The n-th angle is 2*pi - sum(alpha)
        # Ensure convexity: all alpha > 0
        if np.any(alpha <= 0):
            return None, False
        
        alpha_n = 2.0 * np.pi - np.sum(alpha)
        if alpha_n <= 0:
            return None, False
        
        all_alphas = np.concatenate([alpha, [alpha_n]])
        
        # Start at (0,0), direction 0
        # Actually, let's start at (1,0) and turn?
        # Easier: define vertices by cumulative angles.
        # Let vertex 0 be at (0,0)? No, we need to close the loop.
        # Standard way:
        # v_0 = (0,0)
        # v_{i+1} = v_i + L_i * (cos(theta_i), sin(theta_i))
        # But we don't have lengths.
        
        # Alternative: Support function or just optimize coordinates directly with penalty.
        # Coordinate optimization is easier to implement correctly.
        
        return None, False

    # Let's stick to coordinate optimization with a strong convexity penalty.
    # Or use the angle representation properly.
    
    # Angle representation:
    # Vertices $v_i$.
    # Edges $e_i = v_{i+1} - v_i$.
    # Angle of $e_i$ is $\phi_i$.
    # $\phi_{i+1} - \phi_i = \alpha_i$ (turning angle).
    # $\sum \alpha_i = 2\pi$.
    # Lengths $L_i$ are also variables?
    # If we fix angles, the shape is determined up to scale and the lengths must satisfy closure.
    # Closure condition: $\sum L_i e^{i \phi_i} = 0$.
    # This is a system of 2 equations for $L_i$.
    # If we fix $\alpha_i$, the directions $\phi_i$ are fixed (up to rotation).
    # The lengths $L_i$ must be positive.
    # There is a 1D family of solutions for $L_i$?
    # Actually, for a given set of directions, the lengths are determined uniquely (up to scale) if the polygon is convex and closes.
    # Wait, if directions are fixed, the polygon is a "star" of lines.
    # The intersection points are the vertices.
    # So we can construct the polygon from the normal angles?
    
    # Let's use the normal angles (support function angles).
    # $h(\theta)$.
    # This is complex.
    
    # Simplest: Optimize coordinates directly.
    # Initial guess: regular polygon.
    # Objective: -min(vertex_scores) + penalty_if_not_convex.
    
    def compute_scores(polygon):
        # polygon: (20, 2)
        # Normalize
        centroid = np.mean(polygon, axis=0)
        p = polygon - centroid
        dists_to_centroid = np.linalg.norm(p, axis=1)
        mean_dist = np.mean(dists_to_centroid)
        if mean_dist <= 0.01:
            return 0.0
        p = p / mean_dist
        
        # Check convexity
        # Cross product of consecutive edges
        edges = np.roll(p, -1, axis=0) - p
        cross = edges[:, 0] * np.roll(edges, -1, axis=0)[:, 1] - edges[:, 1] * np.roll(edges, -1, axis=0)[:, 0]
        if np.any(cross <= 0):
            return 0.0 # Not strictly convex or self-intersecting
        
        # Check repeated points
        for i in range(n):
            for j in range(i+1, n):
                if np.linalg.norm(p[i] - p[j]) < 1e-5:
                    return 0.0
        
        # Compute vertex scores
        scores = []
        for i in range(n):
            # Distances to other vertices
            d = np.linalg.norm(p[i] - p, axis=1)
            d[i] = np.inf # Exclude self
            d_sorted = np.sort(d)
            
            # Find k
            diffs = d_sorted[1:] - d_sorted[:-1]
            k = np.argmin(diffs)
            
            # Select 4 distances
            if k == 0:
                d_sel = d_sorted[0:4]
            elif k == 17:
                d_sel = d_sorted[15:19]
            else:
                d_sel = d_sorted[k-1:k+3]
            
            D = (d_sorted[k] + d_sorted[k+1]) / 2.0
            if D < 0.1:
                scores.append(0.0)
                continue
                
            # Check min distance
            if d_sorted[0] < 0.05:
                scores.append(0.0)
                continue
                
            raw = 0.0
            for d_val in d_sel:
                ratio = max(d_val / D, D / d_val)
                raw += ratio
            raw /= 4.0
            
            score = min(1.0, 1.0 / raw)
            scores.append(score)
            
        return min(scores)

    def objective(x):
        # x is flattened (20, 2)
        p = x.reshape((n, 2))
        # Soft convexity penalty?
        # Just return -score. If invalid, score is 0, so return 0?
        # We want to maximize score, so minimize -score.
        # If invalid, score is 0. We want to avoid this.
        # Let's add a small penalty for non-convexity to guide the optimizer.
        
        # Check convexity for penalty
        centroid = np.mean(p, axis=0)
        pc = p - centroid
        edges = np.roll(pc, -1, axis=0) - pc
        cross = edges[:, 0] * np.roll(edges, -1, axis=0)[:, 1] - edges[:, 1] * np.roll(edges, -1, axis=0)[:, 0]
        
        # Penalty for negative cross products
        penalty = np.sum(np.maximum(0, -cross)) ** 2
        
        score = compute_scores(p)
        return -score + 0.1 * penalty

    # Initial guess
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    p_init = np.column_stack([np.cos(angles), np.sin(angles)])
    x_init = p_init.flatten()
    
    # Optimize
    res = minimize(objective, x_init, method='Nelder-Mead', options={'maxiter': 1000, 'xatol': 1e-4, 'fatol': 1e-4})
    
    p_opt = res.x.reshape((n, 2))
    
    # Final check and normalization
    score_opt = compute_scores(p_opt)
    
    # If optimization failed or score is bad, return initial
    score_init = compute_scores(p_init)
    
    if score_opt > score_init:
        return p_opt
    else:
        return p_init
