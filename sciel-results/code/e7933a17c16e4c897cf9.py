import numpy as np
from scipy.optimize import minimize
import time

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin
    
    n = 8
    
    def compute_score_and_grad(points):
        """
        Compute the score Delta/n^n and its gradient for a set of points.
        Delta is the product of distances for all ordered pairs (i, j) with i != j.
        The set is scaled to have diameter 2.
        
        Score S = Delta / n^n.
        Since Delta is homogeneous of degree n(n-1), and we scale to diameter D=2,
        let d_ij be the unscaled distances.
        Scaled distance d'_ij = d_ij * (2/D).
        Delta' = prod_{i!=j} d'_ij = Delta * (2/D)^(n(n-1)).
        S = Delta' / n^n.
        
        We want to maximize S.
        Note that S is scale invariant. 
        However, for optimization, we can work with unscaled points and incorporate the scaling.
        Let L = log(S) = log(Delta) - n(n-1) log(D) + n(n-1) log(2) - n log(n).
        Maximizing S is equivalent to maximizing L.
        
        L = sum_{i!=j} log(d_ij) - n(n-1) log(D) - n log(n) + const
        
        Gradient of L w.r.t points P:
        dL/dP = sum_{i!=j} (1/d_ij) * d(d_ij)/dP - n(n-1) * (1/D) * d(D)/dP
        
        d(d_ij)/dP_k = 0 if k != i, j.
        d(d_ij)/dP_i = (P_i - P_j) / d_ij
        d(d_ij)/dP_j = (P_j - P_i) / d_ij
        
        So sum_{i!=j} (1/d_ij) * d(d_ij)/dP_k = sum_{j!=k} (P_k - P_j)/d_kj^2
        
        d(D)/dP_k: D is the max distance. Let (p, q) be the pair achieving max distance.
        d(D)/dP_p = (P_p - P_q)/D
        d(D)/dP_q = (P_q - P_p)/D
        
        This gradient is discontinuous when the identity of the diameter pair changes.
        L-BFGS-B can handle this reasonably well if we are careful, or we can use a smooth approximation of max.
        However, for n=8, the set of diameter pairs is small. 
        Actually, a better approach for the objective function in optimization is to just return -S and let the optimizer handle it, 
        but gradient-based methods need the gradient.
        
        Alternative: Use a smooth surrogate for the diameter?
        Or just use the exact gradient and hope the optimizer navigates the kinks.
        Given the time limit, let's try L-BFGS-B with the exact gradient.
        """
        # Compute distances
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        
        # Mask for i != j
        mask = ~np.eye(n, dtype=bool)
        
        # Current Diameter
        max_dist = np.max(dists[mask])
        if max_dist < 1e-10:
            return 0.0, np.zeros_like(points)
        
        # Log Score
        # Sum log(d_ij) for i != j
        # Avoid log(0)
        dists_nonzero = dists[mask]
        log_prod = np.sum(np.log(dists_nonzero))
        
        # Term: -n(n-1) log(D)
        log_score = log_prod - n*(n-1)*np.log(max_dist) - n*np.log(n) + n*(n-1)*np.log(2)
        
        # Gradient
        grad = np.zeros_like(points)
        
        # Part 1: sum_{j!=k} (P_k - P_j)/d_kj^2
        # dists_sq = dists**2
        # inv_dists_sq = np.where(dists > 0, 1.0/dists**2, 0.0)
        # term1 = np.sum(inv_dists_sq * diff, axis=1)
        # Note: diff[k,j] = P_k - P_j.
        # The sum is over j != k.
        # np.sum over axis=1 sums over j.
        # We need to ensure we don't divide by 0 for k=j.
        # diff[k,k] = 0, so 0 * inf is nan.
        # Set diagonal of inv_dists_sq to 0.
        dists_sq = dists**2
        inv_dists_sq = np.zeros_like(dists_sq)
        mask_inv = dists_sq > 1e-12
        inv_dists_sq[mask_inv] = 1.0 / dists_sq[mask_inv]
        
        # grad[k] = sum_j inv_dists_sq[k,j] * (P_k - P_j)
        grad = np.sum(inv_dists_sq * diff, axis=1)
        
        # Part 2: -n(n-1) * (1/D) * d(D)/dP
        # Find pairs with max distance
        # Due to floating point, there might be multiple. We take the first one or average?
        # The gradient of max is not unique. We pick one.
        max_idx = np.unravel_index(np.argmax(dists), dists.shape)
        p, q = max_idx
        
        if p != q:
            # d(D)/dP_p = (P_p - P_q)/D
            # d(D)/dP_q = (P_q - P_p)/D
            # Term to subtract: n(n-1) * (1/D) * d(D)/dP
            factor = n*(n-1) / max_dist
            vec = points[p] - points[q]
            grad[p] -= factor * vec
            grad[q] -= factor * (-vec) # -vec is P_q - P_p
        else:
            # Degenerate case, shouldn't happen with distinct points
            pass
            
        return log_score, grad

    def objective(x):
        """
        Objective function for scipy minimize.
        Returns -log_score (to minimize).
        """
        points = x.reshape(n, 2)
        score, grad = compute_score_and_grad(points)
        if not np.isfinite(score):
            return 1e10, np.zeros_like(x)
        return -score, -grad.reshape(n*2)

    # Initial guesses
    best_score = -np.inf
    best_points = None
    
    # 1. Regular Octagon
    angles = 2 * np.pi * np.arange(n) / n
    p_oct = np.column_stack([np.cos(angles), np.sin(angles)])
    score_oct, _ = compute_score_and_grad(p_oct)
    best_score = score_oct
    best_points = p_oct.copy()
    
    # 2. Random starts
    np.random.seed(42)
    
    # Try a few different types of starts
    starts = []
    
    # Random points in unit square
    for i in range(5):
        starts.append(np.random.uniform(-1, 1, (n, 2)))
        
    # Two concentric squares (D4 symmetry)
    # Square 1: radius r1, Square 2: radius r2, rotated 45 deg
    # Parametrize by r1, r2
    for r1 in [0.5, 0.8, 1.0, 1.2]:
        for r2 in [0.3, 0.6, 0.9, 1.1]:
            pts = []
            for j in range(4):
                ang1 = 2*np.pi*j/4
                ang2 = 2*np.pi*(j+0.5)/4
                pts.append([r1*np.cos(ang1), r1*np.sin(ang1)])
                pts.append([r2*np.cos(ang2), r2*np.sin(ang2)])
            starts.append(np.array(pts))
            
    # Random cyclic polygons (varying radii or angles)
    for i in range(5):
        # Random angles
        angs = np.sort(np.random.uniform(0, 2*np.pi, n))
        # Random radius 1
        starts.append(np.column_stack([np.cos(angs), np.sin(angs)]))
        
    # Optimize from each start
    for i, x0 in enumerate(starts):
        if time.time() > deadline - 10:
            break
            
        try:
            # Use L-BFGS-B
            res = minimize(objective, x0.flatten(), method='L-BFGS-B', jac=True,
                           options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8})
            
            if res.success or res.fun < -best_score:
                pts = res.x.reshape(n, 2)
                score, _ = compute_score_and_grad(pts)
                if score > best_score:
                    best_score = score
                    best_points = pts.copy()
        except Exception:
            pass

    # If we have time, do a second round with more random starts or finer tuning
    if time.time() < deadline - 20 and best_score < 1.2: # If we haven't found a great solution
         for i in range(10):
             if time.time() > deadline - 5:
                 break
             x0 = np.random.uniform(-1.5, 1.5, (n, 2))
             try:
                 res = minimize(objective, x0.flatten(), method='L-BFGS-B', jac=True,
                                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8})
                 if res.fun < -best_score:
                     pts = res.x.reshape(n, 2)
                     score, _ = compute_score_and_grad(pts)
                     if score > best_score:
                         best_score = score
                         best_points = pts.copy()
             except Exception:
                 pass

    # Final validation and scaling check (though score is scale invariant, we return points)
    # The problem asks for points. The score is computed after scaling to diameter 2.
    # So we can return the optimized points as is.
    
    if best_points is None:
        return [[float(x), float(y)] for x, y in p_oct]
        
    return [[float(x), float(y)] for x, y in best_points]

