import numpy as np
from scipy.optimize import minimize
from scipy.spatial import Voronoi, Delaunay
import sys

N = 26

def construct():
    
    def largest_empty_circle(centers, radii):
        """
        Finds the center and radius of the largest circle that can be added
        to the given set of circles without overlapping, within [0,1]x[0,1].
        Uses Voronoi vertices for interior candidates and boundary checks.
        """
        if len(centers) == 0:
            return (0.5, 0.5, 0.5)
        
        centers = np.array(centers)
        radii = np.array(radii)
        n = len(centers)
        
        best_r = 0
        best_c = None
        
        # 1. Candidate centers from Voronoi vertices
        if n >= 3:
            try:
                vor = Voronoi(centers)
                verts = vor.vertices
                # Filter valid vertices inside [0,1]x[0,1]
                valid_mask = (verts[:, 0] >= 0) & (verts[:, 0] <= 1) & (verts[:, 1] >= 0) & (verts[:, 1] <= 1)
                valid_verts = verts[valid_mask]
                
                if len(valid_verts) > 0:
                    # Vectorized distance calculation to all centers
                    # valid_verts: (M, 2), centers: (N, 2)
                    # dists: (M, N)
                    dists = np.linalg.norm(valid_verts[:, np.newaxis, :] - centers[np.newaxis, :, :], axis=2)
                    # Subtract radii
                    dists_adj = dists - radii[np.newaxis, :]
                    # Min over all centers for each vertex
                    min_dists = np.min(dists_adj, axis=1)
                    
                    # Also need to check distance to boundaries
                    d_bound = np.minimum(
                        np.minimum(valid_verts[:, 0], 1 - valid_verts[:, 0]),
                        np.minimum(valid_verts[:, 1], 1 - valid_verts[:, 1])
                    )
                    
                    # The radius is the min of distance to other circles and distance to boundary
                    candidate_radii = np.minimum(min_dists, d_bound)
                    
                    # Find the max
                    max_idx = np.argmax(candidate_radii)
                    if candidate_radii[max_idx] > best_r:
                        best_r = candidate_radii[max_idx]
                        best_c = valid_verts[max_idx]
                        
            except Exception:
                pass
        
        # 2. Candidate centers on the boundary
        # We can check the intersections of Voronoi edges with the boundary,
        # or simply sample the boundary. Sampling is robust.
        # To be efficient, we can check the midpoints of Voronoi edges that touch the boundary,
        # but simple dense sampling is often sufficient and fast enough for N=26.
        
        # Let's use a slightly smarter boundary check:
        # The largest empty circle on the boundary must be tangent to the boundary
        # and at least one other circle (or corner).
        # So we can check the points on the boundary that are equidistant to the boundary
        # and one circle.
        
        # For each side, the center (x, 0) must satisfy:
        # dist_to_circle_i(x,0) = r_i + r_new
        # r_new = min(x, 1-x, 0, 1) -> wait, y=0 so dist to y=0 is 0.
        # r_new is limited by x, 1-x, and dist to circles.
        
        # Let's just sample the boundary densely, it's O(N * num_samples).
        num_samples = 200
        best_r_bound = 0
        best_c_bound = None
        
        for side in range(4):
            for t in np.linspace(0, 1, num_samples):
                if side == 0: # y=0
                    x, y = t, 0.0
                elif side == 1: # y=1
                    x, y = t, 1.0
                elif side == 2: # x=0
                    x, y = 0.0, t
                else: # x=1
                    x, y = 1.0, t
                
                # Distance to boundary walls (other than the one it's on)
                # If y=0, dist to y=0 is 0. Dist to x=0 is x, x=1 is 1-x, y=1 is 1.
                # So r_max_bound = min(x, 1-x, 1) if y=0.
                # Generally: min(x, 1-x, y, 1-y)
                r_bound = min(x, 1-x, y, 1-y)
                
                # Distance to all circles
                dists_to_circles = np.sqrt((x - centers[:, 0])**2 + (y - centers[:, 1])**2)
                dists_adj = dists_to_circles - radii
                r_circles = np.min(dists_adj)
                
                r = min(r_bound, r_circles)
                
                if r > best_r_bound:
                    best_r_bound = r
                    best_c_bound = np.array([x, y])
                    
        if best_r_bound > best_r:
            best_r = best_r_bound
            best_c = best_c_bound
            
        if best_c is None:
            best_c = np.array([0.5, 0.5])
            best_r = 0.01
            
        return best_c[0], best_c[1], max(best_r, 1e-6)

    def greedy_initialize(n):
        centers = []
        radii = []
        
        # Start with 4 corners to establish the frame
        # Actually, starting with a single center or a few random ones might be better
        # to avoid getting stuck in a symmetric local optimum.
        # Let's start with 1 circle in the center.
        centers.append([0.5, 0.5])
        radii.append(0.5)
        
        for _ in range(n - 1):
            c_x, c_y, r = largest_empty_circle(centers, radii)
            centers.append([c_x, c_y])
            radii.append(r)
            
        return np.array(centers), np.array(radii)

    def energy(params):
        x_ = params[0::3]
        y_ = params[1::3]
        r_ = params[2::3]
        
        pen_bound = 0.0
        # Vectorized boundary penalty
        pen_bound += np.sum(np.maximum(0, r_ - x_)**2)
        pen_bound += np.sum(np.maximum(0, r_ - (1 - x_))**2)
        pen_bound += np.sum(np.maximum(0, r_ - y_)**2)
        pen_bound += np.sum(np.maximum(0, r_ - (1 - y_))**2)
        
        pen_overlap = 0.0
        # Vectorized overlap penalty
        # dist_sq[i, j] for i < j
        # We can compute all pairwise distances
        dists_sq = np.sum((x_[:, np.newaxis] - x_[np.newaxis, :])**2 + 
                          (y_[:, np.newaxis] - y_[np.newaxis, :])**2, axis=2)
        min_dists = r_[:, np.newaxis] + r_[np.newaxis, :]
        
        # We only care about i < j
        iu = np.triu_indices(N, 1)
        dists_sq_iu = dists_sq[iu]
        min_dists_iu = min_dists[iu]
        
        # Overlap = min_dist - sqrt(dist_sq) if dist_sq < min_dist^2
        # To avoid sqrt of negative (shouldn't happen) and division by zero
        # Use: overlap = max(0, min_dist - sqrt(dist_sq))
        # For penalty, we can just use (min_dist - dist)^2 where dist = sqrt(dists_sq)
        
        dists = np.sqrt(dists_sq_iu)
        overlaps = np.maximum(0, min_dists_iu - dists)
        pen_overlap = np.sum(overlaps**2)
        
        lam = 100.0
        return -np.sum(r_) + lam * (pen_bound + pen_overlap)

    def optimize(c_init, r_init):
        params0 = np.zeros(3*N)
        params0[0::3] = c_init[:, 0]
        params0[1::3] = c_init[:, 1]
        params0[2::3] = r_init
        
        bounds = []
        for i in range(N):
            bounds.append((0, 1))
            bounds.append((0, 1))
            bounds.append((0, 0.5))
            
        res = minimize(energy, params0, method='L-BFGS-B', bounds=bounds, 
                       options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10})
        
        params_opt = res.x
        x_opt = params_opt[0::3]
        y_opt = params_opt[1::3]
        r_opt = params_opt[2::3]
        
        # Post-processing: Ensure strict validity
        for _ in range(50):
            changed = False
            for i in range(N):
                r_max_bound = min(x_opt[i], 1-x_opt[i], y_opt[i], 1-y_opt[i])
                if r_opt[i] > r_max_bound:
                    r_opt[i] = r_max_bound
                    changed = True
                    
            for i in range(N):
                for j in range(i+1, N):
                    dist = np.sqrt((x_opt[i]-x_opt[j])**2 + (y_opt[i]-y_opt[j])**2)
                    min_dist = r_opt[i] + r_opt[j]
                    if dist < min_dist:
                        overlap = min_dist - dist
                        if min_dist > 1e-9:
                            scale = dist / min_dist
                            r_opt[i] *= scale
                            r_opt[j] *= scale
                            changed = True
            if not changed:
                break
                
        return x_opt, y_opt, r_opt

    # 1. Generate initial configuration using Greedy Method
    c_init, r_init = greedy_initialize(N)
    
    # 2. Optimize
    x_opt, y_opt, r_opt = optimize(c_init, r_init)
    best_score = np.sum(r_opt)
    best_x, best_y, best_r = x_opt, y_opt, r_opt
    
    # 3. Multi-start
    np.random.seed(42)
    
    # Strategy A: Perturb greedy start
    for i in range(15):
        c_pert = c_init + np.random.normal(0, 0.05, (N, 2))
        c_pert = np.clip(c_pert, 0.01, 0.99)
        r_pert = np.clip(r_init * 0.8, 0.01, 0.4)
        
        x_p, y_p, r_p = optimize(c_pert, r_pert)
        score = np.sum(r_p)
        
        if score > best_score:
            best_score = score
            best_x, best_y, best_r = x_p, y_p, r_p
            
    # Strategy B: Random init
    for i in range(15):
        c_rand = np.random.uniform(0.1, 0.9, (N, 2))
        r_rand = np.random.uniform(0.05, 0.1, N)
        
        x_p, y_p, r_p = optimize(c_rand, r_rand)
        score = np.sum(r_p)
        
        if score > best_score:
            best_score = score
            best_x, best_y, best_r = x_p, y_p, r_p

    # 4. Final validation and formatting
    result = np.zeros((N, 3))
    result[:, 0] = best_x
    result[:, 1] = best_y
    result[:, 2] = best_r
    
    return result
