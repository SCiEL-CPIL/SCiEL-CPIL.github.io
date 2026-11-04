import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

N = 12

def construct():
    start_time = time.time()
    timeout = 110.0  # Safety margin
    
    # Directions for the 6 normals of the hexagon
    # Angles: 0, 60, 120, 180, 240, 300
    angles = np.deg2rad([0, 60, 120, 180, 240, 300])
    u = np.column_stack((np.cos(angles), np.sin(angles)))
    
    # Support function for unit hexagon (aligned)
    # h(theta) = (sqrt(3)/2) * sec(theta mod 60)
    # But for aligned hexes, we only need the apothem for the containment of the "center" 
    # if we treat the exclusion zone correctly.
    
    # Let's define the objective: Minimize S
    # S = 1 + (2/sqrt(3)) * max_k max_i | (x_i - X) dot u_k |
    # We can fix X=0, Y=0 by centering the points before evaluation.
    
    def calc_S(x):
        # x is a flat array of 24 coordinates
        pts = x.reshape(N, 2)
        # Center the points
        center = np.mean(pts, axis=0)
        pts_centered = pts - center
        
        # Calculate max projection in each of the 6 directions
        # proj_i_k = pts_centered[i] dot u[k]
        projs = pts_centered @ u.T  # Shape (N, 6)
        
        # The required apothem for the outer hexagon to contain the centers
        # is max_abs_proj.
        # The inner hexagons have apothem sqrt(3)/2.
        # So the outer apothem A must be max_abs_proj + sqrt(3)/2.
        # S = A * 2 / sqrt(3)
        
        max_abs_proj = np.max(np.abs(projs))
        A = max_abs_proj + np.sqrt(3)/2
        S = A * 2 / np.sqrt(3)
        
        # Check non-overlap
        # For aligned hexes, non-overlap means distance vector is outside hex of apothem sqrt(3).
        # Condition: | d dot u_k | <= sqrt(3) for all k is INSIDE.
        # We want OUTSIDE. So we need min_k | d dot u_k | > sqrt(3)?
        # No. The Minkowski sum is the hexagon of apothem sqrt(3).
        # Two hexes overlap if d is INSIDE the Minkowski sum.
        # So we need d to be OUTSIDE.
        # d is outside if for SOME k, | d dot u_k | > sqrt(3).
        # Wait. The Minkowski sum $P+(-P)$ is the set of d such that they touch/overlap.
        # This set is a hexagon with apothem $\sqrt{3}$.
        # So they are disjoint if d is NOT in this hexagon.
        # d is NOT in the hexagon if $\max_k | d \cdot u_k | > \sqrt{3}$.
        
        valid = True
        for i in range(N):
            for j in range(i+1, N):
                d = pts[i] - pts[j]
                proj_d = np.abs(d @ u.T)
                if np.max(proj_d) <= np.sqrt(3) + 1e-6:
                    valid = False
                    break
            if not valid:
                break
        
        if not valid:
            return 1e9
        return S

    # Initial guess: Honeycomb lattice
    # u_vec = (sqrt(3), 0), v_vec = (sqrt(3)/2, 3/2)
    u_vec = np.array([np.sqrt(3), 0.0])
    v_vec = np.array([np.sqrt(3)/2, 1.5])
    
    # Generate a large pool of lattice points
    lattice_points = []
    for i in range(-3, 4):
        for j in range(-3, 4):
            p = i * u_vec + j * v_vec
            lattice_points.append(p)
    lattice_points = np.array(lattice_points)
    
    # Select 12 points that minimize S
    # We can do a random search over subsets or use a heuristic.
    # Heuristic: Start with a dense cluster and remove/add points.
    
    best_S = 1e9
    best_x = None
    
    # Try a few random initializations from the lattice
    for restart in range(20):
        if time.time() - start_time > timeout:
            break
            
        # Pick 12 random points from the lattice, ensuring they are somewhat close
        # Sort lattice points by distance to origin
        dists = np.linalg.norm(lattice_points, axis=1)
        indices = np.argsort(dists)
        
        # Take the first 20 closest, then pick 12
        candidates = lattice_points[indices[:20]]
        
        # Randomly select 12 from these 20
        if len(candidates) >= 12:
            idx = np.random.choice(len(candidates), 12, replace=False)
            init_pts = candidates[idx]
        else:
            init_pts = candidates
            
        x0 = init_pts.flatten()
        
        # Use Nelder-Mead or L-BFGS-B to refine
        # Since we have discrete lattice start, continuous refinement might break symmetry but could improve.
        # However, the constraint is non-smooth.
        # Let's just evaluate the lattice configuration first.
        
        S_val = calc_S(x0)
        if S_val < best_S:
            best_S = S_val
            best_x = x0
            
    # If we have time, try a local optimization on the best candidate
    if best_x is not None and time.time() - start_time < timeout - 10:
        def objective(x):
            return calc_S(x)
        
        # Use Powell method which doesn't need gradients
        res = minimize(objective, best_x, method='Powell', options={'maxiter': 100, 'xatol': 1e-4, 'fatol': 1e-6})
        if res.success or res.fun < best_S:
            if res.fun < best_S:
                best_S = res.fun
                best_x = res.x
                
    # Final check and formatting
    if best_x is None:
        # Fallback: linear strip
        best_x = np.array([ [i*np.sqrt(3), 0] for i in range(N) ]).flatten()
        
    pts = best_x.reshape(N, 2)
    # Center the points
    center = np.mean(pts, axis=0)
    pts = pts - center
    
    # Calculate final S
    S = calc_S(best_x)
    
    # Format output
    # inner: list of [x, y, angle]
    inner = [[float(p[0]), float(p[1]), 0.0] for p in pts]
    # outer center
    outer_center = [0.0, 0.0]
    # outer side
    outer_side = float(S)
    # outer angle
    outer_angle = 0.0
    
    return inner, outer_center, outer_side, outer_angle
