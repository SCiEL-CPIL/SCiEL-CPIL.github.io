import numpy as np
import itertools
import random

def construct():
    N = 16
    
    # 1. Generate Triangular Lattice Points
    # Basis vectors
    v1 = np.array([1.0, 0.0])
    v2 = np.array([0.5, np.sqrt(3)/2])
    
    # Generate points in a range
    # We need the 19 points of the hexagon of radius 2
    # i, j from -2 to 2 such that max(|i|, |j|, |i+j|) <= 2 ? No, that's hexagonal distance.
    # Let's just generate a grid and filter by distance to origin.
    points = []
    for i in range(-3, 4):
        for j in range(-3, 4):
            p = i * v1 + j * v2
            points.append(p)
            
    points = np.array(points)
    # Calculate distance squared to origin
    dists = np.sum(points**2, axis=1)
    # Sort by distance
    sorted_indices = np.argsort(dists)
    
    # The 19 closest points should be the hexagon of radius 2
    # Let's verify.
    # Dist 0: (0,0)
    # Dist 1: 6 points
    # Dist 3: 12 points
    # Total 19.
    # We need 16 points.
    # We must keep the 7 inner points (dist 0 and 1).
    # We must choose 9 out of the 12 outer points (dist 3).
    
    inner_indices = sorted_indices[:7]
    outer_indices = sorted_indices[7:19]
    
    best_points = None
    best_dmax = float('inf')
    
    # Iterate through combinations of 3 outer points to REMOVE
    # There are 12 outer points. We remove 3.
    # Indices of outer points in the 'points' array are outer_indices
    for remove_combo in itertools.combinations(range(12), 3):
        # Create set of indices to keep
        keep_outer = [i for i in range(12) if i not in remove_combo]
        
        current_indices = list(inner_indices) + [outer_indices[i] for i in keep_outer]
        
        # Get the points
        P = points[current_indices]
        
        # Calculate pairwise distances
        # Flatten to 1D for vectorized distance calculation?
        # Or just loop, N=16 is small.
        dmin = float('inf')
        dmax = 0.0
        for i in range(16):
            for j in range(i+1, 16):
                d = np.linalg.norm(P[i] - P[j])
                if d < dmin: dmin = d
                if d > dmax: dmax = d
                
        if dmax < best_dmax:
            best_dmax = dmax
            best_points = P.copy()
            
    # 2. Local Search Refinement
    # We want to minimize dmax while keeping dmin >= 1.
    # First, scale best_points so that dmin = 1.
    # Calculate dmin of best_points
    dmin_best = float('inf')
    for i in range(16):
        for j in range(i+1, 16):
            d = np.linalg.norm(best_points[i] - best_points[j])
            if d < dmin_best: dmin_best = d
            
    scale = 1.0 / dmin_best
    P = best_points * scale
    
    # Now dmin is 1. We want to minimize dmax.
    # Simple Hill Climbing
    steps = 500
    step_size = 0.05
    random.seed(42)
    
    for _ in range(steps):
        improved = False
        # Calculate current dmax
        dmax_curr = 0.0
        for i in range(16):
            for j in range(i+1, 16):
                d = np.linalg.norm(P[i] - P[j])
                if d > dmax_curr: dmax_curr = d
                
        # Try to move each point
        for i in range(16):
            for _ in range(10): # 10 random directions per point per step
                # Random direction
                angle = random.uniform(0, 2*np.pi)
                dir_vec = np.array([np.cos(angle), np.sin(angle)])
                
                # Small perturbation
                new_P = P.copy()
                new_P[i] += step_size * dir_vec
                
                # Check constraints: all pairwise distances >= 1
                valid = True
                new_dmax = 0.0
                for a in range(16):
                    for b in range(a+1, 16):
                        d = np.linalg.norm(new_P[a] - new_P[b])
                        if d < 1.0 - 1e-9: # Tolerance
                            valid = False
                            break
                        if d > new_dmax: new_dmax = d
                    if not valid: break
                
                if valid and new_dmax < dmax_curr:
                    P = new_P
                    dmax_curr = new_dmax
                    improved = True
                    
        if not improved:
            # Reduce step size
            step_size *= 0.5
            if step_size < 1e-4:
                break
                
    return P
