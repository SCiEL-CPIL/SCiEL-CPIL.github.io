import numpy as np
import time
import random

N = 11
SQRT3 = np.sqrt(3)
MIN_DIST = SQRT3
START_TIME = time.time()
TIME_LIMIT = 110.0  # Safety margin

# Precompute vertices of a unit hexagon centered at origin
# Vertices at angles 0, 60, ...
HEX_VERTS = np.array([
    [np.cos(np.pi * k / 3), np.sin(np.pi * k / 3)] for k in range(6)
])

# Precompute normals for the outer hexagon boundary
# Normals at angles 30, 90, ...
OUTER_NORMALS = np.array([
    [np.cos(np.pi/6 + k * np.pi/3), np.sin(np.pi/6 + k * np.pi/3)] for k in range(6)
])

def get_penalty(centers, S):
    """
    centers: (N, 2) array
    S: scalar
    Returns penalty value. Lower is better.
    """
    penalty = S
    
    # Overlap penalty
    # Distances between all pairs
    # Efficient calculation
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(centers[i] - centers[j])
            if d < MIN_DIST:
                penalty += (MIN_DIST - d) ** 2

    # Outer boundary penalty
    # For each hexagon, check all 6 vertices
    limit = S * SQRT3 / 2
    for i in range(N):
        verts = centers[i] + HEX_VERTS
        # Project vertices onto normals
        # proj = verts @ OUTER_NORMALS.T  (6x6 matrix mult)
        # We need max projection for each normal? No, for each vertex, it must satisfy ALL normals.
        # So for each vertex v, check if max_k (dot(n_k, v)) <= limit.
        # Equivalent to: for each normal k, max_i (dot(n_k, v_i)) <= limit? No.
        # Each vertex must be inside.
        # So for each vertex v:
        #   proj = v @ OUTER_NORMALS.T
        #   if any proj > limit: violation
        for v in verts:
            projs = v @ OUTER_NORMALS.T
            max_proj = np.max(projs)
            if max_proj > limit:
                penalty += (max_proj - limit) ** 2
                
    return penalty

def local_search(S, init_centers):
    """
    Run local search for fixed S to minimize penalty.
    Returns best centers and penalty.
    """
    centers = init_centers.copy()
    best_penalty = get_penalty(centers, S)
    
    # Simple gradient-free local search
    # We'll use a random walk with step size decay
    step_size = 0.5
    for _ in range(500):
        if time.time() - START_TIME > TIME_LIMIT:
            break
        improved = False
        for i in range(N):
            # Try small random moves
            for _ in range(10):
                delta = np.random.normal(0, step_size, 2)
                new_centers = centers.copy()
                new_centers[i] += delta
                p = get_penalty(new_centers, S)
                if p < best_penalty:
                    best_penalty = p
                    centers = new_centers
                    improved = True
                    break
        if not improved:
            step_size *= 0.9
            if step_size < 0.001:
                break
                
    return centers, best_penalty

def construct():
    global START_TIME
    START_TIME = time.time()
    
    best_S = 40.0
    best_centers = None
    
    # Binary search for S
    # Lower bound: sqrt(11) ~ 3.32
    # Upper bound: 10 (row packing is ~9.1, so 10 is safe)
    low = 3.3
    high = 10.0
    
    # We will try to find the minimum S such that penalty can be driven to ~0
    # Or simply minimize the penalty function with S as a variable?
    # Minimizing S + penalty is tricky because penalty grows as S shrinks.
    # Better: Binary search S.
    
    # To speed up, we can check a few candidate S values.
    # Or run a global search on S and centers.
    
    # Let's do a coarse binary search.
    for _ in range(20):
        if time.time() - START_TIME > TIME_LIMIT:
            break
        mid = (low + high) / 2.0
        # Try to find a valid packing for S = mid
        found = False
        for restart in range(5):
            if time.time() - START_TIME > TIME_LIMIT:
                break
            # Initialize centers randomly in a box of size mid
            box_size = mid
            init_centers = np.random.uniform(-box_size, box_size, (N, 2))
            centers, penalty = local_search(mid, init_centers)
            if penalty < 1e-3: # Effectively zero
                found = True
                best_S = mid
                best_centers = centers
                break
        if found:
            high = mid
        else:
            low = mid
            
    # If we found a valid packing, refine it?
    # The binary search gives the smallest S for which we *found* a packing.
    # It might not be the absolute minimum, but it's a valid lower score.
    
    # If no packing found even at S=10 (unlikely), return seed
    if best_centers is None:
        # Fallback to seed-like solution
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
        
    # Format output
    inner = []
    for c in best_centers:
        inner.append([c[0], c[1], 0.0])
        
    return inner, (0.0, 0.0), best_S, 0.0

# EVOLVE-BLOCK-END
