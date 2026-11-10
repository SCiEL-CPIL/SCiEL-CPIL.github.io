# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time
import sys

def no_5_on_a_sphere8() -> np.ndarray:
    """
    Strategy: Deterministic Greedy Construction.
    1. Generate all 512 points in the 8x8x8 grid.
    2. Sort points by a spatial heuristic (e.g., Manhattan distance from center) 
       to prioritize "central" points which may have fewer co-spherical constraints 
       with other central points compared to edge/corner points, or simply to 
       provide a deterministic order that avoids the randomness of the previous 
       repair strategy.
    3. Iterate through the sorted points. For each point, check if adding it to 
       the current set creates any "bad" 5-tuple (5 co-spherical/co-planar points).
       A new point p is safe if for every 4-tuple of existing points, the 
       determinant of the matrix formed by (q_i - p, |q_i|^2 - |p|^2) is >= 0.5.
    4. If safe, add the point.
    5. Return the constructed set.
    
    This is materially different from the repair-then-extend strategy because it 
    builds the solution from scratch in a fixed order, ensuring validity at each 
    step without needing to remove points later. It is also deterministic, which 
    can help in reproducing results and avoiding bad random seeds.
    """
    n = 8
    deadline = time.time() + 595.0  # Safety margin

    # Generate grid
    grid = np.array(list(itertools.product(range(n), range(n), range(n))), dtype=np.int64)
    N = len(grid)
    
    # Precompute squared norms
    sq = (grid ** 2).sum(axis=1)
    
    # Heuristic: Sort by Manhattan distance from the center (3.5, 3.5, 3.5)
    # Points closer to the center are generally "more connected" in terms of 
    # symmetry, but let's try sorting by distance from center to see if it 
    # helps find a larger independent set. Actually, for "no 5 on a sphere", 
    # maybe we want to avoid points that form many spheres. 
    # Let's try a few heuristics and pick the best one within the time limit.
    
    # Heuristic 1: Manhattan distance from center
    center = np.array([3.5, 3.5, 3.5])
    dists = np.abs(grid - center).sum(axis=1)
    order1 = np.argsort(dists)
    
    # Heuristic 2: Manhattan distance from corner (0,0,0)
    dists2 = grid.sum(axis=1)
    order2 = np.argsort(dists2)
    
    # Heuristic 3: Random (fixed seed for reproducibility)
    np.random.seed(42)
    order3 = np.random.permutation(N)
    
    # Heuristic 4: Sort by sum of coordinates squared (radius)
    order4 = np.argsort(sq)
    
    orders = [order1, order2, order3, order4]
    
    best_solution = None
    best_size = 0
    
    for order in orders:
        if time.time() > deadline:
            break
            
        selected = []
        selected_sq = []
        
        # We will iterate through the points in the given order.
        # For each candidate point p, we check if it can be added.
        # A point p is "bad" with the current set S if there exist 4 points q1,q2,q3,q4 in S
        # such that the 5 points p, q1, q2, q3, q4 are co-spherical.
        
        # To check this efficiently:
        # For a fixed p and a set S, we need to check all 4-subsets of S.
        # This is C(|S|, 4) checks.
        # If |S| is large (e.g., 100), C(100,4) ~ 3.9e6. 
        # For 512 points, this is 512 * 3.9e6 ~ 2e9 operations, which might be too slow in Python loops.
        # We need to vectorize the determinant check.
        
        # Let S be the current set of selected points.
        # For a new point p, we form a matrix M where each row corresponds to a point q in S:
        # Row for q: [q_x - p_x, q_y - p_y, q_z - p_z, |q|^2 - |p|^2]
        # We need to check if any 4x4 submatrix of M has determinant < 0.5.
        
        # This is still computationally heavy if done naively for every p.
        # However, we can stop early if we find a violation.
        # Also, we can limit the size of S we check against? No, we must check all.
        
        # Let's implement a fast check using numpy.
        
        def can_add(p, S, S_sq):
            """
            Check if point p can be added to set S without creating a bad 5-tuple.
            S: (k, 3) array of selected points.
            S_sq: (k,) array of squared norms of selected points.
            """
            k = len(S)
            if k < 4:
                return True
                
            p_sq = np.sum(p**2)
            
            # Differences
            diff = S - p  # (k, 3)
            norm_diff = S_sq - p_sq  # (k,)
            
            # Matrix W is (k, 4)
            W = np.column_stack((diff, norm_diff))
            
            # We need to check all 4-combinations of rows in W.
            # If any 4x4 submatrix has |det| < 0.5, return False.
            
            # For efficiency, if k is small, just loop.
            # If k is large, this might be slow.
            # Let's optimize by checking combinations.
            
            # Number of combinations is C(k, 4).
            # For k=50, C(50,4) ~ 230k. 512 points * 230k = 118M checks. 
            # Each check is a 4x4 det. This might be borderline in Python.
            # We can use a heuristic: only check a sample of combinations? 
            # No, that risks invalidity.
            
            # Let's try to vectorize the determinant calculation for all 4-combinations?
            # That's hard because the combinations are dynamic.
            
            # Alternative: Use the fact that det(W[i,j,k,l]) = 0 iff the 5 points are co-spherical.
            # We can compute the determinant for all 4-combinations in a batch?
            # Not easily with standard numpy without loops.
            
            # Let's use a loop but break early.
            # To speed up, we can pre-compute the indices of combinations?
            # Or just use itertools.combinations which is in C.
            
            for combo in itertools.combinations(range(k), 4):
                sub = W[list(combo)]
                d = np.linalg.det(sub)
                if abs(d) < 0.5:
                    return False
                    
            return True

        # Main loop
        for idx in order:
            if time.time() > deadline:
                break
                
            p = grid[idx]
            p_sq = sq[idx]
            
            if can_add(p, np.array(selected), np.array(selected_sq)):
                selected.append(p)
                selected_sq.append(p_sq)
                
            # Optional: Early termination if we've already beaten the best known size
            # and time is running out? No, we want the max size.
            
        # Convert to array
        if selected:
            sol = np.array(selected, dtype=np.int64)
        else:
            sol = np.empty((0, 3), dtype=np.int64)
            
        if len(sol) > best_size:
            best_size = len(sol)
            best_solution = sol
            print(f"Heuristic {orders.index(order)}: Found solution with {best_size} points", file=sys.stderr)
            
    # Final validation to be safe
    def validate_solution(points: np.ndarray) -> bool:
        if len(points) == 0:
            return True
        if np.any(points < 0) or np.any(points >= 8):
            return False
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                if np.array_equal(points[i], points[j]):
                    return False
        k = len(points)
        for i in range(4, k):
            p = points[i]
            prev = points[:i]
            diff = prev - p
            norm_diff = (prev ** 2).sum(axis=1) - (p ** 2).sum()
            w = np.column_stack([diff, norm_diff])
            for combo in itertools.combinations(range(i), 4):
                sub = w[list(combo)]
                if abs(np.linalg.det(sub)) < 0.5:
                    return False
        return True

    if best_solution is not None:
        if not validate_solution(best_solution):
            print("WARNING: Best solution failed validation, returning empty", file=sys.stderr)
            return np.empty((0, 3), dtype=np.int64)
        return best_solution.astype(np.int64)

    # Fallback
    fallback = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.int64)
    return fallback

# EVOLVE-BLOCK-END
