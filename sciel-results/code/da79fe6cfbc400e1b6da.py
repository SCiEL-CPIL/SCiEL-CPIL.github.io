# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time

def no_5_on_a_sphere8() -> np.ndarray:
    n = 8
    # Grid points
    grid = np.array(list(itertools.product(range(n), range(n), range(n))), dtype=np.int64)
    N = len(grid)
    
    # Precompute squared norms
    grid_sq = (grid ** 2).sum(axis=1)
    
    # Precompute all 4-combinations of indices for the grid? No, that's C(512,4) ~ 3.5 billion. Too many.
    # We need a different check strategy.
    
    # The constraint is: No 5 points are co-spherical or co-planar.
    # This is equivalent to: For any set of 5 points, the determinant of the matrix formed by (q_i - p_5, |q_i|^2 - |p_5|^2) is not close to 0.
    # Actually, the test says:
    # "points are taken in the returned order; for each point p with at least 4 earlier points, and every 4 earlier points q1..q4, form the 4x4 matrix with rows (q - p, |q|^2 - |p|^2); if |np.linalg.det| < 0.5 for any such matrix, the result is invalid."
    
    # This means: If we have a set S of points, and we order them p1, p2, ..., pk.
    # For each i > 4, and for each subset {j1, j2, j3, j4} of {1, ..., i-1}, 
    # let p = p_i and q's be p_j1...p_j4.
    # The matrix M has rows:
    # [q1 - p, |q1|^2 - |p|^2]
    # [q2 - p, |q2|^2 - |p|^2]
    # [q3 - p, |q3|^2 - |p|^2]
    # [q4 - p, |q4|^2 - |p|^2]
    # If |det(M)| < 0.5, invalid.
    
    # This condition is symmetric in the 5 points?
    # The condition that 5 points are co-spherical (or co-planar) is independent of which point is designated as 'p'.
    # If 5 points are co-spherical, then for ANY permutation, picking any one as p and the other 4 as q's will result in a singular (or near-singular) matrix.
    # So the constraint is simply: **No 5 points in the set are co-spherical or co-planar.**
    
    # How to check if a set of 5 points is co-spherical/plane?
    # The 5 points (x1,y1,z1), ..., (x5,y5,z5) are co-spherical if the rank of the matrix:
    # [x1 y1 z1 x1^2+y1^2+z1^2 1]
    # [x2 y2 z2 x2^2+y2^2+z2^2 1]
    # ...
    # [x5 y5 z5 x5^2+y5^2+z5^2 1]
    # is less than 5. (Actually, they are co-spherical if the 5x5 determinant is 0).
    # If they are co-planar, they are also co-spherical (sphere of infinite radius).
    # So we just need to check if any 5 points have a 5x5 determinant close to 0.
    
    # Efficient check for a specific set of 5 points:
    def is_bad_5(points_5):
        # points_5 is (5, 3) array
        # Construct 5x5 matrix
        # Rows: [x, y, z, x^2+y^2+z^2, 1]
        sq = (points_5 ** 2).sum(axis=1)
        M = np.column_stack((points_5, sq, np.ones(5)))
        # Compute determinant
        # If |det| < 0.5, it's bad.
        # Note: The determinant scale depends on coordinates.
        # The prompt says: "form the 4x4 matrix with rows (q - p, |q|^2 - |p|^2); if |np.linalg.det| < 0.5"
        # Let's verify if this 4x4 det is proportional to the 5x5 det.
        # The 5x5 matrix is:
        # [x1 y1 z1 x1^2+y1^2+z1^2 1]
        # ...
        # Subtract row 5 from rows 1-4:
        # [x1-x5, y1-y5, z1-z5, (x1^2+y1^2+z1^2) - (x5^2+y5^2+z5^2), 0]
        # This matches the structure of the 4x4 matrix in the prompt, but the prompt has 4 columns: (q-p) and |q|^2-|p|^2.
        # Yes, the 4x4 matrix in the prompt is exactly the top-left 4x4 block of the row-reduced 5x5 matrix.
        # So det(5x5) = det(4x4) * 1 (since the last column of the reduced matrix has 1 at bottom and 0s above? No.)
        # Let's check.
        # M5 = [A  b  1]
        #      [A  b  1]
        #      [A  b  1]
        #      [A  b  1]
        #      [A  b  1]
        # Row ops: Ri - R5 for i=1..4.
        # [Ai-A5  bi-b5  0]
        # ...
        # [A5     b5     1]
        # Expanding along last column: det(M5) = det(M4) * 1 = det(M4).
        # So the condition |det(4x4)| < 0.5 is exactly |det(5x5)| < 0.5.
        
        det_val = np.linalg.det(M)
        return abs(det_val) < 0.5

    # Strategy: Simulated Annealing
    # State: A set of points (indices into grid).
    # Move: Swap one point in the set with one point outside.
    # Cost: Number of "bad" 5-tuples? Or just validity?
    # We want to maximize k (number of points).
    # We can start with a valid set of size k0, and try to grow it.
    # Or start with a large set, remove bad points, then try to add back.
    
    # Better: 
    # 1. Generate a random permutation of all 512 points.
    # 2. Greedily add points to form a valid set. This gives a baseline k.
    # 3. Try to improve by swapping points or adding points.
    
    # To make SA work, we need a fast way to check if adding a point p to a set S is valid.
    # Adding p is invalid if there exist 4 points in S such that {p} U {q1..q4} is a bad 5-tuple.
    # This requires checking C(|S|, 4) tuples.
    # If |S| ~ 40, C(40,4) ~ 91k. 
    # For each tuple, we compute a 5x5 det. That's fast.
    # 91k dets per addition check.
    # If we do 1000 checks, that's 91M dets. NumPy can do that in a few seconds.
    
    # Let's optimize the check for "Can we add p to S?"
    # We need to check all 4-subsets of S.
    # Instead of generating combinations every time, we can precompute? No, S changes.
    
    # Alternative:
    # Maintain the set S.
    # When checking if p can be added, we iterate through all 4-combinations of S.
    # To speed up, we can vectorize the determinant calculation.
    
    def can_add(p_idx, S_indices):
        if len(S_indices) < 4:
            return True
        
        p = grid[p_idx]
        p_sq = grid_sq[p_idx]
        
        # Get coordinates of points in S
        S_pts = grid[S_indices]
        S_sq = grid_sq[S_indices]
        
        # We need to check all 4-combinations of indices in S
        # Let m = len(S_indices)
        # We generate all 4-combinations of range(m)
        # This is the bottleneck.
        
        # Optimization: 
        # If m is small, just do it.
        # If m is large, maybe use a different check?
        
        # Let's try to vectorize the det calculation for all 4-combs at once.
        # Number of combs C = m choose 4.
        # Create an array of shape (C, 4, 3) for the q points.
        # Compute q - p for each.
        # Compute |q|^2 - |p|^2.
        # Stack to (C, 4, 4).
        # Compute det for all C matrices.
        
        m = len(S_indices)
        if m < 4:
            return True
            
        # Generate combinations
        # Using itertools is slow in Python loop.
        # We can use a precomputed list of combinations for small m?
        # m will grow up to ~50. C(50,4) = 230k.
        
        # Let's use a trick:
        # The condition is that the 5 points are not co-spherical.
        # This is equivalent to the rank of the 5x5 matrix being 5.
        
        # Let's just use itertools.combinations for now, but wrap it in a tight loop or vectorize if possible.
        # Actually, generating the index array for combinations is O(C).
        
        try:
            # Use numpy to generate combinations? No direct function.
            # We'll use itertools but try to be efficient.
            combos = np.array(list(itertools.combinations(range(m), 4)), dtype=np.int32)
        except StopIteration:
            return True
            
        if len(combos) == 0:
            return True
            
        # Select points
        # combos shape (C, 4)
        # S_pts[combos] shape (C, 4, 3)
        q_pts = S_pts[combos]
        q_sq = S_sq[combos]
        
        # Compute differences
        # q_pts - p (broadcast)
        diff = q_pts - p # (C, 4, 3)
        sq_diff = q_sq - p_sq # (C, 4)
        
        # Stack
        # We want (C, 4, 4)
        M = np.concatenate([diff, sq_diff[..., np.newaxis]], axis=2)
        
        # Compute determinants
        # np.linalg.det works on stacks of matrices
        dets = np.linalg.det(M)
        
        return not np.any(np.abs(dets) < 0.5)

    # This check is O(m^4) per call.
    # If we call it for every point in a permutation (512 points), and m grows to 40:
    # Sum_{m=4}^{40} C(m,4) ~ C(41,5) ~ 750k operations? No.
    # Sum of C(m,4) for m=4 to 40 is C(41,5) = 749,398.
    # Each operation involves building a 4x4 matrix and computing det.
    # This is feasible for one pass.
    
    # Now, for Simulated Annealing:
    # We want to maximize k.
    # Start with a greedy solution.
    # Then try to swap points to allow adding more?
    # Or just try to add points that were previously rejected?
    
    # Let's implement a "Greedy + Repair" strategy.
    # 1. Greedy pass to get S.
    # 2. Try to add any point not in S. If it fails, identify the conflicting 4-subsets.
    # 3. Try to remove one point from the conflict to allow adding the new point?
    # This is complex.
    
    # Simpler: Random Restart Greedy with Multiple Orders.
    # The current program does this.
    # To improve, we can use a "Backtracking" greedy?
    # Or "Local Search" on the set S.
    
    # Let's try a different angle:
    # **Maximum Independent Set in a Hypergraph**
    # Vertices: 512 points.
    # Hyperedges: All 5-tuples that are co-spherical.
    # We want the largest set of vertices with no hyperedge fully contained.
    
    # This is NP-hard.
    # Heuristic: 
    # 1. Identify "dense" regions of co-spherical 5-tuples.
    # 2. Avoid them.
    
    # What makes 5 points co-spherical?
    # They lie on a sphere.
    # In an 8x8x8 grid, many points lie on simple spheres (e.g., centered at integer or half-integer coords).
    # If we avoid points that are "too symmetric", we might get a larger set.
    
    # Strategy: 
    # Score each point by how many "bad" 5-tuples it participates in?
    # We don't know the bad tuples without checking.
    
    # Let's go back to **Simulated Annealing** with **Swap Moves**.
    # State: Binary vector x of length 512.
    # Objective: Maximize sum(x) subject to no bad 5-tuples.
    # We can define a penalty function:
    # Penalty = sum over all bad 5-tuples T of (1 if all 5 points in T are selected else 0).
    # We want to minimize Penalty - lambda * sum(x).
    # If Penalty == 0, we have a valid set.
    # We can use SA to minimize this.
    
    # But checking "all bad 5-tuples" is expensive.
    # We only need to check if the *current* set S contains any bad 5-tuple.
    # If it does, we have a conflict.
    
    # Let's define the state as the set S.
    # Move: Add a point p not in S.
    # If valid, accept.
    # If invalid, reject (or accept with prob exp(-delta/T)).
    # Move: Remove a point p in S.
    # Always accept (or use to escape local minima).
    # Move: Swap p in S with q not in S.
    # Check if S - {p} + {q} is valid.
    
    # To check validity of S - {p} + {q}:
    # We only need to check 5-tuples that include q.
    # Because 5-tuples not including q were already valid in S (assuming S was valid).
    # Wait, if S was valid, removing p keeps it valid.
    # Adding q creates new 5-tuples containing q.
    # We need to check if any 4-subset of S' = S - {p} forms a bad 5-tuple with q.
    # This is exactly the `can_add` check for q on S'.
    
    # So, if we maintain a valid set S, we can efficiently check swaps.
    
    # Algorithm:
    # 1. Generate a valid set S using Greedy.
    # 2. Loop:
    #    a. Pick a random point q not in S.
    #    b. Try to add q. If valid, accept (k increases).
    #    c. If invalid, pick a random point p in S.
    #    d. Check if S - {p} + {q} is valid.
    #       - This requires checking if q can be added to S - {p}.
    #       - Note: S - {p} is valid.
    #       - We check if q conflicts with any 4-subset of S - {p}.
    #    e. If valid, accept swap.
    #    f. If not, reject.
    #    g. Occasionally, remove a random point p from S to "cool down" or escape?
    
    # This allows us to "move" points around to find a configuration where more points fit.
    
    # Optimization:
    # We can precompute for each point q, which points p are "conflicting" with it?
    # No, conflicts depend on the set S.
    
    # Let's implement this SA.
    
    rng = np.random.default_rng(42)
    
    # Initial Greedy Fill
    def greedy_fill(order_idx):
        S = []
        S_set = set()
        for idx in order_idx:
            if can_add(idx, S):
                S.append(idx)
                S_set.add(idx)
        return np.array(S, dtype=np.int64), S_set

    best_S = np.empty(0, dtype=np.int64)
    best_S_set = set()
    
    start_time = time.time()
    deadline = start_time + 620
    
    # Try a few random starts
    while time.time() < deadline:
        # Random permutation
        order_idx = rng.permutation(N)
        S_arr, S_set = greedy_fill(order_idx)
        
        if len(S_arr) > len(best_S):
            best_S = S_arr
            best_S_set = S_set
            
        # Now try to improve S_arr via swaps
        S = list(S_arr)
        S_set = set(S)
        T = 1.0 # Temperature
        T_min = 0.01
        cooling = 0.999
        
        # List of points not in S
        not_in_S = [i for i in range(N) if i not in S_set]
        
        # We need to be careful with time.
        # Each swap check is O(m^4) ~ 91k dets.
        # If we do 1000 swaps, that's 91M dets.
        # This might be too slow in Python/NumPy if not optimized.
        
        # Let's limit the number of swap attempts per restart.
        max_swaps = 500
        
        for _ in range(max_swaps):
            if time.time() > deadline - 10:
                break
                
            if not not_in_S:
                break
                
            # Pick random q not in S
            q = rng.choice(not_in_S)
            
            # Try to add q
            if can_add(q, S):
                # Accept
                S.append(q)
                S_set.add(q)
                not_in_S.remove(q)
                continue
                
            # Try to swap
            # Pick random p in S
            p = rng.choice(S)
            
            # Check if S - {p} + {q} is valid
            # This is equivalent to: can_add(q, S - {p})
            # We need to remove p from S, check, then restore if rejected.
            
            S.remove(p)
            valid_swap = can_add(q, S)
            
            if valid_swap:
                # Accept swap
                S.append(q)
                S_set.remove(p)
                S_set.add(q)
                not_in_S.remove(q)
                not_in_S.append(p)
            else:
                # Reject, restore
                S.append(p)
            
            # Update best
            if len(S) > len(best_S):
                best_S = np.array(S, dtype=np.int64)
                best_S_set = set(S)
                
            # Cool down
            T *= cooling
            
    return best_S.astype(np.int64)

# EVOLVE-BLOCK-END
