# EVOLVE-BLOCK-START
import numpy as np
import itertools
import random

def no_5_on_a_sphere8() -> np.ndarray:
    n = 8
    grid = np.array(list(itertools.product(range(n), range(n), range(n))), dtype=np.int64)
    N = len(grid)
    
    # Pre-compute squared norms for faster determinant calculation
    # det of rows (q-p, |q|^2 - |p|^2)
    
    def is_valid_set(points):
        """Check if a set of points has no 5 cospherical."""
        m = len(points)
        if m < 5:
            return True
        # Check all 5-combinations? Too slow.
        # Instead, for each point p, check if it forms a bad 5-tuple with 4 others.
        # This is still expensive.
        pass

    def can_add(arr, p):
        """Check if p can be added to arr without creating a cospherical 5-tuple."""
        m = len(arr)
        if m < 4:
            return True
        
        # We need to check if there exist 4 points q1..q4 in arr such that
        # p, q1, q2, q3, q4 are cospherical.
        # This is equivalent to checking if the rank of the matrix formed by
        # lifting p and any 4 from arr is < 5?
        # Actually, 5 points are cospherical iff the 5x5 determinant is 0.
        # Rows: [x, y, z, x^2+y^2+z^2, 1]
        
        # Optimization:
        # If we fix p, the condition for q1..q4 is that they lie on the sphere defined by p and ...?
        # No, a sphere is defined by 4 points.
        # So, for every 4-tuple (q1..q4) in arr, check if p lies on the sphere defined by them.
        
        # Number of 4-tuples is C(m, 4).
        # If m ~ 30, C(30, 4) ~ 27k.
        # For each, we compute a determinant.
        # This is feasible in Python if optimized with NumPy.
        
        combos = itertools.combinations(range(m), 4)
        for combo in combos:
            q = arr[list(combo)]
            # Check if p, q[0], q[1], q[2], q[3] are cospherical
            # Matrix M (5x5):
            # [x, y, z, r2, 1]
            pts = np.vstack([p, q])
            r2 = (pts ** 2).sum(axis=1)
            M = np.column_stack((pts, r2, np.ones(5)))
            det = np.linalg.det(M)
            if abs(det) < 1e-9:
                return False
        return True

    # Local Search Strategy
    # 1. Greedy initialization
    # 2. Local improvement:
    #    - Try to add any point.
    #    - If stuck, try swaps to enable additions.
    
    def local_search(initial_set, max_iters=1000):
        current_set = list(initial_set)
        current_size = len(current_set)
        best_set = current_set[:]
        best_size = current_size
        
        for _ in range(max_iters):
            improved = False
            
            # Phase 1: Try to add points
            for p in grid:
                if p not in current_set:
                    if can_add(np.array(current_set), p):
                        current_set.append(p)
                        improved = True
                        if len(current_set) > best_size:
                            best_size = len(current_set)
                            best_set = current_set[:]
            
            if not improved:
                # Phase 2: Try swaps
                # For each a in current_set, b not in current_set
                # If swapping a->b keeps the set valid, and allows an addition, do it.
                # To keep it simple, we just do the swap if it's valid, 
                # and then retry Phase 1.
                swapped = False
                # Randomize order of swaps to escape local minima
                a_indices = list(range(len(current_set)))
                random.shuffle(a_indices)
                b_candidates = [p for p in grid if p not in current_set]
                random.shuffle(b_candidates)
                
                for ai in a_indices:
                    for b in b_candidates:
                        new_set = current_set[:]
                        new_set[ai] = b
                        # Check validity of new_set
                        # This is expensive: O(m^4)
                        # We only do this if we are stuck.
                        if is_valid_set(np.array(new_set)):
                            current_set = new_set
                            swapped = True
                            break
                    if swapped:
                        break
                
                if not swapped:
                    break # No more improvements possible with this logic
            
        return np.array(best_set)

    # Run multiple restarts
    np.random.seed(42)
    best_overall = None
    best_size = 0
    
    # Greedy init
    order = grid[np.random.permutation(N)]
    greedy_set = []
    for p in order:
        if can_add(np.array(greedy_set), p):
            greedy_set.append(p)
    
    # Local Search
    result = local_search(greedy_set)
    
    if len(result) > best_size:
        best_size = len(result)
        best_overall = result
        
    return best_overall.astype(np.int64)

# EVOLVE-BLOCK-END
