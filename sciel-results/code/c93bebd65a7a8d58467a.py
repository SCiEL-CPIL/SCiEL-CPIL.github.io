# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time

def no_5_on_a_sphere8() -> np.ndarray:
    """
    Finds a large subset of the 8x8x8 grid such that no 5 points are co-spherical.
    Uses a hybrid approach: Randomized Greedy initialization followed by
    Local Search (Add/Swap) to escape local optima.
    """
    n = 8
    grid = np.array(list(itertools.product(range(n), range(n), range(n))), dtype=np.int64)
    N = len(grid)
    sq_norms = (grid ** 2).sum(axis=1)
    
    start_time = time.time()
    deadline = start_time + 580.0  # Safety margin
    
    def get_conflict_indices(S, p, sq_norm_p):
        """
        Checks if adding p to S creates a co-spherical 5-tuple.
        Returns True if valid.
        If invalid, returns (False, indices_of_conflicting_4_points).
        """
        M = len(S)
        if M < 4:
            return True, None
        
        # Construct W matrix (M, 4)
        # Rows: (q - p, |q|^2 - |p|^2)
        W = np.empty((M, 4), dtype=np.float64)
        W[:, 0:3] = S - p
        W[:, 3] = sq_norms[S] - sq_norm_p  # sq_norms[S] works if S is array of indices? 
        # Wait, S is an array of points, not indices. 
        # We need sq_norms for the points in S.
        # Let's pass sq_norms_S directly or compute it.
        
        # Optimization: Precompute sq_norms for the specific points in S?
        # Or just compute (S**2).sum(axis=1)
        sq_norms_S = (S ** 2).sum(axis=1)
        W[:, 3] = sq_norms_S - sq_norm_p
        
        # Check all combinations of 4
        # For efficiency, we can check random subsets first? 
        # No, we need to find *a* conflict to swap.
        # We'll iterate through combinations. 
        # To speed up, we can vectorize the determinant calculation for batches.
        
        # If M is small, loop is fine.
        # If M is large, we need to be careful.
        
        # Let's try a vectorized approach for all combos if M < 40, else sample?
        # Actually, for the swap logic, we just need ONE conflict.
        # We can iterate through combinations and break early.
        
        # To make iteration faster, we can use numpy to compute dets for a batch of combos.
        # But generating the index array for combos is the slow part in Python.
        
        # Strategy:
        # 1. If M < 20, simple loop.
        # 2. If M >= 20, generate all combos, batch det, find first zero.
        
        if M < 25:
            for combo in itertools.combinations(range(M), 4):
                sub = W[list(combo)]
                det = np.linalg.det(sub)
                if abs(det) < 0.5:
                    return False, list(combo)
            return True, None
        else:
            # Batch processing
            # Number of combos C(M, 4)
            # For M=30, C=27405.
            # For M=40, C=91390.
            # For M=50, C=230300.
            
            # Creating the index array
            # idx = np.array(list(itertools.combinations(range(M), 4)))
            # This is slow.
            
            # Alternative: Just loop but with a cap?
            # If we don't find a conflict in the first 1000 random combos, assume valid?
            # Risky.
            
            # Let's stick to the loop but it might be slow for large M.
            # However, the local search will keep M around 15-25 usually for 8x8x8?
            # The max k is likely < 30.
            # So M < 25 branch is likely sufficient.
            
            for combo in itertools.combinations(range(M), 4):
                sub = W[list(combo)]
                det = np.linalg.det(sub)
                if abs(det) < 0.5:
                    return False, list(combo)
            return True, None

    best_set = np.empty((0, 3), dtype=np.int64)
    
    np.random.seed(42)
    
    num_restarts = 100
    
    for r in range(num_restarts):
        if time.time() > deadline:
            break
            
        # 1. Greedy Initialization
        order = grid[np.random.permutation(N)]
        S = np.empty((0, 3), dtype=np.int64)
        
        for p in order:
            if time.time() > deadline:
                break
            sq_p = (p ** 2).sum()
            valid, _ = get_conflict_indices(S, p, sq_p)
            if valid:
                if len(S) == 0:
                    S = p[None, :]
                else:
                    S = np.vstack([S, p])
        
        # 2. Local Search Phase
        S_set = set(map(tuple, S))
        current_size = len(S)
        
        # Try to improve by adding or swapping
        # We will iterate a fixed number of times or until no improvement
        # Since we want max size, we prioritize additions.
        
        improved = True
        max_ls_iterations = 5000
        ls_iter = 0
        
        while improved and ls_iter < max_ls_iterations:
            if time.time() > deadline:
                break
            improved = False
            ls_iter += 1
            
            # Try to add a random point not in S
            # To make this efficient, we pick a random candidate from the grid
            # and check if it's in S. If yes, pick another.
            # Or maintain a list of available points.
            
            # Simpler: Just pick random points from grid.
            # If it's in S, skip.
            
            # We can also try to add points that were rejected in greedy?
            # For now, random sampling.
            
            for _ in range(10): # Try 10 random candidates per LS step
                if time.time() > deadline:
                    break
                
                p_idx = np.random.randint(N)
                p = grid[p_idx]
                if tuple(p) in S_set:
                    continue
                
                sq_p = sq_norms[p_idx]
                valid, conflict_idx = get_conflict_indices(S, p, sq_p)
                
                if valid:
                    # Add it
                    S = np.vstack([S, p])
                    S_set.add(tuple(p))
                    current_size += 1
                    improved = True
                    break # Restart the LS loop with new S
                else:
                    # Conflict found. Try to swap.
                    # conflict_idx contains 4 indices in S that form a bad 5-tuple with p.
                    # We can try swapping one of these 4 points with p.
                    if conflict_idx:
                        # Pick one of the conflicting points to remove
                        # Random choice among the 4
                        swap_idx = conflict_idx[np.random.randint(4)]
                        q = S[swap_idx]
                        
                        # Try swap: S' = S - q + p
                        # Check if S' is valid.
                        # This requires checking all 5-tuples in S'.
                        # Expensive. 
                        # However, we can check if removing q resolves the conflict 
                        # and doesn't create new ones.
                        # New ones would involve p and 4 points from S\{q}.
                        # Since p was conflicting with {q, a, b, c}, 
                        # removing q breaks that specific conflict.
                        # But p might conflict with other sets in S\{q}.
                        # And removing q might break other sets? No, removing points
                        # cannot create new co-spherical sets.
                        # So we only need to check if p is valid with S\{q}.
                        
                        S_minus_q = np.delete(S, swap_idx, axis=0)
                        valid_swap, _ = get_conflict_indices(S_minus_q, p, sq_p)
                        
                        if valid_swap:
                            # Perform swap
                            S = np.vstack([S_minus_q, p])
                            S_set.remove(tuple(q))
                            S_set.add(tuple(p))
                            # Size remains same
                            improved = True
                            break # Restart LS loop

        # Update best
        if len(S) > len(best_set):
            best_set = S.copy()
            # print(f"Restart {r}: New Best {len(S)}")

    return best_set.astype(np.int64)

# EVOLVE-BLOCK-END
