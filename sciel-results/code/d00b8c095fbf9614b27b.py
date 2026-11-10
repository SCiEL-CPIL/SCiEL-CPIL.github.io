import numpy as np
import itertools
import time

def no_5_on_a_sphere8() -> np.ndarray:
    n = 8
    grid = np.array(list(itertools.product(range(n), range(n), range(n))), dtype=np.int64)
    N = len(grid)
    # Precompute squared norms
    sq_norms = np.sum(grid ** 2, axis=1).astype(np.float64)
    
    # Deadline
    start_time = time.time()
    time_limit = 600.0  # Safety margin for 630s limit
    
    def can_add(selected_indices: np.ndarray, cand_idx: int) -> bool:
        """
        Check if adding cand_idx to selected_indices keeps the set valid.
        selected_indices: array of indices into grid.
        """
        m = len(selected_indices)
        if m < 4:
            return True
        
        p_cand = grid[cand_idx]
        sq_cand = sq_norms[cand_idx]
        
        # Get coordinates of selected points
        p_sel = grid[selected_indices]
        sq_sel = sq_norms[selected_indices]
        
        # Construct the matrix W of shape (m, 4)
        # Rows: (p_sel - p_cand, sq_sel - sq_cand)
        W = np.empty((m, 4), dtype=np.float64)
        W[:, 0:3] = p_sel - p_cand
        W[:, 3] = sq_sel - sq_cand
        
        # We need to check all combinations of 4 rows
        # If m is small, itertools.combinations is fine.
        # If m is larger, we might need to be careful.
        # Max m expected ~20-30. C(30,4) = 27405. Fast.
        
        # Vectorized check:
        # Generate all combinations of 4 indices from 0..m-1
        # This is expensive if done every time. 
        # Alternative: Use the fact that if the 4x4 matrix of the *first* 4 is singular, it's bad.
        # But we need ALL 4-subsets to be non-singular.
        
        # Optimization: 
        # If m is large, this is the bottleneck.
        # Let's try to be smart: 
        # The condition "no 5 co-spherical" is equivalent to saying that the lifted points 
        # in R^4 are in "general position" with respect to hyperplanes? No.
        # 5 points p0..p4 are co-spherical iff det([p_i, |p_i|^2, 1]_{i=0..4}) = 0.
        # This is equivalent to the 5 lifted vectors being linearly dependent in R^5?
        # Let v_i = (p_i, |p_i|^2, 1). 
        # Then det(V) = 0 for V being 5x5 matrix of v_i^T.
        
        # In our incremental check:
        # We have p_cand and 4 selected points q1..q4.
        # They are co-spherical iff det of 5x5 matrix is 0.
        # The determinant can be expanded along the row for p_cand.
        # This leads to the 4x4 determinant check in the problem statement.
        
        # So we MUST check all C(m, 4) subsets.
        
        if m < 4:
             return True
             
        # For efficiency, if m is large, we can short-circuit.
        # But let's just do the vectorized det check.
        
        # To avoid generating combinations every time (which creates overhead),
        # we can cache combinations for specific m?
        # Or just use itertools.
        
        # Let's try to vectorize the determinant calculation.
        # We need indices.
        # np.combinations doesn't exist.
        
        # If m <= 20, itertools.combinations is fast enough in Python?
        # 4845 combinations. Looping in Python over 4845 items is slow (~1ms?).
        # 1ms per check * 512 candidates * many swaps = too slow?
        # 512 * 1ms = 0.5s per "try to add" phase.
        # If we do 1000 swaps, that's 500s. Too slow.
        
        # We need a faster `can_add`.
        
        # Idea: 
        # The condition is that the rank of the matrix formed by any 4 rows is 4.
        # This is equivalent to the matrix W (m x 4) having the property that 
        # no 4 rows are linearly dependent.
        # This implies that the null space of W is trivial? No, W is m x 4.
        # If m > 4, rows are in R^4.
        # If there are 4 dependent rows, they span a subspace of dim < 4.
        
        # Can we check this faster?
        # If we pick 4 random rows, if they are independent, it doesn't mean all are.
        # But if we find one dependent set, we return False.
        
        # Let's optimize the combination generation.
        # Precompute combinations for m up to, say, 30.
        
        pass # Placeholder for logic
        
    # Re-implementing can_add with precomputed combos
    
    # Precompute combinations for m=4 to m=30
    max_m = 30
    combos_cache = {}
    for m in range(4, max_m + 1):
        combos_cache[m] = np.array(list(itertools.combinations(range(m), 4)))
        
    def can_add_fast(selected_indices: np.ndarray, cand_idx: int) -> bool:
        m = len(selected_indices)
        if m < 4:
            return True
        
        if m > max_m:
             # Fallback to slow check or reject? 
             # If m is this large, it's likely invalid or we are in a weird state.
             # Let's assume m stays small.
             pass
             
        p_cand = grid[cand_idx]
        sq_cand = sq_norms[cand_idx]
        
        p_sel = grid[selected_indices]
        sq_sel = sq_norms[selected_indices]
        
        W = np.empty((m, 4), dtype=np.float64)
        W[:, 0:3] = p_sel - p_cand
        W[:, 3] = sq_sel - sq_cand
        
        combos = combos_cache[m]
        # W[combos] gives shape (C, 4, 4)
        # np.linalg.det on (C, 4, 4) works?
        # Yes, np.linalg.det accepts (..., M, M).
        
        dets = np.linalg.det(W[combos])
        
        # Check if any determinant is close to 0
        # The problem says |det| < 0.5 is invalid.
        # Since coordinates are integers, det is integer.
        # So |det| < 0.5 implies det == 0.
        return not np.any(np.abs(dets) < 0.5)
        
    def greedy_init():
        order = np.random.permutation(N)
        selected = []
        for idx in order:
            if can_add_fast(np.array(selected), idx):
                selected.append(idx)
        return np.array(selected)
        
    def local_search(selected_arr):
        selected = list(selected_arr)
        unselected = [i for i in range(N) if i not in selected]
        
        improved = True
        while improved and (time.time() - start_time) < time_limit:
            improved = False
            # Try to add
            for idx in unselected:
                if can_add_fast(np.array(selected), idx):
                    selected.append(idx)
                    unselected.remove(idx)
                    improved = True
                    break # Restart add phase
            
            if not improved:
                # Try swaps
                # To avoid O(N^2) full scan every time, we can randomize or limit.
                # But N=512, N^2 = 260k. 
                # Each check is ~1ms? 
                # 260k * 1ms = 260s. Too slow for one pass.
                
                # We need a faster swap check or fewer swaps.
                
                # Optimization: 
                # Only try swaps for a subset of unselected points?
                # Or, check `can_add` for all unselected points against `selected` without `p_out`?
                # No, `p_out` changes the set.
                
                # Let's try to be smarter:
                # If we remove p_out, the set S' is valid.
                # We want to know if p_in can be added to S'.
                
                # Maybe we can limit the number of swaps per iteration?
                # Or use a heuristic: try random swaps.
                
                # Let's try random swaps to escape local minima.
                for _ in range(100): # Limit swaps per phase
                    if (time.time() - start_time) > time_limit: break
                    p_out_idx = np.random.randint(len(selected))
                    p_in_idx = unselected[np.random.randint(len(unselected))]
                    
                    # Check if swap is valid
                    # S' = selected \ {p_out}
                    # Check if p_in can be added to S'
                    
                    # Construct S' indices
                    # This is a bit slow if we do it via list slicing every time.
                    # But m is small.
                    
                    # Create a view or copy of selected without p_out
                    # We can just pass a filtered array.
                    # can_add_fast expects an array of indices.
                    
                    # Optimization: 
                    # can_add_fast(selected_arr, cand)
                    # We need selected_arr to be selected minus p_out.
                    
                    # Let's just do it.
                    # It's 100 checks per "phase". 
                    # 100 * 1ms = 0.1s.
                    # If we need 1000 phases, 100s. Acceptable.
                    
                    sel_list = selected
                    # Create array of selected indices excluding p_out_idx
                    # This is O(m).
                    sel_minus_out = np.array([sel_list[i] for i in range(len(sel_list)) if i != p_out_idx])
                    
                    if can_add_fast(sel_minus_out, p_in_idx):
                        # Perform swap
                        selected[p_out_idx] = p_in_idx
                        unselected.remove(p_in_idx)
                        # p_out is now unselected
                        unselected.append(sel_list[p_out_idx])
                        improved = True
                        break # Restart add phase
                        
        return np.array(selected)
        
    # Main Loop
    best_set = None
    best_len = 0
    
    np.random.seed(42)
    
    # Run multiple restarts
    num_restarts = 50 # We have time
    
    for i in range(num_restarts):
        if (time.time() - start_time) > time_limit:
            break
            
        # Greedy init
        current_set = greedy_init()
        
        # Local Search
        current_set = local_search(current_set)
        
        if len(current_set) > best_len:
            best_len = len(current_set)
            best_set = current_set
            
        # If we found a very large set, maybe stop? 
        # Max possible is 512. Unlikely to reach that.
        # If best_len > 25, it's good.
        
    if best_set is None:
        # Fallback
        best_set = greedy_init()
        
    return grid[best_set]
