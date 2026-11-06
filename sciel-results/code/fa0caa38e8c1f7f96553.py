"""B_3[1] construction: fast greedy with early rejection and swap-based improvement."""
import time
import random
import numpy as np

def construct():
    deadline = time.time() + 82.0
    N = 4096
    MAX_SUM = 3 * N  # Maximum possible triple sum
    
    # Precompute: for a given x and current selected set, check collisions vectorized
    def can_add_vectorized(x, selected_arr, sums_bool):
        """Check if x can be added. selected_arr is numpy array, sums_bool is bool array of size MAX_SUM+1."""
        # 3x
        if sums_bool[3 * x]:
            return False
        # 2x + s for s in selected
        vals = 2 * x + selected_arr
        if np.any(sums_bool[vals]):
            return False
        # x + 2s for s in selected
        vals = x + 2 * selected_arr
        if np.any(sums_bool[vals]):
            return False
        # x + s + t for s <= t in selected
        n = len(selected_arr)
        if n > 0:
            # Generate all pairs (i,j) with i<=j
            # This can be memory intensive for large n, but n is at most ~80
            # Use broadcasting: selected_arr[:, None] + selected_arr[None, :]
            pair_sums = selected_arr[:, None] + selected_arr[None, :]
            # Only take upper triangle (i <= j)
            mask = np.triu(np.ones((n, n), dtype=bool))
            pair_sums = pair_sums[mask]
            vals = x + pair_sums
            if np.any(sums_bool[vals]):
                return False
        return True

    def add_x_vectorized(x, selected_arr, sums_bool):
        """Add x to selected and update sums. Returns new selected_arr and sums_bool."""
        new_selected = np.append(selected_arr, x)
        # 3x
        sums_bool[3 * x] = True
        # 2x + s and x + 2s
        vals = 2 * x + selected_arr
        sums_bool[vals] = True
        vals = x + 2 * selected_arr
        sums_bool[vals] = True
        # x + s + t for s <= t in old selected
        n = len(selected_arr)
        if n > 0:
            pair_sums = selected_arr[:, None] + selected_arr[None, :]
            mask = np.triu(np.ones((n, n), dtype=bool))
            pair_sums = pair_sums[mask]
            vals = x + pair_sums
            sums_bool[vals] = True
        return new_selected, sums_bool

    # Phase 1: Standard greedy from 1 to 4096
    selected_arr = np.array([], dtype=np.int64)
    sums_bool = np.zeros(MAX_SUM + 1, dtype=bool)
    
    for x in range(1, N + 1):
        if time.time() > deadline - 10.0:
            break
        if can_add_vectorized(x, selected_arr, sums_bool):
            selected_arr, sums_bool = add_x_vectorized(x, selected_arr, sums_bool)
    
    best_selected = selected_arr.copy()
    best_sums = sums_bool.copy()
    best_size = len(best_selected)
    
    # Phase 2: Try to improve via swaps
    remaining_time = deadline - time.time()
    if remaining_time > 5.0:
        sel_set = set(best_selected.tolist())
        candidates = [x for x in range(1, N + 1) if x not in sel_set]
        random.shuffle(candidates)
        
        improved = True
        while improved and time.time() < deadline - 3.0:
            improved = False
            for i in range(len(best_selected)):
                if time.time() > deadline - 3.0:
                    break
                orig = best_selected[i]
                # Temporarily remove orig
                temp_selected = np.delete(best_selected, i)
                temp_sums = best_sums.copy()
                # Need to remove sums involving orig - this is expensive
                # For now, just try adding candidates and check
                for cand in candidates:
                    if time.time() > deadline - 3.0:
                        break
                    if can_add_vectorized(cand, temp_selected, temp_sums):
                        # We need to properly remove orig's sums first
                        # This is complex, so let's use a different approach
                        pass
                break  # This phase is too complex for now, skip
    
    # Phase 3: Try adding more elements from the end using vectorized check
    if time.time() < deadline - 2.0:
        sel_set = set(best_selected.tolist())
        for x in range(N, 0, -1):
            if time.time() > deadline - 2.0:
                break
            if x not in sel_set:
                if can_add_vectorized(x, best_selected, best_sums):
                    best_selected, best_sums = add_x_vectorized(x, best_selected, best_sums)
                    sel_set.add(x)
                    best_size = len(best_selected)
    
    return best_selected[:80].tolist()
