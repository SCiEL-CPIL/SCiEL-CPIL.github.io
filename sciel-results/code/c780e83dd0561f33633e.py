import time
import random
import numpy as np

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the exact tensor decomposition for multiplying a 3x5 matrix by a 5x7 matrix.
    Uses Strassen 2x2x2 blocks with an improved packing search (greedy + local search).
    """
    start_time = time.time()
    deadline = start_time + 145.0

    n, m, p = DIMENSIONS  # 3, 5, 7

    U_rows = n * m  # 15
    V_rows = m * p  # 35
    W_rows = p * n  # 21

    U_cols = []
    V_cols = []
    W_cols = []

    def add_term(u_entries, v_entries, w_entries):
        U_cols.append([0] * U_rows)
        V_cols.append([0] * V_rows)
        W_cols.append([0] * W_rows)
        for idx, val in u_entries.items():
            U_cols[-1][idx] = val
        for idx, val in v_entries.items():
            V_cols[-1][idx] = val
        for idx, val in w_entries.items():
            W_cols[-1][idx] = val

    # Generate all possible 2x2x2 blocks
    all_blocks = []
    for i1 in range(n - 1):
        for i2 in range(i1 + 1, n):
            for j1 in range(m - 1):
                for j2 in range(j1 + 1, m):
                    for k1 in range(p - 1):
                        for k2 in range(k1 + 1, p):
                            indices = frozenset((i, j, k) for i in (i1, i2) for j in (j1, j2) for k in (k1, k2))
                            all_blocks.append(((i1, i2, j1, j2, k1, k2), indices))
    
    # Precompute block indices as sets for faster intersection checks
    all_blocks_set = [(bp, set(ind)) for bp, ind in all_blocks]
    all_blocks_dict = {bp: set(ind) for bp, ind in all_blocks}
    
    # Precompute conflict matrix or lists for speed? 
    # 630 blocks is small enough for set intersections.

    def greedy_packing(block_list, used_indices=None):
        """Greedy: pick blocks in given order that don't conflict."""
        if used_indices is None:
            used_indices = set()
        packing = []
        for block_params, block_indices in block_list:
            if not (block_indices & used_indices):
                packing.append(block_params)
                used_indices |= block_indices
        return packing

    def local_search_improve(packing, max_iters=500):
        """
        Try to improve packing by removing 1 block and adding 2 non-conflicting blocks.
        Or removing 2 and adding 3.
        """
        if not packing:
            return packing
        
        used_indices = set()
        for bp in packing:
            used_indices |= all_blocks_dict[bp]
        
        current_size = len(packing)
        best_packing = list(packing)
        
        # List of all blocks for iteration
        block_list = all_blocks_set
        
        improved = True
        while improved and time.time() < deadline - 10.0:
            improved = False
            # Try each block in current packing as a candidate for removal
            for i in range(len(best_packing)):
                if time.time() > deadline - 10.0:
                    break
                
                removed_block = best_packing[i]
                trial_packing = best_packing[:i] + best_packing[i+1:]
                trial_used = set()
                for bp in trial_packing:
                    trial_used |= all_blocks_dict[bp]
                
                # Find all blocks that can be added to trial_packing
                # A block b can be added if b.indices & trial_used == empty
                # AND b is not in trial_packing
                addable = []
                for block_params, block_indices in block_list:
                    if block_params in trial_packing:
                        continue
                    if not (block_indices & trial_used):
                        addable.append((block_params, block_indices))
                
                # Try to find 2 disjoint blocks from addable
                # We want to add 2 blocks that are disjoint from each other.
                # If we find 2, new size = len(trial_packing) + 2 = current_size - 1 + 2 = current_size + 1
                # This is an improvement.
                
                # Optimization: If len(addable) is large, we can be smarter.
                # For now, simple O(N^2) check on addable list.
                # addable size is at most 630. 630^2 = 400k. 
                # Doing this for each of ~10 blocks in packing is 4M ops. 
                # Python might be slow. Let's limit iterations or optimize.
                
                # Better: Just check if we can add 2.
                found_pair = False
                # Sort addable by something? No, just check pairs.
                # To speed up, we can break early if found.
                
                # If we just want ANY improvement, adding 1 block that was previously blocked is also good?
                # No, we removed 1. Adding 1 keeps size same. We need net gain.
                # So we need to add 2.
                
                # Let's try to find 2 disjoint blocks in addable.
                # We can iterate through addable and maintain a list of "compatible" blocks.
                # But 630 is small.
                
                # To avoid O(N^2) in Python which is slow, let's limit the search depth or use a simpler heuristic.
                # Maybe just try to add 1 block that allows adding another?
                
                # Alternative: Randomly pick 2 from addable? No, they must be disjoint.
                
                # Let's use a simple check:
                # If len(addable) >= 2, it's *possible* they are disjoint.
                # We will check pairs.
                
                # To keep it fast, we'll only check a subset of pairs or break early.
                # Actually, 630 blocks is small. 630*630/2 = 200k checks. 
                # 10 blocks in packing * 200k = 2M checks per iteration.
                # 500 iterations = 1B checks. Too slow in Python.
                
                # Optimization: Precompute disjointness? 
                # Or, just do a few random swaps?
                
                # Let's try a different move: 
                # Remove 1 block. Add 1 block that is disjoint from the rest.
                # This maintains size. Not useful for improvement.
                
                # We need a faster way to find 2 disjoint blocks in a set.
                # We can pick a random block from addable, then find any block in addable disjoint from it.
                
                if len(addable) < 2:
                    continue
                    
                # Try a few random pairs from addable
                # Or iterate through addable, for each, check if there's another disjoint one.
                # We can stop as soon as we find one pair.
                
                pair_found = None
                # To speed up, we can just check the first few? No, we need a valid pair.
                # Let's iterate through addable.
                for a_idx, (a_params, a_indices) in enumerate(addable):
                    # Check if any other block in addable is disjoint from a_indices
                    # We can break early if found
                    for b_idx in range(a_idx + 1, len(addable)):
                        b_params, b_indices = addable[b_idx]
                        if not (a_indices & b_indices):
                            pair_found = (a_params, b_params)
                            break
                    if pair_found:
                        break
                
                if pair_found:
                    new_packing = trial_packing + [pair_found[0], pair_found[1]]
                    best_packing = new_packing
                    current_size = len(new_packing)
                    improved = True
                    break # Restart the outer loop with the new packing
        
        return best_packing

    # Phase 1: Greedy with random shuffles
    best_num_blocks = 0
    best_packing = []
    
    # Increase trials
    num_trials = 2000
    for t in range(num_trials):
        if time.time() > deadline - 60.0:
            break
        
        shuffled = list(all_blocks_set)
        random.shuffle(shuffled)
        packing = greedy_packing(shuffled)
        
        if len(packing) > best_num_blocks:
            best_num_blocks = len(packing)
            best_packing = packing
            if best_num_blocks >= 10: # Theoretical max is probably around 8-9?
                break

    # Phase 2: Local search improvement
    if best_num_blocks < 10:
        improved = local_search_improve(best_packing, max_iters=100)
        if len(improved) > best_num_blocks:
            best_num_blocks = len(improved)
            best_packing = improved

    # Construct U, V, W based on the best packing found
    def add_strassen_block(i1, i2, j1, j2, k1, k2):
        a11 = i1 * m + j1
        a12 = i1 * m + j2
        a21 = i2 * m + j1
        a22 = i2 * m + j2
        
        b11 = j1 * p + k1
        b12 = j1 * p + k2
        b21 = j2 * p + k1
        b22 = j2 * p + k2
        
        c11 = k1 * n + i1
        c12 = k2 * n + i1
        c21 = k1 * n + i2
        c22 = k2 * n + i2
        
        add_term({a11: 1, a22: 1}, {b11: 1, b22: 1}, {c11: 1, c22: 1})
        add_term({a21: 1, a22: 1}, {b11: 1}, {c21: 1, c22: -1})
        add_term({a11: 1}, {b12: 1, b22: -1}, {c12: 1, c22: 1})
        add_term({a22: 1}, {b21: 1, b11: -1}, {c11: 1, c21: 1})
        add_term({a11: 1, a12: 1}, {b22: 1}, {c11: -1, c12: 1})
        add_term({a21: 1, a11: -1}, {b11: 1, b12: 1}, {c22: 1})
        add_term({a12: 1, a22: -1}, {b21: 1, b22: 1}, {c11: 1})

    covered = set()
    for block_params in best_packing:
        i1, i2, j1, j2, k1, k2 = block_params
        add_strassen_block(i1, i2, j1, j2, k1, k2)
        for i in (i1, i2):
            for j in (j1, j2):
                for k in (k1, k2):
                    covered.add((i, j, k))

    for i in range(n):
        for j in range(m):
            for k in range(p):
                if time.time() > deadline:
                    break
                if (i, j, k) not in covered:
                    add_term({i * m + j: 1}, {j * p + k: 1}, {k * n + i: 1})

    rank = len(U_cols)
    
    U = [[0] * rank for _ in range(U_rows)]
    V = [[0] * rank for _ in range(V_rows)]
    W = [[0] * rank for _ in range(W_rows)]
    
    for c in range(rank):
        for r in range(U_rows):
            U[r][c] = U_cols[c][r]
        for r in range(V_rows):
            V[r][c] = V_cols[c][r]
        for r in range(W_rows):
            W[r][c] = W_cols[c][r]

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
