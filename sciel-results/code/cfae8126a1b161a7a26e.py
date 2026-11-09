import itertools
import random
import time

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Precomputation for speed
    # 1. List of all t-subsets (tuples)
    all_t_subsets = list(itertools.combinations(range(v), t))
    total_t_subsets = len(all_t_subsets)
    
    # Map tuple to index
    subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    idx_to_subset = [None] * total_t_subsets
    for s, i in subset_to_idx.items():
        idx_to_subset[i] = s
        
    # 2. Precompute extension table: (C_tuple, x) -> S_idx
    # C is a (t-1)-subset, x is a point not in C. S = C U {x}
    all_t_minus_1 = list(itertools.combinations(range(v), t-1))
    ext_table = {}
    for C in all_t_minus_1:
        C_set = set(C)
        for x in range(v):
            if x in C_set:
                continue
            S = tuple(sorted(C + (x,)))
            ext_table[(C, x)] = subset_to_idx[S]
            
    # Lower bound
    num_k_subsets = len(list(itertools.combinations(range(k), t)))
    lower_bound = -(-total_t_subsets // num_k_subsets)
    
    best_blocks = None
    best_len = float('inf')
    
    # Use a fixed seed for reproducibility in evaluation? No, random is fine.
    # But we should ensure we don't spend too much time in precomputation if v is large?
    # v=19 is small.
    
    while time.time() < deadline:
        # Create a random permutation
        perm = list(range(v))
        random.shuffle(perm)
        inv_perm = [0] * v
        for i, p in enumerate(perm):
            inv_perm[p] = i
            
        # Initialize uncovered set with indices of all t-subsets
        # We need to permute the points in the subsets to make the "min" selection random
        # But since we use indices, we need to map the permuted subset to its index in the original space?
        # No, the logic of "uncovered" should be based on the actual points.
        # If we permute points, we are effectively solving the problem on a relabeled set.
        # The structure is isomorphic.
        # However, the greedy choice `min(uncovered)` depends on the labels.
        # If we permute the labels, `min` will pick a different subset structurally.
        # So we should work with the permuted labels for the greedy process,
        # then map back.
        
        # But our `ext_table` and `subset_to_idx` are built on `0..v-1`.
        # If we permute points, the subsets in the permuted space are different tuples.
        # We need to map a permuted subset to an index?
        # Actually, we can just run the greedy algorithm on the permuted labels directly,
        # but we need to check coverage against the *original* set of all t-subsets?
        # No, the set of all t-subsets of `0..v-1` is invariant under permutation of labels.
        # The *specific* subsets change, but the collection is the same.
        # So `uncovered` should contain indices of the *permuted* subsets?
        # But `ext_table` keys are tuples of `0..v-1`.
        
        # Approach:
        # 1. Generate all t-subsets of `range(v)`.
        # 2. Apply permutation to each subset to get `permuted_subsets`.
        # 3. `uncovered` is a set of indices corresponding to these permuted subsets?
        #    No, `subset_to_idx` maps original tuples to indices.
        #    If we use permuted tuples, they are not in `subset_to_idx`.
        
        # Alternative Approach:
        # Don't permute the subsets for the data structure.
        # Instead, permute the *order* in which we consider candidates or subsets?
        # The original code permuted the subsets to make `min` random.
        # We can achieve the same by shuffling the indices in `uncovered`?
        # No, `min` on a set of indices is deterministic.
        
        # Let's stick to the original logic but optimized:
        # We will permute the point labels in the greedy process.
        # This means we treat the points as `perm[0], ..., perm[v-1]`?
        # No, the points are `0..v-1`. The permutation `perm` maps `i -> perm[i]`.
        # If we want to solve on permuted labels, we should map the problem to `0..v-1`
        # such that the "structure" is scrambled.
        
        # Simpler: Just use the original indices, but break ties randomly?
        # The original code relied on `min(uncovered)` being different due to permuted labels.
        # If we use indices, `min(uncovered)` is always the same index (0) initially.
        # This makes the greedy deterministic and might get stuck.
        
        # To maintain randomness with indices:
        # We can shuffle the *mapping* from index to tuple?
        # Or, we can just randomly pick an uncovered subset instead of `min`.
        # Picking a random uncovered subset is a valid greedy heuristic (Random Greedy).
        # It might be slower to converge but avoids the deterministic trap.
        # And it's fast: `random.choice(list(uncovered))` is O(N).
        # N=11k. 335 blocks. 335 * 11k = 3.7M ops. Acceptable.
        
        # Let's use Random Greedy Selection for the starting subset.
        
        uncovered = set(range(total_t_subsets))
        blocks = []
        
        while uncovered:
            # Pick a random uncovered subset
            # To avoid O(N) list conversion every time, we could maintain a list?
            # But `uncovered` changes.
            # For 11k items, `random.choice(list(uncovered))` is ~0.1ms?
            # 335 * 0.1ms = 33ms. Fine.
            
            start_idx = random.choice(list(uncovered))
            start_subset = idx_to_subset[start_idx]
            block = list(start_subset)
            block_set = set(block)
            
            while len(block) < k:
                best_gain = -1
                best_candidates = []
                
                # Precompute combinations of block of size t-1
                # We need to check gain for each x not in block
                # Gain(x) = count of C in Combinations(block, t-1) such that (C, x) in ext_table and idx in uncovered
                
                # Optimization: Iterate over C in Combinations(block, t-1)
                # and update gains for all x?
                # No, we need to know which x is not in block.
                
                gains = [0] * v
                for C in itertools.combinations(block, t - 1):
                    # For each C, check all x not in block
                    # But we can just check all x and filter?
                    # Or iterate x in range(v) and skip if in block?
                    # Since v=19, iterating all x is cheap.
                    for x in range(v):
                        if x in block_set:
                            continue
                        key = (C, x)
                        s_idx = ext_table.get(key)
                        if s_idx is not None and s_idx in uncovered:
                            gains[x] += 1
                
                # Find max gain
                max_gain = max(gains)
                if max_gain == 0:
                    # No gain, break? Or add random?
                    # If no gain, we can break or add a random point.
                    # Breaking might leave block smaller than k?
                    # The contract says block must be size k.
                    # So we must add points even if gain is 0.
                    # Pick random from candidates with gain 0?
                    # Or just pick the first one?
                    pass
                    
                candidates = [x for x in range(v) if x not in block_set and gains[x] == max_gain]
                if not candidates:
                    # Should not happen if v > k
                    break
                    
                next_x = random.choice(candidates)
                block.append(next_x)
                block_set.add(next_x)
                
            block.sort()
            blocks.append(block)
            
            # Remove all t-subsets of this block from uncovered
            for S in itertools.combinations(block, t):
                s_idx = subset_to_idx[S]
                uncovered.discard(s_idx)
                
        # Map blocks back?
        # Since we didn't permute the labels in the logic (we used original 0..v-1),
        # the blocks are already in original labels.
        # The randomness comes from `random.choice` for start subset and ties.
        
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = [list(b) for b in blocks]
            
        if best_len <= lower_bound:
            break
            
    return best_blocks
