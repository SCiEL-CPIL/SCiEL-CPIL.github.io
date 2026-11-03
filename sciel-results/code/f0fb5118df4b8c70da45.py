import itertools
import time
import random

def construct(v, k, t):
    points = range(v)
    
    # 1. Generate all t-subsets and map them to indices
    t_subsets = list(itertools.combinations(points, t))
    t_subset_index = {s: i for i, s in enumerate(t_subsets)}
    n_t_subsets = len(t_subsets)
    all_mask = (1 << n_t_subsets) - 1
    
    # 2. Generate all k-subsets (candidate blocks)
    k_subsets = list(itertools.combinations(points, k))
    n_blocks = len(k_subsets)
    
    # 3. Compute bitmask for each k-subset
    masks = [0] * n_blocks
    # Also precompute which blocks cover each t-subset for B&B heuristic
    # This is a list of lists: blocks_covering_t_subset[t_idx] = [block_idx, ...]
    blocks_covering_t_subset = [[] for _ in range(n_t_subsets)]
    
    for i, block in enumerate(k_subsets):
        mask = 0
        for t_sub in itertools.combinations(block, t):
            idx = t_subset_index[t_sub]
            mask |= (1 << idx)
            blocks_covering_t_subset[idx].append(i)
        masks[i] = mask
        
    # 4. Greedy Set Cover to get initial Upper Bound (UB)
    uncovered = all_mask
    selected_greedy = []
    used_blocks = [False] * n_blocks
    
    while uncovered:
        best_gain = -1
        best_idx = -1
        for i in range(n_blocks):
            if used_blocks[i]:
                continue
            gain = (masks[i] & uncovered).bit_count()
            if gain > best_gain:
                best_gain = gain
                best_idx = i
                if gain == uncovered.bit_count():
                    break
        if best_idx == -1:
            break
        selected_greedy.append(best_idx)
        used_blocks[best_idx] = True
        uncovered &= ~masks[best_idx]
        
    current_best = len(selected_greedy)
    best_solution = selected_greedy[:]

    # 5. Remove Redundant Blocks
    def try_remove_redundant(blocks_list):
        removed = True
        while removed:
            removed = False
            for i in range(len(blocks_list) - 1, -1, -1):
                block_mask = masks[blocks_list[i]]
                # Check if union of others covers all
                # Optimization: If block_mask is subset of union of others, it's redundant.
                # We can check if (all_mask & ~block_mask) is covered by others? No.
                # We need to check if every bit in block_mask is covered by some other block in the list.
                # Let union_others = OR of masks of all blocks except i.
                # If (block_mask & ~union_others) == 0, then i is redundant.
                
                union_others = 0
                for j in range(len(blocks_list)):
                    if j != i:
                        union_others |= masks[blocks_list[j]]
                        
                if (block_mask & ~union_others) == 0:
                    blocks_list.pop(i)
                    removed = True
                    break
        return blocks_list

    best_solution = try_remove_redundant(best_solution)
    current_best = len(best_solution)

    # 6. Branch and Bound with Time Limit
    start_time = time.time()
    time_limit = 100.0 # Keep buffer for overhead
    
    # Precompute max possible gain for lower bound
    max_gain_const = (1 << t) # Wait, max gain is C(k,t)
    from math import comb
    max_gain_const = comb(k, t)
    
    def branch_and_bound(uncovered_mask, current_depth, current_path, used_set):
        nonlocal current_best, best_solution
        
        if not uncovered_mask:
            if current_depth < current_best:
                current_best = current_depth
                best_solution = current_path[:]
            return
            
        # Time Check
        if time.time() - start_time > time_limit:
            return
            
        # Lower Bound Pruning
        # ceil(uncovered_count / max_gain)
        uncovered_count = uncovered_mask.bit_count()
        lb = (uncovered_count + max_gain_const - 1) // max_gain_const
        
        if current_depth + lb >= current_best:
            return
            
        # Heuristic: Pick the uncovered t-subset with fewest available covering blocks
        # To do this efficiently, we iterate through set bits of uncovered_mask
        # and count available blocks.
        # This is O(uncovered_count * avg_degree).
        # avg_degree is ~3000. uncovered_count can be large initially.
        # This might be slow.
        
        # Alternative: Pick the first set bit.
        # Let's try the first set bit for speed, but maybe sample a few?
        # Let's stick to first set bit for now to ensure speed.
        target_bit = uncovered_mask & -uncovered_mask
        target_idx = target_bit.bit_length() - 1
        
        # Get candidates: blocks covering target_idx that are not in used_set
        candidates = []
        for block_idx in blocks_covering_t_subset[target_idx]:
            if block_idx not in used_set:
                # Gain must be > 0 (it covers target_bit, so gain >= 1)
                gain = (masks[block_idx] & uncovered_mask).bit_count()
                candidates.append((gain, block_idx))
                
        if not candidates:
            return # Dead end
            
        # Sort by gain descending
        candidates.sort(key=lambda x: x[0], reverse=True)
        
        # Pruning candidates:
        # If gain is 0, skip (already handled by gain > 0 in candidate selection? No, gain can be 1)
        
        for gain, block_idx in candidates:
            if current_depth + 1 >= current_best:
                break # Even if we pick this, depth+1 >= UB, so no better solution possible?
                      # Wait, if depth+1 == UB, we might find a solution of size UB?
                      # We want strictly less than current_best.
                      # So if depth + 1 >= current_best, we can prune this branch?
                      # Yes, because we are looking for < current_best.
            
            new_uncovered = uncovered_mask & ~masks[block_idx]
            
            # Recursive call
            # used_set is a frozenset or set? Set is faster for add/discard.
            used_set.add(block_idx)
            branch_and_bound(new_uncovered, current_depth + 1, current_path + [block_idx], used_set)
            used_set.discard(block_idx)
            
            # If we found a solution, current_best might have decreased
            # The loop condition `current_depth + 1 >= current_best` will handle pruning subsequent iterations if UB improved.

    # Initialize B&B
    # We start with the empty set? Or can we start from the greedy solution and try to improve?
    # Starting from empty is standard B&B.
    # However, the search space is huge.
    # We can use the greedy solution to set a tight UB.
    # We can also use "Reverse Search" or "Local Search" to find better solutions first.
    
    # Let's try a few random restarts of greedy to improve UB before B&B.
    for _ in range(5):
        uncovered = all_mask
        selected = []
        used = [False] * n_blocks
        
        while uncovered:
            best_gain = -1
            best_idx = -1
            # Randomized tie-breaking or perturbation?
            # Just standard greedy is okay for now.
            for i in range(n_blocks):
                if used[i]:
                    continue
                gain = (masks[i] & uncovered).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i
                    if gain == uncovered.bit_count():
                        break
            if best_idx == -1:
                break
            selected.append(best_idx)
            used[best_idx] = True
            uncovered &= ~masks[best_idx]
            
        selected = try_remove_redundant(selected)
        if len(selected) < current_best:
            current_best = len(selected)
            best_solution = selected[:]
            
        if time.time() - start_time > time_limit * 0.5:
            break

    # Now run B&B
    # To make B&B effective, we should probably not start from depth 0 if the solution is ~300 blocks.
    # B&B from depth 0 will be extremely slow.
    # However, the pruning `current_depth + lb >= current_best` is very strong if `lb` is good.
    # lb = ceil(uncovered / max_gain).
    # Initially, uncovered = C(19,5) = 11628. max_gain = C(8,5) = 56.
    # lb = ceil(11628/56) = 208.
    # If current_best is 300, we explore depth 0.
    # At depth 1, uncovered drops.
    
    # The main issue is the branching factor.
    # At depth 0, we pick a t-subset. It has ~3000 covering blocks.
    # We sort them by gain.
    # We iterate through them.
    # This is 3000 recursive calls at depth 0.
    # Each call goes to depth 1.
    # This is feasible if pruning is effective.
    
    # Let's run B&B.
    initial_used = set()
    branch_and_bound(all_mask, 0, [], initial_used)

    return [list(k_subsets[i]) for i in best_solution]
