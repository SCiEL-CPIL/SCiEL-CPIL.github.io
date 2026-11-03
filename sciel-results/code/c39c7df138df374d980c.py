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
    for i, block in enumerate(k_subsets):
        mask = 0
        for t_sub in itertools.combinations(block, t):
            idx = t_subset_index[t_sub]
            mask |= (1 << idx)
        masks[i] = mask
        
    # Precompute: For each t-subset index, list of blocks that cover it.
    # This is crucial for fast B&B variable selection.
    # Memory: n_t_subsets lists. Total entries = n_blocks * C(k,t).
    # For v=19, k=8, t=5:
    # n_t_subsets = C(19,5) = 11628
    # n_blocks = C(19,8) = 75582
    # C(8,5) = 56
    # Total entries ~ 75582 * 56 ~ 4.2 million.
    # Storing as list of lists of ints is feasible in Python if optimized, or use arrays.
    # Let's use a list of lists.
    blocks_for_t_subset = [[] for _ in range(n_t_subsets)]
    for i in range(n_blocks):
        mask = masks[i]
        # Iterate through set bits in mask
        m = mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            blocks_for_t_subset[idx].append(i)
            m ^= lsb
            
    # 4. Greedy Set Cover to get initial Upper Bound (UB)
    def greedy_cover():
        uncovered = all_mask
        selected = []
        used = [False] * n_blocks
        
        while uncovered:
            best_gain = -1
            best_idx = -1
            # Optimization: Only check blocks that cover *some* uncovered bits?
            # We can iterate all blocks, but that's 75k iterations per step.
            # Steps ~ 300. 75k * 300 = 22.5M ops. Fast enough in Python?
            # Bit count is fast.
            for i in range(n_blocks):
                if used[i]:
                    continue
                gain = (masks[i] & uncovered).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i
            if best_idx == -1:
                break
            selected.append(best_idx)
            used[best_idx] = True
            uncovered &= ~masks[best_idx]
            
        return selected

    initial_greedy = greedy_cover()
    current_best = len(initial_greedy)
    best_solution = initial_greedy[:]
    
    # 5. Branch and Bound
    # State: uncovered_mask, current_path (list of block indices)
    # Heuristic: Pick the uncovered t-subset with the FEWEST available blocks.
    
    def branch_and_bound(uncovered_mask, current_path):
        nonlocal current_best, best_solution
        
        if not uncovered_mask:
            if len(current_path) < current_best:
                current_best = len(current_path)
                best_solution = current_path[:]
            return
            
        # Lower Bound
        # Simple bound: ceil(uncovered_count / max_gain)
        # max_gain is C(k,t).
        # Better bound: Sum of 1/freq? No, freq changes as we pick blocks.
        # Let's use a greedy relaxation lower bound.
        # Greedy LB: Simulate greedy on the remaining uncovered set, but just count steps?
        # Too slow.
        # Use: uncovered_count // max_possible_gain_per_block + 1
        # max_possible_gain_per_block is at most C(k,t) = 56.
        # This is weak.
        
        # Stronger LB:
        # Find the uncovered t-subset with the minimum number of *available* blocks.
        # If min_count == 0, dead end.
        # If min_count == 1, we MUST pick that block.
        
        min_count = n_blocks
        target_idx = -1
        # Iterate through uncovered bits
        m = uncovered_mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            # Count available blocks for this idx
            # We need to know which blocks are *not* in current_path?
            # Actually, blocks in current_path are "used".
            # But any block that covers idx is a candidate.
            # The constraint is that we can pick *any* block from blocks_for_t_subset[idx].
            # Wait, if a block is already in current_path, it's used.
            # But we are branching on *new* blocks.
            # So we need the count of blocks in blocks_for_t_subset[idx] that are NOT in current_path.
            # This is expensive to check every time.
            
            # Alternative: Just use the total count as a proxy? No, if the only block covering idx is already used, we are dead.
            # But if it's used, it would have covered idx already, so idx wouldn't be in uncovered_mask.
            # So, if idx is in uncovered_mask, NO block in current_path covers it.
            # Therefore, ALL blocks in blocks_for_t_subset[idx] are available (not in current_path).
            # So the count is just len(blocks_for_t_subset[idx]).
            
            cnt = len(blocks_for_t_subset[idx])
            if cnt < min_count:
                min_count = cnt
                target_idx = idx
            m ^= lsb
        
        if min_count == 0:
            return # Dead end
            
        # Pruning:
        # Lower bound on remaining blocks:
        # Each block covers at most C(k,t) uncovered bits.
        # But they overlap.
        # A better bound: The number of uncovered bits is U.
        # Each new block can cover at most C(k,t) bits.
        # So we need at least ceil(U / C(k,t)) blocks.
        # This is weak.
        
        # Let's try a slightly better bound:
        # Greedy estimate on the subproblem?
        # Too slow.
        
        # Let's rely on the min_count for branching order and dead ends.
        # And depth pruning: if len(current_path) >= current_best, prune.
        if len(current_path) >= current_best:
            return
            
        # Branch on target_idx
        candidates = blocks_for_t_subset[target_idx]
        
        # Sort candidates by "gain" (how many other uncovered bits they cover)
        # This helps find good solutions earlier to update UB.
        def get_gain(block_idx):
            return (masks[block_idx] & uncovered_mask).bit_count()
            
        # Sorting 1000+ candidates every node is expensive.
        # Limit sorting to top candidates or use a heuristic.
        # For small min_count, the list is small.
        # For large min_count, the list is large.
        # If min_count is large, we might have many options.
        
        # Optimization: If min_count is large, we can just pick the best few.
        # But we must explore all to prove optimality?
        # We are minimizing, so we just need to find *a* solution better than current_best.
        # We don't need to prove optimality if we find a valid cover with fewer blocks.
        # So we can prune branches that don't improve UB.
        
        # Sort candidates by gain descending
        # To save time, if list is huge, maybe sample? No, correctness requires checking.
        # But we can break early if we find a solution?
        # No, we need to check if a *better* solution exists.
        
        candidates.sort(key=get_gain, reverse=True)
        
        for i in candidates:
            new_uncovered = uncovered_mask & ~masks[i]
            new_path = current_path + [i]
            branch_and_bound(new_uncovered, new_path)
            
            # If we found a solution, current_best might have decreased.
            # If len(new_path) >= current_best, we can prune further?
            # The recursion handles this.

    # Start B&B
    # To avoid stack overflow, we might need to limit depth or use iterative deepening?
    # The depth is ~300. Python recursion limit is 1000. Should be fine.
    # But B&B on 75k blocks is HUGE.
    # We need a time limit.
    
    start_time = time.time()
    
    # The B&B above might be too slow if the branching factor is high.
    # The "min_count" strategy is good.
    # Initially, every t-subset is covered by ~3000 blocks.
    # So min_count ~ 3000. Branching factor 3000.
    # Depth 300.
    # 3000^300 is infinite.
    # We MUST have a strong lower bound to prune.
    
    # Let's add a stronger lower bound:
    # Greedy Lower Bound:
    # Estimate the number of blocks needed by running a greedy *simulation* on the uncovered set,
    # but stopping if the cost exceeds the remaining budget.
    
    def greedy_lower_bound(uncovered_mask):
        # This is expensive.
        # Alternative: 
        # Sum of 1 / max_gain?
        # Let's stick to the simple bound and hope the "min_count" pruning is effective enough
        # combined with the initial Greedy UB.
        # If the Greedy UB is tight, B&B might not explore much.
        
        # Actually, for v=19, k=8, t=5, the gap between LB (264) and Greedy (~335) is small.
        # B&B might struggle.
        
        # Let's try to improve the Greedy itself with Local Search (Swap/Remove).
        # The current code does redundancy removal.
        # Let's add a "1-swap" local search:
        # Try replacing a block B in the solution with a block A not in the solution,
        # if A covers all unique parts of B.
        # This keeps the size same.
        # To reduce size, we need to find redundancies.
        
        pass
        
    # Given the time limit and the risk of B&B explosion, I will implement a 
    # **Multi-Start Greedy with Local Search (Redundancy Removal + 1-Swap)**
    # which is more robust and likely to find a better cover than pure B&B that times out.
    
    def remove_redundants(blocks_list):
        # Convert to set of masks for faster union?
        # Or just iterate.
        removed = True
        while removed:
            removed = False
            for i in range(len(blocks_list) - 1, -1, -1):
                block_mask = masks[blocks_list[i]]
                # Check if union of others covers all
                union_others = 0
                for j in range(len(blocks_list)):
                    if j != i:
                        union_others |= masks[blocks_list[j]]
                if union_others == all_mask:
                    blocks_list.pop(i)
                    removed = True
                    break
        return blocks_list

    def local_search(blocks_list):
        # Try to remove redundants first
        blocks_list = remove_redundants(blocks_list)
        
        # Try 1-swap: Replace block i with block j if it doesn't increase size 
        # and allows further reduction? 
        # Actually, 1-swap keeps size constant. 
        # It can enable a subsequent removal.
        
        # Let's try: For each block i in solution, for each block j not in solution:
        # If (masks[i] | masks[j]) covers the same unique parts?
        # Condition: masks[i] & ~union(others) == masks[j] & ~union(others)?
        # No, we need: union(others without i) | masks[j] == all_mask
        # AND union(others without i) | masks[i] == all_mask (which is true by definition of i being in solution)
        # So we just need: union(others without i) | masks[j] == all_mask
        # If this holds, we can swap i for j.
        # This might not reduce size immediately, but might lead to a state where a block is redundant.
        
        # This is O(N^2 * N) which is too slow.
        
        # Let's just do multiple Greedy restarts with randomization.
        return blocks_list

    best_found = initial_greedy[:]
    
    # Time budget for search
    max_time = 60.0
    start_ts = time.time()
    
    # Run B&B with a depth limit or time check?
    # Let's try B&B but with a strict node limit or time check inside.
    
    nodes_visited = 0
    def b_and_b_limited(uncovered_mask, current_path):
        nonlocal current_best, best_solution, nodes_visited
        
        if time.time() - start_ts > max_time:
            return
            
        nodes_visited += 1
        if nodes_visited % 1000 == 0:
            # Check time
            if time.time() - start_ts > max_time:
                return
                
        if not uncovered_mask:
            if len(current_path) < current_best:
                current_best = len(current_path)
                best_solution = current_path[:]
            return
            
        if len(current_path) >= current_best:
            return
            
        # Find target
        min_count = n_blocks
        target_idx = -1
        m = uncovered_mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            cnt = len(blocks_for_t_subset[idx])
            if cnt < min_count:
                min_count = cnt
                target_idx = idx
            m ^= lsb
            
        if min_count == 0:
            return
            
        candidates = blocks_for_t_subset[target_idx]
        
        # Sort by gain
        # Optimization: If candidates is large, sort only a subset?
        # No, sort all.
        candidates.sort(key=lambda i: (masks[i] & uncovered_mask).bit_count(), reverse=True)
        
        for i in candidates:
            new_uncovered = uncovered_mask & ~masks[i]
            b_and_b_limited(new_uncovered, current_path + [i])
            
            if time.time() - start_ts > max_time:
                return

    # Start B&B
    b_and_b_limited(all_mask, [])
    
    # If B&B didn't improve, try randomized greedy restarts
    if time.time() - start_ts < max_time:
        for _ in range(20):
            if time.time() - start_ts > max_time:
                break
            uncovered = all_mask
            selected = []
            used = [False] * n_blocks
            
            while uncovered:
                # Randomized Greedy
                # Pick top K candidates, choose randomly
                gains = []
                for i in range(n_blocks):
                    if not used[i]:
                        gain = (masks[i] & uncovered).bit_count()
                        if gain > 0:
                            gains.append((gain, i))
                if not gains:
                    break
                gains.sort(reverse=True)
                
                # Pick from top 5
                top = gains[:min(5, len(gains))]
                # Weighted random? Or just random from top?
                # Let's pick the best one 80% of the time, random from top 5 20% of the time
                if random.random() < 0.8:
                    idx = top[0][1]
                else:
                    idx = top[random.randint(0, len(top)-1)][1]
                    
                selected.append(idx)
                used[idx] = True
                uncovered &= ~masks[idx]
                
            selected = remove_redundants(selected)
            
            if len(selected) < len(best_found):
                best_found = selected[:]
                current_best = len(best_found)
                best_solution = best_found[:]

    return [list(k_subsets[i]) for i in best_found]
