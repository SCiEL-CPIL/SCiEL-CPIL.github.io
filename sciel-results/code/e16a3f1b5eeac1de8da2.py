import itertools
import time
import sys

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
    # Using Python integers for bitsets is efficient for ~12k bits
    masks = [0] * n_blocks
    for i, block in enumerate(k_subsets):
        mask = 0
        for t_sub in itertools.combinations(block, t):
            idx = t_subset_index[t_sub]
            mask |= (1 << idx)
        masks[i] = mask
        
    # Precompute for each t-subset index, the list of blocks that cover it
    # This is crucial for the "hardest element" heuristic in B&B
    # Memory: 11628 lists. Total entries: n_blocks * C(k,t) = 75582 * 56 = ~4.2M ints.
    # This is feasible.
    blocks_for_t_sub = [[] for _ in range(n_t_subsets)]
    for i in range(n_blocks):
        m = masks[i]
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            blocks_for_t_sub[idx].append(i)
            m ^= lsb

    # Initial Greedy Solution to set an Upper Bound
    def greedy_solve():
        uncovered = all_mask
        selected = []
        used = [False] * n_blocks
        
        while uncovered:
            best_gain = -1
            best_idx = -1
            # To speed up, we can iterate through blocks and check gain
            # But we can also pick a random uncovered subset and look at its blocks?
            # Standard greedy: pick block with max gain.
            # Optimization: Maintain a heap? Or just scan?
            # Scanning 75k blocks * ~300 iterations = 22.5M ops. Fast in Py? Maybe.
            # Let's use the blocks_for_t_sub to speed up?
            # Actually, simple scan is okay for initialization.
            
            for i in range(n_blocks):
                if not used[i]:
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
            
        return selected

    best_solution = greedy_solve()
    current_best = len(best_solution)
    
    # Branch and Bound
    # State: uncovered_mask, current_depth, current_path
    # Pruning:
    # 1. current_depth + lower_bound(uncovered_mask) >= current_best
    # 2. If a t-subset in uncovered_mask has no covering blocks in remaining candidates, fail.
    
    # To make B&B fast, we need a good lower bound.
    # LB1: ceil(uncovered_count / max_possible_gain)
    # max_possible_gain is C(k,t), but effectively less.
    # LB2: Greedy lower bound (expensive).
    # LB3: Based on the "hardest" element. If the hardest element has `d` covering blocks,
    #       and we must pick one, does that help?
    
    # We will use a recursive B&B.
    # To avoid deep recursion issues and overhead, we'll use a stack or limit depth.
    # Given 120s, we can explore a bit.
    
    start_time = time.time()
    timeout = 100 # Leave 20s for cleanup/return
    
    # We need to track "remaining_blocks" or just "available blocks".
    # Since blocks are not removed from the pool (we can reuse? No, set cover uses distinct blocks),
    # we must track which blocks are used.
    # However, in B&B for Set Cover, we usually branch on *which* block covers a specific element.
    # Once a block is chosen, it's "used".
    # We can pass a `used_mask` (bitset of blocks) or a set of used indices.
    # With 75k blocks, a bitset of blocks is 75k bits = ~9KB. Very fast to copy.
    
    all_blocks_mask = (1 << n_blocks) - 1
    
    def lower_bound(uncovered_mask, available_blocks_mask):
        if not uncovered_mask:
            return 0
            
        # Simple bound: uncovered bits / max gain
        # Max gain is C(k,t) = 56
        # But we can do better:
        # Find the t-subset in uncovered with the fewest available blocks.
        # If it has 0, return infinity.
        # If it has 1, we *must* pick that block.
        # This is a strong constraint.
        
        # Iterate through uncovered bits
        u = uncovered_mask
        min_avail = float('inf')
        while u:
            lsb = u & -u
            idx = lsb.bit_length() - 1
            # Count available blocks covering idx
            # blocks_for_t_sub[idx] is a list of block indices.
            # We need to count how many are in available_blocks_mask.
            # This is slow if done naively.
            
            # Optimization: Precompute a bitset for each t-subset indicating which blocks cover it?
            # block_cover_mask[t_idx] = bitset of blocks covering t_idx.
            # Then count = (block_cover_mask[idx] & available_blocks_mask).bit_count()
            
            # Let's precompute block_cover_mask for each t-subset.
            # 11628 bitsets of 75582 bits.
            # 11628 * 75582 / 8 bytes ~ 110 MB. Feasible.
            pass
            
        # For now, use the weak bound to ensure correctness, but it might be slow.
        # We need the precomputed block_cover_mask for speed.
        
        return (uncovered_mask.bit_count() + 55) // 56

    # Precompute block_cover_mask for each t-subset
    # This is the "Conflict Graph" adjacency in a sense?
    # t_idx -> bitset of blocks
    block_cover_mask = [0] * n_t_subsets
    for i in range(n_blocks):
        m = masks[i]
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            block_cover_mask[idx] |= (1 << i)
            m ^= lsb

    def tighter_lower_bound(uncovered_mask, available_blocks_mask):
        if not uncovered_mask:
            return 0
            
        u = uncovered_mask
        min_avail = float('inf')
        while u:
            lsb = u & -u
            idx = lsb.bit_length() - 1
            # Count available blocks covering this t-subset
            avail_count = (block_cover_mask[idx] & available_blocks_mask).bit_count()
            if avail_count == 0:
                return float('inf')
            if avail_count < min_avail:
                min_avail = avail_count
            u ^= lsb
            
        # If min_avail is 1, we are forced to pick that block.
        # This doesn't directly give a count, but it forces the choice.
        # A better LB:
        # Partition uncovered into groups?
        # For now, return the simple bound but check for infeasibility.
        if min_avail == float('inf'):
            return float('inf')
            
        return (uncovered_mask.bit_count() + 55) // 56

    def branch_and_bound(uncovered_mask, available_blocks_mask, current_depth, current_path):
        nonlocal current_best, best_solution
        
        if time.time() - start_time > timeout:
            return
            
        if not uncovered_mask:
            if current_depth < current_best:
                current_best = current_depth
                best_solution = current_path[:]
            return
            
        # Pruning
        lb = tighter_lower_bound(uncovered_mask, available_blocks_mask)
        if current_depth + lb >= current_best:
            return
            
        # Select the "hardest" t-subset
        # One with the fewest available covering blocks
        u = uncovered_mask
        best_idx = -1
        min_avail = float('inf')
        best_block_list = []
        
        while u:
            lsb = u & -u
            idx = lsb.bit_length() - 1
            avail_count = (block_cover_mask[idx] & available_blocks_mask).bit_count()
            if avail_count == 0:
                return # Dead end
            if avail_count < min_avail:
                min_avail = avail_count
                best_idx = idx
                # Get the list of available blocks for this idx
                # We can filter blocks_for_t_sub[idx] by available_blocks_mask
                # But doing this every time is slow.
                # We can just iterate through blocks_for_t_sub[idx] and check bit.
                u ^= lsb
                
        # If we found a t-subset with only 1 available block, we MUST take it.
        # This is a powerful pruning step.
        if min_avail == 1:
            # Find the block
            for b_idx in blocks_for_t_sub[best_idx]:
                if (available_blocks_mask >> b_idx) & 1:
                    # Forced move
                    new_uncovered = uncovered_mask & ~masks[b_idx]
                    new_available = available_blocks_mask & ~(1 << b_idx)
                    branch_and_bound(new_uncovered, new_available, current_depth + 1, current_path + [b_idx])
                    return
                    
        # Otherwise, branch on all available blocks covering best_idx
        # Sort candidates by gain (descending) to find good solutions faster
        candidates = []
        for b_idx in blocks_for_t_sub[best_idx]:
            if (available_blocks_mask >> b_idx) & 1:
                gain = (masks[b_idx] & uncovered_mask).bit_count()
                candidates.append((gain, b_idx))
                
        candidates.sort(reverse=True)
        
        for gain, b_idx in candidates:
            if current_depth + 1 >= current_best:
                break
                
            new_uncovered = uncovered_mask & ~masks[b_idx]
            new_available = available_blocks_mask & ~(1 << b_idx)
            
            # Simple pruning: if new_uncovered is same as uncovered, gain was 0?
            # No, gain > 0 by construction (it covers best_idx).
            
            branch_and_bound(new_uncovered, new_available, current_depth + 1, current_path + [b_idx])

    # Start B&B
    # Initial available is all blocks
    branch_and_bound(all_mask, all_blocks_mask, 0, [])
    
    # If B&B didn't improve, return the greedy result
    # Note: B&B might not finish in time. We rely on the timeout check.
    
    return [list(k_subsets[i]) for i in best_solution]
