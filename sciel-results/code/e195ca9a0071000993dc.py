import itertools
import random
import time

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0
    
    # 1. Precomputation
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t = len(all_t_subsets)
    t_subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    
    all_k_subsets = list(itertools.combinations(range(v), k))
    num_k = len(all_k_subsets)
    
    # block_mask[b] is an integer representing the set of t-subsets covered by block b
    block_mask = [0] * num_k
    # covering_blocks[i] is a list of block indices that cover t-subset i
    covering_blocks = [[] for _ in range(num_t)]
    
    for b_idx, block in enumerate(all_k_subsets):
        mask = 0
        for t_sub in itertools.combinations(block, t):
            idx = t_subset_to_idx[t_sub]
            mask |= (1 << idx)
            covering_blocks[idx].append(b_idx)
        block_mask[b_idx] = mask
        
    def get_unique_mask(counts):
        um = 0
        for i in range(num_t):
            if counts[i] == 1:
                um |= (1 << i)
        return um

    def minimize(active_set, counts):
        changed = True
        while changed:
            changed = False
            unique_mask = get_unique_mask(counts)
            # Iterate over a copy because we modify the set
            for b_idx in list(active_set):
                if b_idx not in active_set:
                    continue
                # Check if b_idx has any unique coverage
                if block_mask[b_idx] & unique_mask:
                    continue
                # It has no unique coverage, so it can be removed
                active_set.remove(b_idx)
                for i in range(num_t):
                    if (block_mask[b_idx] >> i) & 1:
                        counts[i] -= 1
                changed = True
                # Recalculate unique mask immediately? Or at end of loop?
                # Since counts changed, unique_mask is stale.
                # But we are in a while loop, so we recalc at start of next iteration.
                # However, if we remove one, another might become removable.
                # The while loop handles this.
        return active_set, counts

    def greedy_construct():
        active_set = set()
        counts = [0] * num_t
        uncovered_mask = (1 << num_t) - 1
        
        while uncovered_mask:
            best_block = -1
            max_gain = 0
            # Optimization: only consider blocks that cover at least one uncovered subset
            # We can iterate all blocks, 50k is small
            for b_idx in range(num_k):
                gain = (block_mask[b_idx] & uncovered_mask).bit_count()
                if gain > max_gain:
                    max_gain = gain
                    best_block = b_idx
            
            if best_block == -1:
                break
                
            active_set.add(best_block)
            for i in range(num_t):
                if (block_mask[best_block] >> i) & 1:
                    counts[i] += 1
            
            uncovered_mask &= ~block_mask[best_block]
            
        return active_set, counts

    def local_search(active_set, counts):
        # Returns improved active_set and counts
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Shuffle to avoid deterministic loops
            items = list(active_set)
            random.shuffle(items)
            
            for b_idx in items:
                if b_idx not in active_set:
                    continue
                
                # Check if removable
                unique_mask = get_unique_mask(counts)
                if not (block_mask[b_idx] & unique_mask):
                    # Remove
                    active_set.remove(b_idx)
                    for i in range(num_t):
                        if (block_mask[b_idx] >> i) & 1:
                            counts[i] -= 1
                    improved = True
                    # After removal, we should probably restart the local search pass
                    # or at least re-minimize
                    active_set, counts = minimize(active_set, counts)
                    continue
                
                # Find swap candidates
                u_b = block_mask[b_idx] & unique_mask
                if u_b == 0:
                    continue
                    
                # Find the rarest subset in u_b
                # Iterate bits in u_b
                bits = []
                temp = u_b
                while temp:
                    # Get lowest set bit
                    lsb = temp & -temp
                    idx = lsb.bit_length() - 1
                    bits.append(idx)
                    temp ^= lsb
                
                if not bits:
                    continue
                
                # Sort by frequency (length of covering_blocks)
                bits.sort(key=lambda i: len(covering_blocks[i]))
                rarest_idx = bits[0]
                
                found_swap = False
                for cand_idx in covering_blocks[rarest_idx]:
                    if cand_idx == b_idx:
                        continue
                    if cand_idx in active_set:
                        continue
                        
                    # Check if cand covers all of u_b
                    if (block_mask[cand_idx] & u_b) == u_b:
                        # Perform swap
                        active_set.remove(b_idx)
                        active_set.add(cand_idx)
                        
                        # Update counts
                        for i in range(num_t):
                            if (block_mask[b_idx] >> i) & 1:
                                counts[i] -= 1
                            if (block_mask[cand_idx] >> i) & 1:
                                counts[i] += 1
                        
                        # Minimize after swap
                        active_set, counts = minimize(active_set, counts)
                        improved = True
                        found_swap = True
                        break # Restart pass or continue? Breaking inner loop, continuing outer.
                        # Actually, breaking the swap search for this b_idx is good.
                        # The 'improved' flag will cause a new pass.
                
                if found_swap:
                    # Since we made a change, the 'items' list might be stale, 
                    # but we are just using it to iterate. 
                    # The 'improved' flag ensures we loop again.
                    pass

        return active_set, counts

    # Main Logic
    best_solution = None
    best_count = float('inf')
    
    # Run multiple restarts
    num_restarts = 5 # Adjust based on time
    
    for _ in range(num_restarts):
        if time.time() > deadline - 10:
            break
            
        active_set, counts = greedy_construct()
        active_set, counts = minimize(active_set, counts)
        active_set, counts = local_search(active_set, counts)
        
        current_count = len(active_set)
        if current_count < best_count:
            best_count = current_count
            best_solution = set(active_set)
            
    # Final result
    if best_solution is None:
        # Fallback
        active_set, counts = greedy_construct()
        active_set, counts = minimize(active_set, counts)
        best_solution = active_set
        
    result = [list(all_k_subsets[b_idx]) for b_idx in best_solution]
    return result
