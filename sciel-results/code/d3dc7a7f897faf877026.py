import time
import random
import itertools
import sys

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140.0  # 140s out of 150s to be safe
    
    # Precompute all t-subsets and their indices
    # For v=19, k=8, t=5: C(19,5) = 11628
    all_t_subs = list(itertools.combinations(range(v), t))
    n_t_subs = len(all_t_subs)
    
    # Map each t-subset to its index
    t_sub_to_idx = {sub: i for i, sub in enumerate(all_t_subs)}
    
    # Precompute for each k-subset (block), the bitmask of t-subsets it covers
    # A block covers a t-subset if the t-subset is contained in the block
    # For v=19, k=8: C(19,8) = 75582 possible blocks
    # Each block covers C(8,5) = 56 t-subsets
    
    # We'll use Python integers as bitmasks (arbitrary precision)
    # n_t_subs = 11628, which fits in ~116 bits, well within Python int range
    
    # Precompute: for each possible k-subset, compute the set of t-subsets it covers
    # We do this lazily or precompute a lookup
    # Actually, let's precompute for efficiency
    
    # For a given block (sorted tuple of k points), the t-subsets it covers are
    # all combinations of t points chosen from the k points in the block
    def block_to_mask(block):
        """Convert a k-subset (tuple) to a bitmask of covered t-subsets"""
        mask = 0
        for t_sub in itertools.combinations(block, t):
            idx = t_sub_to_idx[t_sub]
            mask |= (1 << idx)
        return mask
    
    # Precompute masks for all possible k-subsets
    all_k_subs = list(itertools.combinations(range(v), k))
    n_k_subs = len(all_k_subs)
    
    # Precompute masks
    k_sub_to_mask = {}
    k_sub_to_idx = {}
    for i, k_sub in enumerate(all_k_subs):
        k_sub_to_idx[k_sub] = i
        k_sub_to_mask[k_sub] = block_to_mask(k_sub)
    
    # Also build a fast lookup: for a given t-subset, which k-subsets contain it?
    # This is useful for greedy selection
    # Actually for greedy we want: given uncovered t-subsets, find block covering most
    # Let's use a different approach:
    
    # Greedy set cover with bitmask
    def greedy_set_cover():
        """Greedy: repeatedly pick block covering most uncovered t-subsets"""
        # uncovered_mask: bit i set means t-sub i is uncovered
        uncovered_mask = (1 << n_t_subs) - 1  # all uncovered
        selected = []
        
        while uncovered_mask:
            # Find block with max coverage of uncovered
            best_idx = -1
            best_count = -1
            for i in range(n_k_subs):
                m = k_sub_to_mask[all_k_subs[i]]
                count = (uncovered_mask & m).bit_count()
                if count > best_count:
                    best_count = count
                    best_idx = i
                # Early termination: if we found a perfect cover
                if best_count == uncovered_mask.bit_count():
                    break
            if best_count <= 0:
                break
            selected.append(best_idx)
            uncovered_mask &= ~k_sub_to_mask[all_k_subs[best_idx]]
        
        return selected
    
    # 1-opt pruning: try removing each block and see if cover still valid
    def prune_blocks(selected_indices):
        """Try to remove redundant blocks"""
        # Count how many times each t-subset is covered
        counts = [0] * n_t_subs
        for idx in selected_indices:
            m = k_sub_to_mask[all_k_subs[idx]]
            # Iterate through set bits
            mm = m
            while mm:
                lsb = mm & -mm
                s_idx = lsb.bit_length() - 1
                counts[s_idx] += 1
                mm ^= lsb
        
        current = selected_indices[:]
        removed = []
        
        # Try removing each block (in random order for diversity)
        random.shuffle(current)
        for idx in current[:]:
            m = k_sub_to_mask[all_k_subs[idx]]
            can_remove = True
            mm = m
            while mm:
                lsb = mm & -mm
                s_idx = lsb.bit_length() - 1
                if counts[s_idx] == 1:
                    can_remove = False
                    break
                mm ^= lsb
            
            if can_remove:
                # Remove it
                mm = m
                while mm:
                    lsb = mm & -mm
                    s_idx = lsb.bit_length() - 1
                    counts[s_idx] -= 1
                    mm ^= lsb
                current.remove(idx)
                removed.append(idx)
        
        return current
    
    # 2-opt: swap a block with another that covers more unique t-subsets
    def two_opt(selected_indices, max_iters=1000):
        """Try swapping blocks to reduce count or improve coverage"""
        # For now, just do 1-opt pruning
        return prune_blocks(selected_indices)
    
    # Local search: try replacing one block with another
    def local_search(selected_indices, deadline):
        """Try to improve by replacing blocks"""
        current = selected_indices[:]
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try removing each block
            random.shuffle(current)
            for idx in current[:]:
                if time.time() >= deadline:
                    break
                # Try removing this block
                new_list = [x for x in current if x != idx]
                # Check if new_list still covers all t-subsets
                # Build coverage mask
                cover = 0
                for j in new_list:
                    cover |= k_sub_to_mask[all_k_subs[j]]
                if cover == (1 << n_t_subs) - 1:
                    # Valid!
                    current = new_list
                    improved = True
                    break
        
        return current
    
    # Main search: greedy + local search + restarts
    best_result = None
    best_count = float('inf')
    
    # First, run greedy to get a starting point
    greedy_result = greedy_set_cover()
    greedy_result = prune_blocks(greedy_result)
    best_count = len(greedy_result)
    best_result = greedy_result[:]
    
    # Now do local search on the greedy result
    ls_result = local_search(best_result, deadline)
    if len(ls_result) < best_count:
        best_count = len(ls_result)
        best_result = ls_result[:]
    
    # Try multiple restarts with randomized greedy
    n_restarts = 0
    while time.time() < deadline - 5.0:  # Leave 5s margin
        # Randomized greedy
        uncovered_mask = (1 << n_t_subs) - 1
        selected = []
        
        while uncovered_mask and time.time() < deadline - 2.0:
            # Find top blocks by coverage, pick randomly from top few
            cov_counts = []
            for i in range(n_k_subs):
                m = k_sub_to_mask[all_k_subs[i]]
                count = (uncovered_mask & m).bit_count()
                if count > 0:
                    cov_counts.append((count, i))
            
            if not cov_counts:
                break
            
            # Sort by coverage descending
            cov_counts.sort(reverse=True)
            
            # Pick from top 5 (or all if fewer)
            top_n = min(5, len(cov_counts))
            best_count_val = cov_counts[0][0]
            candidates = [i for c, i in cov_counts[:top_n]]
            
            chosen = random.choice(candidates)
            selected.append(chosen)
            uncovered_mask &= ~k_sub_to_mask[all_k_subs[chosen]]
        
        if uncovered_mask == 0:
            # Valid cover
            pruned = prune_blocks(selected)
            if len(pruned) < best_count:
                best_count = len(pruned)
                best_result = pruned[:]
        
        n_restarts += 1
    
    # Final local search on best result
    if time.time() < deadline - 2.0:
        final_ls = local_search(best_result, deadline)
        if len(final_ls) < best_count:
            best_count = len(final_ls)
            best_result = final_ls[:]
    
    # Convert back to list of blocks
    if best_result is None:
        # Fallback: return greedy result
        best_result = greedy_set_cover()
    
    blocks = [list(all_k_subs[idx]) for idx in best_result]
    return blocks

if __name__ == "__main__":
    # Quick test
    blocks = construct(19, 8, 5)
    print(f"Number of blocks: {len(blocks)}")
