"""Covering design C(v,k,t) via fast bitmask greedy + redundancy removal."""
import itertools
import time
import random


def construct(v, k, t):
    start = time.time()
    deadline = start + 110.0  # Safety margin under 120s limit

    # 1. Precomputation
    # Generate all k-subsets (blocks)
    all_blocks = [tuple(c) for c in itertools.combinations(range(v), k)]
    num_blocks = len(all_blocks)
    
    # Generate all t-subsets and map them to indices
    all_t_subsets = list(itertools.combinations(range(v), t))
    sub_index = {s: i for i, s in enumerate(all_t_subsets)}
    num_subsets = len(all_t_subsets)
    
    # For each block, compute the bitmask of t-subsets it covers
    block_masks = [0] * num_blocks
    for i, block in enumerate(all_blocks):
        mask = 0
        for t_sub in itertools.combinations(block, t):
            idx = sub_index[t_sub]
            mask |= (1 << idx)
        block_masks[i] = mask

    all_mask = (1 << num_subsets) - 1

    def greedy_with_seed(seed=None):
        """Run greedy set cover with optional random tie-breaking."""
        covered = 0
        selected_indices = []
        
        while covered != all_mask:
            if time.time() > deadline:
                break
            
            best_block_idx = -1
            best_gain = 0
            best_gain_ties = []
            
            for i in range(num_blocks):
                gain = (block_masks[i] & (all_mask ^ covered)).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_gain_ties = [i]
                elif gain == best_gain and gain > 0:
                    best_gain_ties.append(i)
            
            if best_block_idx == -1 or best_gain == 0:
                if not best_gain_ties:
                    break
                # Pick randomly from ties
                if seed is not None:
                    best_block_idx = random.Random(seed).choice(best_gain_ties)
                else:
                    best_block_idx = best_gain_ties[0]
            else:
                best_block_idx = best_gain_ties[0]
            
            selected_indices.append(best_block_idx)
            covered |= block_masks[best_block_idx]
            
        return selected_indices

    def redundancy_removal(selected_set, block_masks, all_mask, deadline):
        """Remove redundant blocks from selected_set."""
        # Compute subset counts for selected blocks
        subset_counts = [0] * num_subsets
        # Precompute subset indices for selected blocks only
        selected_subs = {}
        for idx in selected_set:
            subs = [sub_index[ts] for ts in itertools.combinations(all_blocks[idx], t)]
            selected_subs[idx] = subs
            for s_idx in subs:
                subset_counts[s_idx] += 1
        
        changed = True
        while changed and time.time() < deadline:
            changed = False
            for idx in list(selected_set):
                is_redundant = True
                for s_idx in selected_subs[idx]:
                    if subset_counts[s_idx] <= 1:
                        is_redundant = False
                        break
                if is_redundant:
                    selected_set.remove(idx)
                    for s_idx in selected_subs[idx]:
                        subset_counts[s_idx] -= 1
                    changed = True
                    break
        
        return selected_set, subset_counts, selected_subs

    def local_search_1swap(selected_set, subset_counts, selected_subs, deadline):
        """1-for-1 swap local search."""
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for idx in list(selected_set):
                # Find subsets that would be lost if we remove this block
                lost = [s_idx for s_idx in selected_subs[idx] if subset_counts[s_idx] == 1]
                
                if not lost:
                    # Block is redundant, just remove it
                    selected_set.remove(idx)
                    for s_idx in selected_subs[idx]:
                        subset_counts[s_idx] -= 1
                    del selected_subs[idx]
                    improved = True
                    break
                
                # Try to find a replacement block that covers all lost subsets
                # For speed, precompute a mapping from subset index to blocks that cover it
                # But that's expensive. Instead, iterate over non-selected blocks.
                lost_mask = 0
                for s_idx in lost:
                    lost_mask |= (1 << s_idx)
                
                for j in range(num_blocks):
                    if j in selected_set:
                        continue
                    # Check if block j covers all lost subsets
                    if (block_masks[j] & lost_mask) == lost_mask:
                        # Perform the swap
                        selected_set.remove(idx)
                        for s_idx in selected_subs[idx]:
                            subset_counts[s_idx] -= 1
                        del selected_subs[idx]
                        
                        selected_set.add(j)
                        new_subs = [sub_index[ts] for ts in itertools.combinations(all_blocks[j], t)]
                        selected_subs[j] = new_subs
                        for s_idx in new_subs:
                            subset_counts[s_idx] += 1
                        improved = True
                        break
                if improved:
                    break
        
        return selected_set, subset_counts, selected_subs

    # Run multiple greedy restarts with different seeds, keep the best
    best_selected = None
    best_count = float('inf')
    
    # First, do a standard greedy run
    initial_selected = greedy_with_seed(None)
    selected_set = set(initial_selected)
    selected_set, subset_counts, selected_subs = redundancy_removal(selected_set, block_masks, all_mask, deadline)
    selected_set, subset_counts, selected_subs = local_search_1swap(selected_set, subset_counts, selected_subs, deadline)
    
    # Verify coverage
    final_covered = 0
    for idx in selected_set:
        final_covered |= block_masks[idx]
    if final_covered == all_mask:
        best_selected = set(selected_set)
        best_count = len(selected_set)
    
    # Try a few more restarts if time allows
    rng = random.Random(42)
    for attempt in range(5):
        if time.time() > deadline - 30:
            break
        
        seed = rng.randint(0, 2**31 - 1)
        selected_indices = greedy_with_seed(seed)
        selected_set = set(selected_indices)
        
        selected_set, subset_counts, selected_subs = redundancy_removal(selected_set, block_masks, all_mask, deadline - 15)
        selected_set, subset_counts, selected_subs = local_search_1swap(selected_set, subset_counts, selected_subs, deadline - 10)
        
        # Verify coverage
        final_covered = 0
        for idx in selected_set:
            final_covered |= block_masks[idx]
        
        if final_covered == all_mask and len(selected_set) < best_count:
            best_selected = set(selected_set)
            best_count = len(selected_set)
    
    # If no valid solution found, fall back to pure greedy
    if best_selected is None:
        selected_indices = greedy_with_seed(None)
        best_selected = set(selected_indices)
    
    # Convert indices back to blocks
    result_blocks = [all_blocks[i] for i in best_selected]
    
    return result_blocks
