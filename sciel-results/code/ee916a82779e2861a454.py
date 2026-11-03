import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin

    # Precomputation
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subsets)
    t_subset_to_idx = {ts: i for i, ts in enumerate(all_t_subsets)}
    
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    n_uint64s = (n_t + 63) // 64
    
    # Precompute coverage masks
    coverage_masks = np.zeros((n_blocks, n_uint64s), dtype=np.uint64)
    for i, block in enumerate(all_blocks):
        for ts in itertools.combinations(block, t):
            idx = t_subset_to_idx[ts]
            word_idx = idx // 64
            bit_idx = idx % 64
            coverage_masks[i, word_idx] |= (1 << bit_idx)
            
    if time.time() > deadline - 10.0:
        # Fallback if precomputation is too slow
        return [list(all_blocks[0])]

    # Precompute block_t_indices for local search
    block_t_indices = []
    for block in all_blocks:
        idxs = [t_subset_to_idx[ts] for ts in itertools.combinations(block, t)]
        block_t_indices.append(idxs)

    # Helper to check if a set of t-subset indices is covered by a block
    def block_covers_indices(block_idx, indices):
        if not indices:
            return True
        # Use the mask for a fast check if possible, or just iterate
        # Since indices is small, iteration is fine. But we can use a set for O(1) lookup.
        # However, creating a set for every check is expensive.
        # Let's just iterate and check membership in the list? No, list membership is O(N).
        # Let's precompute sets for the selected blocks? Or just use the mask?
        # Using the mask is better.
        # Create a temporary mask for the indices?
        # Or just check if all bits are set in coverage_masks[block_idx]
        
        # Efficient check:
        # For each index, check if the bit is set in the block's mask.
        # This is O(len(indices)) bitwise ops.
        for idx in indices:
            word_idx = idx // 64
            bit_idx = idx % 64
            if not (coverage_masks[block_idx, word_idx] & (1 << bit_idx)):
                return False
        return True

    # 1. Greedy Set Cover
    covered_mask = np.zeros(n_uint64s, dtype=np.uint64)
    selected_blocks = []
    
    max_iterations = 200
    for _ in range(max_iterations):
        if time.time() > deadline - 10.0:
            break
            
        if np.bitwise_count(covered_mask).sum() == n_t:
            break
            
        not_covered = ~covered_mask
        intersections = coverage_masks & not_covered
        gains = np.bitwise_count(intersections).sum(axis=1)
        
        best_block_idx = np.argmax(gains)
        best_gain = gains[best_block_idx]
        
        if best_gain == 0:
            break
            
        selected_blocks.append(int(best_block_idx))
        covered_mask |= coverage_masks[best_block_idx]

    # 2. Local Search: Block Swapping
    # Goal: Reduce the number of blocks or improve structure to allow removal.
    
    # Compute cover counts
    cover_counts = np.zeros(n_t, dtype=np.int32)
    for b_idx in selected_blocks:
        cover_counts[block_t_indices[b_idx]] += 1
        
    # Set of selected blocks for O(1) lookup
    selected_set = set(selected_blocks)

    improved = True
    while improved and time.time() < deadline - 5.0:
        improved = False
        
        # Try to remove blocks
        for i in range(len(selected_blocks) - 1, -1, -1):
            b_idx = selected_blocks[i]
            t_idxs = block_t_indices[b_idx]
            
            if np.min(cover_counts[t_idxs]) > 1:
                cover_counts[t_idxs] -= 1
                selected_blocks.pop(i)
                selected_set.remove(b_idx)
                improved = True
                break
            
        # If no removal, try swaps
        if not improved and time.time() < deadline - 5.0:
            rng = random.Random(int(time.time() * 1000))
            
            # Build inverse index if not exists (cache it)
            if not hasattr(construct, '_t_to_blocks'):
                t_to_blocks = [[] for _ in range(n_t)]
                for b_idx, t_idxs in enumerate(block_t_indices):
                    for t_idx in t_idxs:
                        t_to_blocks[t_idx].append(b_idx)
                construct._t_to_blocks = t_to_blocks
            else:
                t_to_blocks = construct._t_to_blocks

            for _ in range(200):
                if time.time() > deadline - 5.0:
                    break
                
                if len(selected_blocks) == 0:
                    break
                
                i = rng.randint(0, len(selected_blocks) - 1)
                b_idx = selected_blocks[i]
                t_idxs = block_t_indices[b_idx]
                
                essential_idxs = [idx for idx in t_idxs if cover_counts[idx] == 1]
                
                if not essential_idxs:
                    cover_counts[t_idxs] -= 1
                    selected_blocks.pop(i)
                    selected_set.remove(b_idx)
                    improved = True
                    break
                
                # Pick a random essential t-subset to find candidates
                e = rng.choice(essential_idxs)
                candidates = t_to_blocks[e]
                
                if not candidates:
                    continue
                
                # Sample a few candidates to avoid scanning all
                num_samples = min(10, len(candidates))
                sampled_candidates = rng.sample(candidates, num_samples)
                
                found_swap = False
                for j in sampled_candidates:
                    if j in selected_set:
                        continue
                    
                    # Check if j covers all essential_idxs
                    if block_covers_indices(j, essential_idxs):
                        # Perform swap
                        cover_counts[block_t_indices[i]] -= 1
                        cover_counts[block_t_indices[j]] += 1
                        selected_blocks[i] = j
                        selected_set.remove(i)
                        selected_set.add(j)
                        improved = True
                        found_swap = True
                        break
                        
                if found_swap:
                    break

    # Final Verification
    final_blocks = [list(all_blocks[i]) for i in selected_blocks]
    
    test_counts = np.zeros(n_t, dtype=np.int32)
    for b_idx in selected_blocks:
        test_counts[block_t_indices[b_idx]] += 1
        
    if np.any(test_counts == 0):
        # Invalid. Reconstruct a valid greedy covering quickly.
        covered_mask = np.zeros(n_uint64s, dtype=np.uint64)
        valid_blocks = []
        for _ in range(200):
            not_covered = ~covered_mask
            intersections = coverage_masks & not_covered
            gains = np.bitwise_count(intersections).sum(axis=1)
            best_idx = np.argmax(gains)
            if gains[best_idx] == 0:
                break
            valid_blocks.append(int(best_idx))
            covered_mask |= coverage_masks[best_idx]
            
        return [list(all_blocks[i]) for i in valid_blocks]

    return final_blocks
