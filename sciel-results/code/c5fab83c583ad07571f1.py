import itertools
import time
import random
import sys
import math
import numpy as np
from collections import defaultdict

def construct(v, k, t):
    deadline = time.time() + 115  # Safety margin
    
    # Precomputation
    all_subsets = list(itertools.combinations(range(v), t))
    subset_to_idx = {s: i for i, s in enumerate(all_subsets)}
    n_subsets = len(all_subsets)
    
    # Precompute for each 3-subset, its index
    # We will represent a block's coverage as a boolean vector of length n_subsets
    
    def get_block_mask(block):
        """Return a numpy array of 0s and 1s indicating which 3-subsets are covered by this block."""
        mask = np.zeros(n_subsets, dtype=np.int8)
        for sub in itertools.combinations(block, t):
            mask[subset_to_idx[sub]] = 1
        return mask

    def greedy_construct_whole_block(seed_rng, num_candidates=100, batch_size=50):
        """
        Greedy construction selecting whole blocks.
        At each step, generate a pool of candidate blocks, evaluate their marginal gain
        (number of new subsets covered), and pick the best one.
        """
        blocks = []
        covered = np.zeros(n_subsets, dtype=np.int8)
        num_covered = 0
        
        # To speed up gain calculation, we can precompute the masks for candidate blocks
        # and use vectorized operations.
        
        while num_covered < n_subsets:
            if time.time() > deadline:
                return None
            
            # Generate candidate blocks
            candidates = []
            
            # Strategy 1: Random blocks
            # Ensure some diversity
            for _ in range(num_candidates // 2):
                candidates.append(tuple(sorted(seed_rng.sample(range(v), k))))
            
            # Strategy 2: Blocks centered on uncovered subsets
            # Pick a random uncovered subset and build a block around it
            uncovered_indices = np.where(covered == 0)[0]
            if len(uncovered_indices) > 0:
                # Sample a few uncovered subsets to anchor new blocks
                n_anchors = min(num_candidates // 2, 10)
                anchor_indices = seed_rng.choice(uncovered_indices, size=n_anchors, replace=False)
                for idx in anchor_indices:
                    anchor_sub = all_subsets[idx]
                    # Create a block containing this subset
                    remaining_points = [x for x in range(v) if x not in anchor_sub]
                    extra_points = seed_rng.sample(remaining_points, k - t)
                    block = tuple(sorted(list(anchor_sub) + extra_points))
                    candidates.append(block)
            
            # Remove duplicates
            candidates = list(set(candidates))
            
            if not candidates:
                # Fallback: random block
                candidates = [tuple(sorted(seed_rng.sample(range(v), k)))]
            
            # Evaluate gain for each candidate
            # Gain = number of subsets in candidate that are NOT in 'covered'
            # We can do this by summing (1 - covered) over the mask of the candidate
            
            best_block = None
            best_gain = -1
            
            # Batch processing for speed
            # Since n_subsets is small (2024 for C(24,3)), we can just loop or use simple vectorization
            # If num_candidates is large, we might want to stack masks, but 100 candidates is small enough
            # to just loop with NumPy inside.
            
            for block in candidates:
                # Fast mask creation: we can precompute masks for all possible blocks? No, too many.
                # But we can compute the mask for this specific block.
                # Optimization: Instead of building the full mask every time, we can just check
                # which of the t-subsets of this block are uncovered.
                
                gain = 0
                for sub in itertools.combinations(block, t):
                    idx = subset_to_idx[sub]
                    if covered[idx] == 0:
                        gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    best_block = block
            
            if best_block is None or best_gain == 0:
                # If no gain, pick a random block to break symmetry or handle edge case
                best_block = tuple(sorted(seed_rng.sample(range(v), k)))
            
            blocks.append(list(best_block))
            
            # Update covered
            for sub in itertools.combinations(best_block, t):
                idx = subset_to_idx[sub]
                if covered[idx] == 0:
                    covered[idx] = 1
                    num_covered += 1
        
        return blocks

    def compute_coverage(blocks_list):
        cov_count = np.zeros(n_subsets, dtype=np.int32)
        for block in blocks_list:
            for sub in itertools.combinations(block, t):
                idx = subset_to_idx[sub]
                cov_count[idx] += 1
        return cov_count

    def is_valid_covering(blocks_list):
        cov = compute_coverage(blocks_list)
        return bool(np.all(cov > 0))

    def local_search(blocks):
        """Try to improve by removing redundant blocks and swapping."""
        blocks = list(blocks)
        
        # Remove redundant blocks iteratively
        changed = True
        while changed and time.time() < deadline - 15:
            changed = False
            cov_count = compute_coverage(blocks)
            for i in range(len(blocks)):
                if time.time() > deadline - 15:
                    break
                block_i = blocks[i]
                redundant = True
                for sub in itertools.combinations(block_i, t):
                    idx = subset_to_idx[sub]
                    if cov_count[idx] == 1:
                        redundant = False
                        break
                if redundant:
                    blocks.pop(i)
                    changed = True
                    break
        
        # Try block replacement: replace a block with a better one
        # We can try to replace a block with a random one if it improves the "slack"
        # or just try to remove more.
        max_replacement_attempts = 50
        for _ in range(max_replacement_attempts):
            if time.time() > deadline - 15:
                break
            if len(blocks) == 0:
                break
            
            i = random.randint(0, len(blocks) - 1)
            old_block = blocks[i]
            
            cov_count = compute_coverage(blocks)
            unique_subs = []
            for sub in itertools.combinations(old_block, t):
                idx = subset_to_idx[sub]
                if cov_count[idx] == 1:
                    unique_subs.append(sub)
            
            if not unique_subs:
                blocks.pop(i)
                continue
            
            # Build new block from union of unique subsets
            candidates = set()
            for sub in unique_subs:
                candidates.update(sub)
            
            if len(candidates) > k:
                continue
            
            # Try multiple random completions
            best_new_block = None
            best_new_unique = 0
            for _ in range(3):
                cand = set(candidates)
                while len(cand) < k:
                    x = random.randint(0, v - 1)
                    if x not in cand:
                        cand.add(x)
                
                new_block = tuple(sorted(cand))
                
                # Verify it covers all unique_subs
                valid = True
                for sub in unique_subs:
                    if not all(x in new_block for x in sub):
                        valid = False
                        break
                if not valid:
                    continue
                
                # Count how many new unique subsets this block would cover
                cov_without_i = cov_count.copy()
                for sub in itertools.combinations(old_block, t):
                    idx = subset_to_idx[sub]
                    cov_without_i[idx] -= 1
                
                new_unique = 0
                for sub in itertools.combinations(new_block, t):
                    idx = subset_to_idx[sub]
                    if cov_without_i[idx] == 0:
                        new_unique += 1
                
                if new_unique > best_new_unique:
                    best_new_unique = new_unique
                    best_new_block = new_block
            
            if best_new_block is not None and best_new_unique > len(unique_subs):
                blocks[i] = list(best_new_block)
                blocks[i].sort()
        
        return blocks

    def fast_redundant_delete(blocks):
        """Quick pass: remove blocks whose every t-subset is covered by others."""
        if not blocks:
            return blocks
        cov = np.zeros(n_subsets, dtype=np.int32)
        for block in blocks:
            for sub in itertools.combinations(block, t):
                cov[subset_to_idx[sub]] += 1
        to_remove = set()
        for i, block in enumerate(blocks):
            redundant = True
            for sub in itertools.combinations(block, t):
                idx = subset_to_idx[sub]
                if cov[idx] == 1:
                    redundant = False
                    break
            if redundant:
                to_remove.add(i)
        if to_remove:
            blocks = [b for i, b in enumerate(blocks) if i not in to_remove]
        return blocks

    best_blocks = None
    best_count = float('inf')
    
    # Run multiple greedy restarts
    num_restarts = 0
    while time.time() < deadline - 45:
        seed_rng = random.Random(num_restarts)
        blocks = greedy_construct_whole_block(seed_rng)
        if blocks is None:
            continue
        
        if len(blocks) < best_count:
            blocks = fast_redundant_delete(blocks)
            if len(blocks) < best_count:
                best_count = len(blocks)
                best_blocks = list(blocks)
            
            # Local search refinement
            improved = local_search(blocks)
            if len(improved) < best_count:
                best_count = len(improved)
                best_blocks = improved
        
        num_restarts += 1
    
    # Use remaining time for more local search on the best solution
    if best_blocks is not None and time.time() < deadline - 5:
        # Try a few more local search passes
        for _ in range(5):
            if time.time() > deadline - 2:
                break
            final = local_search(best_blocks)
            if len(final) < best_count:
                best_count = len(final)
                best_blocks = final
    
    if best_blocks is None:
        # Fallback
        seed_rng = random.Random(999)
        blocks = greedy_construct_whole_block(seed_rng)
        if blocks:
            return blocks
        return []
    
    return best_blocks
