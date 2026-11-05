import numpy as np
import itertools
import time
import random
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Leave 5s margin
    
    # Pre-compute all t-subsets
    t_subsets = list(itertools.combinations(range(v), t))
    t_subset_to_idx = {subset: i for i, subset in enumerate(t_subsets)}
    num_t_subsets = len(t_subsets)
    
    # Pre-compute all k-blocks and the t-subsets they cover
    k_blocks = list(itertools.combinations(range(v), k))
    num_k_blocks = len(k_blocks)
    
    # For each block, precompute which t-subsets it covers
    block_t_indices = []
    for block in k_blocks:
        covered = [t_subset_to_idx[subset] for subset in itertools.combinations(block, t)]
        block_t_indices.append(covered)
    
    block_t_indices = np.array(block_t_indices, dtype=np.int32)
    c_k_t = len(block_t_indices[0])  # C(k,t)
    
    def time_left():
        return deadline - time.time()
    
    def greedy_with_weights(weights, max_blocks=None):
        """Greedy set cover with block weights. Returns list of block indices."""
        coverage = np.zeros(num_t_subsets, dtype=np.int32)
        used = np.zeros(num_k_blocks, dtype=bool)
        selected = []
        
        while time_left() > 0.5:
            if max_blocks is not None and len(selected) >= max_blocks:
                break
            
            # Compute gain: number of newly covered t-subsets
            uncovered_mask = (coverage == 0)
            gains = np.sum(uncovered_mask[block_t_indices], axis=1)
            gains[used] = -1
            
            best_idx = np.argmax(gains)
            if gains[best_idx] <= 0:
                break
            
            selected.append(best_idx)
            used[best_idx] = True
            coverage[block_t_indices[best_idx]] += 1
        
        return selected
    
    def remove_redundant(block_indices):
        """Remove blocks that are redundant (all their t-subsets covered by others)."""
        if not block_indices:
            return []
        
        # Count how many selected blocks cover each t-subset
        coverage = np.zeros(num_t_subsets, dtype=np.int32)
        for bi in block_indices:
            coverage[block_t_indices[bi]] += 1
        
        # Try to remove each block
        result = list(block_indices)
        changed = True
        while changed:
            changed = False
            new_result = []
            for bi in result:
                # Check if this block is redundant
                redundant = True
                for t_idx in block_t_indices[bi]:
                    if coverage[t_idx] <= 1:
                        redundant = False
                        break
                if redundant:
                    # Remove it
                    coverage[block_t_indices[bi]] -= 1
                    changed = True
                else:
                    new_result.append(bi)
            result = new_result
        return result
    
    def local_search(block_indices, max_iters=200):
        """Try to improve by swapping one block for another."""
        if not block_indices:
            return block_indices
        
        coverage = np.zeros(num_t_subsets, dtype=np.int32)
        for bi in block_indices:
            coverage[block_t_indices[bi]] += 1
        
        selected_set = set(block_indices)
        current_len = len(block_indices)
        
        for _ in range(max_iters):
            if time_left() < 0.5:
                break
            
            improved = False
            # Try removing each block and see if we can add a better one
            for i, bi in enumerate(block_indices):
                if time_left() < 0.5:
                    break
                
                # Temporarily remove
                coverage[block_t_indices[bi]] -= 1
                
                # Find uncovered t-subsets
                uncovered = np.where(coverage == 0)[0]
                if len(uncovered) == 0:
                    # We can just remove this block
                    block_indices.pop(i)
                    selected_set.discard(bi)
                    improved = True
                    break
                
                # Find a block that covers the most uncovered
                # Only consider blocks not already selected
                gains = np.zeros(num_k_blocks, dtype=np.int32)
                for t_idx in uncovered:
                    # Which blocks cover this t-subset?
                    # This is expensive, so skip for now
                    pass
                
                # Restore
                coverage[block_t_indices[bi]] += 1
            
            if not improved:
                break
        
        return block_indices
    
    # Main search: multiple greedy restarts with different weightings
    best_solution = None
    best_len = float('inf')
    
    num_restarts = 50
    for restart in range(num_restarts):
        if time_left() < 2.0:
            break
        
        # Randomize the order of blocks by shuffling
        np.random.seed(restart * 42 + 7)
        
        # Weighted greedy: bias towards blocks that cover rare t-subsets
        # Compute frequency of each t-subset across all blocks
        freq = np.zeros(num_t_subsets, dtype=np.float64)
        for bi in range(num_k_blocks):
            freq[block_t_indices[bi]] += 1
        
        # Inverse frequency as weight (rarer t-subsets get higher weight)
        # Add some randomness
        weights = 1.0 / (freq + 1.0)
        weights += np.random.uniform(0, 0.1, size=num_t_subsets)
        
        # For weighted greedy, we need to modify the gain calculation
        # Gain = sum of weights for uncovered t-subsets covered by this block
        coverage = np.zeros(num_t_subsets, dtype=np.int32)
        used = np.zeros(num_k_blocks, dtype=bool)
        selected = []
        
        while time_left() > 0.5:
            uncovered_mask = (coverage == 0)
            
            # Weighted gain
            weighted_gains = np.zeros(num_k_blocks, dtype=np.float64)
            for bi in range(num_k_blocks):
                if used[bi]:
                    continue
                t_idxs = block_t_indices[bi]
                weighted_gains[bi] = np.sum(weights[t_idxs] * uncovered_mask[t_idxs])
            
            best_idx = np.argmax(weighted_gains)
            if weighted_gains[best_idx] <= 0:
                break
            
            selected.append(best_idx)
            used[best_idx] = True
            coverage[block_t_indices[best_idx]] += 1
        
        # Post-process: remove redundant blocks
        selected = remove_redundant(selected)
        
        # Local search
        selected = local_search(selected, max_iters=50)
        
        if len(selected) < best_len:
            best_len = len(selected)
            best_solution = list(selected)
            
            # Early stop if we hit a good target
            if best_len <= 151:
                break
    
    if best_solution is None:
        # Fallback: simple greedy
        best_solution = greedy_with_weights(None)
    
    # Final deduplication
    seen = set()
    unique_blocks = []
    for bi in best_solution:
        block = tuple(k_blocks[bi])
        if block not in seen:
            seen.add(block)
            unique_blocks.append(bi)
    
    # Convert to lists of integers
    result = [list(k_blocks[bi]) for bi in unique_blocks]
    
    return result
