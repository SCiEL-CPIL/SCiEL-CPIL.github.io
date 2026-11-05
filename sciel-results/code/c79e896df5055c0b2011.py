import itertools
import random
import sys
import time
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    time_limit = 115  # Leave 5s margin
    
    # Precompute all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    t_subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    M = len(all_t_subsets)  # 11628 for v=19, t=5
    
    # Precompute all k-blocks and their coverage bitmasks
    all_blocks = list(itertools.combinations(range(v), k))
    B = len(all_blocks)  # 75582 for v=19, k=8
    
    # For each block, compute which t-subsets it covers as a bitmask (Python int)
    block_mask = [0] * B
    # Also: for each t-subset, which blocks cover it
    subset_blocks = [[] for _ in range(M)]
    
    for bi, block in enumerate(all_blocks):
        mask = 0
        for ts in itertools.combinations(block, t):
            idx = t_subset_to_idx[ts]
            mask |= (1 << idx)
            subset_blocks[idx].append(bi)
        block_mask[bi] = mask
    
    # Create mapping from block tuple to index
    block_to_idx = {tuple(b): i for i, b in enumerate(all_blocks)}
    
    # Greedy to get initial solution with multiple random restarts
    def greedy_cover(seed=None):
        if seed is not None:
            random.seed(seed)
        uncovered = (1 << M) - 1
        blocks = []
        while uncovered:
            # Pick a random uncovered t-subset
            lsb = uncovered & (-uncovered)
            idx = lsb.bit_length() - 1
            
            # Try all blocks covering this t-subset
            best_block = -1
            best_gain = -1
            for bi in subset_blocks[idx]:
                gain = (block_mask[bi] & uncovered).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_block = bi
            blocks.append(all_blocks[best_block])
            uncovered &= ~block_mask[best_block]
        return blocks
    
    # Get initial greedy solution with multiple restarts
    initial_blocks = greedy_cover()
    best_blocks = initial_blocks
    best_len = len(initial_blocks)
    
    # Try a few random restarts
    for attempt in range(5):
        if time.time() - start_time > time_limit - 30:
            break
        candidate = greedy_cover(seed=attempt * 1000)
        if len(candidate) < best_len:
            best_len = len(candidate)
            best_blocks = candidate
    
    # Reduction: remove redundant blocks
    def reduce_blocks(blocks):
        if not blocks:
            return blocks
        # Count coverage
        coverage_counts = [0] * M
        block_indices = []
        for b in blocks:
            # Find which precomputed block this is
            b_tuple = tuple(b)
            bi = block_to_idx.get(b_tuple)
            if bi is None:
                continue
            block_indices.append(bi)
            for ts in itertools.combinations(b, t):
                idx = t_subset_to_idx[ts]
                coverage_counts[idx] += 1
        
        improved = True
        while improved:
            improved = False
            for i in range(len(block_indices) - 1, -1, -1):
                bi = block_indices[i]
                can_remove = True
                for ts in itertools.combinations(all_blocks[bi], t):
                    idx = t_subset_to_idx[ts]
                    coverage_counts[idx] -= 1
                    if coverage_counts[idx] == 0:
                        can_remove = False
                        break
                if can_remove:
                    block_indices.pop(i)
                    improved = True
                else:
                    for ts in itertools.combinations(all_blocks[bi], t):
                        idx = t_subset_to_idx[ts]
                        coverage_counts[idx] += 1
        
        return [all_blocks[bi] for bi in block_indices]
    
    best_blocks = reduce_blocks(best_blocks)
    best_len = len(best_blocks)
    
    # Improved B&B with better lower bound
    def greedy_bound(uncovered_mask):
        """Estimate lower bound using greedy on uncovered."""
        if uncovered_mask == 0:
            return 0
        count = 0
        remaining = uncovered_mask
        while remaining:
            # Pick an uncovered t-subset with fewest covering blocks (most constrained)
            best_idx = -1
            min_candidates = float('inf')
            
            # Sample a few uncovered subsets to find the most constrained one
            sample_size = min(5, remaining.bit_count())
            temp_remaining = remaining
            for _ in range(sample_size):
                lsb = temp_remaining & (-temp_remaining)
                idx = lsb.bit_length() - 1
                temp_remaining ^= lsb
                
                # Count how many blocks can still help
                count_blocks = 0
                for bi in subset_blocks[idx]:
                    if (block_mask[bi] & remaining) != 0:
                        count_blocks += 1
                
                if count_blocks < min_candidates:
                    min_candidates = count_blocks
                    best_idx = idx
                
                if count_blocks == 0:
                    return float('inf')
            
            if best_idx == -1:
                return float('inf')
            
            # Find best block for this subset
            best_gain = 0
            best_bi = -1
            for bi in subset_blocks[best_idx]:
                gain = (block_mask[bi] & remaining).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_bi = bi
            
            if best_gain == 0:
                return float('inf')
            
            count += 1
            remaining &= ~block_mask[best_bi]
        return count
    
    # B&B
    best_len_bnb = best_len
    best_blocks_bnb = best_blocks
    nodes_explored = 0
    
    def bb_search(uncovered_mask, current_blocks, depth):
        nonlocal best_len_bnb, best_blocks_bnb, nodes_explored
        
        if time.time() - start_time > time_limit:
            return
        
        nodes_explored += 1
        
        if uncovered_mask == 0:
            if len(current_blocks) < best_len_bnb:
                best_len_bnb = len(current_blocks)
                best_blocks_bnb = list(current_blocks)
            return
        
        # Lower bound
        lb = greedy_bound(uncovered_mask)
        if len(current_blocks) + lb >= best_len_bnb:
            return
        
        # Pick an uncovered t-subset - choose one with fewest candidates
        lsb = uncovered_mask & (-uncovered_mask)
        idx = lsb.bit_length() - 1
        
        # Get candidate blocks covering this t-subset, sorted by coverage
        candidates = []
        for bi in subset_blocks[idx]:
            gain = (block_mask[bi] & uncovered_mask).bit_count()
            if gain > 0:
                candidates.append((gain, bi))
        
        if not candidates:
            return
        
        # Sort by gain descending
        candidates.sort(reverse=True)
        
        # Limit branching factor for speed
        max_branch = min(15, len(candidates))
        for gain, bi in candidates[:max_branch]:
            new_mask = uncovered_mask & ~block_mask[bi]
            current_blocks.append(all_blocks[bi])
            bb_search(new_mask, current_blocks, depth + 1)
            current_blocks.pop()
            
            if time.time() - start_time > time_limit:
                return
    
    # Start B&B
    full_mask = (1 << M) - 1
    bb_search(full_mask, [], 0)
    
    if best_len_bnb < best_len:
        best_blocks = best_blocks_bnb
        best_len = best_len_bnb
    
    return best_blocks
