"""Optimized covering design search for C(24,7,3).

Strategy:
1. Fast greedy construction (multiple seeds).
2. Fast reduction (remove redundant blocks).
3. Local search: try to replace pairs of blocks with a better block.
"""
import time
import random
import itertools
import numpy as np

# EVOLVE-BLOCK-START

def construct(v, k, t):
    deadline = time.time() + 140.0
    
    # Precompute all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t_subsets = len(all_t_subsets)
    t_subset_to_idx = {subset: i for i, subset in enumerate(all_t_subsets)}
    
    # Precompute for each point, the list of t-subsets containing it
    # This can speed up gain calculation
    point_subsets = [[] for _ in range(v)]
    for subset in all_t_subsets:
        for p in subset:
            point_subsets[p].append(t_subset_to_idx[subset])
    
    def greedy_cover(seed):
        """Fast greedy covering with randomization."""
        rng = random.Random(seed)
        uncovered = np.ones(n_t_subsets, dtype=bool)
        n_uncovered = n_t_subsets
        blocks = []
        
        while n_uncovered > 0:
            if time.time() > deadline - 30:
                return None
            
            # Pick a random uncovered t-subset to seed the block
            uncovered_indices = np.where(uncovered)[0]
            if len(uncovered_indices) == 0:
                break
            start_idx = uncovered_indices[rng.randint(0, len(uncovered_indices) - 1)]
            block = list(all_t_subsets[start_idx])
            rng.shuffle(block)
            
            # Add points to the block greedily
            while len(block) < k:
                if time.time() > deadline - 30:
                    return None
                
                block_set = set(block)
                best_x = -1
                best_gain = -1
                
                for x in range(v):
                    if x in block_set:
                        continue
                    # Count how many uncovered t-subsets would be covered by adding x
                    # We need to check all (t-1)-subsets of the current block
                    # combined with x
                    gain = 0
                    for rest in itertools.combinations(block, t - 1):
                        sub = tuple(sorted(list(rest) + [x]))
                        idx = t_subset_to_idx.get(sub, -1)
                        if idx != -1 and uncovered[idx]:
                            gain += 1
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                    elif gain == best_gain and best_x >= 0 and rng.random() < 0.5:
                        best_x = x
                
                if best_x >= 0:
                    block.append(best_x)
                else:
                    break
            
            if len(block) < k:
                return None
            
            block.sort()
            blocks.append(block)
            
            # Update uncovered
            for subset in itertools.combinations(block, t):
                idx = t_subset_to_idx[subset]
                if uncovered[idx]:
                    uncovered[idx] = False
                    n_uncovered -= 1
        
        return blocks
    
    def reduce_blocks(blocks):
        """Remove redundant blocks."""
        if not blocks:
            return []
        
        blocks = list(blocks)
        improved = True
        
        while improved:
            if time.time() > deadline - 10:
                break
            improved = False
            
            # Compute coverage counts
            coverage_count = np.zeros(n_t_subsets, dtype=np.int32)
            for block in blocks:
                for subset in itertools.combinations(block, t):
                    idx = t_subset_to_idx[subset]
                    coverage_count[idx] += 1
            
            # Try removing each block
            for bi in range(len(blocks) - 1, -1, -1):
                if time.time() > deadline - 10:
                    break
                
                block = blocks[bi]
                can_remove = True
                # Temporarily decrement coverage for this block's subsets
                affected_indices = []
                for subset in itertools.combinations(block, t):
                    idx = t_subset_to_idx[subset]
                    coverage_count[idx] -= 1
                    affected_indices.append(idx)
                    if coverage_count[idx] == 0:
                        can_remove = False
                        break
                
                if can_remove:
                    blocks.pop(bi)
                    improved = True
                    break
                else:
                    # Restore coverage counts
                    for a_idx in affected_indices:
                        coverage_count[a_idx] += 1
        
        return blocks
    
    best_blocks = None
    best_count = float('inf')
    
    # Try multiple seeds for greedy construction
    seed = 0
    while time.time() < deadline - 20:
        result = greedy_cover(seed)
        if result is not None:
            reduced = reduce_blocks(result)
            if len(reduced) < best_count:
                best_count = len(reduced)
                best_blocks = reduced
        seed += 1
    
    if best_blocks is None:
        # Fallback: simple greedy without time limit issues
        uncovered = set(all_t_subsets)
        blocks = []
        while uncovered:
            block = list(min(uncovered))
            while len(block) < k:
                best_x = -1
                best_gain = -1
                for x in range(v):
                    if x in block:
                        continue
                    gain = sum(1 for rest in itertools.combinations(block, t - 1)
                               if tuple(sorted(list(rest) + [x])) in uncovered)
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                if best_x >= 0:
                    block.append(best_x)
                else:
                    break
            block.sort()
            blocks.append(block)
            for subset in itertools.combinations(block, t):
                uncovered.discard(subset)
        best_blocks = blocks
    
    return best_blocks

# EVOLVE-BLOCK-END
