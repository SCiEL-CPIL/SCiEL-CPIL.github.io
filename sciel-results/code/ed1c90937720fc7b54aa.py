import itertools
import time
import random
import heapq
from collections import defaultdict

# Precomputation for v=19, k=7, t=4
# Since the task says "evaluator passes this task's parameters v, k and t", 
# but the specific task is C(19,7,4), we can hardcode optimizations for this case 
# or keep it general but efficient. 
# Given the "rewrite from scratch" and specific task context, 
# and the fact that v=19 is small, we can precompute all blocks and their t-subsets.

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin

    # For general v, k, t, this might be slow if v is large.
    # But for v=19, k=7, t=4, it's fine.
    # If v is larger, we might need to generate blocks on the fly or use a different heuristic.
    # The prompt implies the task is specifically C(19,7,4).
    
    if v != 19 or k != 7 or t != 4:
        # Fallback to a simpler greedy for other parameters
        return _simple_greedy(v, k, t, deadline)

    # Precompute all t-subsets and map to indices
    all_t_subsets = list(itertools.combinations(range(v), t))
    t_subset_to_idx = {ts: i for i, ts in enumerate(all_t_subsets)}
    n_t_subsets = len(all_t_subsets)
    
    # Precompute all k-subsets (blocks)
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # For each block, store the list of t-subset indices it covers
    block_t_indices = []
    for block in all_blocks:
        indices = []
        for ts in itertools.combinations(block, t):
            indices.append(t_subset_to_idx[ts])
        block_t_indices.append(indices)
        
    # Represent uncovered as a list of bools or a bitset.
    # Python integers can be used as bitsets.
    # n_t_subsets = 3876. This fits in an integer.
    
    def _lazy_greedy(seed):
        rng = random.Random(seed)
        uncovered_mask = (1 << n_t_subsets) - 1  # All bits set to 1
        selected_blocks = []
        
        # Heap items: (-gain, block_idx, last_calculated_gain)
        # We need to lazily update gains.
        # Initial gain for each block is the number of t-subsets it covers.
        # Since all are uncovered initially, gain = len(block_t_indices[b])
        
        heap = []
        for b in range(n_blocks):
            gain = len(block_t_indices[b])
            heapq.heappush(heap, (-gain, b, gain))
            
        while uncovered_mask:
            if time.time() > deadline - 5.0:
                break # Timeout, return partial (invalid)
                
            neg_gain, b, old_gain = heapq.heappop(heap)
            gain = -neg_gain
            
            # Calculate current gain
            current_gain = 0
            for idx in block_t_indices[b]:
                if (uncovered_mask >> idx) & 1:
                    current_gain += 1
                    
            if current_gain != gain:
                # Gain changed, re-insert with new gain
                if current_gain > 0:
                    heapq.heappush(heap, (-current_gain, b, current_gain))
                continue
            
            # If current_gain is 0, we shouldn't have popped it if we check correctly, 
            # but due to laziness, it might be in the heap with old high gain.
            if current_gain == 0:
                continue
                
            # Select block b
            selected_blocks.append(b)
            
            # Update uncovered mask
            for idx in block_t_indices[b]:
                uncovered_mask &= ~(1 << idx)
                
        return selected_blocks

    def _local_search(block_indices):
        """
        Takes a list of block indices, tries to reduce the count.
        """
        blocks = [all_blocks[i] for i in block_indices]
        
        # Compute coverage count for each t-subset
        # cov_count[idx] = number of blocks covering t-subset idx
        cov_count = [0] * n_t_subsets
        
        for b_idx in block_indices:
            for idx in block_t_indices[b_idx]:
                cov_count[idx] += 1
                
        # Check validity
        if any(c == 0 for c in cov_count):
            return block_indices # Invalid, return as is
            
        # Local search loop
        changed = True
        while changed:
            changed = False
            if time.time() > deadline - 2.0:
                break
                
            # 1. Try to remove redundant blocks
            for i in range(len(block_indices)):
                b_idx = block_indices[i]
                redundant = True
                for idx in block_t_indices[b_idx]:
                    if cov_count[idx] == 1:
                        redundant = False
                        break
                if redundant:
                    # Remove block
                    for idx in block_t_indices[b_idx]:
                        cov_count[idx] -= 1
                    block_indices.pop(i)
                    changed = True
                    break # Restart scan after removal
                    
            if not changed:
                # 2. Try swaps? 
                # Swaps are harder. Let's stick to removal for now.
                # If no removal, we are at a local optimum for removal.
                break
                
        return block_indices

    best_blocks = None
    best_size = float('inf')
    
    # Run multiple lazy greedy instances
    num_restarts = 10
    for seed in range(num_restarts):
        if time.time() > deadline - 10.0:
            break
            
        block_indices = _lazy_greedy(seed)
        
        # Post-process with local search
        block_indices = _local_search(block_indices)
        
        # Check validity of the final set
        # (Local search should maintain validity, but double check)
        cov_count = [0] * n_t_subsets
        for b_idx in block_indices:
            for idx in block_t_indices[b_idx]:
                cov_count[idx] += 1
                
        if any(c == 0 for c in cov_count):
            continue # Invalid
            
        if len(block_indices) < best_size:
            best_size = len(block_indices)
            best_blocks = [all_blocks[i] for i in block_indices]
            
    if best_blocks:
        return best_blocks
    else:
        # Fallback if all failed (should not happen)
        return _simple_greedy(v, k, t, deadline)

def _simple_greedy(v, k, t, deadline):
    """
    Fallback greedy for general v, k, t.
    """
    start_time = time.time()
    uncovered = set(itertools.combinations(range(v), t))
    blocks = []
    
    while uncovered:
        if time.time() > deadline:
            break
            
        best_block = None
        max_covered = 0
        
        # Generate candidates. For large v, this is expensive.
        # For v=19, we can iterate all blocks.
        # But this is a fallback.
        
        # Pick a random uncovered t-subset to guide candidate generation?
        # Or just iterate all blocks if v is small.
        
        if v <= 20:
            for block in itertools.combinations(range(v), k):
                covered = 0
                block_set = set(block)
                for ts in itertools.combinations(block, t):
                    if ts in uncovered:
                        covered += 1
                if covered > max_covered:
                    max_covered = covered
                    best_block = block
                    
            if best_block:
                blocks.append(list(best_block))
                block_set = set(best_block)
                for ts in itertools.combinations(best_block, t):
                    if ts in uncovered:
                        uncovered.remove(ts)
        else:
            # Heuristic for larger v
            # Pick random uncovered t-subset
            target = random.choice(list(uncovered))
            block = list(target)
            # Complete greedily
            while len(block) < k:
                best_x = None
                best_gain = 0
                block_set = set(block)
                for x in range(v):
                    if x in block_set:
                        continue
                    gain = 0
                    for rest in itertools.combinations(block, t-1):
                        cand = tuple(sorted(rest + (x,)))
                        if cand in uncovered:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                if best_x is not None:
                    block.append(best_x)
                else:
                    break
            if len(block) == k:
                blocks.append(sorted(block))
                block_set = set(block)
                for ts in itertools.combinations(block, t):
                    if ts in uncovered:
                        uncovered.remove(ts)
                        
    return blocks
