import itertools
import time
import sys
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115  # 5s safety margin

    # 1. Generate all t-subsets and map to indices
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subsets)
    t_subset_to_idx = {ts: i for i, ts in enumerate(all_t_subsets)}
    
    # 2. Generate all k-subsets (blocks) and precompute which t-subsets they cover
    all_k_blocks = list(itertools.combinations(range(v), k))
    n_k = len(all_k_blocks)
    
    # block_to_t_subsets[b] is a list of indices of t-subsets covered by block b
    block_to_t_subsets = []
    # t_sub_to_blocks[ts] is a list of block indices covering t-subset ts
    t_sub_to_blocks = [[] for _ in range(n_t)]
    
    for b_idx, b in enumerate(all_k_blocks):
        covered = []
        for ts in itertools.combinations(b, t):
            idx = t_subset_to_idx[ts]
            covered.append(idx)
            t_sub_to_blocks[idx].append(b_idx)
        block_to_t_subsets.append(covered)
        
    # 3. Greedy Construction
    cover_count = [0] * n_t
    active_blocks = []
    
    # Use a set of uncovered t-subsets for fast membership checking
    uncovered = set(range(n_t))
    
    # Initial gains
    gains = [0] * n_k
    for b_idx in range(n_k):
        gains[b_idx] = len(block_to_t_subsets[b_idx])
        
    # Simple priority queue simulation using a list and sorting is O(N log N) per step if done naively.
    # However, N is ~50k. 150 steps * 50k log 50k is manageable? 
    # Actually, re-scanning all blocks each time is O(N_k * |block_t_subsets|) which is ~1.7M ops per step.
    # 150 steps * 1.7M = 255M ops. In Python, this is likely 20-30 seconds.
    # Let's try to optimize the gain update.
    
    while uncovered:
        if time.time() > deadline:
            break
            
        best_block = -1
        best_gain = -1
        
        # Iterate through all possible blocks to find max gain
        # Optimization: We can skip blocks that are already active (gain 0 effectively, or handled separately)
        # But we need to update gains efficiently.
        
        # For now, do the full scan. If too slow, we'll need a heap.
        for b_idx in range(n_k):
            if b_idx in active_blocks: # This check is O(N) per b_idx if active_blocks is a list. Bad.
                continue
            # Use a set for active blocks for O(1) lookup?
            # Let's maintain a set of active block indices.
            pass
            
    # The above loop is inefficient. Let's use a different strategy for the greedy part.
    # We'll maintain a list of (gain, block_idx) in a heap.
    # But updating gains for all blocks covering a new t-subset requires iterating t_sub_to_blocks.
    
    # Let's restart the greedy with a heap.
    import heapq
    
    active_set = set()
    gains = [len(block_to_t_subsets[b_idx]) for b_idx in range(n_k)]
    # Heap contains (-gain, block_idx)
    heap = [(-gains[b_idx], b_idx) for b_idx in range(n_k)]
    heapq.heapify(heap)
    
    # To handle stale entries, we just check if the gain in heap matches current gain
    # and if the block is not already selected.
    
    while uncovered:
        if time.time() > deadline - 5: # Leave time for local search
            break
            
        best_block = -1
        best_gain = 0
        
        # Pop valid entries
        while heap:
            neg_g, b_idx = heapq.heappop(heap)
            g = -neg_g
            # Check if this entry is stale
            # 1. Is block already selected?
            if b_idx in active_set:
                continue
            # 2. Does the gain match the current gain?
            # We need to track current gains.
            # Let's maintain a gains array.
            if g != gains[b_idx]:
                continue
            
            best_block = b_idx
            best_gain = g
            break
            
        if best_block == -1:
            break
            
        # Select best_block
        active_set.add(best_block)
        active_blocks.append(best_block)
        gains[best_block] = -1 # Mark as selected/invalid
        
        # Update uncovered set and gains of other blocks
        new_covered = []
        for s in block_to_t_subsets[best_block]:
            if s in uncovered:
                uncovered.remove(s)
                new_covered.append(s)
                
        # For each newly covered t-subset, decrease gain of all blocks covering it
        for s in new_covered:
            for b_idx in t_sub_to_blocks[s]:
                if b_idx in active_set:
                    continue
                gains[b_idx] -= 1
                heapq.heappush(heap, (-gains[b_idx], b_idx))
                
    # 4. Local Search: Redundancy Removal and Swaps
    # Rebuild cover_count
    cover_count = [0] * n_t
    for b_idx in active_blocks:
        for s in block_to_t_subsets[b_idx]:
            cover_count[s] += 1
            
    improved = True
    while improved and time.time() < deadline - 2:
        improved = False
        
        # Pass 1: Redundancy Removal
        new_active_blocks = []
        for b_idx in active_blocks:
            redundant = True
            for s in block_to_t_subsets[b_idx]:
                if cover_count[s] == 1:
                    redundant = False
                    break
            if redundant:
                for s in block_to_t_subsets[b_idx]:
                    cover_count[s] -= 1
                improved = True
            else:
                new_active_blocks.append(b_idx)
        
        if improved:
            active_blocks = new_active_blocks
            active_set = set(active_blocks)
            continue
            
        # Pass 2: Swaps
        # For each active block, try to swap with a non-active block that covers its unique t-subsets.
        swapped = False
        for b_idx in active_blocks:
            if time.time() > deadline - 2:
                break
                
            unique_ts = []
            for s in block_to_t_subsets[b_idx]:
                if cover_count[s] == 1:
                    unique_ts.append(s)
                    
            if not unique_ts:
                # Should have been removed, but if it happens, remove it
                for s in block_to_t_subsets[b_idx]:
                    cover_count[s] -= 1
                active_blocks.remove(b_idx)
                active_set.remove(b_idx)
                improved = True
                break
                
            # Find candidate blocks that cover all unique_ts
            # Start with the most constrained unique t-subset (smallest list of covering blocks)
            min_ts = min(unique_ts, key=lambda x: len(t_sub_to_blocks[x]))
            candidates = set(t_sub_to_blocks[min_ts])
            
            for s in unique_ts[1:]:
                if not candidates:
                    break
                candidates &= set(t_sub_to_blocks[s])
                
            candidates.discard(b_idx)
            
            if not candidates:
                continue
                
            # Try to find a valid swap
            # We need to ensure that removing b_idx and adding r doesn't break coverage.
            # Since r covers all unique_ts of b_idx, and all other t-subsets of b_idx have cover_count >= 2,
            # removing b_idx will leave those other t-subsets with cover_count >= 1.
            # So any r in candidates is a valid swap.
            # To improve quality, we might prefer r that covers more uncovered t-subsets?
            # But we are in local search, all t-subsets are covered.
            # So any swap is neutral in terms of coverage validity.
            # However, we want to minimize the number of blocks.
            # A 1-for-1 swap doesn't change the count.
            # So swaps are only useful if they enable further redundancy removal.
            
            # Let's just pick the first candidate.
            r = next(iter(candidates))
            
            # Perform swap
            active_set.remove(b_idx)
            active_set.add(r)
            
            # Update cover counts
            for s in block_to_t_subsets[b_idx]:
                cover_count[s] -= 1
            for s in block_to_t_subsets[r]:
                cover_count[s] += 1
                
            # Replace b_idx with r in active_blocks
            # active_blocks is a list. Find and replace.
            # Since we are iterating, this is tricky.
            # Let's just mark improved and restart the loop.
            improved = True
            swapped = True
            break
            
        if swapped:
            # Rebuild active_blocks list from active_set to avoid list modification issues
            active_blocks = list(active_set)
            
    # Convert block indices back to lists of points
    result = [list(all_k_blocks[b_idx]) for b_idx in active_blocks]
    
    # Ensure distinct blocks
    seen = set()
    final_result = []
    for b in result:
        b_tuple = tuple(b)
        if b_tuple not in seen:
            seen.add(b_tuple)
            final_result.append(b)
            
    return final_result
