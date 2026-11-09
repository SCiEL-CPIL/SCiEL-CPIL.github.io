import heapq
import itertools
from collections import defaultdict

def construct(v, k, t):
    # Precompute all k-blocks
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # Precompute all t-subsets
    all_tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubs)
    
    # Map each t-subset to its index
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}
    
    # For each block, list of t-subset indices it covers
    block_tsubs = []
    for block in all_blocks:
        tsubs_in_block = [tsub_to_idx[ts] for ts in itertools.combinations(block, t)]
        block_tsubs.append(tsubs_in_block)
        
    # For each t-subset index, list of block indices that cover it
    # This is needed to update gains of other blocks
    tsub_to_blocks = [[] for _ in range(n_tsubs)]
    for b_idx, tsubs in enumerate(block_tsubs):
        for s_idx in tsubs:
            tsub_to_blocks[s_idx].append(b_idx)
            
    # current_gain[b] = number of uncovered t-subsets covered by block b
    current_gain = [len(tsubs) for tsubs in block_tsubs]
    
    # cover_count[s] = number of selected blocks covering t-subset s
    cover_count = [0] * n_tsubs
    
    # Heap of (-gain, block_index)
    # We use -gain because heapq is a min-heap
    heap = [(-current_gain[b], b) for b in range(n_blocks)]
    heapq.heapify(heap)
    
    selected = []
    # We don't strictly need a 'remaining' set if we use lazy deletion, 
    # but we need to know if a block is already selected to avoid re-adding it? 
    # Actually, once selected, we don't care about its gain anymore. 
    # Lazy deletion handles stale entries.
    
    uncovered_count = n_tsubs
    
    while uncovered_count > 0 and heap:
        # Pop the best candidate
        neg_gain, b = heapq.heappop(heap)
        gain = -neg_gain
        
        # Check if this entry is stale
        if gain != current_gain[b]:
            continue
            
        if gain <= 0:
            break
            
        # Select block b
        selected.append(all_blocks[b])
        
        # Update cover counts and gains
        for s_idx in block_tsubs[b]:
            if cover_count[s_idx] == 0:
                cover_count[s_idx] = 1
                uncovered_count -= 1
                
                # Decrease gain for all blocks covering this t-subset
                # except the one we just selected (which is no longer in the heap effectively, 
                # but its gain might be referenced by stale entries)
                for other_b in tsub_to_blocks[s_idx]:
                    if other_b != b:
                        current_gain[other_b] -= 1
                        # We do NOT push the new gain to the heap here. 
                        # We rely on the existing entry in the heap being stale 
                        # and being skipped when popped, OR we need to push?
                        # If we don't push, the old entry (with higher gain) will be popped, 
                        # found stale (gain != current_gain), and skipped. 
                        # The new lower gain will never be seen unless we push it.
                        # So we MUST push the new gain to ensure we can select it later if it becomes the best.
                        heapq.heappush(heap, (-current_gain[other_b], other_b))
                        
    return [list(b) for b in selected]
