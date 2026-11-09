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
    tsub_to_blocks = [[] for _ in range(n_tsubs)]
    for b_idx, tsubs in enumerate(block_tsubs):
        for s_idx in tsubs:
            tsub_to_blocks[s_idx].append(b_idx)
    
    # cover_count[s] = number of selected blocks covering t-subset s
    cover_count = [0] * n_tsubs
    # block_gain[b] = number of uncovered t-subsets covered by block b
    block_gain = [len(tsubs) for tsubs in block_tsubs]
    
    selected = []
    remaining = set(range(n_blocks))
    
    while True:
        # Check if all covered
        if all(cover_count[s] > 0 for s in range(n_tsubs)):
            break
        
        # Pick block with max gain
        best_block = -1
        best_gain = -1
        for b in remaining:
            if block_gain[b] > best_gain:
                best_gain = block_gain[b]
                best_block = b
        
        if best_block == -1 or best_gain <= 0:
            break
        
        selected.append(all_blocks[best_block])
        remaining.discard(best_block)
        
        # Update
        for s_idx in block_tsubs[best_block]:
            if cover_count[s_idx] == 0:
                cover_count[s_idx] = 1
                # Decrement gain for all blocks covering this t-subset
                for b in tsub_to_blocks[s_idx]:
                    if b in remaining:
                        block_gain[b] -= 1
    
    return [list(b) for b in selected]
