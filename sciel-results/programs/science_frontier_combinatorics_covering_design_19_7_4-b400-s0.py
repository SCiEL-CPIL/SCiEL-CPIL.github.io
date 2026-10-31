import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit

    # 1. Precomputation
    # Generate all t-subsets
    all_tsubs = list(itertools.combinations(range(v), t))
    tsub_set = set(all_tsubs)
    
    # Map t-subset to index for faster counting if needed, but set is fine
    # Precompute t-subsets for every possible k-block
    # blocks_for_tsub: dict mapping t-subset -> list of k-blocks containing it
    # tsubs_for_block: dict mapping k-block -> list of t-subsets contained in it
    
    blocks_for_tsub = {ts: [] for ts in all_tsubs}
    tsubs_for_block = {}
    
    all_kblocks = list(itertools.combinations(range(v), k))
    
    for B in all_kblocks:
        B_set = set(B)
        # Get all t-subsets of B
        ts_list = []
        for ts in itertools.combinations(B, t):
            ts_list.append(ts)
            blocks_for_tsub[ts].append(B)
        tsubs_for_block[B] = ts_list

    # 2. Greedy Construction
    uncovered = set(all_tsubs)
    blocks = []
    
    while uncovered:
        if time.time() > deadline:
            break
            
        # Heuristic: Pick the uncovered t-subset with the fewest candidate blocks
        # that still have high potential?
        # Simple heuristic: Pick min(uncovered)
        # Better heuristic: Pick the one that is "hardest" to cover.
        # Let's try picking the one with the smallest number of blocks in blocks_for_tsub[ts]
        # that are "active" (i.e., cover at least one other uncovered subset)?
        # This is expensive. Let's stick to min(uncovered) for speed, 
        # or random if we want diversity. 
        # Actually, min(uncovered) is deterministic and fast.
        
        pivot = min(uncovered)
        
        candidates = blocks_for_tsub[pivot]
        
        best_block = None
        max_gain = -1
        
        for B in candidates:
            # Calculate gain: number of uncovered t-subsets covered by B
            gain = 0
            for ts in tsubs_for_block[B]:
                if ts in uncovered:
                    gain += 1
            
            if gain > max_gain:
                max_gain = gain
                best_block = B
                if gain == len(tsubs_for_block[B]):
                    # Can't do better than covering all its t-subsets
                    break
        
        if best_block is not None:
            blocks.append(best_block)
            # Update uncovered
            for ts in tsubs_for_block[best_block]:
                uncovered.discard(ts)
        
        if time.time() > deadline:
            break

    # 3. Post-processing: Remove redundant blocks
    # This can significantly reduce the number of blocks.
    if time.time() < deadline - 5:
        # Build cover counts
        cover_count = {ts: 0 for ts in all_tsubs}
        for B in blocks:
            for ts in tsubs_for_block[B]:
                cover_count[ts] += 1
        
        changed = True
        while changed and time.time() < deadline - 5:
            changed = False
            new_blocks = []
            for B in blocks:
                # Check if B is redundant
                redundant = True
                for ts in tsubs_for_block[B]:
                    if cover_count[ts] == 1:
                        redundant = False
                        break
                
                if redundant:
                    # Remove B
                    for ts in tsubs_for_block[B]:
                        cover_count[ts] -= 1
                    changed = True
                else:
                    new_blocks.append(B)
            blocks = new_blocks
            
    return blocks
