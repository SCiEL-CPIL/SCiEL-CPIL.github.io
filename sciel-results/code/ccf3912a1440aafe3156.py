import time
import itertools
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # All t-subsets
    all_tsubs = list(itertools.combinations(range(v), t))
    uncovered = set(all_tsubs)
    
    # Frequency of each point in current blocks
    freq = [0] * v
    blocks = []
    
    # Helper to check time
    def time_up():
        return time.time() > deadline
    
    # Greedy Phase
    while uncovered and not time_up():
        # 1. Select the "most urgent" uncovered subset
        # Heuristic: Minimize sum of frequencies of its elements
        # If all freqs are 0, pick arbitrary (or first)
        best_sub = None
        best_cost = float('inf')
        
        # Optimization: If many are 0, we can pick one quickly.
        # But iterating is fine for 3876 items.
        for sub in uncovered:
            cost = sum(freq[p] for p in sub)
            if cost < best_cost:
                best_cost = cost
                best_sub = sub
                if cost == 0:
                    break # Can't do better than 0
        
        if best_sub is None:
            break
            
        # 2. Construct a block containing best_sub
        block = list(best_sub)
        
        while len(block) < k:
            # Find point x to add that maximizes gain
            best_x = None
            best_gain = -1
            block_set = set(block)
            
            # Precompute subsets of block of size t-1
            # We need to count how many uncovered sets of form R U {x} exist
            # R is a subset of block of size t-1
            # Gain(x) = sum( 1 for R in C(block, t-1) if (R U {x}) in uncovered )
            
            candidates = [x for x in range(v) if x not in block_set]
            
            # If t-1 == 0, gain is just 1 if {x} is uncovered? No, t>=2 usually.
            # For t=4, t-1=3.
            
            # To speed up, we can group candidates by their potential.
            # But simple iteration is fast enough.
            
            for x in candidates:
                gain = 0
                # Iterate over all subsets R of block of size t-1
                for R in itertools.combinations(block, t-1):
                    # Check if R U {x} is in uncovered
                    # R is a tuple of t-1 elements. x is int.
                    # Create tuple and sort for set lookup
                    # Optimization: R is sorted if block is sorted.
                    # But block is not necessarily sorted yet.
                    # We can keep block sorted.
                    pass # Placeholder, will implement properly below
            
            # Proper implementation of gain calculation
            # Keep block sorted to make combination generation predictable?
            # Actually, itertools.combinations works on any iterable.
            # But for set lookup, we need a canonical tuple.
            
            # Let's sort block to ensure R is sorted.
            block.sort()
            
            for x in candidates:
                gain = 0
                for R in itertools.combinations(block, t-1):
                    # R is sorted. x is int.
                    # We need to check if sorted(R + (x,)) is in uncovered
                    # Since R is sorted, we can insert x and sort.
                    # Or just create tuple and sort.
                    # t is small (4), so R has 3 elements.
                    # tuple(sorted(R + (x,))) is fast.
                    new_sub = tuple(sorted(R + (x,)))
                    if new_sub in uncovered:
                        gain += 1
                        # Early exit if gain exceeds best_gain?
                        # Yes.
                        if gain > best_gain:
                            # We can break inner loop if we know we can't beat best_gain?
                            # Max possible gain is C(len(block), t-1).
                            # But we don't know how many are uncovered.
                            # Just continue.
                            pass
                
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
                    if best_gain == 0:
                        # If no gain, break early?
                        # We still need to pick a point.
                        # But if all gains are 0, we pick the first one.
                        # We can break outer loop if we find 0? No, maybe another x has 0 too.
                        # But if best_gain is 0, we can't do better.
                        # So we can break.
                        break

            if best_x is None:
                # Should not happen if uncovered is not empty
                break
            
            block.append(best_x)
        
        block.sort()
        blocks.append(block)
        
        # Update uncovered and freq
        for sub in itertools.combinations(block, t):
            if sub in uncovered:
                uncovered.remove(sub)
                # Update freq?
                # Freq is updated per point in block.
        
        for p in block:
            freq[p] += 1
    
    # Post-processing: Remove redundant blocks
    # Count coverage for each t-subset
    if not time_up() and blocks:
        cover_count = {}
        for sub in all_tsubs:
            cover_count[sub] = 0
        
        for block in blocks:
            for sub in itertools.combinations(block, t):
                cover_count[sub] += 1
        
        # Try to remove blocks
        # A block is redundant if for all sub in C(block, t), cover_count[sub] > 1
        # AND it is the only block covering some sub? No.
        # If cover_count[sub] == 1, the block covering it is essential.
        # So a block can be removed only if it does NOT uniquely cover any sub.
        
        # Let's identify essential blocks.
        # Actually, simpler:
        # For each block, check if it is the unique cover for any of its t-subsets.
        # If yes, keep it.
        # If no, remove it and decrement counts.
        
        # To do this efficiently, we can iterate blocks.
        # But removing a block changes cover_count, which might make other blocks non-essential.
        # So we should iterate until no more removals.
        
        removed_any = True
        while removed_any and not time_up():
            removed_any = False
            new_blocks = []
            for block in blocks:
                is_essential = False
                for sub in itertools.combinations(block, t):
                    if cover_count[sub] == 1:
                        is_essential = True
                        break
                if not is_essential:
                    # Remove block
                    for sub in itertools.combinations(block, t):
                        cover_count[sub] -= 1
                    removed_any = True
                else:
                    new_blocks.append(block)
            blocks = new_blocks
    
    return blocks
