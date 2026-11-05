import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Precompute all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t_subsets = len(all_t_subsets)
    
    best_blocks = None
    best_count = float('inf')
    
    # Run randomized greedy multiple times
    while time.time() < deadline:
        uncovered = set(all_t_subsets)
        blocks = []
        
        # Shuffle the order in which we consider t-subsets to start new blocks
        # This randomizes the "greedy" aspect
        shuffled_subsets = all_t_subsets[:]
        random.shuffle(shuffled_subsets)
        
        for subset in shuffled_subsets:
            if subset not in uncovered:
                continue
            
            # Start a new block with this subset
            block = list(subset)
            
            # Extend the block to size k
            while len(block) < k:
                best_point = -1
                max_gain = -1
                
                # Try each point not in the block
                for x in range(v):
                    if x in block:
                        continue
                    
                    # Calculate gain: how many new t-subsets are covered
                    # by adding x to the current block prefix
                    # New t-subsets must contain x and (t-1) elements from current block
                    # Actually, any t-subset of (block + [x]) that was not in block
                    # Since t=3, and block size grows, we just check all t-subsets of the new set
                    # But we only care about those that are in 'uncovered'
                    
                    # Optimization: 
                    # The new subsets covered are those that include x and t-1 elements from 'block'.
                    # Wait, no. The block is being built. The covered subsets are ALL t-subsets of the final block.
                    # When we add x, the *newly* covered subsets are those that contain x and t-1 elements from the current 'block'.
                    # Let's verify: 
                    # Current block P. Subsets of P are already considered/covered (or will be).
                    # New block P' = P U {x}.
                    # Subsets of P' are Subsets(P) U {Subsets containing x}.
                    # Subsets containing x are formed by x and (t-1) elements from P.
                    # So we just need to count how many such subsets are in 'uncovered'.
                    
                    gain = 0
                    # Generate combinations of (t-1) elements from 'block'
                    # For each such combination 'rest', the subset is sorted(rest + (x,))
                    # Check if it's in uncovered
                    for rest in itertools.combinations(block, t - 1):
                        # Create the t-subset tuple
                        # Need to sort to match the canonical form in 'uncovered'
                        # Since 'block' is not necessarily sorted, and 'rest' is a combo from it,
                        # we should ensure the tuple is sorted.
                        # Actually, 'uncovered' contains sorted tuples.
                        # 'rest' is a tuple of elements from 'block'.
                        # We form a set {x} U set(rest), then sort it.
                        # Or simply: tuple(sorted(list(rest) + [x]))
                        new_sub = tuple(sorted(list(rest) + [x]))
                        if new_sub in uncovered:
                            gain += 1
                    
                    if gain > max_gain:
                        max_gain = gain
                        best_point = x
                
                # If no point increases the count (max_gain == 0), we still need to fill the block
                # to size k, but it won't cover any *new* uncovered subsets.
                # We should pick an arbitrary point to fill it, or break if we don't need to?
                # The block MUST have size k.
                # If max_gain is 0, any point is equally good (gain 0).
                # We pick the first one to save time, or randomize?
                # Randomizing might help avoid bias.
                if best_point == -1:
                    # Pick a random point not in block
                    candidates = [x for x in range(v) if x not in block]
                    if candidates:
                        best_point = random.choice(candidates)
                    else:
                        # Should not happen if v >= k
                        break
                
                block.append(best_point)
            
            # Block is complete. Sort it for consistency (though not strictly required for validity, good for dedup)
            block.sort()
            blocks.append(block)
            
            # Update uncovered set
            # Remove all t-subsets of this block from uncovered
            for sub in itertools.combinations(block, t):
                uncovered.discard(tuple(sub))
            
            # Early exit if all covered
            if not uncovered:
                break
        
        # Check if this run is better
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks
            
            # Optional: if we hit a very good number, we could stop or tighten the deadline
            # But let's keep searching for a bit longer to see if we can beat it.
            
    return best_blocks
