import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # 10s buffer before 120s limit

    all_t_subsets = list(itertools.combinations(range(v), t))
    
    # Helper to check time
    def time_up():
        return time.time() > deadline

    # Greedy Construction
    def greedy_construct():
        uncovered = set(all_t_subsets)
        blocks = []
        # Shuffle to allow different runs if we restart
        points = list(range(v))
        
        while uncovered:
            if time_up():
                return blocks
            # Pick the first uncovered subset (or random for diversity)
            # Using min for determinism in single run, but we can randomize
            # Let's pick a random uncovered subset to break symmetry
            target = random.choice(list(uncovered))
            block = list(target)
            
            # Fill the block
            while len(block) < k:
                if time_up():
                    break
                best_point = -1
                max_gain = -1
                # Candidates are points not in block
                candidates = [x for x in range(v) if x not in block]
                for x in candidates:
                    # Calculate gain: how many uncovered subsets are covered by block + [x]
                    # A subset S is covered if S is subset of block + [x]
                    # This is equivalent to: S is in uncovered AND S is subset of (block + [x])
                    # Since S has size t, and block has size < k, 
                    # S must contain x and t-1 elements from block?
                    # No, S can be any t-subset of block+[x].
                    # If S does not contain x, it must be subset of block.
                    # But if S is subset of block, it was already covered by previous steps?
                    # No, we add points one by one.
                    # Actually, if S is subset of block, it was covered when the last point of S was added?
                    # Not necessarily, we haven't added the block to 'blocks' yet.
                    # But we are building ONE block.
                    # The 'uncovered' set represents subsets NOT covered by FINALIZED blocks.
                    # So if S is subset of current 'block', it is not yet covered by finalized blocks.
                    # So gain should count ALL S in uncovered that are subset of block+[x].
                    
                    gain = 0
                    # Optimization: only check subsets of block+[x]
                    # block+[x] has size len(block)+1
                    # t-subsets of it
                    # If len(block) < t-1, no t-subset can be formed?
                    # If len(block) < t-1, we must add t-1-len(block) points from future steps?
                    # No, we just add 1 point at a time.
                    # If len(block) < t-1, then block+[x] has size < t. No t-subsets. Gain 0.
                    if len(block) + 1 >= t:
                        for s in itertools.combinations(block + [x], t):
                            if s in uncovered:
                                gain += 1
                    if gain > max_gain:
                        max_gain = gain
                        best_point = x
                
                if best_point == -1:
                    break
                block.append(best_point)
            
            block.sort()
            blocks.append(block)
            
            # Update uncovered
            for s in itertools.combinations(block, t):
                if s in uncovered:
                    uncovered.remove(s)
        
        return blocks

    def local_search(blocks):
        # counts[s] = number of blocks covering s
        counts = {s: 0 for s in all_t_subsets}
        
        # Initialize counts
        for b in blocks:
            for s in itertools.combinations(b, t):
                counts[s] += 1
        
        # Precompute subsets for each block? No, blocks change.
        
        improved = True
        while improved and not time_up():
            improved = False
            
            # 1. Redundancy Removal
            new_blocks = []
            for b in blocks:
                # Check if b is redundant
                redundant = True
                for s in itertools.combinations(b, t):
                    if counts[s] == 1:
                        redundant = False
                        break
                if redundant:
                    # Remove b
                    for s in itertools.combinations(b, t):
                        counts[s] -= 1
                    improved = True
                else:
                    new_blocks.append(b)
            blocks = new_blocks
            
            # 2. Swap Search
            # Try to swap points in blocks
            # To be efficient, limit swaps per pass or use random sampling
            # We will iterate through blocks and try swaps
            for i, b in enumerate(blocks):
                if time_up():
                    return blocks
                
                # Try swapping each point in b with each point not in b
                # This is O(k * v * (cost of check))
                # Cost of check: 
                #   1. Temporarily remove b from counts
                #   2. Check if any s in b has count 0
                #   3. If yes, check if s is in b'
                #   4. If all such s are in b', accept
                #   5. Add b' to counts
                
                # Optimization: Only try a few random swaps per block per pass?
                # Or all? k=8, v=19. 8 * 11 = 88 swaps per block.
                # Number of blocks ~ 300. 300 * 88 = 26400 checks.
                # Each check involves iterating t-subsets of b (126 subsets).
                # 26400 * 126 ~ 3.3M ops. Should be fast in Python if optimized.
                
                for p in b:
                    for q in range(v):
                        if q == p or q in b:
                            continue
                        
                        b_new = b.copy()
                        b_new.remove(p)
                        b_new.append(q)
                        b_new.sort()
                        
                        # Check validity
                        # 1. Remove b from counts
                        for s in itertools.combinations(b, t):
                            counts[s] -= 1
                        
                        # 2. Check for uncovered subsets in b
                        valid = True
                        # We only care about subsets of b that became uncovered (count 0)
                        # And check if they are covered by b_new
                        for s in itertools.combinations(b, t):
                            if counts[s] == 0:
                                # This subset was only covered by b
                                # It must be covered by b_new
                                if s not in set(itertools.combinations(b_new, t)):
                                    valid = False
                                    break
                        
                        if not valid:
                            # Revert counts
                            for s in itertools.combinations(b, t):
                                counts[s] += 1
                            continue
                        
                        # 3. If valid, accept swap
                        # Add b_new to counts
                        for s in itertools.combinations(b_new, t):
                            counts[s] += 1
                        
                        blocks[i] = b_new
                        improved = True
                        # Break out of q loop? Or continue to try more swaps?
                        # Continuing might find better swaps, but takes time.
                        # Let's break to restart the pass with new structure
                        break
                    if improved:
                        break
                if improved:
                    break
        
        return blocks

    best_blocks = []
    
    # Run multiple restarts if time permits
    # First run is greedy + local search
    blocks = greedy_construct()
    blocks = local_search(blocks)
    best_blocks = blocks
    
    # If time remains, try more restarts
    while time.time() < deadline - 5.0:
        blocks = greedy_construct()
        blocks = local_search(blocks)
        if len(blocks) < len(best_blocks):
            best_blocks = blocks
        # If we hit the lower bound or a very good number, we can stop
        # Lower bound is 264. If we get close, stop?
        # 335 is best known. If we get < 335, great.
        if len(best_blocks) <= 300:
            break

    return best_blocks
