"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115  # 115s out of 120s limit, leaving 5s safety margin
    
    # Precompute all t-subsets for efficiency
    all_tsubsets = list(itertools.combinations(range(v), t))
    uncovered = set(all_tsubsets)
    blocks = []
    
    while uncovered and time.time() < deadline:
        # Pick a random uncovered t-subset to start the block (randomization for diversity)
        target = random.choice(list(uncovered))
        block = list(target)
        
        # Greedily add points to maximize coverage of uncovered t-subsets
        while len(block) < k:
            block_set = set(block)
            best_gain = -1
            best_point = None
            
            for x in range(v):
                if x in block_set:
                    continue
                
                # Calculate how many uncovered t-subsets this point would help cover
                gain = 0
                for rest in itertools.combinations(block, t - 1):
                    subset = tuple(sorted(rest + (x,)))
                    if subset in uncovered:
                        gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    best_point = x
            
            if best_point is None or best_gain <= 0:
                # If no point adds coverage, just pick a random unused point
                available = [x for x in range(v) if x not in block_set]
                if available:
                    best_point = random.choice(available)
                else:
                    break
            
            block.append(best_point)
        
        if len(block) == k:
            block.sort()
            blocks.append(tuple(block))
            
            # Remove all covered t-subsets
            for tsub in itertools.combinations(block, t):
                uncovered.discard(tsub)
        
        # Check if we're running out of time
        if time.time() > deadline - 5:
            break
    
    # Convert to list of lists for return
    return [list(b) for b in blocks]
# EVOLVE-BLOCK-END
