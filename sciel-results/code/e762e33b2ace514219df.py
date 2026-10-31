"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random


def construct(v, k, t):
    deadline = time.time() + 110.0  # safety margin before 120s limit
    
    # Precompute all t-subsets
    all_tsubs = list(itertools.combinations(range(v), t))
    # Map tuple to index for fast lookup if we used array, but we will use set of tuples
    # Actually, set of tuples is fine for membership checks.
    
    uncovered = set(all_tsubs)
    blocks = []
    
    # To make it deterministic and fast, we can iterate through uncovered subsets
    # But greedy picking the "best" next point is key.
    
    # We need to pick a starting t-subset.
    # Standard greedy: pick any uncovered, then fill it.
    # To improve, we could pick the uncovered subset that is "hardest" to cover, 
    # but for small v, any is fine. Let's just pick the first one from the set.
    # To avoid bias, we can shuffle or just use min().
    
    while uncovered:
        if time.time() > deadline:
            break
            
        # Pick a target t-subset to cover
        # Using min() for determinism
        target = min(uncovered)
        
        # Initialize block with the target elements
        block = list(target)
        block_set = set(block)
        
        # Fill the block to size k
        while len(block) < k:
            best_x = -1
            best_gain = -1
            
            # Candidates are points not in block
            for x in range(v):
                if x in block_set:
                    continue
                
                # Gain: number of uncovered t-subsets S such that S \subseteq block U {x}
                # This implies x in S and S \ {x} \subseteq block
                # So S is formed by x and t-1 elements from block
                
                gain = 0
                # Generate all t-1 combinations from block
                for combo in itertools.combinations(block, t - 1):
                    # Candidate subset is combo + (x,)
                    cand = tuple(sorted(combo + (x,)))
                    if cand in uncovered:
                        gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            
            if best_x == -1:
                # Fallback: pick any available point
                for x in range(v):
                    if x not in block_set:
                        best_x = x
                        break
            
            block.append(best_x)
            block_set.add(best_x)
        
        block.sort()
        blocks.append(block)
        
        # Update uncovered set
        # Remove all t-subsets of the new block
        for sub in itertools.combinations(block, t):
            uncovered.discard(sub)
            
    return blocks
# EVOLVE-BLOCK-END
