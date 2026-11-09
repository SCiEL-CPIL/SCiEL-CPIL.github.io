import time
import random
from itertools import combinations

# Global deadline guard
START_TIME = time.time()
TIME_LIMIT = 140.0  # Safety margin for 150s limit

def construct():
    """
    Constructs the Steiner system S(5,6,12) using seeded random greedy search.
    Returns a list of 132 distinct 6-subsets of 0..11.
    """
    points = list(range(12))
    all_6_subsets = list(combinations(points, 6))
    # Precompute 5-subsets for each 6-subset for faster checking
    # subset_to_5s maps a 6-tuple to a list of 5-tuples
    subset_to_5s = {b: list(combinations(b, 5)) for b in all_6_subsets}
    
    best_blocks = []
    best_count = 0
    
    # Try multiple random seeds to find the optimal 132 blocks
    # The optimal solution is unique up to isomorphism, but random greedy can find it.
    for seed in range(1000):
        if time.time() - START_TIME > TIME_LIMIT:
            break
            
        random.seed(seed)
        covered_5s = set()
        blocks = []
        
        # Create a list of all 6-subsets and shuffle it
        candidates = all_6_subsets[:]
        random.shuffle(candidates)
        
        for b in candidates:
            if time.time() - START_TIME > TIME_LIMIT:
                break
                
            # Check if any 5-subset of b is already covered
            # Optimization: break early if a conflict is found
            conflict = False
            for s5 in subset_to_5s[b]:
                if s5 in covered_5s:
                    conflict = True
                    break
                    
            if not conflict:
                blocks.append(list(b))
                covered_5s.update(subset_to_5s[b])
                
                # If we reached the optimum, we can stop immediately
                if len(blocks) == 132:
                    return blocks
        
        # Update best result
        if len(blocks) > best_count:
            best_count = len(blocks)
            best_blocks = blocks
            
        # If we found the optimum, return
        if best_count == 132:
            return best_blocks

    return best_blocks

# EVOLVE-BLOCK-START
# The construct function is defined above, but we need to ensure the interface matches.
# The task says "Keep the function names and signatures of the seed program."
# The seed program had `def construct():` inside the block or outside?
# The prompt shows:
# """Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# # EVOLVE-BLOCK-START
# def construct():
#     ...
# # EVOLVE-BLOCK-END
#
# So I should put the entire logic inside the EVOLVE-BLOCK markers if possible, 
# or just replace the content of the block. 
# However, helper functions can be outside if they are not part of the "interface" 
# but the prompt implies the code block is the whole program.
# Let's structure it cleanly.
# EVOLVE-BLOCK-END

# Actually, the prompt asks to "Output the COMPLETE program in a single
