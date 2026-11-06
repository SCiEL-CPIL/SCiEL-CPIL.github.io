# EVOLVE-BLOCK-START
import time
from itertools import combinations

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Points are 0..11. Let 11 be infinity.
    # Z_11 is 0..10.
    # 1. Find all 5-subsets of Z_11 that are (11,5,2) difference sets.
    diff_sets = []
    for D in combinations(range(11), 5):
        D_set = set(D)
        # Check difference condition
        # Count differences
        diff_counts = [0] * 11
        for d1 in D:
            for d2 in D:
                if d1 != d2:
                    diff_counts[(d1 - d2) % 11] += 1
        # Every non-zero element should appear exactly 2 times
        is_diff_set = True
        for x in range(1, 11):
            if diff_counts[x] != 2:
                is_diff_set = False
                break
        if is_diff_set:
            diff_sets.append(tuple(D))
            
    # There should be 66 such sets.
    # Construct blocks
    blocks = []
    
    # Type 1: {infinity} U D
    for D in diff_sets:
        block = list(D) + [11]
        block.sort()
        blocks.append(tuple(block))
        
    # Type 2: Z_11 \ D
    for D in diff_sets:
        D_set = set(D)
        complement = tuple(sorted([x for x in range(11) if x not in D_set]))
        blocks.append(complement)
        
    # Verify and return
    # Convert to list of lists as required
    return [list(b) for b in blocks]
# EVOLVE-BLOCK-END
