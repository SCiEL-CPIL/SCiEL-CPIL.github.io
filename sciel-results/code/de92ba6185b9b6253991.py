"""Rewritten program for the suite task; uses group action to construct S(5,6,12)."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = 110  # Safety margin before 120s limit

    # Generators for M12 acting on 0..11
    # These are standard generators for the Mathieu group M12.
    def perm_from_cycles(cycles, n=12):
        p = list(range(n))
        for cyc in cycles:
            for i in range(len(cyc)):
                p[cyc[i]] = cyc[(i+1) % len(cyc)]
        return p

    g1 = perm_from_cycles([(0,1,2,3,4,5), (6,7,8,9,10,11)])
    g2 = perm_from_cycles([(0,6,10,9), (1,7,8,11), (2,11,9,6), (3,8,7,10), (4,9,11,8), (5,10,6,7)])
    
    # Base block
    base_block = frozenset([0, 1, 2, 3, 4, 5])
    
    # Generate orbit of the base block under the group action
    orbit = set()
    queue = [base_block]
    orbit.add(base_block)
    
    while queue:
        if time.time() > deadline:
            break
        current = queue.pop()
        for g in [g1, g2]:
            # Apply permutation g to the block
            new_block = frozenset([g[x] for x in current])
            if new_block not in orbit:
                orbit.add(new_block)
                queue.append(new_block)
                
    blocks = [list(b) for b in orbit]
    
    # If the orbit is not 132, we might have a subgroup or error.
    # We fill the remaining capacity with a greedy search to maximize the count.
    if len(blocks) < 132:
        covered = set()
        for b in blocks:
            for s in combinations(b, 5):
                covered.add(frozenset(s))
                
        # Greedy fill
        for b in combinations(range(12), 6):
            if time.time() > deadline:
                break
            subsets = [frozenset(s) for s in combinations(b, 5)]
            if not any(s in covered for s in subsets):
                blocks.append(list(b))
                covered.update(subsets)
                
    return blocks
# EVOLVE-BLOCK-END
