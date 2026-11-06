import time
from collections import deque
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 10.0  # Safety margin
    
    # Generators for M12
    # Permutation as list: p[i] is image of i
    g1 = list(range(12))
    for i in range(12):
        g1[i] = (i + 1) % 12
        
    g2 = [0]*12
    # (0, 1, 4, 8, 5)
    g2[0] = 1
    g2[1] = 4
    g2[4] = 8
    g2[8] = 5
    g2[5] = 0
    # (2, 3, 11, 9, 7)
    g2[2] = 3
    g2[3] = 11
    g2[11] = 9
    g2[9] = 7
    g2[7] = 2
    # Others fixed
    g2[6] = 6
    g2[10] = 10

    def apply_perm(p, s):
        return frozenset(p[x] for x in s)

    # Start block
    b0 = frozenset([0, 1, 2, 3, 4, 5])
    
    visited = set()
    visited.add(b0)
    queue = deque([b0])
    blocks = []
    
    while queue:
        if time.time() > deadline:
            break
        b = queue.popleft()
        blocks.append(list(b))
        
        # Apply g1
        nb1 = apply_perm(g1, b)
        if nb1 not in visited:
            visited.add(nb1)
            queue.append(nb1)
            
        # Apply g2
        nb2 = apply_perm(g2, b)
        if nb2 not in visited:
            visited.add(nb2)
            queue.append(nb2)
            
    # Validate
    if len(blocks) == 132:
        # Check for repeated 5-subsets
        covered = set()
        valid = True
        for b in blocks:
            subs = list(combinations(b, 5))
            for s in subs:
                if s in covered:
                    valid = False
                    break
                covered.add(s)
            if not valid:
                break
        
        if valid:
            return blocks
            
    # Fallback: Greedy
    covered, fallback_blocks = set(), []
    for b in combinations(range(12), 6):
        if time.time() > deadline:
            break
        subsets = list(combinations(b, 5))
        if not any(s in covered for s in subsets):
            fallback_blocks.append(list(b))
            covered.update(subsets)
            
    return fallback_blocks
