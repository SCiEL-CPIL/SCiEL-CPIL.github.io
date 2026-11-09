import time
import random

def construct():
    N = 512
    
    # Precompute 4-APs and their incidence lists
    aps = []
    aps_with = [[] for _ in range(N + 1)]
    
    for a in range(1, N + 1):
        max_d = (N - a) // 3
        for d in range(1, max_d + 1):
            x1 = a
            x2 = a + d
            x3 = a + 2 * d
            x4 = a + 3 * d
            ap = (x1, x2, x3, x4)
            idx = len(aps)
            aps.append(ap)
            for x in ap:
                aps_with[x].append(idx)
                
    # Optimization: Store the other 3 elements for each x in each AP
    # conflicts[x] is a list of tuples (y1, y2, y3)
    conflicts = [[] for _ in range(N + 1)]
    for idx, ap in enumerate(aps):
        for i in range(4):
            x = ap[i]
            others = tuple(ap[j] for j in range(4) if j != i)
            conflicts[x].append(others)

    def can_add(x, in_S):
        for others in conflicts[x]:
            # Check if all 3 are in S
            if in_S[others[0]] and in_S[others[1]] and in_S[others[2]]:
                return False
        return True

    best_size = 0
    best_set = set()
    start_time = time.time()
    deadline = start_time + 85.0
    
    # Use a fixed seed for reproducibility? No, random is better for exploration.
    # But if we want to ensure we beat the seed, randomization is key.
    
    while time.time() < deadline:
        in_S = [False] * (N + 1)
        current_size = 0
        candidates = list(range(1, N + 1))
        random.shuffle(candidates)
        
        for x in candidates:
            if can_add(x, in_S):
                in_S[x] = True
                current_size += 1
                
        if current_size > best_size:
            best_size = current_size
            best_set = {i for i in range(1, N + 1) if in_S[i]}
            
    return sorted(best_set)
