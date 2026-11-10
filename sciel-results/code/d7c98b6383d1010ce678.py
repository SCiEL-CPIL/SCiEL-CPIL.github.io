import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # 50s safety margin
    
    n = 512
    # Precompute all 4-APs
    # An AP is (a, d) -> a, a+d, a+2d, a+3d
    # We need to check if all 4 are in [1, n]
    
    # For each x, store the other 3 elements for each AP containing x
    # aps[x] is a list of (y1, y2, y3) such that {x, y1, y2, y3} is a 4-AP
    aps = [[] for _ in range(n + 1)]
    
    for a in range(1, n + 1):
        max_d = (n - a) // 3
        for d in range(1, max_d + 1):
            x1 = a
            x2 = a + d
            x3 = a + 2 * d
            x4 = a + 3 * d
            # Add to each element's list
            aps[x1].append((x2, x3, x4))
            aps[x2].append((x1, x3, x4))
            aps[x3].append((x1, x2, x4))
            aps[x4].append((x1, x2, x3))
    
    best_size = 0
    best_set = set()
    
    while time.time() < deadline:
        # Randomized Greedy
        S = set()
        conflict_count = [0] * (n + 1)
        
        # Order: random shuffle
        order = list(range(1, n + 1))
        random.shuffle(order)
        
        for x in order:
            if conflict_count[x] == 0:
                S.add(x)
                # Update conflict counts for others
                for (y1, y2, y3) in aps[x]:
                    # For y1: other 3 are x, y2, y3. If y2, y3 in S, then y1's conflict increases
                    if y2 in S and y3 in S:
                        conflict_count[y1] += 1
                    if y1 in S and y3 in S:
                        conflict_count[y2] += 1
                    if y1 in S and y2 in S:
                        conflict_count[y3] += 1
        
        # Local Search: Try to add any valid element
        # We can try multiple passes
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Find all valid elements
            valids = [x for x in range(1, n + 1) if x not in S and conflict_count[x] == 0]
            if valids:
                # Add one (or all? Adding one might block others, so add one at a time or all if independent)
                # Let's add one random valid element
                x = random.choice(valids)
                S.add(x)
                for (y1, y2, y3) in aps[x]:
                    if y2 in S and y3 in S:
                        conflict_count[y1] += 1
                    if y1 in S and y3 in S:
                        conflict_count[y2] += 1
                    if y1 in S and y2 in S:
                        conflict_count[y3] += 1
                improved = True
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = S.copy()
            
    return sorted(best_set)
