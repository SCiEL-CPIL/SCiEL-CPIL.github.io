import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 80.0
    
    N = 180
    S = set(range(1, 46)) | set(range(91, 181))
    
    pairs_for_a = {a: [] for a in range(46, 91)}
    conflicts = {a: set() for a in range(46, 91)}
    
    for a in range(46, 91):
        a2 = a * a
        limit = int(math.isqrt(a2))
        seen = set()
        for d in range(1, limit + 1):
            if a2 % d == 0:
                d1 = d
                d2 = a2 // d
                b1 = a + d1
                c1 = a + d2
                if 91 <= b1 <= 180 and 91 <= c1 <= 180 and b1 != c1:
                    pair = (min(b1, c1), max(b1, c1))
                    if pair not in seen:
                        seen.add(pair)
                        pairs_for_a[a].append(pair)
                        conflicts[a].add(b1)
                        conflicts[a].add(c1)
                        
    x_to_a = {x: [] for x in range(91, 181)}
    for a in range(46, 91):
        for x in conflicts[a]:
            x_to_a[x].append(a)
            
    current_S = set(S)
    
    for a in range(90, 45, -1):
        if time.time() > deadline:
            break
            
        pairs = pairs_for_a[a]
        if not pairs:
            current_S.add(a)
            continue
            
        # Check if a single x covers all pairs
        involved_x = set()
        for p in pairs:
            involved_x.add(p[0])
            involved_x.add(p[1])
            
        candidates = [x for x in involved_x if x in current_S]
        
        found_common = False
        best_x = None
        min_cost = float('inf')
        
        for x in candidates:
            # Check if x is in all pairs
            in_all = all(x in p for p in pairs)
            if in_all:
                cost = len([a_prime for a_prime in x_to_a[x] if a_prime < a])
                if cost < min_cost:
                    min_cost = cost
                    best_x = x
                    found_common = True
                    
        if found_common and min_cost == 0:
            current_S.remove(best_x)
            current_S.add(a)
            
    return list(current_S)
