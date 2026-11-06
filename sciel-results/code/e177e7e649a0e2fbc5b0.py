import time
import random

def construct():
    start_time = time.time()
    DEADLINE = 85.0
    
    n = 256
    
    # Precompute all 3-term APs
    triples = []
    triple_idx = {}  # (a,b,c) -> index
    for d in range(1, n // 2 + 1):
        for a in range(1, n - 2 * d + 1):
            b = a + d
            c = a + 2 * d
            idx = len(triples)
            triples.append((a, b, c))
            triple_idx[(a, b, c)] = idx
    
    # Precompute all 4-term APs
    quads = []
    for d in range(1, n // 3 + 1):
        for a in range(1, n - 3 * d + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            quads.append((a, b, c, e))
    
    # Membership lists
    triple_of = [[] for _ in range(n + 1)]
    for i, (a, b, c) in enumerate(triples):
        triple_of[a].append(i)
        triple_of[b].append(i)
        triple_of[c].append(i)
    
    quad_of = [[] for _ in range(n + 1)]
    for i, (a, b, c, e) in enumerate(quads):
        quad_of[a].append(i)
        quad_of[b].append(i)
        quad_of[c].append(i)
        quad_of[e].append(i)
    
    def count_3aps(in_set):
        cnt = 0
        for a, b, c in triples:
            if in_set[a] and in_set[b] and in_set[c]:
                cnt += 1
        return cnt
    
    def has_4ap(in_set):
        for a, b, c, e in quads:
            if in_set[a] and in_set[b] and in_set[c] and in_set[e]:
                return True
        return False
    
    def can_add(in_set, x):
        if in_set[x]:
            return False
        for qi in quad_of[x]:
            a, b, c, e = quads[qi]
            others = [v for v in (a, b, c, e) if v != x]
            if all(in_set[v] for v in others):
                return False
        return True
    
    def gain_3aps(in_set, x):
        if in_set[x]:
            return 0
        g = 0
        for ti in triple_of[x]:
            a, b, c = triples[ti]
            others = [v for v in (a, b, c) if v != x]
            if all(in_set[v] for v in others):
                g += 1
        return g
    
    def loss_3aps(in_set, x):
        if not in_set[x]:
            return 0
        l = 0
        for ti in triple_of[x]:
            a, b, c = triples[ti]
            others = [v for v in (a, b, c) if v != x]
            if all(in_set[v] for v in others):
                l += 1
        return l
    
    # Phase 1: Multiple greedy starts
    best_in_set = [False] * (n + 1)
    best_score = 0
    
    for trial in range(30):
        if time.time() - start_time > DEADLINE:
            break
        
        in_set = [False] * (n + 1)
        elems = list(range(1, n + 1))
        random.shuffle(elems)
        
        for x in elems:
            if time.time() - start_time > DEADLINE:
                break
            if can_add(in_set, x):
                in_set[x] = True
        
        score = count_3aps(in_set)
        if score > best_score:
            best_score = score
            best_in_set = in_set[:]
    
    # Phase 2: Local search
    in_set = best_in_set[:]
    current_score = best_score
    
    # First, try pure additions
    changed = True
    while changed and time.time() - start_time < DEADLINE:
        changed = False
        for x in range(1, n + 1):
            if time.time() - start_time > DEADLINE:
                break
            if not in_set[x] and can_add(in_set, x):
                g = gain_3aps(in_set, x)
                if g > 0:
                    in_set[x] = True
                    current_score += g
                    changed = True
    
    # Then, try swaps
    for iteration in range(100):
        if time.time() - start_time > DEADLINE:
            break
        
        improved = False
        # Try removing each element and adding a better one
        for x in range(1, n + 1):
            if time.time() - start_time > DEADLINE:
                break
            if not in_set[x]:
                continue
            
            lx = loss_3aps(in_set, x)
            
            # Try each non-member y
            for y in range(1, n + 1):
                if time.time() - start_time > DEADLINE:
                    break
                if in_set[y]:
                    continue
                
                # Temporarily remove x
                in_set[x] = False
                
                # Check if y can be added
                ok = True
                for qi in quad_of[y]:
                    a, b, c, e = quads[qi]
                    others = [v for v in (a, b, c, e) if v != y]
                    if all(in_set[v] for v in others):
                        ok = False
                        break
                
                if ok:
                    gy = gain_3aps(in_set, y)
                    net = gy - lx
                    if net > 0:
                        in_set[y] = True
                        current_score += net
                        improved = True
                        break
                    else:
                        in_set[x] = True
                else:
                    in_set[x] = True
                
                if improved:
                    break
            
            if improved:
                break
    
    result = [i for i in range(1, n + 1) if in_set[i]]
    if not result:
        result = [1]
    return sorted(result)
