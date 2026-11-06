def construct():
    import random
    import time
    from collections import defaultdict
    
    # Enumerate all forbidden triples
    triples = []
    triple_set = set()
    for b in range(1, 181):
        for c in range(b + 1, 181):
            s = b + c
            prod = b * c
            if prod % s == 0:
                a = prod // s
                if 1 <= a <= 180 and a != b and a != c:
                    t = tuple(sorted((a, b, c)))
                    if t not in triple_set:
                        triple_set.add(t)
                        triples.append(t)
    
    # Precompute conflict pairs
    # conflict_pairs[x] is a list of tuples (y, z) such that {x, y, z} is a forbidden triple.
    conflict_pairs = defaultdict(list)
    for t in triples:
        a, b, c = t
        conflict_pairs[a].append((b, c))
        conflict_pairs[b].append((a, c))
        conflict_pairs[c].append((a, b))
        
    def is_valid_addition(x, S):
        for y, z in conflict_pairs[x]:
            if y in S and z in S:
                return False
        return True

    def get_blocking_elements(x, S):
        blockers = set()
        for y, z in conflict_pairs[x]:
            if y in S and z in S:
                blockers.add(y)
                blockers.add(z)
        return blockers

    start_time = time.time()
    timeout = 55.0
    best_size = 0
    best_set = set()
    
    def run_sa():
        nonlocal best_size, best_set
        S = set()
        elements = list(range(1, 181))
        random.shuffle(elements)
        for x in elements:
            if is_valid_addition(x, S):
                S.add(x)
        
        current_size = len(S)
        T = 5.0
        cooling = 0.9995
        max_iters = 50000
        
        for _ in range(max_iters):
            if time.time() - start_time > timeout:
                return
            
            if current_size > best_size:
                best_size = current_size
                best_set = set(S)
                
            x = random.randint(1, 180)
            if x not in S:
                if is_valid_addition(x, S):
                    S.add(x)
                    current_size += 1
                else:
                    blockers = get_blocking_elements(x, S)
                    if blockers:
                        y = random.choice(list(blockers))
                        S.remove(y)
                        if is_valid_addition(x, S):
                            S.add(x)
                        else:
                            S.add(y)
            
            T *= cooling
            if T < 0.01:
                break

    while time.time() - start_time < timeout:
        run_sa()
        
    return list(sorted(best_set))
