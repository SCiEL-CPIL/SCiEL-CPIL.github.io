def construct():
    import random
    import math
    import time

    n = 128
    elements = list(range(1, n + 1))
    
    def get_lcm(a, b):
        return a * b // math.gcd(a, b)

    # Precompute lcm matrix
    lcm_mat = [[0]*(n+1) for _ in range(n+1)]
    for a in range(1, n+1):
        for b in range(1, n+1):
            lcm_mat[a][b] = get_lcm(a, b)
            
    # Precompute bad triples
    bad_triples_set = set()
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            L = lcm_mat[a][b]
            for c in range(b + 1, n + 1):
                if lcm_mat[a][c] == L and lcm_mat[b][c] == L:
                    bad_triples_set.add((a, b, c))
    
    # Precompute triple_map: (x, y) -> set of z such that (x,y,z) is bad
    triple_map = {}
    for a, b, c in bad_triples_set:
        triple_map[(a,b)] = triple_map.get((a,b), set()) | {c}
        triple_map[(b,a)] = triple_map.get((b,a), set()) | {c}
        triple_map[(a,c)] = triple_map.get((a,c), set()) | {b}
        triple_map[(c,a)] = triple_map.get((c,a), set()) | {b}
        triple_map[(b,c)] = triple_map.get((b,c), set()) | {a}
        triple_map[(c,b)] = triple_map.get((c,b), set()) | {a}
        
    # Precompute conflicts: x -> list of (u,v) such that (x,u,v) is bad
    conflicts = {x: [] for x in elements}
    for a, b, c in bad_triples_set:
        conflicts[a].append((b, c))
        conflicts[b].append((a, c))
        conflicts[c].append((a, b))

    def greedy_solve(seed):
        rng = random.Random(seed)
        S = []
        available = set(elements)
        blocked = set()
        
        while available:
            candidates = [x for x in available if x not in blocked]
            if not candidates:
                break
            
            min_cost = float('inf')
            best_candidates = []
            
            for x in candidates:
                cost = 0
                for u, v in conflicts[x]:
                    if u in available and v in available:
                        if u not in blocked and v not in blocked:
                            cost += 1
                
                if cost < min_cost:
                    min_cost = cost
                    best_candidates = [x]
                elif cost == min_cost:
                    best_candidates.append(x)
                    
            x = rng.choice(best_candidates)
            S.append(x)
            available.remove(x)
            
            # Update blocked
            # New pairs are (x, u) for u in S[:-1]
            for u in S[:-1]:
                if (x, u) in triple_map:
                    blocked.update(triple_map[(x, u)] & available)
            
        return S

    best_size = 0
    best_set = []
    
    start_time = time.time()
    max_time = 55.0
    
    # Run many iterations
    # Estimate time per iteration:
    # 128 steps.
    # Each step: iterate candidates (up to 128), for each iterate conflicts (small).
    # Update blocked: iterate S (up to 128), lookup set, update set.
    # Should be < 1ms per iteration.
    # 10000 iterations = 10s. Safe.
    
    for i in range(20000):
        if time.time() - start_time > max_time:
            break
        S = greedy_solve(i)
        if len(S) > best_size:
            best_size = len(S)
            best_set = S
            
    if not best_set:
        best_set = [1]
        
    return best_set
