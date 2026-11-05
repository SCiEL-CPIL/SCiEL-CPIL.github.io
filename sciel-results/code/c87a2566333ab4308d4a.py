import math

def is_square(n):
    if n < 0:
        return False
    r = int(math.isqrt(n))
    return r * r == n

def construct():
    import random
    
    N = 256
    # Precompute adjacency sets for all pairs (i, j) where i+j is a square
    adj = [set() for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            if is_square(i + j):
                adj[i].add(j)
                adj[j].add(i)
                
    def local_search(S_init):
        S = set(S_init)
        T = set(range(1, N + 1)) - S
        
        # Step 1: Ensure S is an independent set by removing conflicting vertices
        # We remove the vertex with the highest degree within S
        changed = True
        while changed:
            changed = False
            if not S:
                break
            # Calculate degree of each vertex in S
            degs = {v: len(adj[v] & S) for v in S}
            max_deg = max(degs.values())
            if max_deg == 0:
                break
            # Remove one vertex with max degree
            # If multiple, pick arbitrary (first found)
            for v in S:
                if degs[v] == max_deg:
                    S.remove(v)
                    T.add(v)
                    changed = True
                    break
        
        # Step 2: Local Search to maximize |S|
        # Move: Remove u from S, then greedily add vertices from T that are safe
        improved = True
        while improved:
            improved = False
            s_list = list(S)
            for u in s_list:
                if u not in S:
                    continue
                S_new = S - {u}
                T_new = T | {u}
                
                # Find vertices in T_new that are not adjacent to any in S_new
                candidates = []
                for v in T_new:
                    if not (adj[v] & S_new):
                        candidates.append(v)
                        
                # Greedily add an independent subset of candidates to S_new
                # We use a greedy min-degree heuristic on the subgraph induced by candidates
                remaining = set(candidates)
                added_any = False
                while remaining:
                    # Pick vertex with min degree in remaining
                    min_deg = float('inf')
                    best_v = None
                    for v in remaining:
                        deg = len(adj[v] & remaining)
                        if deg < min_deg:
                            min_deg = deg
                            best_v = v
                    S_new.add(best_v)
                    remaining -= (adj[best_v] | {best_v})
                    added_any = True
                    
                if len(S_new) > len(S):
                    S = S_new
                    T = set(range(1, N + 1)) - S
                    improved = True
                    break # Restart the loop with the updated S
        
        return S

    # Generate diverse initial sets
    inits = []
    
    # Modular partitions
    R0 = [i for i in range(1, N + 1) if i % 4 == 0]
    R1 = [i for i in range(1, N + 1) if i % 4 == 1]
    R2 = [i for i in range(1, N + 1) if i % 4 == 2]
    R3 = [i for i in range(1, N + 1) if i % 4 == 3]
    
    inits.append(R1 + R2)
    inits.append(R0 + R3)
    inits.append(R1 + R3)
    inits.append(R0 + R2)
    inits.append(R1)
    inits.append(R2)
    inits.append(R0)
    inits.append(R3)
    
    # Random inits
    random.seed(42)
    for _ in range(10):
        size = random.randint(80, 120)
        rand_set = random.sample(range(1, N + 1), size)
        inits.append(rand_set)
        
    best_set = set()
    for init in inits:
        res = local_search(init)
        if len(res) > len(best_set):
            best_set = res
            
    return list(best_set)
