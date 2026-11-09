import time
import sys
import random

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s safety margin
    
    n = P['n']
    edges = P['edges']
    
    # Build original adjacency sets
    orig_adj = [set() for _ in range(n)]
    for u, v in edges:
        orig_adj[u].add(v)
        orig_adj[v].add(u)
        
    def greedy_solve(seed):
        random.seed(seed)
        
        # Track uncovered edges
        uncovered = [set() for _ in range(n)]
        for u, v in edges:
            uncovered[u].add(v)
            uncovered[v].add(u)
            
        bicliques = []
        
        while any(uncovered[u] for u in range(n)):
            if time.time() > deadline:
                return None # Timeout in this run
                
            best_L = None
            best_R = None
            max_covered = 0
            
            active = [u for u in range(n) if uncovered[u]]
            if not active:
                break
                
            # Sort by degree
            active_sorted = sorted(active, key=lambda x: len(uncovered[x]), reverse=True)
            
            # Candidates
            candidates = []
            
            # Size 1
            for u in active_sorted:
                R = set(uncovered[u])
                covered = len(R)
                candidates.append((covered, [u], list(R)))
                
            # Size 2
            K = 20
            top_active = active_sorted[:K]
            for i in range(len(top_active)):
                u = top_active[i]
                for j in range(i+1, len(top_active)):
                    v = top_active[j]
                    R = uncovered[u].intersection(uncovered[v])
                    R.discard(u)
                    R.discard(v)
                    if R:
                        covered = 2 * len(R)
                        candidates.append((covered, [u, v], list(R)))
                        
            # Size 3
            M = 10
            top_active_m = active_sorted[:M]
            for i in range(len(top_active_m)):
                u = top_active_m[i]
                for j in range(i+1, len(top_active_m)):
                    v = top_active_m[j]
                    common_uv = uncovered[u].intersection(uncovered[v])
                    common_uv.discard(u)
                    common_uv.discard(v)
                    if not common_uv:
                        continue
                    w_candidates = list(common_uv)
                    if len(w_candidates) > 15:
                        w_candidates = sorted(w_candidates, key=lambda x: len(uncovered[x]), reverse=True)[:15]
                    for w in w_candidates:
                        R = common_uv.intersection(uncovered[w])
                        R.discard(u)
                        R.discard(v)
                        R.discard(w)
                        if R:
                            covered = 3 * len(R)
                            candidates.append((covered, [u, v, w], list(R)))
                            
            if not candidates:
                # Fallback
                for u in active:
                    v = next(iter(uncovered[u]))
                    candidates.append((1, [u], [v]))
                    break
            
            # Sort candidates by coverage, then randomize ties
            candidates.sort(key=lambda x: x[0], reverse=True)
            
            # Pick from top 5 to introduce randomness
            top_k = min(5, len(candidates))
            chosen = random.choice(candidates[:top_k])
            
            _, best_L, best_R = chosen
            
            # Validate
            valid = True
            for u in best_L:
                for v in best_R:
                    if v not in orig_adj[u]:
                        valid = False
                        break
                if not valid:
                    break
            
            if not valid:
                # Fallback to single edge if validation fails (shouldn't happen)
                for u in active:
                    if uncovered[u]:
                        v = next(iter(uncovered[u]))
                        best_L = [u]
                        best_R = [v]
                        break
            
            bicliques.append([best_L, best_R])
            
            # Update uncovered
            for u in best_L:
                for v in best_R:
                    if v in uncovered[u]:
                        uncovered[u].discard(v)
                    if u in uncovered[v]:
                        uncovered[v].discard(u)
                        
        return bicliques

    best_bicliques = None
    best_count = float('inf')
    
    # Run multiple restarts
    seed = 42
    while time.time() < deadline:
        result = greedy_solve(seed)
        if result is not None:
            count = len(result)
            if count < best_count:
                best_count = count
                best_bicliques = result
        seed += 1
        # Check time
        if time.time() > deadline - 5:
            break

    if best_bicliques is None:
        # Fallback to a single greedy run without randomization if all timed out?
        # Or just return a valid trivial partition
        # Construct trivial partition: each edge is a biclique
        trivial = []
        for u, v in edges:
            trivial.append([[u], [v]])
        return trivial

    # Convert to required format
    result = []
    for L, R in best_bicliques:
        result.append([[int(x) for x in L], [int(y) for y in R]])
        
    return result
