"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys
from collections import defaultdict

sys.setrecursionlimit(10000)

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def construct():
    start_time = time.time()
    deadline = start_time + 78  # safety margin
    
    n = P['n']
    edges = P['edges']
    edge_set = set()
    for a, b in edges:
        if a > b:
            a, b = b, a
        edge_set.add((a, b))
    E = len(edge_set)
    
    edge_to_idx = {}
    for i, (a, b) in enumerate(sorted(edge_set)):
        edge_to_idx[(a, b)] = i
    
    adj = [set() for _ in range(n)]
    for a, b in edges:
        adj[a].add(b)
        adj[b].add(a)
    
    # Generate candidate bicliques
    candidates = []
    seen_masks = set()
    
    def add_candidate(L, R):
        if not L or not R:
            return
        L = list(L)
        R = list(R)
        for u in L:
            for v in R:
                a, b = (u, v) if u < v else (v, u)
                if (a, b) not in edge_to_idx:
                    return
        mask = 0
        for u in L:
            for v in R:
                a, b = (u, v) if u < v else (v, u)
                mask |= (1 << edge_to_idx[(a, b)])
        if mask == 0 or mask in seen_masks:
            return
        seen_masks.add(mask)
        candidates.append((mask, list(L), list(R)))
    
    # Maximal bicliques from vertex pairs
    for u in range(n):
        for v in adj[u]:
            if u < v:
                common = adj[u] & adj[v]
                L = [u] + sorted([w for w in common if w != v])
                R = [v]
                # Remove v from L if present
                if v in L:
                    L.remove(v)
                add_candidate(L, R)
                # Also try with v on left
                L2 = [v] + sorted([w for w in common if w != u])
                R2 = [u]
                if u in L2:
                    L2.remove(u)
                add_candidate(L2, R2)
    
    # Stars
    for u in range(n):
        if adj[u]:
            add_candidate([u], sorted(adj[u]))
    
    # Pairs of edges sharing a vertex
    for u in range(n):
        nbrs = list(adj[u])
        for i in range(len(nbrs)):
            for j in range(i + 1, len(nbrs)):
                v, w = nbrs[i], nbrs[j]
                if w in adj[v]:
                    add_candidate([u, v], [w])
                    add_candidate([u], [v, w])
    
    # Triplets
    for u in range(n):
        nbrs = list(adj[u])
        for i in range(len(nbrs)):
            for j in range(i + 1, len(nbrs)):
                v, w = nbrs[i], nbrs[j]
                if w in adj[v]:
                    for k in range(j + 1, len(nbrs)):
                        x = nbrs[k]
                        if x in adj[v] and x in adj[w]:
                            add_candidate([u, v], [w, x])
                            add_candidate([u, w], [v, x])
                            add_candidate([u, x], [v, w])
    
    # Random bicliques
    random.seed(42)
    for _ in range(500):
        u = random.choice(list(range(n)))
        if not adj[u]:
            continue
        v = random.choice(list(adj[u]))
        common = adj[u] & adj[v]
        L = [u]
        R = [v]
        for w in common:
            if random.random() < 0.4:
                if w < v:
                    L.append(w)
                else:
                    R.append(w)
        if v in L:
            L.remove(v)
        if u in R:
            R.remove(u)
        add_candidate(L, R)
    
    # Full mask
    full_mask = (1 << E) - 1
    
    # Sort candidates by size descending
    candidates.sort(key=lambda x: x[0].bit_count(), reverse=True)
    
    # Greedy to get upper bound
    remaining = set(edge_set)
    greedy_sol = []
    while remaining:
        best_bic = None
        best_count = 0
        for (mask, L, R) in candidates:
            count = 0
            for u in L:
                for v in R:
                    a, b = (u, v) if u < v else (v, u)
                    if (a, b) in remaining:
                        count += 1
            if count > best_count:
                best_count = count
                best_bic = (mask, L, R)
        if best_bic is None or best_count == 0:
            e = min(remaining)
            u, v = e
            greedy_sol.append(([u], [v]))
            remaining.discard(e)
        else:
            mask, L, R = best_bic
            greedy_sol.append((L, R))
            for u in L:
                for v in R:
                    a, b = (u, v) if u < v else (v, u)
                    remaining.discard((a, b))
    
    best_count = len(greedy_sol)
    best_solution = list(greedy_sol)
    
    # Branch and bound with bitsets
    def solve(remaining_mask, current_sol):
        nonlocal best_solution, best_count, deadline
        
        if time.time() > deadline:
            return
        
        if remaining_mask == 0:
            if len(current_sol) < best_count:
                best_count = len(current_sol)
                best_solution = list(current_sol)
            return
        
        if len(current_sol) >= best_count:
            return
        
        # Find edge with fewest covering candidates
        edge_idx = 0
        while not (remaining_mask & (1 << edge_idx)):
            edge_idx += 1
        
        covering = []
        for i, (mask, L, R) in enumerate(candidates):
            if (mask & remaining_mask) == mask and (mask & (1 << edge_idx)):
                covering.append(i)
        
        if not covering:
            return
        
        covering.sort(key=lambda i: candidates[i][0].bit_count(), reverse=True)
        
        for i in covering:
            mask, L, R = candidates[i]
            current_sol.append((L, R))
            solve(remaining_mask & ~mask, current_sol)
            current_sol.pop()
            
            if time.time() > deadline:
                return
            if len(current_sol) + 1 >= best_count:
                break
    
    solve(full_mask, [])
    
    if best_solution is None:
        best_solution = [([u], [v for a, v in P['edges'] if a == u]) 
                        for u in range(P['n']) if any(a == u for a, v in P['edges'])]
    
    result = []
    for L, R in best_solution:
        L_list = [int(x) for x in L]
        R_list = [int(x) for x in R]
        if L_list and R_list:
            result.append([L_list, R_list])
    
    return result
# EVOLVE-BLOCK-END
