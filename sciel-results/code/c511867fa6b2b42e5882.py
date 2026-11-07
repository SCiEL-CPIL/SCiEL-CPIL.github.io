import time
import random
import numpy as np
from scipy.sparse.csgraph import laplacian
from scipy.sparse import csr_matrix

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}


def construct():
    n = P['n']
    edges = P['edges']
    
    # Precompute adjacency sets
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # Normalize edges to tuples for set operations
    edge_set = set(tuple(sorted(e)) for e in edges)
    
    def get_biclique_edges(L, R):
        res = set()
        for u in L:
            for v in R:
                res.add((u, v) if u < v else (v, u))
        return res
    
    def expand_biclique(seed_u, seed_v, remaining_edges):
        L = {seed_u}
        R = {seed_v}
        changed = True
        while changed:
            changed = False
            new_L = set(L)
            for w in range(n):
                if w not in L and w not in R:
                    if all((w, r) in remaining_edges or (r, w) in remaining_edges for r in R):
                        if all(r in adj[w] for r in R):
                            new_L.add(w)
                            changed = True
            L = new_L
            
            new_R = set(R)
            for w in range(n):
                if w not in L and w not in R:
                    if all((w, l) in remaining_edges or (l, w) in remaining_edges for l in L):
                        if all(l in adj[w] for l in L):
                            new_R.add(w)
                            changed = True
            R = new_R
        return L, R
    
    def solve_greedy(seed_order):
        remaining = set(edge_set)
        bicliques = []
        
        while remaining:
            best_biclique = None
            best_size = -1
            best_edges = set()
            
            for u, v in seed_order:
                if (u, v) not in remaining and (v, u) not in remaining:
                    continue
                    
                L, R = expand_biclique(u, v, remaining)
                
                b_edges = set()
                for lu in L:
                    for rv in R:
                        e = (lu, rv) if lu < rv else (rv, lu)
                        b_edges.add(e)
                        
                if b_edges.issubset(remaining):
                    size = len(b_edges)
                    if size > best_size:
                        best_size = size
                        best_biclique = [list(L), list(R)]
                        best_edges = b_edges
            
            if best_biclique is None:
                u, v = next(iter(remaining))
                best_biclique = [[u], [v]]
                best_edges = {(u, v) if u < v else (v, u)}
            
            remaining -= best_edges
            bicliques.append(best_biclique)
            
        return bicliques
    
    def merge_pass(bicliques, adj, deadline):
        changed = True
        while changed and time.time() < deadline:
            changed = False
            n_b = len(bicliques)
            for i in range(n_b):
                if time.time() > deadline:
                    break
                for j in range(i + 1, n_b):
                    l1, r1 = bicliques[i]
                    l2, r2 = bicliques[j]
                    merged = None
                    for (new_l, new_r) in [
                        (set(l1)|set(l2), set(r1)&set(r2)),
                        (set(l1)&set(l2), set(r1)|set(r2)),
                        (set(l1)|set(r2), set(r1)&set(l2)),
                        (set(l2)|set(r1), set(r2)&set(l1)),
                    ]:
                        if not new_l or not new_r:
                            continue
                        if new_l & new_r:
                            continue
                        valid = True
                        for u in new_l:
                            for v in new_r:
                                if v not in adj[u]:
                                    valid = False
                                    break
                            if not valid:
                                break
                        if valid:
                            # Check that merged covers exactly the union of edges
                            merged_edges = set()
                            for u in new_l:
                                for v in new_r:
                                    merged_edges.add((min(u,v), max(u,v)))
                            
                            old_edges = set()
                            for u in l1:
                                for v in r1:
                                    old_edges.add((min(u,v), max(u,v)))
                            for u in l2:
                                for v in r2:
                                    old_edges.add((min(u,v), max(u,v)))
                            
                            if merged_edges == old_edges:
                                merged = (list(new_l), list(new_r))
                                break
                    if merged:
                        bicliques[i] = merged
                        bicliques.pop(j)
                        changed = True
                        break
                if changed:
                    break
        return bicliques
    
    def local_search(bicliques, adj, deadline):
        """Try to improve by moving vertices between bicliques"""
        best = bicliques[:]
        best_count = len(bicliques)
        
        while time.time() < deadline:
            improved = False
            for i in range(len(best)):
                if time.time() > deadline:
                    break
                for j in range(len(best)):
                    if i == j:
                        continue
                    l1, r1 = best[i]
                    l2, r2 = best[j]
                    
                    # Try moving a vertex from l1 to l2 or r1 to r2
                    for v in l1[:]:
                        # Try moving v from l1 to l2
                        if v in r2:
                            continue
                        new_l1 = [x for x in l1 if x != v]
                        new_l2 = l2 + [v]
                        if not new_l1:
                            continue
                        # Check if new_l1, r1 is still valid
                        valid = True
                        for u in new_l1:
                            for w in r1:
                                if w not in adj[u]:
                                    valid = False
                                    break
                            if not valid:
                                break
                        if valid:
                            # Check if new_l2, r2 is valid
                            valid2 = True
                            for u in new_l2:
                                for w in r2:
                                    if w not in adj[u]:
                                        valid2 = False
                                        break
                                    if not valid2:
                                        break
                            if valid2:
                                # Check edge coverage
                                edges1 = get_biclique_edges(new_l1, r1)
                                edges2 = get_biclique_edges(new_l2, r2)
                                all_edges = set()
                                for bi, bj in best:
                                    all_edges |= get_biclique_edges(bi, bj)
                                new_all = edges1 | edges2
                                for k in range(len(best)):
                                    if k != i and k != j:
                                        lk, rk = best[k]
                                        new_all |= get_biclique_edges(lk, rk)
                                if new_all == all_edges and len(new_all) == len(all_edges):
                                    best[i] = [new_l1, r1]
                                    best[j] = [new_l2, r2]
                                    improved = True
                                    break
                    if improved:
                        break
                if improved:
                    break
            
            if not improved:
                break
        
        return best
    
    start_time = time.time()
    deadline = start_time + 55
    
    best_solution = None
    best_count = float('inf')
    
    # Initial deterministic run
    initial_order = list(edge_set)
    sol = solve_greedy(initial_order)
    sol = merge_pass(sol, adj, deadline)
    best_solution = sol
    best_count = len(sol)
    
    # Try different strategies
    strategies = [
        # Sort by degree of endpoints
        sorted(edge_set, key=lambda e: (len(adj[e[0]]) + len(adj[e[1]]), e[0], e[1])),
        # Sort by min degree
        sorted(edge_set, key=lambda e: (min(len(adj[e[0]]), len(adj[e[1]])), e[0], e[1])),
        # Sort by max degree
        sorted(edge_set, key=lambda e: (max(len(adj[e[0]]), len(adj[e[1]])), e[0], e[1])),
    ]
    
    for strategy in strategies:
        if time.time() > deadline:
            break
        sol = solve_greedy(strategy)
        sol = merge_pass(sol, adj, deadline)
        if len(sol) < best_count:
            best_count = len(sol)
            best_solution = sol
    
    # Random restarts
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        shuffled = list(edge_set)
        random.shuffle(shuffled)
        
        sol = solve_greedy(shuffled)
        sol = merge_pass(sol, adj, deadline)
        if len(sol) < best_count:
            best_count = len(sol)
            best_solution = sol
            
        if best_count <= 5:
            break
    
    # Final local search
    if time.time() < deadline:
        best_solution = local_search(best_solution, adj, deadline)
        best_count = len(best_solution)
    
    return best_solution
