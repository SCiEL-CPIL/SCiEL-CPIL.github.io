import time
import sys
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    from random import Random

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def solve_instance(n, k, p, seed):
        edges = instance(n, k, p, seed)
        
        # Build adjacency
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        C = list(range(k))
        I = list(range(k, n))
        
        # Internal edges of C
        internal_edges = set((u, v) for u in range(k) for v in range(u+1, k))
        
        # Used edges in C
        used_edges = set()
        
        partition = []
        
        # Sort I by degree in C (ascending) - process smaller neighborhoods first
        I_sorted = sorted(I, key=lambda v: len(adj[v] & set(C)))
        
        for v in I_sorted:
            N_v = adj[v] & set(C)
            if not N_v:
                continue
            
            n_v = len(N_v)
            if n_v == 1:
                u = N_v.pop()
                partition.append(sorted([v, u]))
                continue
                
            # Build graph H on N_v using used_edges
            h_vertices = list(N_v)
            h_adj = {u: set() for u in h_vertices}
            
            for i in range(len(h_vertices)):
                u = h_vertices[i]
                for j in range(i+1, len(h_vertices)):
                    w = h_vertices[j]
                    e = (min(u, w), max(u, w))
                    if e in used_edges:
                        h_adj[u].add(w)
                        h_adj[w].add(u)
                        
            # DSATUR-style greedy coloring of H
            colors = {}
            sat_degree = {u: 0 for u in h_vertices}
            color_set = {u: set() for u in h_vertices}
            uncolored = set(h_vertices)
            
            while uncolored:
                best = max(uncolored, key=lambda x: (sat_degree[x], len(h_adj[x])))
                used_colors = color_set[best]
                c = 0
                while c in used_colors:
                    c += 1
                colors[best] = c
                color_set[best].add(c)
                uncolored.discard(best)
                for w in h_adj[best]:
                    if w in uncolored:
                        if c not in color_set[w]:
                            color_set[w].add(c)
                            sat_degree[w] = len(color_set[w])
                
            color_count = max(colors.values()) + 1 if colors else 0
            
            color_classes = {}
            for u, c in colors.items():
                if c not in color_classes:
                    color_classes[c] = []
                color_classes[c].append(u)
                
            for c in range(color_count):
                S = color_classes.get(c, [])
                if len(S) < 1:
                    continue
                    
                clique = S + [v]
                partition.append(sorted(clique))
                
                for i in range(len(S)):
                    for j in range(i+1, len(S)):
                        used_edges.add((min(S[i], S[j]), max(S[i], S[j])))
                            
        # Remaining edges in C - use bitmask-based max clique extraction
        remaining_edges = internal_edges - used_edges
        rem_set = set(remaining_edges)
        
        if rem_set:
            # Build adjacency bitmasks for remaining graph
            rem_adj = [0] * k
            for u, v in rem_set:
                rem_adj[u] |= (1 << v)
                rem_adj[v] |= (1 << u)
            
            def find_max_clique(start_mask, adj_bm):
                """Bron-Kerbosch with pivot to find one maximum clique."""
                best_clique = []
                
                def bk(R, P, X):
                    nonlocal best_clique
                    if not P and not X:
                        if len(R) > len(best_clique):
                            best_clique = list(R)
                        return
                    # Choose pivot
                    pivot_candidates = P | X
                    if pivot_candidates:
                        # Pick vertex with max degree in P
                        pivot = max(pivot_candidates, key=lambda v: bin(adj_bm[v] & P).count('1'))
                        candidates = P & ~(1 << pivot)
                    else:
                        candidates = P
                    
                    for v in list(candidates):
                        if not (candidates & (1 << v)):
                            continue
                        R.append(v)
                        P_new = P & adj_bm[v]
                        X_new = X & adj_bm[v]
                        bk(R, P_new, X_new)
                        R.pop()
                        P &= ~(1 << v)
                        X |= (1 << v)
                
                bk([], start_mask, 0)
                return best_clique
            
            while rem_set:
                # Find a large clique in the remaining graph
                # Start from the vertex with highest degree in remaining graph
                degs = [bin(rem_adj[v]).count('1') for v in range(k)]
                max_v = max(range(k), key=lambda v: degs[v])
                
                # Try multiple starting points for diversity
                best_found = None
                candidates_to_try = sorted(range(k), key=lambda v: -degs[v])[:min(5, k)]
                
                for sv in candidates_to_try:
                    if not (rem_adj[sv]):
                        continue
                    # Find max clique containing sv
                    # Start with sv, then find max clique in neighbors of sv within remaining
                    start = (1 << sv) | (rem_adj[sv] & rem_set_mask if False else 0)
                    # Actually, let's just find max clique in the subgraph induced by rem
                    # Use a simpler approach: greedy from sv
                    clique = [sv]
                    mask = rem_adj[sv]
                    while mask:
                        # Pick vertex with max common neighbors in clique
                        best_w = None
                        best_common = -1
                        m = mask
                        while m:
                            lsb = m & (-m)
                            w = lsb.bit_length() - 1
                            m -= lsb
                            # Count how many vertices in clique are adjacent to w
                            common = 0
                            for cv in clique:
                                if (1 << cv) & rem_adj[w]:
                                    common += 1
                            if common > best_common:
                                best_common = common
                                best_w = w
                        if best_w is None:
                            break
                        clique.append(best_w)
                        mask &= rem_adj[best_w]
                    
                    if best_found is None or len(clique) > len(best_found):
                        best_found = clique
                
                if best_found is None or len(best_found) < 2:
                    # Fallback: take any edge
                    e = next(iter(rem_set))
                    best_found = [e[0], e[1]]
                
                clique_list = sorted(best_found)
                partition.append(clique_list)
                
                # Remove edges of this clique from rem_set and update adj
                for i in range(len(clique_list)):
                    for j in range(i+1, len(clique_list)):
                        a, b = clique_list[i], clique_list[j]
                        rem_set.discard((min(a, b), max(a, b)))
                        rem_adj[a] &= ~(1 << b)
                        rem_adj[b] &= ~(1 << a)
                
                if time.time() > deadline - 30:
                    # Emergency: just pack remaining edges as pairs
                    while rem_set:
                        e = next(iter(rem_set))
                        partition.append(sorted([e[0], e[1]]))
                        rem_set.discard(e)
                        rem_adj[e[0]] &= ~(1 << e[1])
                        rem_adj[e[1]] &= ~(1 << e[0])
                    break
                        
        return partition

    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    results = []
    for spec in specs:
        if time.time() > deadline:
            while len(results) < 3:
                results.append([])
            return results
        res = solve_instance(*spec)
        cleaned = []
        for clique in res:
            cleaned.append([int(x) for x in clique])
        results.append(cleaned)
        
    return results
