import time
import sys
from itertools import combinations
from collections import defaultdict
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    from random import Random

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def solve_instance(n, k, p, seed):
        edges = instance(n, k, p, seed)
        
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        C_set = set(range(k))
        I = list(range(k, n))
        
        I_neigh = {}
        for v in I:
            N_v = adj[v] & C_set
            I_neigh[v] = frozenset(N_v) if N_v else None
        
        used_edges = set()
        partition = []
        
        I_sorted = sorted([v for v in I if I_neigh[v] is not None], 
                         key=lambda v: len(I_neigh[v]))
        
        for v in I_sorted:
            N_v = set(I_neigh[v])
            n_v = len(N_v)
            
            if n_v == 1:
                u = next(iter(N_v))
                partition.append(sorted([v, u]))
                continue
            
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
        
        internal_edges = set((u, v) for u in range(k) for v in range(u+1, k))
        remaining_edges = internal_edges - used_edges
        
        if remaining_edges:
            rem_set = set(remaining_edges)
            rem_adj = [set() for _ in range(k)]
            for u, v in rem_set:
                rem_adj[u].add(v)
                rem_adj[v].add(u)
            
            while rem_set:
                best_edge = max(rem_set, key=lambda e: len(rem_adj[e[0]] & rem_adj[e[1]] - {e[0], e[1]}))
                u, v = best_edge
                current_clique = {u, v}
                
                changed = True
                while changed:
                    changed = False
                    candidates = set()
                    for w in range(k):
                        if w in current_clique:
                            continue
                        if all(w in rem_adj[x] for x in current_clique):
                            candidates.add(w)
                    
                    if candidates:
                        best_w = max(candidates, key=lambda w: len(rem_adj[w]))
                        current_clique.add(best_w)
                        changed = True
                        for x in current_clique:
                            if x != best_w:
                                rem_adj[x].discard(best_w)
                                rem_adj[best_w].discard(x)
                
                clique_list = sorted(list(current_clique))
                if len(clique_list) >= 2:
                    partition.append(clique_list)
                
                for i in range(len(clique_list)):
                    for j in range(i+1, len(clique_list)):
                        a, b = clique_list[i], clique_list[j]
                        rem_set.discard((min(a, b), max(a, b)))
        
        # Local search: repeatedly try to merge cliques
        best = [list(c) for c in partition]
        best_count = len(best)
        
        for _ in range(500):
            if time.time() > deadline - 15:
                break
            
            improved = True
            while improved:
                improved = False
                for i in range(len(best)):
                    if i >= len(best):
                        break
                    for j in range(i+1, len(best)):
                        if j >= len(best):
                            break
                        c1 = set(best[i])
                        c2 = set(best[j])
                        union = c1 | c2
                        is_clique = all(v in adj[u] for u in union for v in union if u < v)
                        if is_clique and len(union) >= 2:
                            best.pop(j)
                            best[i] = sorted(union)
                            improved = True
                            break
                    if improved:
                        break
            
            if len(best) < best_count:
                best_count = len(best)
        
        return best

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
