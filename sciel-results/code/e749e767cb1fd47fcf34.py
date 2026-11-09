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

    def dsatur_color(adj_dict, vertices):
        """DSATUR greedy coloring. Returns dict vertex -> color."""
        if not vertices:
            return {}
        colors = {}
        color_set = {u: set() for u in vertices}
        uncolored = set(vertices)
        sat_degree = {u: 0 for u in vertices}
        
        while uncolored:
            best = max(uncolored, key=lambda x: (sat_degree[x], len(adj_dict.get(x, set()))))
            used_colors = color_set[best]
            c = 0
            while c in used_colors:
                c += 1
            colors[best] = c
            color_set[best].add(c)
            uncolored.discard(best)
            for w in adj_dict.get(best, set()):
                if w in uncolored:
                    if c not in color_set[w]:
                        color_set[w].add(c)
                        sat_degree[w] = len(color_set[w])
        return colors

    def cover_remaining_edges(remaining_edges, k):
        """Greedy cover of remaining edges in C with cliques."""
        if not remaining_edges:
            return []
        
        rem_set = set(remaining_edges)
        partition = []
        
        while rem_set:
            deg = defaultdict(int)
            for u, v in rem_set:
                deg[u] += 1
                deg[v] += 1
            
            if not deg:
                break
                
            start_v = max(deg.keys(), key=lambda x: deg[x])
            current_clique = {start_v}
            
            changed = True
            while changed:
                changed = False
                candidates = set()
                for v in range(k):
                    if v in current_clique:
                        continue
                    is_connected = True
                    for x in current_clique:
                        e = (min(v, x), max(v, x))
                        if e not in rem_set:
                            is_connected = False
                            break
                    if is_connected:
                        candidates.add(v)
                
                if not candidates:
                    break
                    
                best_c = max(candidates, key=lambda v: deg[v])
                current_clique.add(best_c)
                changed = True
            
            clique_list = sorted(list(current_clique))
            if len(clique_list) >= 2:
                partition.append(clique_list)
                
            for i in range(len(clique_list)):
                for j in range(i+1, len(clique_list)):
                    a, b = clique_list[i], clique_list[j]
                    rem_set.discard((min(a, b), max(a, b)))
                    
        return partition

    def solve_instance(n, k, p, seed):
        edges = instance(n, k, p, seed)
        
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        C = list(range(k))
        I = list(range(k, n))
        C_set = set(C)
        
        internal_edges = set((u, v) for u in range(k) for v in range(u+1, k))
        
        neighborhood_groups = defaultdict(list)
        for v in I:
            N_v = frozenset(adj[v] & C_set)
            if N_v:
                neighborhood_groups[N_v].append(v)
        
        best_partition = None
        best_count = float('inf')
        
        # Strategy 1: Ascending size
        # Strategy 2: Descending size
        # Strategy 3-6: Random shuffles
        
        orders = []
        items = list(neighborhood_groups.items())
        
        # Ascending
        orders.append(sorted(items, key=lambda item: len(item[0])))
        # Descending
        orders.append(sorted(items, key=lambda item: -len(item[0])))
        
        # Random shuffles
        rng = Random(seed + 1000)
        for _ in range(4):
            shuffled = items[:]
            rng.shuffle(shuffled)
            orders.append(shuffled)
            
        for order in orders:
            used_edges = set()
            partition = []
            
            for N_v_frozen, group_vertices in order:
                N_v = set(N_v_frozen)
                n_v = len(N_v)
                
                if n_v == 0:
                    continue
                    
                if n_v == 1:
                    u = next(iter(N_v))
                    for v in group_vertices:
                        partition.append(sorted([v, u]))
                        # Mark cross edge? No, cross edges are not in internal_edges.
                        # But we need to track which C-edges are used.
                        # The clique {v, u} uses no C-edges.
                        continue
                
                h_vertices = list(N_v)
                
                # Build complement of used-edge graph on N_v
                comp_adj = {u: set() for u in h_vertices}
                for i in range(len(h_vertices)):
                    u = h_vertices[i]
                    for j in range(i+1, len(h_vertices)):
                        w = h_vertices[j]
                        e = (min(u, w), max(u, w))
                        if e not in used_edges:
                            comp_adj[u].add(w)
                            comp_adj[w].add(u)
                            
                colors = dsatur_color(comp_adj, h_vertices)
                
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
                        
                    clique = S + group_vertices
                    partition.append(sorted(clique))
                        
                    for i in range(len(S)):
                        for j in range(i+1, len(S)):
                            used_edges.add((min(S[i], S[j]), max(S[i], S[j])))
                            
            remaining_edges = internal_edges - used_edges
            rem_cliques = cover_remaining_edges(remaining_edges, k)
            partition.extend(rem_cliques)
            
            count = len(partition)
            if count < best_count:
                best_count = count
                best_partition = partition
                
        return best_partition if best_partition else []

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
