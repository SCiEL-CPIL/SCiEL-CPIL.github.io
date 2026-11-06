import time
import sys
from random import Random
from collections import deque

def construct():
    def instance(n, d, seed):
        rng = Random(seed)
        degree, edges = [0] * n, set()
        for _ in range(4 * n * d):
            u, v = rng.randrange(n), rng.randrange(n)
            e = (min(u, v), max(u, v))
            if u != v and degree[u] < d and degree[v] < d and e not in edges:
                edges.add(e)
                degree[u] += 1
                degree[v] += 1
        return sorted(edges)

    specs = [(40, 4, 101), (60, 5, 102), (80, 6, 103)]

    def fallback():
        out = []
        for spec in specs:
            edges = instance(*spec)
            adj = {}
            for u, v in edges:
                adj.setdefault(u, set()).add(v)
                adj.setdefault(v, set()).add(u)
            colour = {}
            for e in edges:
                near = {x for x in e} | {y for x in e for y in adj[x]}
                taken = {c for f, c in colour.items() if set(f) & near}
                colour[e] = next(c for c in range(len(edges)) if c not in taken)
            out.append([colour[e] for e in edges])
        return out

    def compress(color):
        m = len(color)
        if m == 0:
            return []
        mapping = {}
        new = [0] * m
        nxt = 0
        for i, c in enumerate(color):
            if c not in mapping:
                mapping[c] = nxt
                nxt += 1
            new[i] = mapping[c]
        return new

    def solve_instance(n, d, seed, deadline):
        edges = instance(n, d, seed)
        m = len(edges)
        if m == 0:
            return []
        
        # Build conflict graph
        adj_orig = [set() for _ in range(n)]
        for u, v in edges:
            adj_orig[u].add(v)
            adj_orig[v].add(u)
            
        neigh = [[] for _ in range(m)]
        for i in range(m):
            a, b = edges[i]
            for j in range(i + 1, m):
                c, dd = edges[j]
                if a == c or a == dd or b == c or b == dd:
                    neigh[i].append(j)
                    neigh[j].append(i)
                else:
                    if c in adj_orig[a] or dd in adj_orig[a] or c in adj_orig[b] or dd in adj_orig[b]:
                        neigh[i].append(j)
                        neigh[j].append(i)
                        
        deg = [len(neigh[i]) for i in range(m)]
        
        # DSATUR greedy coloring
        def dsatur_coloring(order=None):
            if order is None:
                order = list(range(m))
            color = [-1] * m
            # Track saturation degree (number of different colors in neighborhood)
            sat = [set() for _ in range(m)]
            # Sort by degree initially
            order = sorted(order, key=lambda x: deg[x], reverse=True)
            
            for v in order:
                # Find smallest available color
                forbidden = set()
                for u in neigh[v]:
                    if color[u] != -1:
                        forbidden.add(color[u])
                c = 0
                while c in forbidden:
                    c += 1
                color[v] = c
                # Update saturation of neighbors
                for u in neigh[v]:
                    if color[u] != -1:
                        sat[u].add(color[v])
            
            return color, max(color) + 1
        
        # Try multiple random orderings
        rng = Random(seed * 7919 + 13)
        best_color = None
        best_cost = m  # Upper bound
        
        # First, deterministic DSATUR
        c0, k0 = dsatur_coloring()
        best_color = c0
        best_cost = k0
        
        # Try several random restarts
        for restart in range(50):
            if time.time() > deadline - 5.0:
                break
            perm = list(range(m))
            rng.shuffle(perm)
            c, k = dsatur_coloring(perm)
            if k < best_cost:
                best_color = c
                best_cost = k
        
        # Now try to reduce colors using backtracking
        # Lower bound: max degree in conflict graph + 1
        lower_bound = max(deg) + 1
        
        # Try to find coloring with best_cost - 1, best_cost - 2, ... down to lower_bound
        current_color = best_color[:]
        current_K = best_cost
        
        def can_color_with_K(K, start_color=None, max_nodes=500000):
            """Backtracking to find K-coloring. Returns coloring or None."""
            if time.time() > deadline - 3.0:
                return None
            
            # Use the start_color as a guide for ordering
            if start_color is not None:
                # Order vertices by how constrained they are given start_color
                order = sorted(range(m), key=lambda v: len(neigh[v]), reverse=True)
            else:
                order = sorted(range(m), key=lambda v: deg[v], reverse=True)
            
            color = [-1] * m
            nodes_visited = [0]
            
            def backtrack(idx):
                if time.time() > deadline - 3.0:
                    return False
                if nodes_visited[0] >= max_nodes:
                    return False
                if idx == m:
                    return True
                nodes_visited[0] += 1
                
                v = order[idx]
                forbidden = set()
                for u in neigh[v]:
                    if color[u] != -1:
                        forbidden.add(color[u])
                
                # Try colors 0 to K-1
                for c in range(K):
                    if c not in forbidden:
                        color[v] = c
                        if backtrack(idx + 1):
                            return True
                        color[v] = -1
                
                return False
            
            if backtrack(0):
                return color
            return None
        
        # Try to reduce colors step by step
        for target_K in range(current_K - 1, lower_bound - 1, -1):
            if time.time() > deadline - 3.0:
                break
            
            # Try multiple times with different node limits
            found = False
            for attempt in range(3):
                if time.time() > deadline - 3.0:
                    break
                max_nodes = 200000 + attempt * 300000
                result = can_color_with_K(target_K, current_color, max_nodes)
                if result is not None:
                    current_color = result
                    current_K = target_K
                    found = True
                    break
            
            if not found:
                break  # Can't reduce further
        
        return compress(current_color)

    try:
        start = time.time()
        deadline = start + 130.0  # Safety margin before 150s kill
        out = []
        for idx, (n, d, seed) in enumerate(specs):
            remaining = deadline - time.time()
            # Allocate time proportionally, but ensure minimum
            share = remaining / (len(specs) - idx)
            share = min(share, 40.0)  # Cap per instance
            share = max(share, 1.0)
            instance_deadline = start + (time.time() - start) + share
            out.append(solve_instance(n, d, seed, instance_deadline))
        return out
    except Exception:
        return fallback()
