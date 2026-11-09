import time
import random
import sys
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # 10s margin for safety

    def instance(n, d, seed):
        rng = random.Random(seed)
        degree, edges = [0] * n, set()
        for _ in range(4 * n * d):
            u, v = rng.randrange(n), rng.randrange(n)
            e = (min(u, v), max(u, v))
            if u != v and degree[u] < d and degree[v] < d and e not in edges:
                edges.add(e)
                degree[u] += 1
                degree[v] += 1
        return sorted(edges)

    def solve_instance(edges, adj):
        m = len(edges)
        if m == 0:
            return []
        
        # Build incompatibility graph
        # Two edges are incompatible if they share a vertex OR are joined by an edge.
        # We want to color such that each color class is a set of mutually compatible edges.
        # This is equivalent to coloring the incompatibility graph.
        
        # For efficiency, we'll build a list of lists of incompatible edges.
        # Since m is small (~200-500), O(m^2) is fine.
        incompatible = [set() for _ in range(m)]
        
        for i in range(m):
            u1, v1 = edges[i]
            for j in range(i + 1, m):
                u2, v2 = edges[j]
                # Check if they form an induced matching
                # 1. Disjoint vertices
                if u1 == u2 or u1 == v2 or v1 == u2 or v1 == v2:
                    incompatible[i].add(j)
                    incompatible[j].add(i)
                    continue
                # 2. No edge between {u1, v1} and {u2, v2}
                if u2 in adj[u1] or v2 in adj[u1] or u2 in adj[v1] or v2 in adj[v1]:
                    incompatible[i].add(j)
                    incompatible[j].add(i)
        
        # Greedy Coloring of the incompatibility graph
        # We want to minimize the number of colors.
        # Standard greedy: process vertices in some order, assign smallest available color.
        # To improve results, we can sort vertices by degree in the incompatibility graph (descending)
        # or use a simple heuristic.
        
        # Let's try a simple greedy with a decent ordering.
        # Order by degree in incompatibility graph (descending) is a standard heuristic for graph coloring.
        degrees = [len(incompatible[i]) for i in range(m)]
        order = sorted(range(m), key=lambda x: -degrees[x])
        
        colors = [-1] * m
        current_color = 0
        
        for v in order:
            # Find the smallest color not used by neighbors
            used_colors = set()
            for u in incompatible[v]:
                if colors[u] != -1:
                    used_colors.add(colors[u])
            
            c = 0
            while c in used_colors:
                c += 1
            colors[v] = c
            if c + 1 > current_color:
                current_color = c + 1
        
        return colors

    out = []
    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        if time.time() > deadline:
            # If we run out of time, we must return something valid.
            # Since we haven't processed this instance, we can't return a valid coloring for it yet.
            # But the loop structure allows us to check before starting an instance.
            # If we are in the middle of an instance, we'd need to handle it inside solve_instance.
            # Given the speed of the new algorithm, this is unlikely.
            pass
            
        edges = instance(*spec)
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        colors = solve_instance(edges, adj)
        out.append(colors)
        
    return out
