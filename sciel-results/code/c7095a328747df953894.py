import time
import math
import numpy as np
from random import Random

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

    def build_conflict(n, edges):
        m = len(edges)
        if m == 0:
            return [set() for _ in range(m)]
        
        adj_orig = [set() for _ in range(n)]
        for u, v in edges:
            adj_orig[u].add(v)
            adj_orig[v].add(u)
        
        conflict = [set() for _ in range(m)]
        
        for i in range(m):
            a, b = edges[i]
            for j in range(i + 1, m):
                c, dd = edges[j]
                if a == c or a == dd or b == c or b == dd:
                    conflict[i].add(j)
                    conflict[j].add(i)
                else:
                    if c in adj_orig[a] or dd in adj_orig[a] or c in adj_orig[b] or dd in adj_orig[b]:
                        conflict[i].add(j)
                        conflict[j].add(i)
        
        return conflict

    def dsatur_color(conflict, m, deadline, rng):
        """
        DSATUR (Degree of Saturation) graph coloring heuristic.
        At each step, pick the uncolored vertex with the highest saturation degree
        (number of distinct colors in its colored neighbors), breaking ties by degree.
        """
        if m == 0:
            return [], 0
        
        color = [-1] * m
        color_sets = [set() for _ in range(m)]  # colors used by neighbors
        uncolored = set(range(m))
        num_colors = 0
        
        # Precompute degrees in conflict graph
        degrees = [len(conflict[i]) for i in range(m)]
        
        while uncolored and time.time() < deadline:
            # Pick vertex with max saturation, then max degree
            best_v = -1
            best_sat = -1
            best_deg = -1
            for v in uncolored:
                sat = len(color_sets[v])
                deg = degrees[v]
                if sat > best_sat or (sat == best_sat and deg > best_deg):
                    best_sat = sat
                    best_deg = deg
                    best_v = v
            
            if best_v == -1:
                break
            
            # Assign smallest available color
            forbidden = color_sets[best_v]
            c = 0
            while c in forbidden:
                c += 1
            
            color[best_v] = c
            if c >= num_colors:
                num_colors = c + 1
            
            uncolored.remove(best_v)
            
            # Update color sets of uncolored neighbors
            for u in conflict[best_v]:
                if u in uncolored:
                    color_sets[u].add(c)
        
        return color, num_colors

    def greedy_peeling_color(conflict, m, deadline, rng):
        """Greedy peeling: repeatedly extract independent sets."""
        if m == 0:
            return [], 0
        
        color = [-1] * m
        active = set(range(m))
        current_color = 0
        
        while active and time.time() < deadline:
            # Greedy: pick min-degree vertex in remaining subgraph
            iset = set()
            remaining = set(active)
            
            while remaining:
                # Find min-degree vertex
                best_v = -1
                best_deg = float('inf')
                for v in remaining:
                    deg = len(conflict[v] & remaining)
                    if deg < best_deg:
                        best_deg = deg
                        best_v = v
                
                iset.add(best_v)
                remaining.discard(best_v)
                remaining -= conflict[best_v]
            
            for v in iset:
                color[v] = current_color
            
            active -= iset
            current_color += 1
        
        return color, current_color

    def local_search_reduce(color, conflict, m, deadline):
        """
        Try to reduce colors by removing empty color classes and reassigning.
        Also try swapping colors to enable removal of a color.
        """
        if m == 0:
            return color, 0
        
        num_colors = max(color) + 1
        if num_colors == 0:
            return color, 0
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # Try to remove each color from highest to lowest
            for c in range(num_colors - 1, -1, -1):
                # Find all vertices with color c
                verts_c = [i for i in range(m) if color[i] == c]
                if not verts_c:
                    num_colors -= 1
                    continue
                
                # Try to reassign each vertex in verts_c to another color
                trial_color = color[:]
                success = True
                
                for v in verts_c:
                    placed = False
                    # Try existing colors (excluding c)
                    for c2 in range(num_colors):
                        if c2 == c:
                            continue
                        # Check if v can take color c2
                        can_add = True
                        for u in conflict[v]:
                            if trial_color[u] == c2:
                                can_add = False
                                break
                        if can_add:
                            trial_color[v] = c2
                            placed = True
                            break
                    
                    if not placed:
                        success = False
                        break
                
                if success:
                    color = trial_color
                    num_colors -= 1
                    improved = True
                    break
        
        return color, num_colors

    def solve_instance(n, d, seed, deadline):
        edges = instance(n, d, seed)
        m = len(edges)
        if m == 0:
            return []
        
        conflict = build_conflict(n, edges)
        
        rng = Random(seed * 7919 + 42)
        
        best_coloring = None
        best_num_colors = float('inf')
        
        # Try multiple strategies
        strategies = [
            ("dsatur", dsatur_color),
            ("peeling", greedy_peeling_color),
        ]
        
        attempt = 0
        while time.time() < deadline - 0.5:
            attempt += 1
            strat_name, strat_func = strategies[attempt % len(strategies)]
            attempt_rng = Random(rng.random())
            
            coloring, num_colors = strat_func(conflict, m, deadline - 0.5, attempt_rng)
            
            # Local search to reduce
            coloring, num_colors = local_search_reduce(coloring, conflict, m, deadline - 0.5)
            
            if num_colors < best_num_colors:
                best_num_colors = num_colors
                best_coloring = coloring[:]
            
            # Check if we've reached a good bound
            en_bound = int(5 * d * d / 4)
            if best_num_colors <= en_bound:
                break
        
        if best_coloring is None:
            # Fallback: simple greedy
            color = [-1] * m
            active = set(range(m))
            c = 0
            while active and time.time() < deadline:
                v = min(active)
                color[v] = c
                active.discard(v)
                active -= conflict[v]
                c += 1
            best_coloring = color
        
        return compress(best_coloring)

    try:
        start = time.time()
        total_budget = 140.0
        per_instance = total_budget / len(specs)

        out = []
        for idx, (n, d, seed) in enumerate(specs):
            remaining = start + total_budget - time.time()
            share = min(per_instance, remaining / (len(specs) - idx))
            share = max(share, 5.0)
            instance_deadline = time.time() + share
            out.append(solve_instance(n, d, seed, instance_deadline))

        return out
    except Exception:
        # Fallback: simple greedy
        out = []
        for spec in specs:
            n, d, seed = spec
            edges = instance(n, d, seed)
            if not edges:
                out.append([])
                continue
            adj = {}
            for u, v in edges:
                adj.setdefault(u, set()).add(v)
                adj.setdefault(v, set()).add(u)
            colour = {}
            for e in edges:
                near = {x for x in e} | {y for x in e for y in adj.get(x, set())}
                taken = set()
                for f, c in colour.items():
                    if set(f) & near:
                        taken.add(c)
                c = 0
                while c in taken:
                    c += 1
                colour[e] = c
            out.append([colour[e] for e in edges])
        return out
