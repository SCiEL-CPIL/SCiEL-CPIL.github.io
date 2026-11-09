import time
import random
import sys
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 110.0

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

    def solve_instance(edges, adj, time_budget):
        m = len(edges)
        if m == 0:
            return []
        
        # Build compatibility graph (edges that CAN share a color)
        # Two edges are compatible if they don't share a vertex and aren't joined by an edge
        compatible = [set() for _ in range(m)]
        
        for i in range(m):
            u1, v1 = edges[i]
            for j in range(i + 1, m):
                u2, v2 = edges[j]
                # Check if they share a vertex
                if u1 == u2 or u1 == v2 or v1 == u2 or v1 == v2:
                    continue
                # Check if they are joined by an edge
                if u2 in adj[u1] or v2 in adj[u1] or u2 in adj[v1] or v2 in adj[v1]:
                    continue
                # They are compatible
                compatible[i].add(j)
                compatible[j].add(i)
        
        # Maximum clique in compatibility graph = maximum induced matching
        # Use Bron-Kerbosch with pivot for finding maximum clique
        best_clique_size = 0
        best_cliques = []  # Store found cliques for coloring
        
        def bron_kerbosch(R, P, X):
            nonlocal best_clique_size
            if time.time() > deadline:
                return
            if not P and not X:
                if len(R) > best_clique_size:
                    best_clique_size = len(R)
                    best_cliques.append(set(R))
                return
            # Pivot selection: choose vertex with max degree in P ∪ X
            pivot = max(P | X, key=lambda v: len(compatible[v] & (P | X))) if (P | X) else None
            candidates = P - (compatible[pivot] if pivot is not None else set())
            for v in list(candidates):
                new_R = R | {v}
                new_P = P & compatible[v]
                new_X = X & compatible[v]
                bron_kerbosch(new_R, new_P, new_X)
                P.remove(v)
                X.add(v)
        
        # Find several large cliques to build a good initial coloring
        # Use a greedy approach to extract maximal cliques
        remaining = set(range(m))
        found_cliques = []
        
        while remaining and time.time() < deadline - 10:
            # Greedily build a clique from remaining vertices
            clique = set()
            candidates = set(remaining)
            while candidates:
                # Pick vertex with max degree in candidates
                v = max(candidates, key=lambda x: len(compatible[x] & candidates))
                clique.add(v)
                remaining.discard(v)
                candidates = candidates & compatible[v]
            if len(clique) > 1:
                found_cliques.append(clique)
        
        # Now do local search: try to merge color classes
        # Initial coloring: assign each found clique a color, rest get new colors
        colors = [-1] * m
        color_idx = 0
        for clique in found_cliques:
            for v in clique:
                colors[v] = color_idx
            color_idx += 1
        
        # Color remaining vertices
        for v in range(m):
            if colors[v] == -1:
                colors[v] = color_idx
                color_idx += 1
        
        current_color_count = color_idx
        
        # Local search: try to merge color classes
        def try_merge(source, target):
            nonlocal current_color_count
            target_edges = [i for i in range(m) if colors[i] == target]
            if not target_edges:
                return True
            
            source_edges = set(i for i in range(m) if colors[i] == source)
            
            # Check conflicts
            conflicts = []
            for i in target_edges:
                for j in source_edges:
                    if j in compatible[i]:
                        continue  # compatible, no conflict
                    else:
                        conflicts.append(i)
                        break
            
            if not conflicts:
                for i in target_edges:
                    colors[i] = source
                # Recompute color count
                used = set(colors)
                current_color_count = len(used)
                return True
            
            if len(conflicts) > 3:
                return False
            
            old_colors = colors[:]
            
            for i in target_edges:
                colors[i] = source
            
            resolved = True
            for i in conflicts:
                valid = set(range(current_color_count + 1))
                for j in compatible[i]:
                    if colors[j] != -1:
                        valid.discard(colors[j])
                # Also need to check incompatibility - actually compatible set already handles this
                # An edge can go to color c if all edges with color c are compatible with i
                valid_final = set()
                for c in valid:
                    ok = True
                    for j in range(m):
                        if colors[j] == c and j not in compatible[i]:
                            ok = False
                            break
                    if ok:
                        valid_final.add(c)
                
                if not valid_final:
                    resolved = False
                    break
                colors[i] = min(valid_final)
            
            if resolved:
                used = set(colors)
                if len(used) < current_color_count:
                    current_color_count = len(used)
                    return True
            
            colors[:] = old_colors
            return False
        
        rng = random.Random(42)
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            used_colors = set(colors)
            if len(used_colors) <= 1:
                break
            
            color_counts = defaultdict(int)
            for c in colors:
                color_counts[c] += 1
            
            sorted_colors = sorted(color_counts.keys(), key=lambda x: color_counts[x])
            target = rng.choice(sorted_colors[:len(sorted_colors)//2 + 1])
            source = rng.choice([c for c in used_colors if c != target])
            
            if try_merge(source, target):
                improved = True
        
        return colors

    out = []
    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        if time.time() > deadline:
            break
        edges = instance(*spec)
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        colors = solve_instance(edges, adj, 30.0)
        out.append(colors)
        
    return out
