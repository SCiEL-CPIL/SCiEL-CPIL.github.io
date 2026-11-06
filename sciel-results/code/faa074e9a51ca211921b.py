"""Strong edge colouring via greedy + local refinement with multiple restarts."""
import time
from random import Random
from collections import defaultdict

def construct():
    deadline = time.time() + 135.0  # safety margin before 150s kill

    def instance(n, d, seed):
        rng = Random(seed)
        degree = [0] * n
        edges = set()
        for _ in range(4 * n * d):
            u, v = rng.randrange(n), rng.randrange(n)
            e = (min(u, v), max(u, v))
            if u != v and degree[u] < d and degree[v] < d and e not in edges:
                edges.add(e)
                degree[u] += 1
                degree[v] += 1
        return sorted(edges)

    def build_conflict(edges, adj):
        """Build conflict graph for strong edge coloring."""
        n_edges = len(edges)
        if n_edges == 0:
            return []
        
        edge_to_idx = {e: i for i, e in enumerate(edges)}
        vert_edges = defaultdict(list)
        for i, (u, v) in enumerate(edges):
            vert_edges[u].append(i)
            vert_edges[v].append(i)
        
        conflict = [set() for _ in range(n_edges)]
        for u, v in edges:
            ei = edge_to_idx[(u, v)]
            for w in adj.get(u, set()):
                for ej in vert_edges.get(w, []):
                    if ej != ei:
                        conflict[ei].add(ej)
                        conflict[ej].add(ei)
            for w in adj.get(v, set()):
                for ej in vert_edges.get(w, []):
                    if ej != ei:
                        conflict[ei].add(ej)
                        conflict[ej].add(ei)
        return conflict

    def greedy_color(order, conflict, n_edges):
        """Greedy coloring given an edge order."""
        color = [-1] * n_edges
        for e in order:
            forbidden = set()
            for f in conflict[e]:
                if color[f] != -1:
                    forbidden.add(color[f])
            c = 0
            while c in forbidden:
                c += 1
            color[e] = c
        return color

    def local_refine(color, conflict, n_edges, max_iter=5000, deadline=None):
        """Iteratively recolor edges to lower colors if possible."""
        for _ in range(max_iter):
            if deadline is not None and time.time() > deadline:
                break
            improved = False
            for e in range(n_edges):
                old_c = color[e]
                if old_c == 0:
                    continue
                for new_c in range(old_c):
                    ok = True
                    for f in conflict[e]:
                        if color[f] == new_c:
                            ok = False
                            break
                    if ok:
                        color[e] = new_c
                        improved = True
                        break
            if not improved:
                break
        return color

    def reduce_max_color(color, conflict, n_edges):
        """Try to eliminate the highest color."""
        max_c = max(color) if color else 0
        if max_c <= 0:
            return color
        for e in range(n_edges):
            if color[e] == max_c:
                for new_c in range(max_c):
                    ok = True
                    for f in conflict[e]:
                        if color[f] == new_c:
                            ok = False
                            break
                    if ok:
                        color[e] = new_c
                        break
        return color

    def solve_instance(n, d, seed):
        edges = instance(n, d, seed)
        n_edges = len(edges)
        if n_edges == 0:
            return []
        
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        conflict = build_conflict(edges, adj)
        
        best_color = None
        best_count = float('inf')
        
        # Strategy 1: Greedy with descending conflict degree
        order1 = sorted(range(n_edges), key=lambda e: -len(conflict[e]))
        color1 = greedy_color(order1, conflict, n_edges)
        color1 = local_refine(color1, conflict, n_edges, max_iter=5000, deadline=deadline)
        color1 = reduce_max_color(color1, conflict, n_edges)
        color1 = local_refine(color1, conflict, n_edges, max_iter=3000, deadline=deadline)
        count1 = max(color1) + 1
        best_color = color1[:]
        best_count = count1
        
        # Strategy 2: Greedy with ascending conflict degree
        order2 = sorted(range(n_edges), key=lambda e: len(conflict[e]))
        color2 = greedy_color(order2, conflict, n_edges)
        color2 = local_refine(color2, conflict, n_edges, max_iter=5000, deadline=deadline)
        color2 = reduce_max_color(color2, conflict, n_edges)
        color2 = local_refine(color2, conflict, n_edges, max_iter=3000, deadline=deadline)
        count2 = max(color2) + 1
        if count2 < best_count:
            best_count = count2
            best_color = color2[:]
        
        # Strategy 3+: Randomized restarts
        for restart in range(20):
            if time.time() > deadline - 10:
                break
            rng = Random(seed * 1000 + restart)
            shuffled = list(range(n_edges))
            rng.shuffle(shuffled)
            color_r = greedy_color(shuffled, conflict, n_edges)
            color_r = local_refine(color_r, conflict, n_edges, max_iter=4000, deadline=deadline)
            color_r = reduce_max_color(color_r, conflict, n_edges)
            color_r = local_refine(color_r, conflict, n_edges, max_iter=2000, deadline=deadline)
            count_r = max(color_r) + 1
            if count_r < best_count:
                best_count = count_r
                best_color = color_r[:]
        
        return best_color

    out = []
    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        if time.time() > deadline - 5:
            edges = instance(*spec)
            adj = defaultdict(set)
            for u, v in edges:
                adj[u].add(v)
                adj[v].add(u)
            conflict = build_conflict(edges, adj)
            n_edges = len(edges)
            order = sorted(range(n_edges), key=lambda e: -len(conflict[e])) if n_edges > 0 else []
            color = greedy_color(order, conflict, n_edges) if n_edges > 0 else []
            out.append(color)
        else:
            out.append(solve_instance(*spec))
    
    return out
