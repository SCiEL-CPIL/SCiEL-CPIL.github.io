import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

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

    def build_conflicts(edges, n):
        edge_index = {e: i for i, e in enumerate(edges)}
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        n_edges = len(edges)
        conflicts = [0] * n_edges
        
        for i, (u, v) in enumerate(edges):
            near_vertices = set(adj[u]) | set(adj[v])
            near_vertices.add(u)
            near_vertices.add(v)
            
            for x in near_vertices:
                for y in adj[x]:
                    e2 = (min(x, y), max(x, y))
                    if e2 in edge_index:
                        j = edge_index[e2]
                        if j != i:
                            conflicts[i] |= (1 << j)
                            conflicts[j] |= (1 << i)
                            
        return conflicts

    def greedy_color(edges, conflicts, n_edges, order=None):
        """Greedy coloring with optional custom order."""
        if order is None:
            order = list(range(n_edges))
        
        color = [-1] * n_edges
        color_classes = []
        
        for idx in order:
            placed = False
            for c, edges_in_class in enumerate(color_classes):
                ok = True
                for e in edges_in_class:
                    if conflicts[idx] & (1 << e):
                        ok = False
                        break
                if ok:
                    color[idx] = c
                    edges_in_class.add(idx)
                    placed = True
                    break
            if not placed:
                color[idx] = len(color_classes)
                color_classes.append({idx})
        
        return color, len(color_classes)

    def local_search(color, conflicts, n_edges, deadline, max_iters=5000):
        """Local search: move edges to reduce colors, eliminate empty colors."""
        n_colors = max(color) + 1 if color else 0
        if n_colors == 0:
            return color, 0
        
        classes = [set() for _ in range(n_colors)]
        for i, c in enumerate(color):
            classes[c].add(i)
        
        best_n_colors = n_colors
        best_color = color[:]
        
        for iteration in range(max_iters):
            if time.time() > deadline - 5.0:
                break
            
            improved = False
            
            # Try to eliminate the highest color
            for c in range(n_colors - 1, -1, -1):
                if not classes[c]:
                    continue
                
                # Try to reassign all edges in color c to other colors
                temp_assign = {}
                success = True
                for e in list(classes[c]):
                    placed = False
                    for c2 in range(n_colors):
                        if c2 == c:
                            continue
                        ok = True
                        for x in classes[c2]:
                            if conflicts[e] & (1 << x):
                                ok = False
                                break
                        if ok:
                            temp_assign[e] = c2
                            placed = True
                            break
                    if not placed:
                        success = False
                        break
                
                if success:
                    for e, c2 in temp_assign.items():
                        classes[c].discard(e)
                        classes[c2].add(e)
                        color[e] = c2
                    n_colors -= 1
                    improved = True
                    if n_colors < best_n_colors:
                        best_n_colors = n_colors
                        best_color = color[:]
                    break
            
            # If no color eliminated, try moving individual edges to lower colors
            if not improved:
                for c in range(n_colors - 1, 0, -1):
                    if not classes[c]:
                        continue
                    for e in list(classes[c]):
                        for c2 in range(c):
                            ok = True
                            for x in classes[c2]:
                                if conflicts[e] & (1 << x):
                                    ok = False
                                    break
                            if ok:
                                classes[c].discard(e)
                                classes[c2].add(e)
                                color[e] = c2
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
            
            if not improved:
                break
        
        # Return best found
        if best_n_colors < n_colors:
            return best_color, best_n_colors
        return color, n_colors

    def multi_restart_greedy(edges, conflicts, n_edges, deadline):
        """Try multiple greedy orderings and keep the best."""
        best_color = None
        best_n_colors = float('inf')
        
        # Order 1: natural order
        c, nc = greedy_color(edges, conflicts, n_edges)
        c, nc = local_search(c, conflicts, n_edges, deadline)
        if nc < best_n_colors:
            best_n_colors = nc
            best_color = c[:]
        
        # Order 2: by degree of endpoints (high degree first)
        deg = [0] * len(edges)
        for i, (u, v) in enumerate(edges):
            # Approximate degree from conflicts
            deg[i] = bin(conflicts[i]).count('1')
        order2 = sorted(range(n_edges), key=lambda i: -deg[i])
        c, nc = greedy_color(edges, conflicts, n_edges, order2)
        c, nc = local_search(c, conflicts, n_edges, deadline)
        if nc < best_n_colors:
            best_n_colors = nc
            best_color = c[:]
        
        # Order 3: random shuffles
        rng = random.Random(42)
        for _ in range(3):
            order3 = list(range(n_edges))
            rng.shuffle(order3)
            c, nc = greedy_color(edges, conflicts, n_edges, order3)
            c, nc = local_search(c, conflicts, n_edges, deadline)
            if nc < best_n_colors:
                best_n_colors = nc
                best_color = c[:]
        
        return best_color, best_n_colors

    results = []
    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        n, d, seed = spec
        edges = instance(n, d, seed)
        n_edges = len(edges)
        conflicts = build_conflicts(edges, n)
        
        color, n_colors = multi_restart_greedy(edges, conflicts, n_edges, deadline)
        
        results.append(color)
        
    return results
