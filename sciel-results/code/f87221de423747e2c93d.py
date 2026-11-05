import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

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

    def build_line_square_graph(edges):
        m = len(edges)
        if m == 0:
            return []
        edge_index = {e: i for i, e in enumerate(edges)}
        adj = [set() for _ in range(m)]
        
        orig_adj = defaultdict(set)
        for u, v in edges:
            orig_adj[u].add(v)
            orig_adj[v].add(u)

        for i in range(m):
            u, v = edges[i]
            for j in range(i + 1, m):
                x, y = edges[j]
                # Check if they share a vertex
                if u in (x, y) or v in (x, y):
                    adj[i].add(j)
                    adj[j].add(i)
                else:
                    # Check if they are connected by an edge in G
                    # This means there exists an edge (a,b) in G such that
                    # (a,b) shares a vertex with (u,v) and (a,b) shares a vertex with (x,y).
                    # Let's iterate over neighbors of u and v.
                    found = False
                    for w in orig_adj[u]:
                        if (min(w, x), max(w, x)) in edge_index or (min(w, y), max(w, y)) in edge_index:
                            found = True
                            break
                    if not found:
                        for w in orig_adj[v]:
                            if (min(w, x), max(w, x)) in edge_index or (min(w, y), max(w, y)) in edge_index:
                                found = True
                                break
                    if found:
                        adj[i].add(j)
                        adj[j].add(i)
        
        return adj

    def greedy_color(adj, m, order):
        colors = [-1] * m
        for v in order:
            used = set()
            for u in adj[v]:
                if colors[u] != -1:
                    used.add(colors[u])
            c = 0
            while c in used:
                c += 1
            colors[v] = c
        return colors

    def local_search_color(adj, m, colors, rng):
        # Try to reduce number of colors by removing highest colors
        max_color = max(colors) if colors else 0
        # We will try to remove colors from max_color down to 1
        # If we successfully remove a color, we decrement max_color
        # If not, we stop trying to remove that specific color in this pass?
        # Actually, we can try again later, but for simplicity, we just do one pass downwards.
        # To be more effective, we can loop this process multiple times.
        
        improved = True
        while improved and max_color > 0:
            improved = False
            k = max_color
            vertices_k = [i for i in range(m) if colors[i] == k]
            if not vertices_k:
                max_color -= 1
                continue
            
            rng.shuffle(vertices_k)
            
            for v in vertices_k:
                used = set()
                for u in adj[v]:
                    if colors[u] < k:
                        used.add(colors[u])
                
                c = 0
                while c < k and c in used:
                    c += 1
                
                if c < k:
                    colors[v] = c
                    improved = True
            
            if any(colors[i] == k for i in range(m)):
                # Failed to remove color k
                # We might have moved some vertices, but k is still used.
                # We should not decrement max_color.
                # To prevent infinite loop if no improvement is possible for k,
                # we break if no vertex was recolored?
                # 'improved' is True if any vertex was recolored.
                # If improved is True but k is still used, we loop again?
                # This could lead to oscillation.
                # Better: if we can't remove k, we just stop trying to remove k for now.
                # But we might be able to remove k-1?
                # Let's just break the while loop if we can't remove the current max.
                # Or, we can just continue to next k? No, max_color is the highest.
                # If we can't remove max_color, we can't reduce the number of colors further
                # using this strategy without more complex moves.
                break
            else:
                max_color -= 1

        return colors

    out = []
    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        edges = instance(*spec)
        m = len(edges)
        if m == 0:
            out.append([])
            continue
            
        adj = build_line_square_graph(edges)
        
        best_colors = None
        best_num_colors = float('inf')
        
        # Run multiple restarts
        num_restarts = 20
        for r in range(num_restarts):
            if time.time() > deadline - 5:
                break
                
            rng = random.Random(1000 + r)
            
            # Different ordering strategies
            if r % 3 == 0:
                order = sorted(range(m), key=lambda i: len(adj[i]), reverse=True)
            elif r % 3 == 1:
                order = sorted(range(m), key=lambda i: len(adj[i]))
            else:
                order = list(range(m))
                rng.shuffle(order)
            
            colors = greedy_color(adj, m, order)
            colors = local_search_color(adj, m, colors, rng)
            
            num_colors = max(colors) + 1
            if num_colors < best_num_colors:
                best_num_colors = num_colors
                best_colors = colors.copy()
                
            if time.time() > deadline - 10:
                break

        if best_colors is None:
            order = sorted(range(m), key=lambda i: len(adj[i]), reverse=True)
            best_colors = greedy_color(adj, m, order)
            
        out.append(best_colors)

    return out
