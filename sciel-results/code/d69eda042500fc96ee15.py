"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 150s limit
    
    n = 20
    best_edges = [[0, v, 0] for v in range(1, n)]  # Star as fallback (19 edges)
    best_count = len(best_edges)
    
    all_pairs = [(u, v) for u in range(n) for v in range(u+1, n)]
    
    def check_valid(adj, u, v, c):
        """Check if adding edge (u,v) with colour c creates any non-rainbow 4-cycle."""
        # 4-cycles containing (u,v): u-v-x-y-u where x~v, y~u, x~y
        for x in range(n):
            if x == u or x == v:
                continue
            if adj[v][x] == -1:
                continue
            cvx = adj[v][x]
            if c == cvx:
                continue  # quick reject for this x
            for y in range(n):
                if y == u or y == v or y == x:
                    continue
                if adj[u][y] == -1:
                    continue
                if adj[x][y] == -1:
                    continue
                cyu = adj[y][u]
                cxy = adj[x][y]
                # Check: c, cvx, cxy, cyu all distinct
                if c == cxy or c == cyu or cvx == cxy or cvx == cyu or cxy == cyu:
                    return False
        return True
    
    def try_complete(adj, edges, order, deadline):
        """Try to add edges from order to adj, respecting time limit."""
        for u, v in order:
            if time.time() > deadline:
                break
            # Try colours in a smart order: prefer unused colors first
            used_colors = set()
            for x in range(n):
                if adj[u][x] != -1:
                    used_colors.add(adj[u][x])
                if adj[v][x] != -1:
                    used_colors.add(adj[v][x])
            
            # Try unused colors first, then used
            colors = list(range(n))
            colors.sort(key=lambda c: (c in used_colors, c))
            random.shuffle([c for c in colors if c in used_colors])
            random.shuffle([c for c in colors if c not in used_colors])
            colors = [c for c in colors if c not in used_colors] + [c for c in colors if c in used_colors]
            
            for c in colors:
                if check_valid(adj, u, v, c):
                    adj[u][v] = c
                    adj[v][u] = c
                    edges.append([u, v, c])
                    break
        return edges
    
    for attempt in range(500):
        if time.time() > deadline:
            break
        
        # Different strategies for edge ordering
        strategy = attempt % 5
        
        if strategy == 0:
            # Pure random
            order = all_pairs[:]
            random.shuffle(order)
        elif strategy == 1:
            # Start with a random spanning tree, then add the rest
            tree = []
            unvisited = set(range(n))
            visited = {0}
            unvisited.remove(0)
            tree_edges = []
            while unvisited:
                for v in unvisited:
                    if random.random() < 0.5:
                        break
                # Find a random unvisited vertex and connect to a random visited vertex
                v = random.choice(list(unvisited))
                u = random.choice(list(visited))
                tree_edges.append((min(u,v), max(u,v)))
                visited.add(v)
                unvisited.remove(v)
            rest = [p for p in all_pairs if p not in set(tree_edges)]
            random.shuffle(rest)
            order = tree_edges + rest
        elif strategy == 2:
            # Random but biased towards high-degree potential
            order = all_pairs[:]
            # Weight by some heuristic
            random.shuffle(order)
        elif strategy == 3:
            # Reverse random
            order = all_pairs[:]
            random.shuffle(order)
            order.reverse()
        else:
            # Random with local clustering
            order = all_pairs[:]
            # Group by vertex proximity
            for i in range(0, len(order) - 1, 2):
                if random.random() < 0.3:
                    order[i], order[i+1] = order[i+1], order[i]
        
        adj = [[-1]*n for _ in range(n)]
        edges = []
        
        # First pass: add edges from order
        edges = try_complete(adj, edges, order, deadline)
        
        # Second pass: try to add any remaining edges
        if time.time() < deadline - 5:
            remaining = [p for p in all_pairs if adj[p[0]][p[1]] == -1]
            random.shuffle(remaining)
            edges = try_complete(adj, edges, remaining, deadline)
        
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = edges[:]
    
    return best_edges
# EVOLVE-BLOCK-END
