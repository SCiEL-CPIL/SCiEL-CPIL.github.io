import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s kill
    n = 20
    num_colors = 20
    
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    edge_set = set(all_edges)
    m = len(all_edges)
    
    best_count = 0
    best_edges = []
    
    def check_rainbow(adj, neighbors):
        """Verify all 4-cycles are rainbow. Returns True if valid."""
        for a in range(n):
            for b in neighbors[a]:
                if b <= a:
                    continue
                ca = adj[a][b]
                for c in neighbors[b]:
                    if c == a or c <= b:
                        continue
                    cb = adj[b][c]
                    if ca == cb:
                        return False
                    for d in neighbors[c]:
                        if d == b or d <= c:
                            continue
                        cd = adj[c][d]
                        if ca == cd or cb == cd:
                            return False
                        if d in neighbors[a]:
                            da = adj[a][d]
                            if ca == da or cb == da or cd == da:
                                return False
        return True
    
    def build_adj(edges):
        adj = [[-1] * n for _ in range(n)]
        neighbors = [set() for _ in range(n)]
        for u, v, c in edges:
            adj[u][v] = c
            adj[v][u] = c
            neighbors[u].add(v)
            neighbors[v].add(u)
        return adj, neighbors
    
    def greedy_build(seed, init_edges=None):
        rng = random.Random(seed)
        adj = [[-1] * n for _ in range(n)]
        neighbors = [set() for _ in range(n)]
        edges = []
        
        if init_edges:
            for u, v, c in init_edges:
                adj[u][v] = c
                adj[v][u] = c
                neighbors[u].add(v)
                neighbors[v].add(u)
                edges.append([u, v, c])
            existing = set((min(u,v), max(u,v)) for u, v, c in edges)
            remaining = [e for e in all_edges if e not in existing]
        else:
            remaining = all_edges[:]
        
        rng.shuffle(remaining)
        
        for (u, v) in remaining:
            forbidden = set()
            impossible = False
            
            for a in neighbors[u]:
                if a == v:
                    continue
                ca = adj[u][a]
                for b in neighbors[v]:
                    if b == u or b == a:
                        continue
                    if adj[a][b] != -1:
                        cb = adj[a][b]
                        cv = adj[b][v]
                        if ca == cb or cb == cv or ca == cv:
                            impossible = True
                            break
                        forbidden.add(ca)
                        forbidden.add(cb)
                        forbidden.add(cv)
                if impossible:
                    break
            
            if not impossible:
                colors = list(range(num_colors))
                rng.shuffle(colors)
                for c in colors:
                    if c not in forbidden:
                        adj[u][v] = c
                        adj[v][u] = c
                        neighbors[u].add(v)
                        neighbors[v].add(u)
                        edges.append([u, v, c])
                        break
        
        return edges, adj, neighbors
    
    def try_add_edge(adj, neighbors, u, v):
        """Try to add edge (u,v) with a valid color. Returns color or -1."""
        forbidden = set()
        impossible = False
        
        for a in neighbors[u]:
            if a == v:
                continue
            ca = adj[u][a]
            for b in neighbors[v]:
                if b == u or b == a:
                    continue
                if adj[a][b] != -1:
                    cb = adj[a][b]
                    cv = adj[b][v]
                    if ca == cb or cb == cv or ca == cv:
                        impossible = True
                        break
                    forbidden.add(ca)
                    forbidden.add(cb)
                    forbidden.add(cv)
            if impossible:
                break
        
        if impossible:
            return -1
        
        for c in range(num_colors):
            if c not in forbidden:
                return c
        return -1
    
    def local_search(seed, edges, adj, neighbors, max_iter=5000):
        """Local search: try to add missing edges, and if stuck, remove one and retry."""
        rng = random.Random(seed)
        current_edges = edges[:]
        cur_adj = [[adj[i][j] for j in range(n)] for i in range(n)]
        cur_neighbors = [set(neighbors[i]) for i in range(n)]
        
        cur_count = len(current_edges)
        existing = set((min(e[0], e[1]), max(e[0], e[1])) for e in current_edges)
        missing = [(u, v) for u, v in all_edges if (u, v) not in existing]
        
        no_improve = 0
        for it in range(max_iter):
            if time.time() > deadline:
                break
            if not missing:
                break
            
            # Try to add a missing edge
            rng.shuffle(missing)
            added = False
            for (u, v) in missing:
                c = try_add_edge(cur_adj, cur_neighbors, u, v)
                if c != -1:
                    cur_adj[u][v] = c
                    cur_adj[v][u] = c
                    cur_neighbors[u].add(v)
                    cur_neighbors[v].add(u)
                    current_edges.append([u, v, c])
                    cur_count += 1
                    missing.remove((u, v))
                    added = True
                    no_improve = 0
                    break
            
            if not added:
                no_improve += 1
                if no_improve > 20 and cur_count > 10:
                    # Remove a random edge and try again
                    idx = rng.randint(0, len(current_edges) - 1)
                    u, v, c = current_edges[idx]
                    cur_adj[u][v] = -1
                    cur_adj[v][u] = -1
                    cur_neighbors[u].discard(v)
                    cur_neighbors[v].discard(u)
                    current_edges.pop(idx)
                    cur_count -= 1
                    missing.append((u, v))
                    no_improve = 0
        
        return current_edges, cur_adj, cur_neighbors, cur_count
    
    # Phase 1: Multiple greedy restarts
    num_restarts = 300
    for i in range(num_restarts):
        if time.time() > deadline:
            break
        edges, adj, neighbors = greedy_build(i)
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = edges[:]
    
    # Phase 2: Local search on best solutions
    # Keep top few solutions for local search
    top_solutions = []
    for i in range(min(50, num_restarts)):
        if time.time() > deadline:
            break
        edges, adj, neighbors = greedy_build(i)
        top_solutions.append((len(edges), edges, adj, neighbors))
    
    top_solutions.sort(key=lambda x: -x[0])
    
    for count, edges, adj, neighbors in top_solutions[:10]:
        if time.time() > deadline:
            break
        new_edges, new_adj, new_neighbors, new_count = local_search(
            random.randint(0, 10**9), edges, adj, neighbors, max_iter=3000
        )
        if new_count > best_count:
            best_count = new_count
            best_edges = new_edges[:]
    
    # Phase 3: More aggressive local search with different seeds
    while time.time() < deadline:
        seed = random.randint(0, 10**9)
        edges, adj, neighbors = greedy_build(seed)
        new_edges, new_adj, new_neighbors, new_count = local_search(
            seed + 1, edges, adj, neighbors, max_iter=2000
        )
        if new_count > best_count:
            best_count = new_count
            best_edges = new_edges[:]
    
    # Final verification
    if best_edges:
        adj, neighbors = build_adj(best_edges)
        if not check_rainbow(adj, neighbors):
            # Fallback: rebuild from scratch with simple greedy
            edges, adj, neighbors = greedy_build(0)
            best_edges = edges
            best_count = len(edges)
    
    if not best_edges:
        best_edges = [[0, v, v % 20] for v in range(1, 20)]
    
    return best_edges
