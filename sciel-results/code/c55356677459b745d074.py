import random
import time

def construct():
    n = 20
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Adjacency matrix for fast access
    # G[i][j] is 1 if edge exists
    
    def is_valid(G):
        for u in range(n):
            for v in range(u+1, n):
                for w in range(v+1, n):
                    common = 0
                    for x in range(n):
                        if G[u][x] and G[v][x] and G[w][x]:
                            common += 1
                            if common > 2:
                                return False
        return True

    def get_violation(G):
        for u in range(n):
            for v in range(u+1, n):
                for w in range(v+1, n):
                    common = []
                    for x in range(n):
                        if G[u][x] and G[v][x] and G[w][x]:
                            common.append(x)
                            if len(common) > 2:
                                return (u, v, w, common)
        return None

    def can_add(G, u, v):
        if G[u][v]:
            return True
        # Check triples involving u
        # Adding (u,v) means v is added to N(u).
        # For any x, y, if v is in N(x) and N(y), then v is added to N(u) intersect N(x) intersect N(y).
        # We need to ensure that for no x, y is the count already >= 2.
        # i.e., if v in N(x) and v in N(y), we need |N(u) intersect N(x) intersect N(y)| < 2.
        # Wait, if |...| == 2, adding v makes it 3. So we need |...| < 2, i.e., <= 1.
        
        # Actually, simpler: just check all triples involving u or v after hypothetical addition?
        # That's O(n^2) checks, each O(n) -> O(n^3) per edge. Too slow for many edges?
        # n=20. 20^3 = 8000. 190 edges. 1.5M ops. Fast enough.
        
        # Let's use the optimized check.
        # Triples containing u: {u, x, y}
        # Common neighbors of {u, x, y} is N(u) intersect N(x) intersect N(y).
        # New N(u) = N(u) union {v}.
        # So new common neighbors = (N(u) intersect N(x) intersect N(y)) union ({v} intersect N(x) intersect N(y)).
        # The size increases by 1 if v is in N(x) intersect N(y).
        # So if v in N(x) and v in N(y), we must have |N(u) intersect N(x) intersect N(y)| <= 1.
        
        for x in range(n):
            if x == u or x == v: continue
            if not G[v][x]: continue # v not in N(x)
            for y in range(n):
                if y == u or y == v or y == x: continue
                if not G[v][y]: continue # v not in N(y)
                # v is in N(x) and N(y).
                # Count common neighbors of u, x, y in current G.
                cnt = 0
                for z in range(n):
                    if G[u][z] and G[x][z] and G[y][z]:
                        cnt += 1
                        if cnt > 1:
                            return False
        # Triples containing v: {v, x, y}
        # Similar logic, symmetric.
        for x in range(n):
            if x == u or x == v: continue
            if not G[u][x]: continue # u not in N(x)
            for y in range(n):
                if y == u or y == v or y == x: continue
                if not G[u][y]: continue # u not in N(y)
                cnt = 0
                for z in range(n):
                    if G[v][z] and G[x][z] and G[y][z]:
                        cnt += 1
                        if cnt > 1:
                            return False
                            
        return True

    def count_edges(G):
        return sum(sum(row) for row in G) // 2

    def random_graph(p):
        G = [[0]*n for _ in range(n)]
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < p:
                    G[i][j] = 1
                    G[j][i] = 1
        return G

    best_G = None
    best_count = 0

    # Try a few different starting densities
    for p in [0.4, 0.5, 0.6]:
        G = random_graph(p)
        
        # Phase 1: Remove violations
        while time.time() < deadline:
            viol = get_violation(G)
            if viol is None:
                break
            u, v, w, common = viol
            # Remove a random edge from the violation
            # The violation is caused by common neighbors of u,v,w.
            # We need to break the connection.
            # Any edge (u, c) or (v, c) or (w, c) for c in common will help.
            # Actually, removing (u, c) removes c from N(u).
            # So c is no longer a common neighbor of u,v,w.
            c = random.choice(common)
            # Choose which endpoint to disconnect
            # Randomly pick u, v, or w
            center = random.choice([u, v, w])
            G[center][c] = 0
            G[c][center] = 0
            
        if time.time() > deadline:
            break

        # Phase 2: Add edges
        changed = True
        while changed and time.time() < deadline:
            changed = False
            # Shuffle non-edges to avoid bias
            non_edges = [(i, j) for i in range(n) for j in range(i+1, n) if not G[i][j]]
            random.shuffle(non_edges)
            for u, v in non_edges:
                if can_add(G, u, v):
                    G[u][v] = 1
                    G[v][u] = 1
                    changed = True
        
        # Phase 3: Local Search (Simulated Annealing style or just random kick)
        # Try removing one random edge and then re-adding greedily
        for _ in range(50): # Max 50 kicks per start
            if time.time() > deadline:
                break
            # Pick a random edge to remove
            edges = [(i, j) for i in range(n) for j in range(i+1, n) if G[i][j]]
            if not edges:
                break
            u, v = random.choice(edges)
            G[u][v] = 0
            G[v][u] = 0
            
            # Re-add greedily
            changed = True
            while changed and time.time() < deadline:
                changed = False
                non_edges = [(i, j) for i in range(n) for j in range(i+1, n) if not G[i][j]]
                random.shuffle(non_edges)
                for a, b in non_edges:
                    if can_add(G, a, b):
                        G[a][b] = 1
                        G[b][a] = 1
                        changed = True

        curr_count = count_edges(G)
        if curr_count > best_count:
            best_count = curr_count
            best_G = [row[:] for row in G]

    # Convert best_G to list of edges
    if best_G is None:
        # Fallback to seed
        return [[a, b] for a in range(2) for b in range(2, 20)]
        
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if best_G[i][j]:
                edges.append([i, j])
    return edges
