import random
import time

N = 20
C = 20

def _edge_ok(adj, colour_of, u, v, c):
    """Check whether adding edge (u,v) with colour c keeps every 4-cycle rainbow."""
    adj_v = adj[v]
    adj_u = adj[u]
    
    for x in adj_v:
        if x == u:
            continue
        cvx = colour_of[v][x]
        
        # Potential y's are neighbors of both u and x
        common = adj_u & adj[x]
        
        if cvx == c:
            # If cvx == c, we cannot have any 4-cycle u-v-x-y-u.
            # This requires that there is NO valid y.
            # Valid y must be in common, y != v, y != x.
            # If there is any such y, return False.
            has_valid_y = False
            for y in common:
                if y != v and y != x:
                    has_valid_y = True
                    break
            if has_valid_y:
                return False
            continue

        # If cvx != c, check all valid y's for colour conflicts
        for y in common:
            if y == v or y == x:
                continue
            cxy = colour_of[x][y]
            cyu = colour_of[y][u]
            
            # Check distinctness: c, cvx, cxy, cyu
            if c == cvx or c == cxy or c == cyu:
                return False
            if cvx == cxy or cvx == cyu:
                return False
            if cxy == cyu:
                return False
    return True

def _greedy_construct(seed):
    """Run a single greedy construction pass."""
    adj = [set() for _ in range(N)]
    colour_of = [[-1] * N for _ in range(N)]
    placed = []
    
    # Generate all potential edges
    potential_edges = [(i, j) for i in range(N) for j in range(i + 1, N)]
    rng = random.Random(seed)
    rng.shuffle(potential_edges)
    
    for (u, v) in potential_edges:
        cols = list(range(C))
        rng.shuffle(cols)
        
        for c in cols:
            if _edge_ok(adj, colour_of, u, v, c):
                adj[u].add(v)
                adj[v].add(u)
                colour_of[u][v] = c
                colour_of[v][u] = c
                placed.append([u, v, c])
                break
    return placed

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin under 120s limit
    
    best_edges = []
    max_edges = 0
    
    # Try multiple random restarts to find a denser graph
    # Each restart is very fast, so we can do many
    seed = 42
    while time.time() < deadline:
        edges = _greedy_construct(seed)
        if len(edges) > max_edges:
            max_edges = len(edges)
            best_edges = edges
        seed += 1
        
    return best_edges
