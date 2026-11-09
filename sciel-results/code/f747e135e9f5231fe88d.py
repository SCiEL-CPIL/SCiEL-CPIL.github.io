import random
import time

N = 20
C = 20
DEADLINE = time.time() + 110.0  # Safety margin: start + 110s, well under 150s


def _edge_ok(adj, colour_of, u, v, c):
    """Check whether adding edge (u,v) with colour c keeps every 4-cycle rainbow."""
    # A 4-cycle containing (u,v) is u-v-x-y-u
    # x is neighbor of v, y is neighbor of u
    # (x,y) must be an edge
    # Colors: c_uv=c, c_vx, c_xy, c_yu must be distinct
    
    adj_v = adj[v]
    adj_u = adj[u]
    
    # To speed up, iterate over the smaller adjacency list?
    # The structure is symmetric in the loop bounds but the check is specific.
    # x in adj[v], y in adj[u].
    
    for x in adj_v:
        if x == u:
            continue
        cvx = colour_of[v][x]
        if cvx == c:
            # If cvx == c, then any y that forms a cycle with x will fail because c==cvx
            # But we still need to check if such y exists.
            # If no y exists, it's fine.
            pass
        
        for y in adj_u:
            if y == v or y == x:
                continue
            if x in adj[y]:
                # Cycle u-v-x-y-u exists
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


def _try_build(seed, edge_order, rng):
    adj = [set() for _ in range(N)]
    colour_of = [[-1] * N for _ in range(N)]
    placed = []
    
    # Shuffle edge order using the provided rng
    order = list(edge_order)
    rng.shuffle(order)
    
    for (u, v) in order:
        if time.time() > DEADLINE - 5.0:
            break # Abort build if close to deadline
        
        if u in adj[v]:
            continue
            
        # Try colors in a random order
        # Generating a new shuffle for 20 items is fast, but let's use rng
        cols = list(range(C))
        rng.shuffle(cols)
        
        added = False
        for c in cols:
            if _edge_ok(adj, colour_of, u, v, c):
                adj[u].add(v)
                adj[v].add(u)
                colour_of[u][v] = c
                colour_of[v][u] = c
                placed.append([u, v, c])
                added = True
                break
        # If no color worked, we skip this edge
    return placed


def construct():
    all_edges = [(u, v) for u in range(N) for v in range(u + 1, N)]
    best = []
    rng = random.Random(12345)
    
    # We will run multiple seeds
    # Estimate time per build: ~0.01 - 0.05s
    # We have ~100s available.
    
    while time.time() < DEADLINE - 10.0:
        seed = rng.randint(0, 10**9)
        # Reuse a single RNG instance for speed, but re-seed or just keep using it?
        # Using the same rng instance continuously is fine for randomization.
        cand = _try_build(seed, all_edges, rng)
        
        if len(cand) > len(best):
            best = cand
            
    return best
