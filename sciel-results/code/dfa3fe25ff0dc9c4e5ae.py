import random
import time

N = 20
C = 20

def _edge_ok(adj_mask, colour_of, u, v, c):
    """Check whether adding edge (u,v) with colour c keeps every 4-cycle rainbow."""
    # adj_mask[i] is a bitmask of neighbors of i
    # colour_of[i][j] is the color of edge (i,j), or -1 if not present
    
    nu = adj_mask[u]
    nv = adj_mask[v]
    
    # Iterate over neighbors x of v
    # We need to check 4-cycles u-v-x-y-u
    # x must be in N(v)
    # y must be in N(u) and N(x)
    
    # Mask for neighbors of v, excluding u
    # Actually u is not in N(v) yet, so nv is fine. But we should be careful if u was already a neighbor? 
    # The function is called before adding, so u is not in nv.
    
    # Iterate x in nv
    x_iter = nv
    while x_iter:
        lsb = x_iter & (-x_iter)
        x = lsb.bit_length() - 1
        x_iter -= lsb
        
        cvx = colour_of[v][x]
        
        if cvx == c:
            # If cvx == c, any 4-cycle u-v-x-y-u would have c appearing twice (on uv and vx)
            # So we must ensure no such y exists.
            # y must be in N(u) and N(x).
            # Check if N(u) & N(x) is non-empty.
            if (adj_mask[u] & adj_mask[x]):
                return False
            continue
        
        # For each y in N(u) & N(x), check colors
        # y cannot be v (v not in N(u) usually? No, v is not in N(u) before adding)
        # y cannot be x (x not in N(u) usually? x is in N(v). Is x in N(u)? 
        # If x is in N(u), then u-x is an edge. 
        # The cycle is u-v-x-y-u. 
        # Vertices are u, v, x, y. They must be distinct.
        # u != v (distinct)
        # v != x (x in N(v), so x != v)
        # x != y (y in N(x), so y != x)
        # y != u (y in N(u), so y != u)
        # u != x? If u and x are adjacent, is it a 4-cycle? 
        # A 4-cycle subgraph has 4 distinct vertices. 
        # If u and x are adjacent, the set of vertices is {u, v, x, y}. 
        # Edges are (u,v), (v,x), (x,y), (y,u). 
        # If (u,x) exists, it's a chord, but the cycle itself is still defined by the 4 vertices. 
        # The problem states "Every 4-cycle (as a subgraph)". 
        # Usually this means the induced subgraph on 4 vertices contains a cycle of length 4? 
        # Or any simple cycle of length 4 in the graph. 
        # "checked over all 4-cycles". 
        # Standard definition: A cycle is a sequence of distinct vertices v1...vk v1. 
        # So u, v, x, y must be distinct. 
        # If u and x are adjacent, they are still distinct vertices. 
        # So we just need to ensure u, v, x, y are distinct. 
        # u != v, v != x, x != y, y != u are guaranteed by definition of neighbors. 
        # What about u != x? 
        # If u == x, then v and u are neighbors, and x is u. 
        # But x is a neighbor of v. So u is a neighbor of v. 
        # But we are adding (u,v). So u is not currently a neighbor of v. 
        # So u != x is guaranteed because x is in N(v) and u is not in N(v).
        
        # So we just iterate y in N(u) & N(x).
        y_iter = adj_mask[u] & adj_mask[x]
        while y_iter:
            lsb_y = y_iter & (-y_iter)
            y = lsb_y.bit_length() - 1
            y_iter -= lsb_y
            
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

def _greedy_construct(seed, deadline):
    """Run a single greedy construction pass with random edge and color ordering."""
    adj_mask = [0] * N
    colour_of = [[-1] * N for _ in range(N)]
    placed = []
    
    potential_edges = [(i, j) for i in range(N) for j in range(i + 1, N)]
    rng = random.Random(seed)
    rng.shuffle(potential_edges)
    
    for (u, v) in potential_edges:
        if time.time() > deadline:
            break
        
        cols = list(range(C))
        rng.shuffle(cols)
        
        for c in cols:
            if _edge_ok(adj_mask, colour_of, u, v, c):
                # Add edge
                adj_mask[u] |= (1 << v)
                adj_mask[v] |= (1 << u)
                colour_of[u][v] = c
                colour_of[v][u] = c
                placed.append([u, v, c])
                break
    return placed

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin under 120s limit
    
    best_edges = []
    max_edges = 0
    
    seed = 42
    while time.time() < deadline:
        edges = _greedy_construct(seed, deadline)
        if len(edges) > max_edges:
            max_edges = len(edges)
            best_edges = edges
        seed += 1
    
    return best_edges
