import time
import random
import itertools

N = 16
ALL_PAIRS = list(itertools.combinations(range(N), 2))

def _creates_cube(u, v, adj):
    """Return True if adding edge (u,v) to adjacency `adj` (list of int bitsets)
    creates a 3-cube (Q3) as a subgraph.
    """
    Nu = adj[u] & ~(1 << v)
    Nv = adj[v] & ~(1 << u)
    
    # Get lists of neighbors
    L_u = [i for i in range(N) if (Nu >> i) & 1]
    L_v = [i for i in range(N) if (Nv >> i) & 1]
    
    if len(L_u) < 2 or len(L_v) < 2:
        return False
        
    for i in range(len(L_u)):
        x = L_u[i]
        for j in range(i+1, len(L_u)):
            w = L_u[j]
            # We need y in N(x) \cap N(v) and z in N(w) \cap N(v)
            # such that y != z and y,z distinct from x,w
            mask_y = adj[x] & Nv & ~(1 << x) & ~(1 << w)
            mask_z = adj[w] & Nv & ~(1 << x) & ~(1 << w)
            
            if not mask_y or not mask_z:
                continue
                
            # Check if there exist y in mask_y and z in mask_z with y != z
            # This fails only if mask_y == mask_z and it is a single bit
            if mask_y == mask_z and (mask_y & (mask_y - 1)) == 0:
                continue
                
            return True
    return False


def construct():
    start = time.time()
    deadline = start + 110.0  # safety margin before 120s limit

    best_edges = None
    best_count = -1

    def time_left():
        return deadline - time.time()

    # Greedy with randomized edge order, many trials.
    rng = random.Random(12345)

    trial = 0
    while time_left() > 5.0:
        trial += 1
        # Random edge ordering (shuffle a copy)
        order = ALL_PAIRS[:]
        rng.shuffle(order)

        adj = [0] * N
        edges = []
        count = 0
        
        for e in order:
            if time_left() < 1.0:
                break
            u, v = e
            if (adj[u] >> v) & 1:
                continue
            if not _creates_cube(u, v, adj):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                edges.append([u, v])
                count += 1

        if count > best_count:
            best_count = count
            best_edges = edges

    if best_edges is None:
        # fallback: K3,13
        best_edges = [[a, b] for a in range(3) for b in range(3, 16)]
    return best_edges
