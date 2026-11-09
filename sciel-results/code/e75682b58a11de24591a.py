import random
import time

# EVOLVE-BLOCK-START
def _is_k33_free(adj):
    """Check no two vertices have >=3 common neighbors."""
    n = len(adj)
    for u in range(n):
        for v in range(u + 1, n):
            if (adj[u] & adj[v]).bit_count() >= 3:
                return False
    return True

def _try_add_edge(adj, u, v):
    """Try adding edge (u,v). Return True if graph stays K3,3-free."""
    # Adding edge (u,v) increases common neighbors of (u,x) by 1 for x in N(v),
    # and of (v,x) by 1 for x in N(u).
    for x in range(20):
        if x == u or x == v:
            continue
        # Check pair (u, x): common neighbors = N(u) & N(x) plus v if v in N(x)
        if adj[x] & (1 << v):
            if (adj[u] & adj[x]).bit_count() >= 3:
                return False
        # Check pair (v, x): common neighbors = N(v) & N(x) plus u if u in N(x)
        if adj[x] & (1 << u):
            if (adj[v] & adj[x]).bit_count() >= 3:
                return False
    return True

def _greedy_fill(adj, order):
    """Try to add edges in given order, keeping K3,3-free."""
    for u, v in order:
        if not (adj[u] >> v) & 1:
            if _try_add_edge(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
    return adj

def _local_search(adj, max_iters=200000):
    """Try to improve by swapping edges."""
    n = 20
    improved = True
    iters = 0
    while improved and iters < max_iters:
        improved = False
        iters += 1
        # Try removing an edge and adding a different one
        for u in range(n):
            for v in range(u + 1, n):
                if (adj[u] >> v) & 1:
                    # Try removing (u,v)
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                    # Try adding a random non-edge
                    added = False
                    for _ in range(5):
                        a = random.randint(0, n - 1)
                        b = random.randint(0, n - 1)
                        if a == b or (adj[a] >> b) & 1:
                            continue
                        if _try_add_edge(adj, a, b):
                            adj[a] |= (1 << b)
                            adj[b] |= (1 << a)
                            added = True
                            improved = True
                            break
                    if not added:
                        # Restore
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
    return adj

def construct():
    start = time.time()
    deadline = start + 110.0
    n = 20
    
    best_adj = None
    best_count = -1
    
    # Base: complete bipartite K10,10 (100 edges, K3,3-free)
    base_adj = [0] * n
    for i in range(10):
        for j in range(10, 20):
            base_adj[i] |= (1 << j)
            base_adj[j] |= (1 << i)
    base_count = 100
    
    # Try multiple random orderings for greedy fill
    rng = random.Random(42)
    
    # Generate all non-edges in the base
    non_edges = []
    for i in range(10):
        for j in range(i + 1, 10):
            non_edges.append((i, j))
    for i in range(10, 20):
        for j in range(i + 1, 20):
            non_edges.append((i, j))
    
    # Try several random orderings
    for trial in range(30):
        if time.time() > deadline:
            break
        order = non_edges[:]
        rng.shuffle(order)
        
        adj = base_adj[:]
        adj = _greedy_fill(adj, order)
        
        count = sum(adj[u].bit_count() for u in range(n)) // 2
        if count > best_count:
            best_count = count
            best_adj = adj[:]
        
        # Local search
        if time.time() < deadline - 20:
            adj = _local_search(adj, max_iters=50000)
            count = sum(adj[u].bit_count() for u in range(n)) // 2
            if count > best_count:
                best_count = count
                best_adj = adj[:]
    
    # Convert to edge list
    if best_adj is None:
        return [[i, j] for i in range(10) for j in range(10, 20)]
    
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if (best_adj[u] >> v) & 1:
                edges.append([u, v])
    
    return edges
# EVOLVE-BLOCK-END
