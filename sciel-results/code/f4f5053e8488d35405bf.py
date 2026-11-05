import time
import random
import sys

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # 10s margin for fallback and safety
    
    # If parameters don't match the specific (4,10) cage problem, 
    # fallback to the seed construction which is generic.
    if k != 4 or g != 10:
        return _seed_construct(k, g)
    
    best_valid = None
    
    # We know the lower bound is 243 and best known is 320.
    # We try to find a valid graph in this range.
    # We iterate N in increasing order to find the smallest one.
    # We allocate time per N.
    
    # If we find a valid graph at N, we return it immediately 
    # because we are checking smaller N first.
    
    for N in range(250, 330, 2):
        if time.time() > deadline - 15:
            break
        
        # Run a short search for this N
        # Limit time per N to ensure we try multiple Ns
        # Total time ~ 110s. Range 250-330 is 40 steps.
        # 110/40 ~ 2.7s per N.
        
        result = _search_for_cage(N, g, deadline - 10)
        if result:
            return result
            
    if best_valid:
        return best_valid
        
    # Fallback to seed
    return _seed_construct(k, g)

def _search_for_cage(N, g, deadline):
    # Try a few restarts or a single SA run
    # Given the short time per N, a single SA run is better.
    
    # Generate initial random 4-regular graph
    edges = _gen_random_4_reg(N)
    if not edges:
        return None
    
    adj = _build_adj(N, edges)
    current_girth = _get_girth(N, adj, g) # Early exit if < g
    
    if current_girth >= g:
        return N, edges
    
    T = 1.0
    cooling = 0.9995
    best_girth = current_girth
    best_edges = edges.copy()
    
    # Precompute edge set for O(1) lookup
    edge_set = set((min(u,v), max(u,v)) for u,v in edges)
    
    while time.time() < deadline:
        # Perform a 2-opt swap
        # Pick two random edges
        # To ensure we don't pick the same edge twice, pick indices
        idx1 = random.randint(0, len(edges)-1)
        idx2 = random.randint(0, len(edges)-1)
        if idx1 == idx2:
            continue
            
        u, v = edges[idx1]
        x, y = edges[idx2]
        
        # Check for loops and multi-edges in the new configuration
        # New edges: (u,x) and (v,y)
        # We remove (u,v) and (x,y)
        
        # Check loops
        if u == x or u == y or v == x or v == y:
            continue
            
        # Check multi-edges
        # (u,x) must not exist (unless it's one of the edges being removed, 
        # but we checked distinct indices, so (u,x) is not (u,v) or (x,y) 
        # unless u=x or u=y etc which we checked)
        # However, (u,x) might be an existing edge elsewhere.
        
        new_e1 = (min(u,x), max(u,x))
        new_e2 = (min(v,y), max(v,y))
        
        if new_e1 in edge_set or new_e2 in edge_set:
            continue
            
        # Also need to ensure (u,x) != (v,y)
        if new_e1 == new_e2:
            continue
            
        # Perform swap
        edges[idx1] = (u, x)
        edges[idx2] = (v, y)
        
        edge_set.remove((min(u,v), max(u,v)))
        edge_set.remove((min(x,y), max(x,y)))
        edge_set.add(new_e1)
        edge_set.add(new_e2)
        
        # Update Adjacency
        # Rebuilding adj is O(N+E). 
        # For N=300, this is fast.
        adj = _build_adj(N, edges)
        
        new_girth = _get_girth(N, adj, g)
        
        # Acceptance criterion
        if new_girth >= best_girth:
            best_girth = new_girth
            best_edges = edges.copy()
            if new_girth >= g:
                return N, edges
        else:
            # Reject and revert
            edges[idx1] = (u, v)
            edges[idx2] = (x, y)
            edge_set.add((min(u,v), max(u,v)))
            edge_set.add((min(x,y), max(x,y)))
            edge_set.remove(new_e1)
            edge_set.remove(new_e2)
            # adj is not used for revert, we just restore edges
            
        T *= cooling
        
    return None

def _gen_random_4_reg(N):
    # Configuration model
    for _ in range(100):
        stubs = [i for i in range(N) for _ in range(4)]
        random.shuffle(stubs)
        edges = set()
        valid = True
        for i in range(0, 2*N, 2):
            u = stubs[i]
            v = stubs[i+1]
            if u == v:
                valid = False
                break
            if (u, v) in edges or (v, u) in edges:
                valid = False
                break
            edges.add((min(u,v), max(u,v)))
        if valid:
            return list(edges)
    return None

def _build_adj(N, edges):
    adj = [[] for _ in range(N)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    return adj

def _get_girth(N, adj, g):
    # Returns girth if >= g, else returns a value < g (e.g. the min girth found)
    # We only care if it is >= g.
    # So we can stop early if we find a cycle < g.
    
    min_girth = float('inf')
    
    for start in range(N):
        dist = [-1] * N
        parent = [-1] * N
        dist[start] = 0
        queue = [start]
        head = 0
        while head < len(queue):
            u = queue[head]
            head += 1
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                elif v != parent[u]:
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < min_girth:
                        min_girth = cycle_len
                        if min_girth < g:
                            return min_girth
    return min_girth

def _seed_construct(k, g):
    # Original seed code
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index, order, edges = {start: 0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges
