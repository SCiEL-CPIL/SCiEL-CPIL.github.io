import time
import random
import math

N = 20
DEADLINE_MARGIN = 5.0

def _bit_count(x):
    return x.bit_count()

def _can_add(adj, u, v):
    """
    Check if edge (u, v) can be added to adj without creating K2,3.
    adj is a list of ints, where adj[i] is a bitmask of neighbors.
    """
    # 1. Direct common neighbors of u and v must be <= 2
    if _bit_count(adj[u] & adj[v]) > 2:
        return False
    
    # 2. For any w in N(v), w != u:
    #    Adding (u,v) adds v to N(u).
    #    If v is in N(w), then v becomes a common neighbor of u and w.
    #    So we need current common neighbors of u and w to be <= 1.
    mask_v = adj[v]
    for w in range(N):
        if w == u or w == v:
            continue
        if (mask_v >> w) & 1:
            if _bit_count(adj[u] & adj[w]) > 1:
                return False
                
    # 3. For any w in N(u), w != v:
    #    Adding (u,v) adds u to N(v).
    #    If u is in N(w), then u becomes a common neighbor of v and w.
    #    So we need current common neighbors of v and w to be <= 1.
    mask_u = adj[u]
    for w in range(N):
        if w == u or w == v:
            continue
        if (mask_u >> w) & 1:
            if _bit_count(adj[v] & adj[w]) > 1:
                return False
                
    return True

def _add_edge(adj, u, v):
    adj[u] |= (1 << v)
    adj[v] |= (1 << u)

def _remove_edge(adj, u, v):
    adj[u] &= ~(1 << v)
    adj[v] &= ~(1 << u)

def _count_edges(adj):
    # Sum bits in all masks, divide by 2
    total = 0
    for i in range(N):
        total += _bit_count(adj[i])
    return total // 2

def _greedy_init(seed):
    rng = random.Random(seed)
    adj = [0] * N
    all_edges = [(u, v) for u in range(N) for v in range(u + 1, N)]
    rng.shuffle(all_edges)
    
    for u, v in all_edges:
        if _can_add(adj, u, v):
            _add_edge(adj, u, v)
            
    return adj, _count_edges(adj)

def _polish(adj, edges):
    """Try to add any remaining valid edges greedily."""
    improved = True
    while improved:
        improved = False
        for u in range(N):
            for v in range(u + 1, N):
                if not ((adj[u] >> v) & 1):
                    if _can_add(adj, u, v):
                        _add_edge(adj, u, v)
                        edges += 1
                        improved = True
    return adj, edges

def _local_search(adj, edges, rng, time_limit, deadline):
    """
    Local search: try to remove one edge and add two edges, or swap edges.
    """
    best_adj = adj[:]
    best_edges = edges
    start_time = time.time()
    
    all_edges = [(u, v) for u in range(N) for v in range(u + 1, N)]
    
    while time.time() < deadline - 0.1:
        # Try a swap: remove edge (a,b), try to add (c,d)
        a, b = rng.choice(all_edges)
        if not ((adj[a] >> b) & 1):
            continue
        
        _remove_edge(adj, a, b)
        edges -= 1
        
        # Try to add a random missing edge
        c, d = rng.choice(all_edges)
        if (c == a and d == b) or (c == b and d == a):
            c, d = rng.choice(all_edges)
        
        if not ((adj[c] >> d) & 1) and _can_add(adj, c, d):
            _add_edge(adj, c, d)
            edges += 1
            
            # Now try to add another edge to compensate
            e, f = rng.choice(all_edges)
            if (e == a and f == b) or (e == b and f == a) or (e == c and f == d) or (e == d and f == c):
                e, f = rng.choice(all_edges)
            
            if not ((adj[e] >> f) & 1) and _can_add(adj, e, f):
                _add_edge(adj, e, f)
                edges += 1
                
                if edges > best_edges:
                    best_edges = edges
                    best_adj = adj[:]
            else:
                # Revert the second add
                _remove_edge(adj, e, f) if ((adj[e] >> f) & 1) else None
                # Keep the first add if it improved
                if edges > best_edges:
                    best_edges = edges
                    best_adj = adj[:]
        else:
            # Revert
            _add_edge(adj, a, b)
            edges += 1
            continue
        
        # Polish
        adj, edges = _polish(adj, edges)
        if edges > best_edges:
            best_edges = edges
            best_adj = adj[:]
    
    return best_adj, best_edges

def _simulated_annealing(adj_init, edges_init, seed, time_limit):
    rng = random.Random(seed)
    adj = adj_init[:]
    edges = edges_init
    best_adj = adj[:]
    best_edges = edges
    
    # Precompute all possible edges
    all_edges = [(u, v) for u in range(N) for v in range(u + 1, N)]
    
    # Initial temperature
    T = 2.0
    T_min = 0.1
    cooling_rate = 0.9995
    
    start_time = time.time()
    deadline = start_time + time_limit
    
    while time.time() < deadline:
        # Choose a move
        r = rng.random()
        
        if r < 0.5:
            # Try to add a random missing edge
            attempts = 0
            added = False
            while attempts < 30:
                u = rng.randint(0, N-1)
                v = rng.randint(0, N-1)
                if u == v:
                    attempts += 1
                    continue
                if u > v:
                    u, v = v, u
                # Check if missing
                if not ((adj[u] >> v) & 1):
                    if _can_add(adj, u, v):
                        _add_edge(adj, u, v)
                        edges += 1
                        added = True
                        if edges > best_edges:
                            best_edges = edges
                            best_adj = adj[:]
                        break
                attempts += 1
                
        elif r < 0.8:
            # Try to remove a random existing edge
            attempts = 0
            removed = False
            while attempts < 30:
                u = rng.randint(0, N-1)
                v = rng.randint(0, N-1)
                if u == v:
                    attempts += 1
                    continue
                if u > v:
                    u, v = v, u
                if (adj[u] >> v) & 1:
                    # Delta E = -1 (worse)
                    # Accept with prob exp(-1/T)
                    if rng.random() < math.exp(-1.0 / T):
                        _remove_edge(adj, u, v)
                        edges -= 1
                        removed = True
                        break
                attempts += 1
        else:
            # Swap: Remove one, Add one
            # Pick random existing edge to remove
            attempts_rm = 0
            u_rm, v_rm = -1, -1
            while attempts_rm < 30:
                u = rng.randint(0, N-1)
                v = rng.randint(0, N-1)
                if u == v:
                    attempts_rm += 1
                    continue
                if u > v:
                    u, v = v, u
                if (adj[u] >> v) & 1:
                    u_rm, v_rm = u, v
                    break
                attempts_rm += 1
            
            if u_rm != -1:
                # Remove it temporarily
                _remove_edge(adj, u_rm, v_rm)
                edges -= 1
                
                # Try to add a random missing edge
                attempts_add = 0
                added = False
                while attempts_add < 30:
                    u = rng.randint(0, N-1)
                    v = rng.randint(0, N-1)
                    if u == v or (u == u_rm and v == v_rm) or (u == v_rm and v == u_rm):
                        attempts_add += 1
                        continue
                    if u > v:
                        u, v = v, u
                    if not ((adj[u] >> v) & 1):
                        if _can_add(adj, u, v):
                            _add_edge(adj, u, v)
                            edges += 1
                            added = True
                            if edges > best_edges:
                                best_edges = edges
                                best_adj = adj[:]
                            break
                    attempts_add += 1
                
                if not added:
                    # Revert removal
                    _add_edge(adj, u_rm, v_rm)
                    edges += 1

        # Cool down
        T *= cooling_rate
        if T < T_min:
            T = T_min
    
    # Final polish
    adj, edges = _polish(adj, edges)
    if edges > best_edges:
        best_edges = edges
        best_adj = adj[:]
    
    return best_adj, best_edges

def construct():
    global deadline
    deadline = time.time() + 140.0
    
    best_adj = None
    best_edges = 0
    
    # Run multiple SA restarts with different seeds
    seeds = [42, 123, 456, 789, 101, 202, 303, 404, 505, 606, 707, 808, 909, 1010, 1111, 1212, 1313, 1414, 1515, 1616]
    
    for seed in seeds:
        if time.time() > deadline - DEADLINE_MARGIN:
            break
            
        # Greedy init
        adj, edges = _greedy_init(seed)
        
        # SA with local search
        time_per_restart = max(1.0, (deadline - time.time() - DEADLINE_MARGIN) / max(1, len(seeds) - seeds.index(seed)))
        adj_sa, edges_sa = _simulated_annealing(adj, edges, seed, time_per_restart)
        
        # Local search polish
        rng = random.Random(seed + 10000)
        adj_ls, edges_ls = _local_search(adj_sa, edges_sa, rng, time_per_restart / 2, deadline)
        
        if edges_ls > best_edges:
            best_edges = edges_ls
            best_adj = adj_ls
            
    if best_adj is None:
        # Fallback
        adj = [0] * N
        best_adj = adj
        best_edges = 0
    
    # Final polish
    best_adj, best_edges = _polish(best_adj, best_edges)
        
    # Convert to edge list
    edges = []
    for u in range(N):
        for v in range(u + 1, N):
            if (best_adj[u] >> v) & 1:
                edges.append([u, v])
                
    return edges
