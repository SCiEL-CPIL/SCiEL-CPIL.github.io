import time
import random

N = 20
DEADLINE_MARGIN = 5.0
TOTAL_EDGES = N * (N - 1) // 2

# Precompute edge indices
EDGE_LIST = [(u, v) for u in range(N) for v in range(u + 1, N)]
EDGE_INDEX = {(u, v): i for i, (u, v) in enumerate(EDGE_LIST)}
EDGE_INDEX.update({(v, u): i for i, (u, v) in enumerate(EDGE_LIST)})

def _count_edges(adj):
    return sum(adj[u].count(1) for u in range(N)) // 2

def _can_add(adj, u, v):
    """
    Check if edge (u, v) can be added.
    adj is a list of bitmasks.
    """
    # Check if already present
    if adj[u] & (1 << v):
        return True

    # 1. Check C(u, v) <= 2
    # Common neighbors of u and v
    common_uv = adj[u] & adj[v]
    if common_uv.bit_count() > 2:
        return False

    # 2. For all w in N(v), check C(u, w) <= 1
    # C(u, w) = popcount(adj[u] & adj[w])
    # We need C(u, w) + (1 if v~w else 0) <= 2
    # Since w is in N(v), v~w is true, so we need C(u, w) + 1 <= 2 => C(u, w) <= 1
    neighbors_v = adj[v]
    # Iterate through set bits in neighbors_v
    temp = neighbors_v
    while temp:
        lsb = temp & -temp
        w = lsb.bit_length() - 1
        temp -= lsb
        if w == u:
            continue
        common_uw = adj[u] & adj[w]
        if common_uw.bit_count() > 1:
            return False

    # 3. For all w in N(u), check C(v, w) <= 1
    neighbors_u = adj[u]
    temp = neighbors_u
    while temp:
        lsb = temp & -temp
        w = lsb.bit_length() - 1
        temp -= lsb
        if w == v:
            continue
        common_vw = adj[v] & adj[w]
        if common_vw.bit_count() > 1:
            return False

    return True

def _add_edge(adj, u, v):
    adj[u] |= (1 << v)
    adj[v] |= (1 << u)

def _remove_edge(adj, u, v):
    adj[u] &= ~(1 << v)
    adj[v] &= ~(1 << u)

def _greedy_build(seed):
    rng = random.Random(seed)
    adj = [0] * N
    edges_to_try = list(range(TOTAL_EDGES))
    rng.shuffle(edges_to_try)
    
    for idx in edges_to_try:
        u, v = EDGE_LIST[idx]
        if _can_add(adj, u, v):
            _add_edge(adj, u, v)
            
    return adj

def _local_search(adj, seed, time_limit):
    rng = random.Random(seed)
    adj = [b for b in adj]
    current_edges = _count_edges(adj)
    best_edges = current_edges
    best_adj = [b for b in adj]
    
    start_time = time.time()
    
    while time.time() - start_time < time_limit:
        improved = False
        
        # 1. Try to add a random missing edge
        missing = [i for i in range(TOTAL_EDGES) if not (adj[EDGE_LIST[i][0]] & (1 << EDGE_LIST[i][1]))]
        if missing:
            rng.shuffle(missing)
            for idx in missing[:50]: # Try a few random missing edges
                u, v = EDGE_LIST[idx]
                if _can_add(adj, u, v):
                    _add_edge(adj, u, v)
                    current_edges += 1
                    improved = True
                    if current_edges > best_edges:
                        best_edges = current_edges
                        best_adj = [b for b in adj]
                    break
        
        if not improved:
            # 2. Try a swap: remove one edge, then try to add edges
            # This can help escape local optima where no single edge can be added
            present = [i for i in range(TOTAL_EDGES) if (adj[EDGE_LIST[i][0]] & (1 << EDGE_LIST[i][1]))]
            if present:
                rng.shuffle(present)
                for idx_rm in present[:10]:
                    u_rm, v_rm = EDGE_LIST[idx_rm]
                    _remove_edge(adj, u_rm, v_rm)
                    current_edges -= 1
                    
                    added = 0
                    # Try to add edges, prioritizing those incident to u_rm or v_rm
                    # This is a heuristic to find a "good" swap
                    candidates = [i for i in range(TOTAL_EDGES) if not (adj[EDGE_LIST[i][0]] & (1 << EDGE_LIST[i][1]))]
                    rng.shuffle(candidates)
                    
                    for idx_add in candidates:
                        u_add, v_add = EDGE_LIST[idx_add]
                        if _can_add(adj, u_add, v_add):
                            _add_edge(adj, u_add, v_add)
                            current_edges += 1
                            added += 1
                            if current_edges > best_edges:
                                best_edges = current_edges
                                best_adj = [b for b in adj]
                            # If we added at least as many as we removed, we might be better off
                            # But usually we want to maximize edges, so adding 1 is good, adding 2 is great
                            if added >= 1:
                                break
                            
                    if added == 0:
                        # Revert
                        _add_edge(adj, u_rm, v_rm)
                        current_edges += 1
                    elif added == 1:
                        # We removed 1 and added 1, net 0 change. 
                        # This move might open up future possibilities, so we keep it.
                        pass
                    else:
                        # Net gain, great!
                        pass

    return best_adj, best_edges

def construct():
    global deadline
    deadline = time.time() + 150.0
    
    best_adj = None
    best_edges = 0
    
    # Phase 1: Multiple greedy builds
    seeds = [42, 123, 456, 789, 101, 202, 303, 404, 505, 606, 707, 808, 909, 1010, 1111, 2222, 3333, 4444, 5555, 6666]
    
    for seed in seeds:
        if time.time() > deadline - DEADLINE_MARGIN:
            break
        adj = _greedy_build(seed)
        edges = _count_edges(adj)
        if edges > best_edges:
            best_edges = edges
            best_adj = adj
    
    if best_adj is None:
        # Fallback: 5 copies of K4
        adj = [0] * N
        for k in range(5):
            for a in range(4):
                for b in range(a + 1, 4):
                    u = 4 * k + a
                    v = 4 * k + b
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        best_adj = adj
        best_edges = _count_edges(adj)
    
    # Phase 2: Local search on best
    remaining_time = deadline - DEADLINE_MARGIN - time.time()
    if remaining_time > 2.0:
        local_time = remaining_time
        best_adj, best_edges = _local_search(best_adj, 42, local_time)
    
    # Convert to edge list
    edges = []
    for u in range(N):
        for v in range(u + 1, N):
            if best_adj[u] & (1 << v):
                edges.append([u, v])
    
    return edges

deadline = 0
