import time
import random

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin for 150s kill
    
    best_result = None
    best_n = 0
    
    # Try n=47 first, then 46, then 45
    for n in [47, 46, 45]:
        min_deg_req = max(1, n - 39)
        
        time_left = deadline - (time.time() - start_time)
        if time_left < 5:
            break
        
        # Allocate time based on n
        if n == 47:
            t_limit = min(70.0, time_left - 2)
        elif n == 46:
            t_limit = min(50.0, time_left - 2)
        else:
            t_limit = min(25.0, time_left - 2)
            
        if t_limit <= 0:
            continue
        
        result = _solve_c4_free(n, min_deg_req, t_limit)
        if result:
            if n > best_n:
                best_n = n
                best_result = result
            return result
        
        # If we found nothing for this n, continue to smaller n
        if best_result and best_n >= n:
            break
    
    # Fallback: return best found or a trivial valid graph
    if best_result:
        return best_result
    
    # Trivial valid graph: empty graph on n=39 (min_deg = 0, but we need >= 1)
    # Actually n=40, min_deg=1: just add one edge
    return {'n': 40, 'edges': [[0, 1]]}


def _creates_c4(adj, u, v):
    """Check if adding edge (u,v) to graph adj creates a C4.
    A C4 exists if there is x in N(u) with a neighbor y in N(v), x != v, y != u.
    Since u,v are not yet adjacent, v not in N(u), u not in N(v).
    So we check: exists x in N(u) such that adj[x] & adj[v] != 0.
    """
    nu = adj[u]
    nv = adj[v]
    for x in range(len(adj)):
        if (nu >> x) & 1:
            if adj[x] & nv:
                return True
    return False


def _find_c4(adj, n):
    """Find a C4 in the graph. Returns (u, v, x, y) or None.
    C4: u-x-y-v-u where all edges exist.
    For each edge (u,v), check if u and v have >= 2 common neighbors.
    """
    for u in range(n):
        for v in range(u + 1, n):
            if adj[u] & (1 << v):
                common = adj[u] & adj[v]
                # Need at least 2 common neighbors
                if common.bit_count() >= 2:
                    # Find two distinct common neighbors
                    bits = common
                    first = (bits & -bits).bit_length() - 1
                    bits &= bits - 1
                    if bits:
                        second = (bits & -bits).bit_length() - 1
                        return (u, v, first, second)
    return None


def _min_degree(adj):
    return min(a.bit_count() for a in adj)


def _greedy_build(n, min_deg, adj=None, time_limit=None, start_time=None):
    """Greedy construction: add edges one by one, always maintaining C4-freeness.
    Returns adj list if min degree requirement is met, else None.
    """
    if adj is None:
        adj = [0] * n
    
    if time_limit is None or start_time is None:
        start_time = time.time()
        time_limit = 10.0
    
    # Try to add edges to increase minimum degree
    # Strategy: repeatedly find vertex with min degree, try to add an edge to it
    max_iters = 50000
    for it in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        
        # Find vertex with minimum degree
        min_v = min(range(n), key=lambda i: adj[i].bit_count())
        min_d = adj[min_v].bit_count()
        
        if min_d >= min_deg:
            # Check if entire graph is C4-free
            c4 = _find_c4(adj, n)
            if c4 is None:
                return adj
        
        if min_d >= min_deg:
            # All degrees meet requirement, but there might be C4s
            # Try to remove edges to break C4s
            c4 = _find_c4(adj, n)
            if c4 is not None:
                u, v, x, y = c4
                # Remove edge (x, y) or (u, x) to break the C4
                # Remove (u, x)
                adj[u] &= ~(1 << x)
                adj[x] &= ~(1 << u)
                continue
            else:
                return adj
        
        # Try to add an edge to min_v
        candidates = []
        for v in range(n):
            if v != min_v and not (adj[min_v] & (1 << v)):
                candidates.append(v)
        
        random.shuffle(candidates)
        added = False
        for v in candidates:
            if not _creates_c4(adj, min_v, v):
                adj[min_v] |= (1 << v)
                adj[v] |= (1 << min_v)
                added = True
                break
        
        if not added:
            # Stuck: try removing a random edge to open up possibilities
            # Find a random edge to remove
            for _ in range(100):
                u = random.randint(0, n - 1)
                v = random.randint(0, n - 1)
                if u != v and (adj[u] & (1 << v)):
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                    break
    
    # Final check
    if _min_degree(adj) >= min_deg:
        c4 = _find_c4(adj, n)
        if c4 is None:
            return adj
    
    return None


def _polarity_seed(n, min_deg, time_limit, start_time):
    """Use polarity graph structure as seed, then greedy repair.
    Polarity graphs are known to be C4-free with good degree properties.
    """
    # Create a polarity graph: partition vertices into pairs, connect within pair
    # and across pairs based on a rule.
    # For n up to 47, we can use a simple construction.
    
    adj = [0] * n
    
    # Partition into blocks of size 2 (or 1 if odd)
    blocks = []
    i = 0
    while i < n:
        block = list(range(i, min(i + 2, n)))
        blocks.append(block)
        i += 2
    
    # For each pair of blocks, decide if they're connected based on index
    for bi in range(len(blocks)):
        for bj in range(bi + 1, len(blocks)):
            # Simple rule: connect if (bi + bj) is even
            if (bi + bj) % 2 == 0:
                for u in blocks[bi]:
                    for v in blocks[bj]:
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
    
    # Now greedily try to improve minimum degree
    result = _greedy_build(n, min_deg, adj, time_limit, start_time)
    return result


def _random_seed(n, min_deg, time_limit, start_time):
    """Random initialization with greedy completion."""
    adj = [0] * n
    
    # Random initialization with moderate density
    p = min_deg / (n - 1) * 0.8
    for u in range(n):
        for v in range(u + 1, n):
            if random.random() < p:
                # Only add if it doesn't create a C4
                if not _creates_c4(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
    
    # Greedy completion
    result = _greedy_build(n, min_deg, adj, time_limit, start_time)
    return result


def _solve_c4_free(n, min_deg, time_limit):
    """Try multiple strategies to find a C4-free graph with min degree >= min_deg."""
    start_time = time.time()
    
    strategies = [
        _greedy_build_from_empty,
        _polarity_seed,
        _random_seed,
        _greedy_build_from_empty,  # Another attempt with different random seed
        _random_seed,
        _polarity_seed,
    ]
    
    for strategy in strategies:
        time_left = time_limit - (time.time() - start_time)
        if time_left < 2:
            break
        
        # Allocate time per strategy
        t_for_strategy = min(20.0, time_left - 1)
        if t_for_strategy <= 0:
            break
        
        adj = strategy(n, min_deg, t_for_strategy, time.time())
        
        if adj is not None:
            # Verify
            if _min_degree(adj) >= min_deg and _find_c4(adj, n) is None:
                edges = []
                for u in range(n):
                    for v in range(u + 1, n):
                        if adj[u] & (1 << v):
                            edges.append([u, v])
                return {'n': n, 'edges': edges}
    
    # Final attempt: pure greedy from empty with more time
    time_left = time_limit - (time.time() - start_time)
    if time_left > 5:
        adj = _greedy_build_from_empty(n, min_deg, time_left - 1, time.time())
        if adj is not None:
            if _min_degree(adj) >= min_deg and _find_c4(adj, n) is None:
                edges = []
                for u in range(n):
                    for v in range(u + 1, n):
                        if adj[u] & (1 << v):
                            edges.append([u, v])
                return {'n': n, 'edges': edges}
    
    return None


def _greedy_build_from_empty(n, min_deg, time_limit, start_time):
    """Pure greedy from empty graph."""
    return _greedy_build(n, min_deg, adj=[0] * n, time_limit=time_limit, start_time=start_time)


def _verify_and_return(n, adj, min_deg):
    """Verify adj is C4-free and has min degree >= min_deg, return result dict or None."""
    if _min_degree(adj) >= min_deg and _find_c4(adj, n) is None:
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u] & (1 << v):
                    edges.append([u, v])
        return {'n': n, 'edges': edges}
    return None

