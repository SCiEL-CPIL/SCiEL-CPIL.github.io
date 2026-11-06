import time
import random
import math

def _creates_c4_fast(adj_int, n, u, v):
    """Check if adding edge (u,v) creates a C4. Returns True if C4 would form."""
    nu = adj_int[u]
    nv = adj_int[v]
    
    # C4: u-x-v-y-u where x~y, x!=y, x!=v, y!=u
    # Check: exists x in N(u), y in N(v) with x~y, x!=y, x!=v, y!=u
    # Equivalent: for some y in N(v)\{u}, N(y) & N(u) & ~(1<<v) & ~(1<<y) != 0
    
    deg_u = bin(nu).count('1')
    deg_v = bin(nv).count('1')
    
    if deg_u <= deg_v:
        temp = nu
        while temp:
            lsb = temp & -temp
            x = lsb.bit_length() - 1
            if adj_int[x] & nv & ~(1 << u) & ~(1 << x):
                return True
            temp ^= lsb
        return False
    else:
        temp = nv
        while temp:
            lsb = temp & -temp
            y = lsb.bit_length() - 1
            if adj_int[y] & nu & ~(1 << v) & ~(1 << y):
                return True
            temp ^= lsb
        return False

def _verify_c4_free(adj_int, n):
    """Verify graph is C4-free. Returns True if valid."""
    for u in range(n):
        nu = adj_int[u]
        temp = nu
        while temp:
            lsb1 = temp & -temp
            x = lsb1.bit_length() - 1
            temp ^= lsb1
            # For each pair (u, x) in E, check no y in N(x) with y~v for v in N(u)\{x}
            # C4: u-x-y-v-u means x~y and v~u and x~u and v~y
            # For fixed u,x: need y in N(x)\{u} such that N(y) & (N(u)\{x}) != 0
            nx = adj_int[x] & ~(1 << u)
            inner = nx
            while inner:
                lsb2 = inner & -inner
                y = lsb2.bit_length() - 1
                inner ^= lsb2
                if adj_int[y] & (nu & ~(1 << x)):
                    return False
    return True

def _build_random_c4_free(n, seed, target_min_deg):
    """Build a random C4-free graph by greedily adding edges."""
    random.seed(seed)
    adj_int = [0] * n
    degrees = [0] * n
    
    # Generate all possible edges
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    random.shuffle(all_edges)
    
    # Greedily add edges if they don't create C4
    for u, v in all_edges:
        if not _creates_c4_fast(adj_int, n, u, v):
            adj_int[u] |= (1 << v)
            adj_int[v] |= (1 << u)
            degrees[u] += 1
            degrees[v] += 1
    
    return adj_int, degrees

def _build_greedy_c4_free(n, seed, target_min_deg):
    """Build C4-free graph with greedy degree balancing."""
    random.seed(seed)
    adj_int = [0] * n
    degrees = [0] * n
    
    # Start with a random sparse graph
    for _ in range(n * 2):
        u = random.randint(0, n-1)
        v = random.randint(0, n-1)
        if u != v and not (adj_int[u] & (1 << v)):
            if not _creates_c4_fast(adj_int, n, u, v):
                adj_int[u] |= (1 << v)
                adj_int[v] |= (1 << u)
                degrees[u] += 1
                degrees[v] += 1
    
    # Now greedily add edges to boost min degree
    for _ in range(200):
        # Find vertex with lowest degree
        min_deg = min(degrees)
        if min_deg >= target_min_deg:
            break
        low_verts = [i for i in range(n) if degrees[i] == min_deg]
        u = random.choice(low_verts)
        
        # Try to add edge from u to a random non-neighbor
        non_neigh_mask = ((1 << n) - 1) & ~(adj_int[u] | (1 << u))
        if non_neigh_mask:
            # Pick random non-neighbor
            temp = non_neigh_mask
            candidates = []
            while temp:
                lsb = temp & -temp
                candidates.append(lsb.bit_length() - 1)
                temp ^= lsb
            random.shuffle(candidates)
            
            for v in candidates:
                if not _creates_c4_fast(adj_int, n, u, v):
                    adj_int[u] |= (1 << v)
                    adj_int[v] |= (1 << u)
                    degrees[u] += 1
                    degrees[v] += 1
                    break
    
    return adj_int, degrees

def _build_polarity_graph(n, seed):
    """
    Build a polarity graph on n vertices.
    Uses quadratic non-residues mod p for prime p >= n.
    """
    def is_prime(m):
        if m < 2:
            return False
        if m == 2:
            return True
        if m % 2 == 0:
            return False
        for i in range(3, int(math.isqrt(m)) + 1, 2):
            if m % i == 0:
                return False
        return True
    
    p = n
    while not is_prime(p):
        p += 1
    
    # Build quadratic residues mod p
    qr = set()
    for x in range(p):
        qr.add((x * x) % p)
    
    # Build polarity graph on p vertices
    polarity_adj = [0] * p
    for i in range(p):
        for j in range(i+1, p):
            diff = (i - j) % p
            sq = (diff * diff) % p
            if sq not in qr:
                polarity_adj[i] |= (1 << j)
                polarity_adj[j] |= (1 << i)
    
    # Take first n vertices
    adj_int = [0] * n
    degrees = [0] * n
    for i in range(n):
        adj_int[i] = polarity_adj[i] & ((1 << n) - 1)
        degrees[i] = bin(adj_int[i]).count('1')
    
    return adj_int, degrees

def _local_search(adj_int, n, min_deg_required, time_limit, seed):
    """
    Local search to maximize minimum degree while maintaining C4-free property.
    Uses aggressive edge additions with multiple attempts per vertex.
    """
    random.seed(seed)
    start_time = time.time()
    degrees = [bin(adj_int[i]).count('1') for i in range(n)]
    
    best_min_deg = min(degrees)
    best_adj = adj_int[:]
    
    max_iter = 300
    for iteration in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        if min(degrees) >= min_deg_required:
            break
        
        sorted_verts = sorted(range(n), key=lambda i: degrees[i])
        
        improved = False
        for u in sorted_verts:
            if time.time() - start_time > time_limit:
                break
            if degrees[u] >= min_deg_required:
                continue
            
            non_neigh_mask = ((1 << n) - 1) & ~(adj_int[u] | (1 << u))
            if not non_neigh_mask:
                continue
            
            non_neighbors = []
            temp = non_neigh_mask
            while temp:
                lsb = temp & -temp
                non_neighbors.append(lsb.bit_length() - 1)
                temp ^= lsb
            
            random.shuffle(non_neighbors)
            
            for v in non_neighbors:
                if time.time() - start_time > time_limit:
                    break
                if degrees[u] >= min_deg_required:
                    break
                if _creates_c4_fast(adj_int, n, u, v):
                    continue
                adj_int[u] |= (1 << v)
                adj_int[v] |= (1 << u)
                degrees[u] += 1
                degrees[v] += 1
                improved = True
        
        if not improved:
            low_vert = min(range(n), key=lambda i: degrees[i])
            if degrees[low_vert] >= min_deg_required:
                break
            
            edge_list = []
            for a in range(n):
                temp = adj_int[a] & ((1 << a) - 1)
                while temp:
                    lsb = temp & -temp
                    edge_list.append((a, lsb.bit_length() - 1))
                    temp ^= lsb
            
            random.shuffle(edge_list)
            
            swapped = False
            for (x, y) in edge_list[:150]:
                if time.time() - start_time > time_limit:
                    break
                if x == low_vert or y == low_vert:
                    continue
                    
                adj_int[x] ^= (1 << y)
                adj_int[y] ^= (1 << x)
                degrees[x] -= 1
                degrees[y] -= 1
                
                candidates = set()
                temp_x = adj_int[x]
                while temp_x:
                    lsb = temp_x & -temp_x
                    candidates.add(lsb.bit_length() - 1)
                    temp_x ^= lsb
                temp_y = adj_int[y]
                while temp_y:
                    lsb = temp_y & -temp_y
                    candidates.add(lsb.bit_length() - 1)
                    temp_y ^= lsb
                candidates.add(x)
                candidates.add(y)
                
                cand_list = [z for z in candidates if z != low_vert and not (adj_int[low_vert] & (1 << z))]
                random.shuffle(cand_list)
                
                found = False
                for z in cand_list[:8]:
                    if not _creates_c4_fast(adj_int, n, low_vert, z):
                        adj_int[low_vert] |= (1 << z)
                        adj_int[z] |= (1 << low_vert)
                        degrees[low_vert] += 1
                        degrees[z] += 1
                        found = True
                        break
                
                if not found:
                    adj_int[x] |= (1 << y)
                    adj_int[y] |= (1 << x)
                    degrees[x] += 1
                    degrees[y] += 1
                else:
                    swapped = True
                    break
            
            if not swapped:
                break
        
        if min(degrees) > best_min_deg:
            best_min_deg = min(degrees)
            best_adj = adj_int[:]
    
    return best_adj, [bin(best_adj[i]).count('1') for i in range(n)]

def _construct_c4_free(n, min_deg_required, time_limit, seed):
    """Construct a C4-free graph on n vertices with min degree >= min_deg_required."""
    start_time = time.time()
    
    def _make_edges(adj_int, n):
        edges = []
        for u in range(n):
            temp = adj_int[u]
            while temp:
                lsb = temp & -temp
                v = lsb.bit_length() - 1
                if u < v:
                    edges.append([u, v])
                temp ^= lsb
        return edges
    
    # Try multiple initialization strategies
    strategies = []
    
    # Strategy 1: Polarity graph (deterministic but good structure)
    for s in range(5):
        strategies.append(('polarity', s))
    
    # Strategy 2: Random greedy C4-free
    for s in range(10):
        strategies.append(('random', s))
    
    # Strategy 3: Greedy degree-balanced
    for s in range(10):
        strategies.append(('greedy', s))
    
    best_result = None
    best_min_deg = -1
    
    for strategy_name, sub_seed in strategies:
        elapsed = time.time() - start_time
        remaining = time_limit - elapsed
        if remaining < 2.0:
            break
        
        s = seed + sub_seed
        
        if strategy_name == 'polarity':
            adj_int, degrees = _build_polarity_graph(n, s)
        elif strategy_name == 'random':
            adj_int, degrees = _build_random_c4_free(n, s, min_deg_required)
        else:  # greedy
            adj_int, degrees = _build_greedy_c4_free(n, s, min_deg_required)
        
        if min(degrees) >= min_deg_required:
            if _verify_c4_free(adj_int, n):
                return {'n': n, 'edges': _make_edges(adj_int, n)}
            continue
        
        remaining = time_limit - (time.time() - start_time)
        if remaining < 3.0:
            continue
        
        # Allocate time: earlier strategies get more time
        t_ls = min(remaining * 0.7, 6.0)
        best_adj, best_degrees = _local_search(adj_int, n, min_deg_required, t_ls, s)
        
        if min(best_degrees) >= min_deg_required:
            if _verify_c4_free(best_adj, n):
                return {'n': n, 'edges': _make_edges(best_adj, n)}
        
        if min(best_degrees) > best_min_deg:
            best_min_deg = min(best_degrees)
            best_result = {'n': n, 'edges': _make_edges(best_adj, n)}
    
    # Return best found even if not meeting requirement (caller will check)
    if best_result and best_min_deg >= min_deg_required:
        return best_result
    
    return None

def construct():
    start_time = time.time()
    deadline = 115.0
    
    # Try n=47, 46, 45, ...
    # n=47 -> min_deg 8
    # n=46 -> min_deg 7
    # n=45 -> min_deg 6
    
    time_allocation = {
        47: 45.0,
        46: 40.0,
        45: 15.0,
        44: 8.0,
        43: 5.0,
    }
    
    for n in range(47, 38, -1):
        if time.time() - start_time > deadline:
            break
        remaining = deadline - (time.time() - start_time)
        if remaining < 3:
            break
            
        min_deg_req = max(1, n - 39)
        
        t_limit = min(time_allocation.get(n, 5.0), remaining - 1)
        
        result = _construct_c4_free(n, min_deg_req, t_limit, 42)
        if result:
            return result
            
    return {'n': 39, 'edges': []}
