import time
import random
import itertools

# EVOLVE-BLOCK-START

def _find_k222(adj, n):
    """Find a K2,2,2 subgraph. Returns tuple (a,b,u,v,w,x) or None."""
    for a in range(n):
        for b in range(a+1, n):
            if not (adj[a] >> b) & 1:
                continue
            cn_mask = adj[a] & adj[b]
            cn_mask &= ~(1 << a)
            cn_mask &= ~(1 << b)
            
            if cn_mask.bit_count() < 4:
                continue
            
            cn_list = []
            m = cn_mask
            while m:
                lsb = m & (-m)
                idx = lsb.bit_length() - 1
                cn_list.append(idx)
                m ^= lsb
            
            cn_bitset = cn_mask
            for i in range(len(cn_list)):
                u = cn_list[i]
                adj_u = adj[u]
                for j in range(i+1, len(cn_list)):
                    v = cn_list[j]
                    common = adj_u & adj[v] & cn_bitset
                    if common.bit_count() >= 2:
                        m2 = common
                        w = (m2 & (-m2)).bit_length() - 1
                        m2 ^= (1 << w)
                        x = (m2 & (-m2)).bit_length() - 1
                        return (a, b, u, v, w, x)
    return None

def _build_adj_from_colors(colors, n):
    """Build adjacency bitmask list from colors list."""
    adj = [0] * n
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            if colors[idx] == 0:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
            idx += 1
    return adj

def _get_colors_from_adj(adj, n):
    """Get colors list from adjacency bitmasks."""
    colors = []
    for i in range(n):
        for j in range(i+1, n):
            if (adj[i] >> j) & 1:
                colors.append(0)
            else:
                colors.append(1)
    return colors

def _edge_index(i, j, n):
    if i > j:
        i, j = j, i
    return i*(n-1) - i*(i-1)//2 + (j - i - 1)

def _paley_adj(q):
    """Generate Paley graph adjacency bitmasks on q vertices (q prime, q ≡ 1 mod 4)."""
    squares = set()
    for x in range(1, q):
        squares.add(x * x % q)
    
    adj = [0] * q
    for i in range(q):
        for j in range(i+1, q):
            diff = (j - i) % q
            if diff in squares:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    return adj

def _is_valid_fast(adj, n):
    """Check both color classes are K2,2,2-free. Returns True if valid."""
    full = (1 << n) - 1
    comp = [0] * n
    for i in range(n):
        comp[i] = full ^ adj[i] ^ (1 << i)
    
    if _find_k222(adj, n) is not None:
        return False
    if _find_k222(comp, n) is not None:
        return False
    return True

def _flip_edge_adj(adj, i, j):
    """Flip edge (i,j) in adjacency bitmasks in-place."""
    adj[i] ^= (1 << j)
    adj[j] ^= (1 << i)

def _search_n30(deadline, rng):
    n = 30
    total_edges = n*(n-1)//2
    
    # Precompute all 12 edges of a K2,2,2 for faster access
    def get_witness_edges(a, b, u, v, w, x):
        return [(a,b), (a,u), (a,v), (a,w), (a,x), (b,u), (b,v), (b,w), (b,x), (u,w), (u,x), (v,w), (v,x)]
    
    # Strategy 1: Start from Paley-29 + 1 vertex
    paley29 = _paley_adj(29)
    
    # Use time-based steps to ensure we use the full budget effectively
    # Start with 50 Paley-based restarts, then 150 Random restarts
    # Each restart gets a fixed time slice, e.g., 0.5s
    
    total_restarts = 200
    restart_time_slice = 0.4 # seconds per restart
    
    for restart in range(total_restarts):
        if time.time() > deadline - 5: # Safety margin
            break
        
        if restart < 50:
            # Start from Paley-29 with vertex 29 connected randomly
            adj = [0] * n
            for i in range(29):
                adj[i] = paley29[i]
            # Connect vertex 29 randomly
            for i in range(29):
                if rng.randint(0, 1):
                    adj[i] |= (1 << 29)
                    adj[29] |= (1 << i)
            adj = list(adj)
        else:
            # Random initialization
            adj = [0] * n
            for i in range(n):
                for j in range(i+1, n):
                    if rng.randint(0, 1):
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
        
        # Local search with time limit per restart
        restart_deadline = time.time() + restart_time_slice
        while time.time() < restart_deadline and time.time() < deadline - 2:
            witness = _find_k222(adj, n)
            if witness is None:
                full = (1 << n) - 1
                comp = [0] * n
                for i in range(n):
                    comp[i] = full ^ adj[i] ^ (1 << i)
                witness_comp = _find_k222(comp, n)
                if witness_comp is None:
                    return _get_colors_from_adj(adj, n)
                else:
                    a, b, u, v, w, x = witness_comp
                    edges = get_witness_edges(a, b, u, v, w, x)
                    edge = rng.choice(edges)
                    i, j = edge
                    if i > j: i, j = j, i
                    _flip_edge_adj(adj, i, j)
            else:
                a, b, u, v, w, x = witness
                edges = get_witness_edges(a, b, u, v, w, x)
                edge = rng.choice(edges)
                i, j = edge
                if i > j: i, j = j, i
                _flip_edge_adj(adj, i, j)
                
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    rng = random.Random(42)
    
    # 1. Try to find n=30
    colors_30 = _search_n30(deadline, rng)
    if colors_30 is not None:
        return {'n': 30, 'colors': colors_30}
        
    # 2. Fallback to n=29 (Paley)
    q = 29
    adj = _paley_adj(q)
    if _is_valid_fast(adj, q):
        colors = _get_colors_from_adj(adj, q)
        return {'n': q, 'colors': colors}
    
    # 3. Fallback to smaller
    for q_try in [17, 13, 5]:
        adj_t = _paley_adj(q_try)
        if _is_valid_fast(adj_t, q_try):
            colors = _get_colors_from_adj(adj_t, q_try)
            return {'n': q_try, 'colors': colors}
            
    return {'n': 1, 'colors': []}

# EVOLVE-BLOCK-END
