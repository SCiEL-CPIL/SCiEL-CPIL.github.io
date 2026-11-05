import time
import random

_START = time.time()
_DEADLINE = 140  # seconds, leave 10s margin for the 150s limit

def _elapsed():
    return time.time() - _START

def _has_k4(adj, n):
    """Check if graph has a K4. Returns the K4 vertices if found, else None."""
    for i in range(n):
        for j in range(i + 1, n):
            if not (adj[i] & (1 << j)):
                continue
            common = adj[i] & adj[j] & ~((1 << (j + 1)) - 1)
            if common == 0:
                continue
            bits = []
            c = common
            while c:
                lsb = c & -c
                idx = lsb.bit_length() - 1
                bits.append(idx)
                c ^= lsb
            for k_idx in range(len(bits)):
                k = bits[k_idx]
                for l_idx in range(k_idx + 1, len(bits)):
                    l = bits[l_idx]
                    if adj[k] & (1 << l):
                        return [i, j, k, l]
    return None

def _has_ind_6(adj, n):
    """Check if graph has an independent set of size 6. Returns the set if found, else None."""
    def backtrack(candidates, current_set):
        if len(current_set) == 6:
            return current_set[:]
        if candidates.bit_count() < 6 - len(current_set):
            return None
        
        # Pick vertex that minimizes remaining candidates after inclusion
        # Heuristic: choose vertex with max degree in induced subgraph of candidates?
        # Or just pick the first one? The original picks min remaining candidates.
        best_v = -1
        best_count = n + 1
        c = candidates
        while c:
            lsb = c & -c
            v = lsb.bit_length() - 1
            # If we include v, candidates become candidates & ~adj[v] & ~(1<<v)
            new_cand = candidates & ~adj[v] & ~(1 << v)
            cnt = new_cand.bit_count()
            if cnt < best_count:
                best_count = cnt
                best_v = v
            c ^= lsb
        
        if best_v == -1:
            return None
            
        v = best_v
        new_candidates = candidates & ~((1 << v) | adj[v])
        
        # Branch 1: Include v
        res = backtrack(new_candidates, current_set + [v])
        if res:
            return res
        
        # Branch 2: Exclude v
        res = backtrack(candidates & ~(1 << v), current_set)
        if res:
            return res
        
        return None

    return backtrack((1 << n) - 1, [])

def _random_adj(n, p_edge):
    """Create a random adjacency list with edge probability p_edge."""
    adj = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if random.random() < p_edge:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    return adj

def _adj_to_edges(adj, n):
    """Convert adjacency list to edge list."""
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i] & (1 << j):
                edges.append([i, j])
    return edges

def _try_local_search(n, adj_init, time_limit, seed=None):
    """Try to find a valid graph on n vertices using local search from initial adj."""
    if seed is not None:
        random.seed(seed)
    start = time.time()
    
    # Copy initial adjacency
    adj = list(adj_init)
    
    # Local search loop
    while time.time() - start < time_limit:
        # Check for K4
        k4 = _has_k4(adj, n)
        if k4:
            # Remove a random edge from the K4
            pairs = [(k4[0], k4[1]), (k4[0], k4[2]), (k4[0], k4[3]),
                     (k4[1], k4[2]), (k4[1], k4[3]), (k4[2], k4[3])]
            u, v = random.choice(pairs)
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
            continue
        
        # Check for Ind 6
        ind6 = _has_ind_6(adj, n)
        if ind6:
            # Add a random edge between two vertices in the independent set
            pairs = [(ind6[i], ind6[j]) for i in range(6) for j in range(i + 1, 6)]
            u, v = random.choice(pairs)
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            continue
        
        # If no K4 and no Ind 6, we found a valid graph
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i] & (1 << j):
                    edges.append([i, j])
        return edges
    
    return None

def construct():
    # Start with a small valid graph and try to grow it
    # Or just try random local search for increasing n
    
    best_n = 1
    best_edges = []
    
    # We know n=32 is achievable. Let's aim for 33, 34, 35.
    # Start from n=30 and go up.
    
    for n in range(30, 36):
        if _elapsed() > _DEADLINE - 10:
            break
            
        # Allocate time based on difficulty
        # Larger n is harder, so give it more time? Or less time if we fail quickly?
        # Let's give 10s for n=30, 15s for n=31, 20s for n=32, 30s for n=33, 40s for n=34, 50s for n=35
        # But we have 140s total.
        # Let's just try each n with a fixed time slice if we haven't found it yet.
        
        time_slice = 10.0
        if n >= 33:
            time_slice = 20.0
        if n >= 34:
            time_slice = 30.0
        if n >= 35:
            time_slice = 40.0
            
        if _elapsed() + time_slice > _DEADLINE - 5:
            time_slice = _DEADLINE - 5 - _elapsed()
            if time_slice < 1.0:
                break
                
        found = False
        # Try multiple restarts
        for attempt in range(5):
            if _elapsed() > _DEADLINE - 5:
                break
            adj_init = _random_adj(n, 0.4)
            edges = _try_local_search(n, adj_init, min(time_slice/5, _DEADLINE - 5 - _elapsed()))
            if edges is not None:
                best_n = n
                best_edges = edges
                found = True
                break
                
        if found:
            continue
            
    # If we didn't find anything above 30, try to ensure we have at least 32
    if best_n < 32:
        # Fallback: try n=32 with remaining time
        if _elapsed() < _DEADLINE - 5:
            adj_init = _random_adj(32, 0.4)
            edges = _try_local_search(32, adj_init, _DEADLINE - 5 - _elapsed())
            if edges is not None:
                best_n = 32
                best_edges = edges
    return {'n': best_n, 'edges': best_edges}
