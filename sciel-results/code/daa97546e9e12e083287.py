import time
import random
import sys

_START = time.time()
_DEADLINE = 143  # seconds, leave 7s margin for the 150s limit

def _elapsed():
    return time.time() - _START

def _has_k4(adj, n):
    """Check if graph has a K4. Returns the K4 vertices if found, else None."""
    for i in range(n):
        for j in range(i + 1, n):
            if not (adj[i] & (1 << j)):
                continue
            common = adj[i] & adj[j]
            common &= ~((1 << (j + 1)) - 1)
            if common == 0:
                continue
            c = common
            while c:
                lsb = c & -c
                k = lsb.bit_length() - 1
                c ^= lsb
                rest = common & ~((1 << (k + 1)) - 1)
                if rest & adj[k]:
                    r = rest & adj[k]
                    if r:
                        lsb2 = r & -r
                        l = lsb2.bit_length() - 1
                        return [i, j, k, l]
    return None

def _has_ind_6(adj, n):
    """Check if graph has an independent set of size 6. Returns the set if found, else None."""
    def backtrack(candidates, current_set):
        if len(current_set) == 6:
            return current_set[:]
        if candidates.bit_count() < 6 - len(current_set):
            return None
        
        best_v = -1
        best_count = n + 1
        c = candidates
        while c:
            lsb = c & -c
            v = lsb.bit_length() - 1
            new_cand = candidates & ~adj[v]
            cnt = new_cand.bit_count()
            if cnt < best_count:
                best_count = cnt
                best_v = v
            c ^= lsb
        
        if best_v == -1:
            return None
            
        v = best_v
        new_candidates = candidates & ~((1 << v) | adj[v])
        
        res = backtrack(new_candidates, current_set + [v])
        if res:
            return res
        
        res = backtrack(candidates & ~(1 << v), current_set)
        if res:
            return res
        
        return None

    return backtrack((1 << n) - 1, [])

def _try_local_search(n, adj_init, time_limit, seed=None):
    """Try to find a valid graph on n vertices using local search from initial adj."""
    if seed is not None:
        random.seed(seed)
    start = time.time()
    
    adj = list(adj_init)
    
    while time.time() - start < time_limit:
        k4 = _has_k4(adj, n)
        if k4:
            pairs = [(k4[0], k4[1]), (k4[0], k4[2]), (k4[0], k4[3]),
                     (k4[1], k4[2]), (k4[1], k4[3]), (k4[2], k4[3])]
            u, v = random.choice(pairs)
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
            continue
        
        ind6 = _has_ind_6(adj, n)
        if ind6:
            pairs = [(ind6[i], ind6[j]) for i in range(6) for j in range(i + 1, 6)]
            u, v = random.choice(pairs)
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            continue
        
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i] & (1 << j):
                    edges.append([i, j])
        return edges
    
    return None

def _random_adj(n, p_edge):
    """Create a random adjacency list with edge probability p_edge."""
    adj = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if random.random() < p_edge:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    return adj

def _circulant_adj(n, shifts):
    """Create a circulant graph adjacency list."""
    adj = [0] * n
    for i in range(n):
        for s in shifts:
            j = (i + s) % n
            if j != i:
                adj[i] |= (1 << j)
    return adj

def _is_valid(adj, n):
    """Check if the graph is valid (no K4, no independent set of 6)."""
    if _has_k4(adj, n):
        return False
    if _has_ind_6(adj, n):
        return False
    return True

def construct():
    best_n = 0
    best_edges = []
    
    # Try circulant graphs first - these are structured and might hit n=35
    # For circulant graphs, we need to find shifts that avoid K4 and independent 6-sets
    for n in range(18, 36):
        if _elapsed() > _DEADLINE - 5:
            break
        
        # For circulant graphs, try different shift sets
        # The degree should be around 16-18 for n=35 (Schläfli-like)
        target_degree = 16 if n == 35 else max(8, n // 2 - 2)
        
        # Try random shift sets of appropriate size
        for attempt in range(50):
            if _elapsed() > _DEADLINE - 3:
                break
            
            # Choose random shifts
            possible_shifts = list(range(1, n // 2 + 1))
            random.shuffle(possible_shifts)
            k = target_degree // 2  # Each shift gives degree 2 (symmetric)
            if k < 1:
                k = 1
            if k > len(possible_shifts):
                k = len(possible_shifts)
            
            shifts = possible_shifts[:k]
            adj = _circulant_adj(n, shifts)
            
            if _is_valid(adj, n):
                best_n = n
                best_edges = []
                for i in range(n):
                    for j in range(i + 1, n):
                        if adj[i] & (1 << j):
                            best_edges.append([i, j])
                break
    
    # If we haven't found n>=30, try local search
    if best_n < 30:
        for n in range(18, min(36, best_n + 15)):
            if _elapsed() > _DEADLINE - 5:
                break
            
            time_slice = max(0.5, (_DEADLINE - 5 - _elapsed()) / max(1, (min(36, best_n + 15) - n)))
            
            for attempt in range(3):
                if _elapsed() > _DEADLINE - 3:
                    break
                
                p_edge = 0.35 + 0.05 * attempt
                adj_init = _random_adj(n, p_edge)
                edges = _try_local_search(n, adj_init, min(time_slice / 3, 5.0))
                if edges is not None:
                    best_n = n
                    best_edges = edges
                    break
    
    # Fallback to Paley 17 if nothing else worked
    if best_n < 17:
        p = 17
        squares = {x * x % p for x in range(1, p)}
        best_edges = [[i, j] for i in range(p) for j in range(i + 1, p) if (j - i) % p in squares]
        best_n = p
    
    return {'n': best_n, 'edges': best_edges}
