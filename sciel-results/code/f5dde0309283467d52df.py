import time
import random

_START = time.time()
_DEADLINE = 145  # seconds, leave 5s margin for the 150s limit

def _elapsed():
    return time.time() - _START

def _has_k4(adj, n):
    """Check if graph has a K4. Returns the K4 vertices if found, else None."""
    for i in range(n):
        for j in range(i + 1, n):
            if not (adj[i] & (1 << j)):
                continue
            # Common neighbors of i and j, strictly greater than j to avoid duplicates
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
    """
    Check if graph has an independent set of size 6.
    Returns the set if found, else None.
    Uses a branch-and-bound with bitmasks for speed.
    """
    # Precompute non-adjacency masks (complement of adj, excluding self)
    # comp[i] has bits set for all j != i such that (i,j) is NOT an edge
    all_mask = (1 << n) - 1
    comp = [0] * n
    for i in range(n):
        # ~adj[i] & all_mask & ~(1 << i)
        comp[i] = all_mask & ~(adj[i] | (1 << i))
    
    def backtrack(candidates, current_set):
        if len(current_set) == 6:
            return current_set[:]
        if candidates.bit_count() < 6 - len(current_set):
            return None
        
        # Heuristic: Pick vertex that minimizes remaining candidates
        # This is a standard MIS heuristic
        best_v = -1
        best_count = n + 1
        c = candidates
        while c:
            lsb = c & -c
            v = lsb.bit_length() - 1
            new_cand = candidates & comp[v]
            cnt = new_cand.bit_count()
            if cnt < best_count:
                best_count = cnt
                best_v = v
            c ^= lsb
        
        if best_v == -1:
            return None
            
        v = best_v
        # Branch 1: Include v
        new_candidates = candidates & comp[v]
        res = backtrack(new_candidates, current_set + [v])
        if res:
            return res
        
        # Branch 2: Exclude v
        res = backtrack(candidates & ~(1 << v), current_set)
        if res:
            return res
        
        return None

    return backtrack(all_mask, [])

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

def _random_adj(n, p_edge):
    """Create a random adjacency list with edge probability p_edge."""
    adj = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if random.random() < p_edge:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    return adj

def construct():
    # Strategy: Target n=35 first (Schläfli graph size), then 34, etc.
    # This is structurally different from incremental search.
    # We use multiple random restarts for each n, focusing time on higher n.
    
    best_n = 1
    best_edges = []
    
    # We will try n from 35 down to 18
    # Time allocation: Higher n gets more time initially.
    
    for n in range(35, 17, -1):
        if _elapsed() > _DEADLINE - 5:
            break
        
        # Time slice for this n
        # 35: 40s, 34: 20s, 33: 10s, 32: 5s, ...
        if n >= 35:
            time_slice = 40.0
        elif n >= 34:
            time_slice = 20.0
        elif n >= 33:
            time_slice = 10.0
        elif n >= 32:
            time_slice = 5.0
        else:
            time_slice = 2.0
            
        if _elapsed() + time_slice > _DEADLINE - 3:
            time_slice = _DEADLINE - 3 - _elapsed()
            if time_slice < 0.5:
                break
        
        found = False
        t_start = time.time()
        
        # Multiple restarts
        # For n=35, we do fewer but longer runs.
        # For smaller n, we do more shorter runs.
        if n >= 33:
            restarts = 3
        else:
            restarts = 5
            
        for r in range(restarts):
            if time.time() - t_start > time_slice:
                break
            
            # Estimate remaining time for this restart
            rem_time = time_slice - (time.time() - t_start)
            if rem_time < 0.5:
                break
            
            # Initial probability
            # Schläfli graph density is 16/34 ~ 0.47
            p_edge = 0.45 + 0.05 * r
            
            adj_init = _random_adj(n, p_edge)
            
            # Run local search
            edges = _try_local_search(n, adj_init, rem_time)
            
            if edges is not None:
                best_n = n
                best_edges = edges
                found = True
                break
        
        if found:
            # If we found a valid graph for n, we can stop if n is high enough
            # Or continue to see if we can find a larger one? 
            # Since we iterate downwards, the first found is the largest.
            # So we can return immediately if we found one for n>=35?
            # Actually, if we find n=35, that's the max possible known.
            # If we find n=34, we should check if we have time to try 35 again?
            # No, we tried 35 first. If it failed, we move to 34.
            # So the first success is the best.
            return {'n': best_n, 'edges': best_edges}
    
    # Fallback: If we didn't find anything in the loop (unlikely), 
    # try a quick search for n=20 to ensure we return something valid.
    if best_n < 17:
        n = 20
        adj_init = _random_adj(n, 0.5)
        edges = _try_local_search(n, adj_init, _DEADLINE - 5 - _elapsed())
        if edges:
            best_n = n
            best_edges = edges
        else:
            # Absolute fallback: Empty graph on 5 vertices (alpha=5, no K4)
            best_n = 5
            best_edges = []
            
    return {'n': best_n, 'edges': best_edges}
