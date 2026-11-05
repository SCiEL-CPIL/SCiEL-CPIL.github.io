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
            # Common neighbors of i and j, with index > j
            common = adj[i] & adj[j] & ~((1 << (j + 1)) - 1)
            if common == 0:
                continue
            # Extract bits
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
        
        # Heuristic: Pick vertex that minimizes remaining candidates after inclusion
        # This is equivalent to picking the vertex with the most neighbors in 'candidates'
        # because new_candidates = candidates & ~adj[v]
        # Minimizing |new_candidates| is maximizing |adj[v] & candidates|
        best_v = -1
        best_new_count = n + 1
        c = candidates
        while c:
            lsb = c & -c
            v = lsb.bit_length() - 1
            new_cand = candidates & ~adj[v]
            cnt = new_cand.bit_count()
            if cnt < best_new_count:
                best_new_count = cnt
                best_v = v
            c ^= lsb
        
        if best_v == -1:
            return None
            
        v = best_v
        # Remove v from candidates for the 'exclude' branch
        candidates_excl = candidates & ~(1 << v)
        
        # Branch 1: Include v
        # If we include v, we cannot include any of its neighbors
        new_candidates_incl = candidates_excl & ~adj[v]
        res = backtrack(new_candidates_incl, current_set + [v])
        if res:
            return res
        
        # Branch 2: Exclude v
        res = backtrack(candidates_excl, current_set)
        if res:
            return res
        
        return None

    return backtrack((1 << n) - 1, [])

def _try_local_search(n, time_limit):
    """Try to find a valid graph on n vertices using local search with restarts."""
    start = time.time()
    restart_interval = 2.0  # Restart if no progress for 2 seconds
    last_restart = start
    
    while time.time() - start < time_limit:
        # Initialize random graph
        # p=0.5 is a good balance for Ramsey graphs
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < 0.5:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        
        last_progress = time.time()
        
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
                last_progress = time.time()
                continue
            
            # Check for Ind 6
            ind6 = _has_ind_6(adj, n)
            if ind6:
                # Add a random edge between two vertices in the independent set
                pairs = [(ind6[i], ind6[j]) for i in range(6) for j in range(i + 1, 6)]
                u, v = random.choice(pairs)
                adj[u] |= (1 << v)
                adj[j] |= (1 << u) # Correction: adj[v]
                adj[v] |= (1 << u)
                last_progress = time.time()
                continue
            
            # If no K4 and no Ind 6, we found a valid graph
            edges = []
            for i in range(n):
                for j in range(i + 1, n):
                    if adj[i] & (1 << j):
                        edges.append([i, j])
            return edges
        
        # If we exit the inner loop without returning, it means time_limit was reached
        # But we check time at the top of the outer loop too.
        # Actually, if we exit inner loop, we should check if we should restart or stop.
        # The outer loop condition `time.time() - start < time_limit` handles stopping.
        # If we are still within time_limit, we restart.
        if time.time() - start >= time_limit:
            break

    return None

def construct():
    # Base case: Paley 17
    p = 17
    squares = {x * x % p for x in range(1, p)}
    best_edges = [[i, j] for i in range(p) for j in range(i + 1, p) if (j - i) % p in squares]
    best_n = p
    
    # Try to find larger graphs
    # Iterate from 18 up to 35
    # Use dynamic time allocation
    
    for n in range(18, 36):
        if _elapsed() > _DEADLINE - 2:
            break
        
        # Remaining time
        rem = _DEADLINE - _elapsed()
        # Number of steps left (including current)
        steps_left = 36 - n
        
        # Heuristic: allocate proportional to remaining steps, but with a cap
        # to ensure we don't spend too much on one n if it's easy
        # If n is small, it should be fast. If n is large, it might be hard.
        # Capping at 10s prevents one hard n from eating all time, but
        # dynamic allocation `rem / steps_left` naturally gives more to later steps
        # if earlier steps are fast.
        time_slice = min(10.0, rem / steps_left)
        
        # Ensure at least 0.5s if possible
        if time_slice < 0.5:
            time_slice = 0.5
            
        edges = _try_local_search(n, time_slice)
        if edges is not None:
            best_n = n
            best_edges = edges
    
    return {'n': best_n, 'edges': best_edges}
