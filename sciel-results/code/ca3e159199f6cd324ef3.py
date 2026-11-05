"""Combined program: targeted local repair + grow strategy for K4-free, no-ind-6 graphs."""
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
        
        # Pick vertex with fewest non-neighbors in candidates (best pruning)
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
        
        # Branch 1: Include v
        res = backtrack(new_candidates, current_set + [v])
        if res:
            return res
        
        # Branch 2: Exclude v
        candidates ^= (1 << v)
        res = backtrack(candidates, current_set)
        if res:
            return res
        
        return None

    return backtrack((1 << n) - 1, [])

def _count_k4(adj, n):
    """Count all K4s in the graph."""
    count = 0
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
                        count += 1
    return count

def _count_ind6(adj, n):
    """Count all independent sets of size 6 in the graph."""
    count = 0
    def backtrack(candidates, current_set):
        nonlocal count
        if len(current_set) == 6:
            count += 1
            return
        if candidates.bit_count() < 6 - len(current_set):
            return
        
        # Pick vertex with fewest non-neighbors
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
            return
            
        v = best_v
        new_candidates = candidates & ~((1 << v) | adj[v])
        
        # Branch 1: Include v
        backtrack(new_candidates, current_set + [v])
        
        # Branch 2: Exclude v
        candidates ^= (1 << v)
        backtrack(candidates, current_set)

    backtrack((1 << n) - 1, [])
    return count

def _paley_graph(p):
    """Construct the Paley graph on p vertices."""
    if p % 4 != 1:
        return None
    squares = {x * x % p for x in range(1, p)}
    adj = [0] * p
    edges = []
    for i in range(p):
        for j in range(i + 1, p):
            if (j - i) % p in squares:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
                edges.append([i, j])
    return adj, edges

def _try_local_search(n, time_limit, init_adj=None):
    """Try to find a valid graph on n vertices using targeted local repair."""
    start = time.time()
    
    if init_adj is not None:
        adj = init_adj[:]
    else:
        # Initialize random graph
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < 0.5:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
    
    # Local search with targeted repair
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

def _try_grow(adj, n, time_limit):
    """Try to grow a valid graph on n vertices to n+1 by adding a new vertex."""
    start = time.time()
    new_n = n + 1
    new_adj = adj + [0]
    
    # Try different degrees for the new vertex
    for degree in range(0, new_n):
        if time.time() - start > time_limit:
            break
        
        # Try random subsets of neighbors
        attempts = 0
        while attempts < 50 and time.time() - start < time_limit:
            attempts += 1
            # Randomly select neighbors for new vertex
            neighbors = random.sample(range(n), degree)
            for v in neighbors:
                new_adj[new_n - 1] |= (1 << v)
                new_adj[v] |= (1 << (new_n - 1))
            
            # Check if valid
            k4 = _has_k4(new_adj, new_n)
            if k4 is None:
                ind6 = _has_ind_6(new_adj, new_n)
                if ind6 is None:
                    edges = []
                    for i in range(new_n):
                        for j in range(i + 1, new_n):
                            if new_adj[i] & (1 << j):
                                edges.append([i, j])
                    return edges, new_n
            
            # Reset new vertex connections
            for v in range(n):
                new_adj[v] &= ~(1 << (new_n - 1))
            new_adj[new_n - 1] = 0
    
    return None, n

def construct():
    # Base case: Paley 17
    p = 17
    squares = {x * x % p for x in range(1, p)}
    best_edges = [[i, j] for i in range(p) for j in range(i + 1, p) if (j - i) % p in squares]
    best_n = p
    
    # Try to find larger graphs using grow strategy
    for n in range(18, 36):
        if _elapsed() > _DEADLINE - 5:
            break
        
        # Time slice: decrease as n increases
        time_slice = 2.0
        if n > 25:
            time_slice = 1.0
        if n > 30:
            time_slice = 0.5
        
        # First try growing from the previous best
        if best_n == n - 1:
            # Reconstruct adj from best_edges
            adj = [0] * best_n
            for u, v in best_edges:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
            
            result, new_n = _try_grow(adj, best_n, time_slice)
            if result is not None:
                best_n = new_n
                best_edges = result
                continue
        
        # If grow failed or not applicable, try local search
        edges = _try_local_search(n, time_slice)
        if edges is not None:
            best_n = n
            best_edges = edges
    
    return {'n': best_n, 'edges': best_edges}
