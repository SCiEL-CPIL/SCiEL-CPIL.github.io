import time
import random
import sys

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
    all_mask = (1 << n) - 1
    comp = [0] * n
    for i in range(n):
        comp[i] = all_mask & ~(adj[i] | (1 << i))
    
    def backtrack(candidates, current_set):
        if len(current_set) == 6:
            return current_set[:]
        if candidates.bit_count() < 6 - len(current_set):
            return None
        
        # Heuristic: Pick vertex that minimizes remaining candidates
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

def _make_paley(q):
    """Create Paley graph of order q (q must be prime ≡ 1 mod 4)."""
    if q < 5:
        return None
    # Check if q is prime
    if q < 2:
        return None
    for i in range(2, int(q**0.5) + 1):
        if q % i == 0:
            return None
    # Check q ≡ 1 mod 4
    if q % 4 != 1:
        return None
    squares = {x * x % q for x in range(1, q)}
    n = q
    adj = [0] * n
    for i in range(n):
        for j in range(n):
            if i != j and (j - i) % q in squares:
                adj[i] |= (1 << j)
    return adj

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

def _random_adj(n, p_edge, rng=None):
    """Create a random adjacency list with edge probability p_edge."""
    if rng is None:
        rng = random
    adj = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < p_edge:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    return adj

def _perturb_paley(paley_adj, n, rng, num_flips):
    """Create a perturbed version of the Paley graph by flipping some edges."""
    adj = list(paley_adj)
    for _ in range(num_flips):
        i = rng.randrange(n)
        j = rng.randrange(n)
        if i == j:
            continue
        if (adj[j] >> i) & 1:
            adj[i] &= ~(1 << j)
            adj[j] &= ~(1 << i)
        else:
            adj[i] |= (1 << j)
            adj[j] |= (1 << i)
    return adj

def construct():
    # Strategy: 
    # 1. Try to repair a Paley graph of size 37 (if it exists and is valid or close).
    # 2. If that fails, use Paley-based seeds for local search on n=35, 34, etc.
    
    best_n = 1
    best_edges = []
    
    # Step 1: Try Paley graph of order 37
    # Paley(37) is a graph on 37 vertices.
    # If we can make it valid, we get n=37.
    # If not, we can try to prune it to n=36, 35.
    
    paley_37 = _make_paley(37)
    if paley_37 is not None:
        # Try to repair Paley(37) directly
        if _elapsed() < _DEADLINE - 10:
            edges = _try_local_search(37, paley_37, 10.0, seed=42)
            if edges is not None:
                return {'n': 37, 'edges': edges}
        
        # Try to prune Paley(37) to 36 vertices
        # Remove one vertex and its edges, then repair
        if _elapsed() < _DEADLINE - 10:
            for v_to_remove in range(37):
                if _elapsed() > _DEADLINE - 5:
                    break
                # Create subgraph on 36 vertices
                n_sub = 36
                adj_sub = [0] * n_sub
                for i in range(n_sub):
                    orig_i = i if i < v_to_remove else i + 1
                    mask_i = paley_37[orig_i]
                    # Remap bits
                    new_mask = 0
                    for j in range(n_sub):
                        orig_j = j if j < v_to_remove else j + 1
                        if mask_i & (1 << orig_j):
                            new_mask |= (1 << j)
                    adj_sub[i] = new_mask
                
                edges = _try_local_search(36, adj_sub, 2.0, seed=v_to_remove)
                if edges is not None:
                    return {'n': 36, 'edges': edges}
        
        # Try to prune Paley(37) to 35 vertices
        if _elapsed() < _DEADLINE - 5:
            for v1 in range(37):
                for v2 in range(v1 + 1, 37):
                    if _elapsed() > _DEADLINE - 2:
                        break
                    n_sub = 35
                    adj_sub = [0] * n_sub
                    for i in range(n_sub):
                        orig_i = i
                        if orig_i >= v1: orig_i += 1
                        if orig_i >= v2: orig_i += 1
                        
                        mask_i = paley_37[orig_i]
                        new_mask = 0
                        for j in range(n_sub):
                            orig_j = j
                            if orig_j >= v1: orig_j += 1
                            if orig_j >= v2: orig_j += 1
                            if mask_i & (1 << orig_j):
                                new_mask |= (1 << j)
                        adj_sub[i] = new_mask
                    
                    edges = _try_local_search(35, adj_sub, 0.5, seed=v1*37+v2)
                    if edges is not None:
                        return {'n': 35, 'edges': edges}
    
    # Step 2: Fallback to local search with Paley-based or random seeds
    # We iterate n from 35 down to 20
    # We use the Paley graph of size 37 as a source for "good" seeds by taking random subgraphs or perturbations
    
    for n in range(35, 19, -1):
        if _elapsed() > _DEADLINE - 5:
            break
        
        # Time slice for this n
        if n >= 35:
            time_slice = 20.0
        elif n >= 33:
            time_slice = 10.0
        else:
            time_slice = 5.0
            
        if _elapsed() + time_slice > _DEADLINE - 3:
            time_slice = _DEADLINE - 3 - _elapsed()
            if time_slice < 0.5:
                break
        
        found = False
        t_start = time.time()
        
        # Try a few restarts
        # Use seeds derived from Paley if possible, otherwise random
        # For n=35, we can take a random 35-vertex subgraph of Paley(37)
        
        restarts = 5
        for r in range(restarts):
            if time.time() - t_start > time_slice:
                break
            
            rem_time = time_slice - (time.time() - t_start)
            if rem_time < 0.5:
                break
            
            # Generate initial graph
            if paley_37 is not None and n <= 37:
                # Take a random subset of vertices from Paley(37)
                # Or just perturb a random subgraph
                # Simpler: Random graph with p=0.45 is also good, 
                # but let's try to mix in some structure if n is close to 37.
                # For now, stick to random graphs for speed and reliability,
                # but if we have time, we can try structured seeds.
                pass
            
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
            return {'n': best_n, 'edges': best_edges}
    
    # Absolute fallback: Empty graph on 5 vertices
    if best_n < 1:
        best_n = 5
        best_edges = []
        
    return {'n': best_n, 'edges': best_edges}
