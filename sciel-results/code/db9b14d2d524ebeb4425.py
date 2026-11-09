import time
import random
import sys

def _alpha_bnb_fast(adj, n, limit=10):
    """
    Fast branch-and-bound: returns True if alpha >= limit, False if alpha < limit.
    """
    best = [0]
    
    def bnb(cand, sz):
        if sz >= limit:
            best[0] = limit
            return True
        if sz + cand.bit_count() < limit:
            return False
        if cand == 0:
            if sz > best[0]:
                best[0] = sz
            return False
        
        # Pick vertex with max degree in cand for better pruning
        best_v = -1
        best_deg = -1
        c = cand
        while c:
            lsb = c & -c
            v = lsb.bit_length() - 1
            deg = (adj[v] & cand).bit_count()
            if deg > best_deg:
                best_deg = deg
                best_v = v
            c ^= lsb
        
        if best_v == -1:
            return False

        v_bit = 1 << best_v
        
        # Branch 1: Include best_v
        if bnb(cand & ~v_bit & ~adj[best_v], sz + 1):
            return True
        
        # Branch 2: Exclude best_v
        if bnb(cand & ~v_bit, sz):
            return True
        
        if sz > best[0]:
            best[0] = sz
        return False

    bnb((1 << n) - 1, 0)
    return best[0]

def _is_triangle_free_fast(adj, n):
    """Check triangle-free using bitmask operations."""
    for u in range(n):
        v_bits = adj[u]
        while v_bits:
            lsb = v_bits & -v_bits
            v = lsb.bit_length() - 1
            if u < v:
                if (adj[u] & adj[v]) != 0:
                    return False
            v_bits ^= lsb
    return True

def _find_independent_set(adj, n, limit=10):
    """Find an independent set of size 'limit' if it exists."""
    found = [None]
    
    def bnb(cand, sz, path):
        if found[0] is not None:
            return
        if sz >= limit:
            found[0] = list(path)
            return
        if sz + cand.bit_count() < limit:
            return
        if cand == 0:
            return
        
        # Pick vertex with min degree in cand (more likely to be in IS)
        best_v = -1
        best_deg = n + 1
        c = cand
        while c:
            lsb = c & -c
            v = lsb.bit_length() - 1
            deg = (adj[v] & cand).bit_count()
            if deg < best_deg:
                best_deg = deg
                best_v = v
            c ^= lsb
        
        if best_v == -1:
            return

        v_bit = 1 << best_v
        
        path.append(best_v)
        bnb(cand & ~v_bit & ~adj[best_v], sz + 1, path)
        path.pop()
        
        if found[0] is not None:
            return
        bnb(cand & ~v_bit, sz, path)
        
    bnb((1 << n) - 1, 0, [])
    return found[0]

def _build_cayley(n, S):
    """Build Cayley graph on Z_n with connection set S (must be symmetric)."""
    adj = [0] * n
    S_set = set(S)
    for i in range(n):
        for s in S_set:
            j = (i + s) % n
            if j != i:
                adj[i] |= (1 << j)
    return adj

def _check_sum_free_symmetric(n, S):
    """Check S is symmetric (-S = S mod n) and sum-free."""
    S_set = set(S)
    # Symmetric
    for s in S_set:
        if (-s) % n not in S_set and s != 0:
            return False
    # Sum-free
    for s1 in S_set:
        for s2 in S_set:
            if (s1 + s2) % n in S_set:
                return False
    return True

def _try_cayley_constructions(deadline):
    """Try Cayley graph constructions for n from 40 down to 30."""
    best = None
    
    for n in range(40, 29, -1):
        if time.time() > deadline:
            break
        
        candidates = []
        
        # Modulo-based sum-free sets: S = {s : s mod m in R}
        for m in range(3, n):
            for r_size in range(1, m // 2 + 1):
                for start in range(m):
                    R = set((start + i) % m for i in range(r_size))
                    # Check if S = {s in 1..n-1 : s mod m in R} is symmetric
                    S = set()
                    for s in range(1, n):
                        if s % m in R:
                            S.add(s)
                    # Check symmetry
                    sym = True
                    for s in S:
                        if (-s) % n not in S and s != 0:
                            sym = False
                            break
                    if not sym:
                        continue
                    # Check sum-free
                    sf = True
                    S_list = list(S)
                    for s1 in S_list:
                        for s2 in S_list:
                            if (s1 + s2) % n in S:
                                sf = False
                                break
                        if not sf:
                            break
                    if sf and len(S) >= 2:
                        candidates.append(S)
        
        # Also try specific known patterns
        # For n divisible by 7: S = {s : s mod 7 in {1,2,3}}
        if n % 7 == 0:
            S = set(s for s in range(1, n) if s % 7 in (1, 2, 3))
            if _check_sum_free_symmetric(n, S):
                candidates.append(S)
        
        # For n divisible by 5: S = {s : s mod 5 in {1,2}}
        if n % 5 == 0:
            S = set(s for s in range(1, n) if s % 5 in (1, 2))
            if _check_sum_free_symmetric(n, S):
                candidates.append(S)
        
        # Sort by size descending (more edges = smaller alpha)
        candidates.sort(key=len, reverse=True)
        
        for S in candidates[:30]:
            if time.time() > deadline:
                break
            adj = _build_cayley(n, S)
            if not _is_triangle_free_fast(adj, n):
                continue
            if not _alpha_bnb_fast(adj, n, 10):
                edges = []
                for u in range(n):
                    v_bits = adj[u]
                    while v_bits:
                        lsb = v_bits & -v_bits
                        v = lsb.bit_length() - 1
                        if u < v:
                            edges.append([u, v])
                        v_bits ^= lsb
                if best is None or n > best['n']:
                    best = {'n': n, 'edges': edges}
                break
    
    return best

def _local_search(n, seed, time_limit):
    """
    Local search to find a triangle-free graph with alpha < 10.
    """
    rng = random.Random(seed)
    start_time = time.time()
    
    # Initialize with a random triangle-free graph
    # Strategy: Start with bipartite, then add edges carefully?
    # Or just random edges that don't form triangles.
    adj = [0] * n
    edges_to_try = [(i, j) for i in range(n) for j in range(i + 1, n)]
    rng.shuffle(edges_to_try)
    
    for u, v in edges_to_try:
        if (adj[u] & adj[v]) == 0:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            
    # Check if already valid
    alpha = _alpha_bnb_fast(adj, n, limit=10)
    if alpha < 10:
        return adj
        
    # Local Search Loop
    while time.time() - start_time < time_limit:
        alpha = _alpha_bnb_fast(adj, n, limit=10)
        if alpha < 10:
            return adj
            
        is_set = _find_independent_set(adj, n, limit=10)
        if is_set is None:
            return adj # Should be caught above
            
        # Try to add an edge within is_set
        modified = False
        # Shuffle to avoid bias
        is_set_copy = is_set[:]
        rng.shuffle(is_set_copy)
        
        for i in range(len(is_set_copy)):
            for j in range(i + 1, len(is_set_copy)):
                u, v = is_set_copy[i], is_set_copy[j]
                if (adj[u] & (1 << v)) == 0: # Not already connected
                    if (adj[u] & adj[v]) == 0: # Adding edge doesn't create triangle
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                        modified = True
                        break
            if modified:
                break
                
        if not modified:
            # Stuck: Restart with a new random graph
            adj = [0] * n
            edges_to_try = [(i, j) for i in range(n) for j in range(i + 1, n)]
            rng.shuffle(edges_to_try)
            for u, v in edges_to_try:
                if (adj[u] & adj[v]) == 0:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    
            # Quick check after restart
            if _alpha_bnb_fast(adj, n, limit=10) < 10:
                return adj

    return None

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Fallback: Andrasfai graph And(3) on 26 vertices
    # Actually, let's use a known valid graph if we can construct one quickly.
    # But we will rely on local search.
    
    best_result = None
    
    # Try Cayley constructions first (deterministic, fast)
    result = _try_cayley_constructions(deadline)
    if result:
        best_result = result
    
    # Try n=39 first (known to be possible) via local search if not found by Cayley
    if not best_result or best_result['n'] < 39:
        for seed in range(100):
            if time.time() > deadline:
                break
            adj = _local_search(39, seed, time_limit=0.5)
            if adj is not None:
                edges = []
                for u in range(39):
                    v_bits = adj[u]
                    while v_bits:
                        lsb = v_bits & -v_bits
                        v = lsb.bit_length() - 1
                        if u < v:
                            edges.append([u, v])
                        v_bits ^= lsb
                best_result = {'n': 39, 'edges': edges}
                break
            
    if best_result:
        # If we have time, try n=40
        if time.time() < deadline - 10:
            for seed in range(100):
                if time.time() > deadline - 5:
                    break
                adj = _local_search(40, seed, time_limit=0.5)
                if adj is not None:
                    edges = []
                    for u in range(40):
                        v_bits = adj[u]
                        while v_bits:
                            lsb = v_bits & -v_bits
                            v = lsb.bit_length() - 1
                            if u < v:
                                edges.append([u, v])
                            v_bits ^= lsb
                    best_result = {'n': 40, 'edges': edges}
                    break
    
    if best_result:
        return best_result
        
    # Final fallback: Return a valid smaller graph
    # Construct Andrásfai graph And(3) on 26 vertices
    n_fallback = 26
    edges_fallback = []
    for i in range(n_fallback):
        for j in range(i + 1, n_fallback):
            if (j - i) % n_fallback % 3 == 1:
                edges_fallback.append([i, j])
                
    return {'n': n_fallback, 'edges': edges_fallback}
