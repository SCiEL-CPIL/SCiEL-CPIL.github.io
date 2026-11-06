# EVOLVE-BLOCK-START
import time
import numpy as np
import itertools
from math import comb

def _build_adj(n, orientation):
    adj = np.zeros((n, n), dtype=np.uint8)
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            bit = orientation[idx]
            idx += 1
            if bit == 0:
                adj[i, j] = 1
            else:
                adj[j, i] = 1
    return adj

def _is_valid(n, adj):
    """Check if every 4-set has a common dominator. Returns True if valid."""
    # For each 4-set, check if any vertex dominates all 4
    # A vertex v dominates {a,b,c,d} if adj[v,a] and adj[v,b] and adj[v,c] and adj[v,d]
    # We need at least one such v for every 4-set.
    
    # Precompute in-neighborhoods: in_neigh[x] = set of v such that v->x
    # dominators({a,b,c,d}) = in_neigh[a] & in_neigh[b] & in_neigh[c] & in_neigh[d]
    # Use bitsets for speed
    
    n_words = (n + 63) // 64
    in_neigh = [np.zeros(n_words, dtype=np.uint64) for _ in range(n)]
    
    for i in range(n):
        for j in range(n):
            if i != j and adj[i, j] == 1:
                in_neigh[j][i // 64] |= (1 << (i % 64))
    
    for quad in itertools.combinations(range(n), 4):
        a, b, c, d = quad
        # AND the four in-neighborhood bitsets
        result = in_neigh[a].copy()
        result &= in_neigh[b]
        result &= in_neigh[c]
        result &= in_neigh[d]
        if not np.any(result):
            return False
    return True

def _is_valid_sampled(n, adj, max_checks=2000, seed=42):
    """Sampled validity check for speed."""
    rng = np.random.default_rng(seed)
    n_words = (n + 63) // 64
    in_neigh = [np.zeros(n_words, dtype=np.uint64) for _ in range(n)]
    
    for i in range(n):
        for j in range(n):
            if i != j and adj[i, j] == 1:
                in_neigh[j][i // 64] |= (1 << (i % 64))
    
    for _ in range(max_checks):
        quad = rng.choice(n, size=4, replace=False)
        a, b, c, d = quad
        result = in_neigh[a].copy()
        result &= in_neigh[b]
        result &= in_neigh[c]
        result &= in_neigh[d]
        if not np.any(result):
            return False
    return True

def _local_search(n, orientation_init, max_time=10.0, seed=42):
    """
    Local search from a starting orientation.
    Returns (orientation, bad_count) if found valid, else None.
    """
    start_time = time.time()
    rng = np.random.default_rng(seed)
    
    orientation = orientation_init[:]
    adj = _build_adj(n, orientation)
    
    # Precompute
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    pair_to_idx = {}
    for idx, (i, j) in enumerate(pairs):
        pair_to_idx[(i, j)] = idx
        pair_to_idx[(j, i)] = idx
    
    # Build D sets
    D = [set() for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j and adj[i, j] == 1:
                D[i].add(j)
    
    # Build cover counts
    cover_counts = {}
    for v in range(n):
        if len(D[v]) >= 4:
            for S in itertools.combinations(D[v], 4):
                if S in cover_counts:
                    cover_counts[S] += 1
                else:
                    cover_counts[S] = 1
    
    total_4sets = comb(n, 4)
    bad_count = total_4sets - len(cover_counts)
    
    if bad_count == 0:
        return orientation, 0
    
    current_orientation = orientation[:]
    current_adj = adj.copy()
    current_D = [set(d) for d in D]
    
    # Simulated annealing style
    temperature = 5.0
    cooling = 0.99
    
    while time.time() - start_time < max_time:
        i, j = rng.integers(0, n, size=2)
        if i == j:
            continue
        
        if current_adj[i, j] == 1:
            u, v = i, j
        else:
            u, v = j, i
        
        # u dominates v, flip to v dominates u
        lost_sets = []
        D_u_without_v = current_D[u] - {v}
        if len(D_u_without_v) >= 3:
            for combo in itertools.combinations(D_u_without_v, 3):
                S = (v,) + combo
                lost_sets.append(S)
        
        gained_sets = []
        if len(current_D[v]) >= 3:
            for combo in itertools.combinations(current_D[v], 3):
                S = (u,) + combo
                gained_sets.append(S)
        
        delta_bad = 0
        
        for S in lost_sets:
            if S in cover_counts:
                old_count = cover_counts[S]
                if old_count == 1:
                    delta_bad += 1
                    del cover_counts[S]
                else:
                    cover_counts[S] = old_count - 1
        
        for S in gained_sets:
            if S in cover_counts:
                cover_counts[S] += 1
            else:
                cover_counts[S] = 1
                delta_bad -= 1
        
        # Accept if improvement, or with probability exp(-delta/T) if worsening
        if delta_bad <= 0:
            accept = True
        else:
            accept = rng.random() < np.exp(-delta_bad / temperature)
        
        if accept:
            current_adj[u, v] = 0
            current_adj[v, u] = 1
            current_D[u].remove(v)
            current_D[v].add(u)
            
            idx = pair_to_idx[(u, v)]
            if u < v:
                current_orientation[idx] = 1
            else:
                current_orientation[idx] = 0
            
            if len(cover_counts) == total_4sets:
                return current_orientation, 0
        else:
            # Reject: undo the changes
            for S in lost_sets:
                if S in cover_counts:
                    cover_counts[S] += 1
                else:
                    cover_counts[S] = 1
            for S in gained_sets:
                if S in cover_counts:
                    cover_counts[S] -= 1
                    if cover_counts[S] == 0:
                        del cover_counts[S]
        
        temperature *= cooling
    
    return None, bad_count

def _paleyn_tournament(p):
    """Paley tournament on p vertices."""
    squares = set()
    for x in range(1, p):
        squares.add(x * x % p)
    
    orientation = []
    for i in range(p):
        for j in range(i + 1, p):
            diff = (j - i) % p
            if diff in squares:
                orientation.append(0)
            else:
                orientation.append(1)
    return p, orientation

def _try_remove_vertex(n, orientation, adj):
    """
    Try removing each vertex and check if the remaining n-1 tournament is valid.
    Returns (n-1, new_orientation) if found, else None.
    """
    for removed in range(n):
        # Build new orientation for n-1 vertices
        new_n = n - 1
        new_orientation = []
        idx = 0
        for i in range(n):
            if i == removed:
                continue
            for j in range(i + 1, n):
                if j == removed:
                    continue
                new_orientation.append(orientation[idx])
                idx += 1
        
        new_adj = _build_adj(new_n, new_orientation)
        
        # Check validity (use sampled for speed)
        if _is_valid_sampled(new_n, new_adj, max_checks=1000, seed=42):
            return new_n, new_orientation
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Strategy:
    # 1. Start with Paley 67 (known valid)
    # 2. Try local search from Paley 67 to see if we can find improvements
    # 3. Try vertex removal to reduce n
    # 4. Try local search for smaller n from random starts
    # 5. Return the smallest valid n found
    
    best_n = 91
    best_orientation = None
    
    # Step 1: Get Paley 67
    n67, orient67 = _paleyn_tournament(67)
    adj67 = _build_adj(n67, orient67)
    
    # Verify Paley 67 is valid
    if _is_valid_sampled(n67, adj67, max_checks=5000, seed=42):
        best_n = n67
        best_orientation = orient67[:]
    else:
        # Fallback to Paley 83
        n83, orient83 = _paleyn_tournament(83)
        best_n = n83
        best_orientation = orient83[:]
    
    # Step 2: Try to reduce n via vertex removal
    if time.time() < deadline - 20:
        result = _try_remove_vertex(best_n, best_orientation, _build_adj(best_n, best_orientation))
        if result is not None:
            new_n, new_orient = result
            # Verify more thoroughly
            new_adj = _build_adj(new_n, new_orient)
            if _is_valid_sampled(new_n, new_adj, max_checks=3000, seed=123):
                best_n = new_n
                best_orientation = new_orient
    
    # Step 3: Try local search from Paley 67 for more reductions
    if time.time() < deadline - 15:
        result, bad = _local_search(n67, orient67, max_time=10.0, seed=42)
        if result is not None:
            # Found valid, try to reduce
            adj_res = _build_adj(n67, result)
            rem = _try_remove_vertex(n67, result, adj_res)
            if rem is not None:
                new_n, new_orient = rem
                new_adj = _build_adj(new_n, new_orient)
                if _is_valid_sampled(new_n, new_adj, max_checks=2000, seed=777):
                    if new_n < best_n:
                        best_n = new_n
                        best_orientation = new_orient
    
    # Step 4: Try local search for smaller n from random starts
    # Focus on n values between 31 and best_n - 1
    if time.time() < deadline - 10:
        for n in range(31, best_n):
            if time.time() > deadline - 5:
                break
            
            # Allocate time based on how close to best_n
            remaining = deadline - time.time() - 5
            attempts = 3
            time_per = remaining / (attempts * (best_n - n)) if (best_n - n) > 0 else 2.0
            time_per = max(1.0, min(3.0, time_per))
            
            for attempt in range(attempts):
                if time.time() > deadline - 5:
                    break
                
                # Random init
                rng = np.random.default_rng(42 + n * 100 + attempt)
                orient = [int(rng.integers(0, 2)) for _ in range(n * (n - 1) // 2)]
                
                result, bad = _local_search(n, orient, max_time=time_per, seed=42 + n + attempt)
                if result is not None:
                    adj_res = _build_adj(n, result)
                    if _is_valid_sampled(n, adj_res, max_checks=1000, seed=42):
                        if n < best_n:
                            best_n = n
                            best_orientation = result
                        break
    
    # Final safety check
    if best_orientation is None:
        n, orientation = _paleyn_tournament(83)
        best_n = n
        best_orientation = orientation
    
    return {'n': best_n, 'orientation': best_orientation}
# EVOLVE-BLOCK-END
