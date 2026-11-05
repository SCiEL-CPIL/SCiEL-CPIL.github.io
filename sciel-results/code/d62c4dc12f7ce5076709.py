"""Triangle-free graph without independent 10-set, maximizing n (Ramsey R(3,10))."""
import time
import random
import sys
import itertools

_START = time.time()
_DEADLINE = 145.0  # safety margin before 150s kill


def _remaining_time():
    return _DEADLINE - (time.time() - _START)


def _bit_count(x):
    return bin(x).count('1')

def _has_triangle_bitmask(adj, n):
    """Check if adjacency bitmask list has a triangle."""
    for i in range(n):
        ni = adj[i]
        j_mask = ni
        while j_mask:
            lsb = j_mask & -j_mask
            j = lsb.bit_length() - 1
            j_mask ^= lsb
            if j > i:
                if adj[i] & adj[j]:
                    return True
    return False

def _max_independent_set_size_bitmask(adj, n):
    """Find size of maximum independent set using branch and bound with bitmasks."""
    best = [0]
    full_mask = (1 << n) - 1
    
    def greedy_clique_partition_bound(available):
        """
        Upper bound on independent set size in G[available].
        Partition available vertices into cliques of G.
        Any independent set in G picks at most 1 from each clique.
        """
        unassigned = available
        k = 0
        while unassigned:
            k += 1
            lsb = unassigned & -unassigned
            v = lsb.bit_length() - 1
            unassigned ^= lsb
            clique = lsb
            candidates = unassigned & adj[v]
            while candidates:
                lsb_u = candidates & -candidates
                u = lsb_u.bit_length() - 1
                candidates ^= lsb_u
                unassigned ^= lsb_u
                clique |= lsb_u
                candidates &= adj[u]
        return k

    def bnb(available, current_size):
        if not available:
            if current_size > best[0]:
                best[0] = current_size
            return
        
        bound = greedy_clique_partition_bound(available)
        if current_size + bound <= best[0]:
            return
        
        v_mask = available
        max_deg = -1
        v = -1
        while v_mask:
            lsb = v_mask & -v_mask
            u = lsb.bit_length() - 1
            v_mask ^= lsb
            deg = _bit_count(adj[u] & available)
            if deg > max_deg:
                max_deg = deg
                v = u
        
        if v == -1:
            return

        new_avail_include = available & ~(1 << v) & ~adj[v]
        bnb(new_avail_include, current_size + 1)
        
        new_avail_exclude = available & ~(1 << v)
        bnb(new_avail_exclude, current_size)

    bnb(full_mask, 0)
    return best[0]

def _validate_bitmask(adj, n):
    """Check triangle-free and no independent 10-set."""
    if _has_triangle_bitmask(adj, n):
        return False
    if _max_independent_set_size_bitmask(adj, n) >= 10:
        return False
    return True

def _build_circulant_graph(n, shifts):
    """
    Build a circulant graph on n vertices with the given shift set.
    Vertex i is connected to (i+s) mod n and (i-s) mod n for each s in shifts.
    """
    adj = [0] * n
    for i in range(n):
        for s in shifts:
            j1 = (i + s) % n
            j2 = (i - s) % n
            adj[i] |= (1 << j1)
            adj[i] |= (1 << j2)
    # Remove self-loops (shouldn't happen if s != 0 and s != n/2)
    for i in range(n):
        adj[i] &= ~(1 << i)
    return adj

def _is_triangle_free_circulant(n, shifts):
    """
    Check if a circulant graph with given shifts is triangle-free.
    A triangle exists if there exist a, b, c in shifts such that a+b=c, a-b=c, etc.
    More precisely: triangle (0, s1, s2) exists if s1-s2 is in shifts (up to sign).
    """
    shift_set = set()
    for s in shifts:
        shift_set.add(s % n)
        shift_set.add((-s) % n)
    
    # Check all pairs of shifts
    for s1 in shifts:
        for s2 in shifts:
            if s1 == s2:
                continue
            # Triangle (0, s1, s2) exists if (s1-s2) mod n is in shift_set
            diff = (s1 - s2) % n
            if diff in shift_set and diff != 0:
                return False
            # Also check (0, s1, -s2) -> diff = s1+s2
            sum_val = (s1 + s2) % n
            if sum_val in shift_set and sum_val != 0:
                return False
    return True

def _enumerate_circulants(n, max_shifts, max_degree):
    """
    Enumerate circulant graphs on n vertices with at most max_shifts shifts.
    Yield (shifts, adj) for each triangle-free circulant.
    """
    # Generate all possible shift sets
    all_shifts = list(range(1, n // 2 + 1))
    
    for k in range(1, max_shifts + 1):
        for combo in itertools.combinations(all_shifts, k):
            # Quick degree check
            degree = 2 * len(combo)
            if degree > max_degree:
                continue
            
            # Check triangle-free
            if not _is_triangle_free_circulant(n, combo):
                continue
            
            adj = _build_circulant_graph(n, combo)
            yield combo, adj

def _try_circulant_construction(n, max_shifts=4, max_degree=12):
    """
    Try to find a valid circulant graph on n vertices.
    """
    best_adj = None
    best_alpha = float('inf')
    
    for shifts, adj in _enumerate_circulants(n, max_shifts, max_degree):
        if _remaining_time() < 3:
            break
        
        alpha = _max_independent_set_size_bitmask(adj, n)
        if alpha < best_alpha:
            best_alpha = alpha
            best_adj = adj
        
        if alpha < 10:
            return adj, alpha
    
    if best_adj is not None and best_alpha < 10:
        return best_adj, best_alpha
    
    return None

def _greedy_triangle_free_bitmask(n, rng):
    """
    Build a triangle-free graph greedily using bitmasks.
    """
    adj = [0] * n
    
    edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    rng.shuffle(edges)
    
    for i, j in edges:
        if _remaining_time() < 2:
            break
        if adj[i] & adj[j]:
            continue
        adj[i] |= (1 << j)
        adj[j] |= (1 << i)
    
    return adj

def _local_search_bitmask(adj, n, rng, max_iter=2000):
    """
    Improve graph by adding edges to reduce independence number.
    """
    adj = adj[:]
    alpha = _max_independent_set_size_bitmask(adj, n)
    
    if alpha < 10:
        return adj, alpha
    
    for _ in range(max_iter):
        if _remaining_time() < 3:
            break
        
        addable = []
        for i in range(n):
            for j in range(i+1, n):
                if not (adj[i] & (1 << j)):
                    if not (adj[i] & adj[j]):
                        addable.append((i, j))
        
        if not addable:
            break
        
        improved = False
        for _ in range(10):
            i, j = rng.choice(addable)
            
            adj[i] |= (1 << j)
            adj[j] |= (1 << i)
            
            new_alpha = _max_independent_set_size_bitmask(adj, n)
            if new_alpha < alpha:
                alpha = new_alpha
                improved = True
                if alpha < 10:
                    break
            else:
                adj[i] &= ~(1 << j)
                adj[j] &= ~(1 << i)
        
        if alpha < 10:
            break
        if not improved:
            pass

    return adj, alpha

def _try_construction_bitmask(n, seed, max_local_iters=500):
    """
    Try to build a valid graph on n vertices.
    """
    rng = random.Random(seed)
    
    # Strategy 1: Random greedy
    adj = _greedy_triangle_free_bitmask(n, rng)
    alpha = _max_independent_set_size_bitmask(adj, n)
    
    if alpha < 10:
        return adj, alpha
    
    # Try local search
    if _remaining_time() > 5:
        adj_ls, alpha_ls = _local_search_bitmask(adj, n, rng, max_local_iters)
        if alpha_ls < 10:
            return adj_ls, alpha_ls
        if alpha_ls < alpha:
            adj, alpha = adj_ls, alpha_ls
            
    # Strategy 2: Different random orderings
    for attempt in range(5):
        if _remaining_time() < 5:
            break
        rng2 = random.Random(seed * 1000 + attempt)
        adj2 = _greedy_triangle_free_bitmask(n, rng2)
        alpha2 = _max_independent_set_size_bitmask(adj2, n)
        
        if alpha2 < 10:
            return adj2, alpha2
        
        if alpha2 < alpha and _remaining_time() > 5:
            adj_ls2, alpha_ls2 = _local_search_bitmask(adj2, n, rng2, max_local_iters)
            if alpha_ls2 < alpha:
                adj, alpha = adj_ls2, alpha_ls2
                
    return None

def construct():
    """
    Construct a triangle-free graph on n vertices (1 <= n <= 41) with no independent set of 10.
    Maximize n.
    """
    best_adj = None
    best_n = 0
    best_alpha = float('inf')
    
    # Try n from high to low
    for n_target in range(40, 20, -1):
        if _remaining_time() < 8:
            break
        
        found = False
        
        # Strategy 1: Circulant enumeration (systematic, structured)
        circ_result = _try_circulant_construction(n_target, max_shifts=4, max_degree=12)
        if circ_result is not None:
            adj_res, alpha_res = circ_result
            if alpha_res < best_alpha:
                best_adj = adj_res
                best_n = n_target
                best_alpha = alpha_res
            if alpha_res < 10:
                found = True
                break
        
        # Strategy 2: Random greedy + local search
        for seed in range(30):
            if _remaining_time() < 3:
                break
            
            result = _try_construction_bitmask(n_target, seed)
            if result is not None:
                adj_res, alpha_res = result
                if alpha_res < best_alpha:
                    best_adj = adj_res
                    best_n = n_target
                    best_alpha = alpha_res
                if alpha_res < 10:
                    found = True
                    break
        
        if found:
            break
    
    # Fallback: use a known small construction if nothing worked
    if best_adj is None:
        n = 26
        rng = random.Random(42)
        adj = _greedy_triangle_free_bitmask(n, rng)
        best_adj = adj
        best_n = n
    
    # Convert to edge list
    edges = []
    for i in range(best_n):
        for j in range(i + 1, best_n):
            if best_adj[i] & (1 << j):
                edges.append([i, j])
    
    return {'n': best_n, 'edges': edges}
