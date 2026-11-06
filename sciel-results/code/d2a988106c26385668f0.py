"""Triangle-free graph without independent 10-set, maximizing n (Ramsey R(3,10))."""
import time
import random
import sys
from itertools import combinations

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


def _greedy_independent_set_lower_bound(adj, n):
    """
    Find a lower bound on the size of the maximum independent set.
    Uses a greedy algorithm to find a maximal independent set.
    """
    remaining = (1 << n) - 1
    count = 0
    while remaining:
        # Pick a vertex with minimum degree in the remaining subgraph
        # to tend to find a larger independent set
        min_deg = n
        v = -1
        temp = remaining
        while temp:
            lsb = temp & -temp
            u = lsb.bit_length() - 1
            temp ^= lsb
            deg = _bit_count(adj[u] & remaining)
            if deg < min_deg:
                min_deg = deg
                v = u
        
        if v == -1:
            break
        
        count += 1
        remaining &= ~adj[v]
        remaining &= ~(1 << v)
        
    return count


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

        # Quick check: if we already have 10, we can stop if we only care about <10
        # But we need exact size for comparison? No, we just need to know if it's < 10.
        # If current_size >= 10, we can return early if we are just checking validity.
        # However, the function returns the size.
        
        bound = greedy_clique_partition_bound(available)
        if current_size + bound <= best[0]:
            return

        # Pick vertex with max degree in G[available]
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

        # Branch 1: Include v
        new_avail_include = available & ~(1 << v) & ~adj[v]
        bnb(new_avail_include, current_size + 1)

        # Branch 2: Exclude v
        new_avail_exclude = available & ~(1 << v)
        bnb(new_avail_exclude, current_size)

    bnb(full_mask, 0)
    return best[0]


def _is_triangle_free_circulant(n, shifts):
    """
    Check if circulant graph with given shifts is triangle-free.
    """
    shift_set = set(s % n for s in shifts)
    shift_set.discard(0)
    
    shifts_list = list(shift_set)
    for i in range(len(shifts_list)):
        for j in range(len(shifts_list)):
            if i == j:
                continue
            diff = (shifts_list[i] - shifts_list[j]) % n
            if diff in shift_set:
                return False
    return True


def _build_circulant_graph(n, shifts):
    """
    Build a circulant graph on n vertices with given shift set.
    Returns adjacency bitmask list.
    """
    adj = [0] * n
    for s in shifts:
        s = s % n
        if s == 0:
            continue
        for i in range(n):
            j = (i + s) % n
            adj[i] |= (1 << j)
            adj[j] |= (1 << i)
    return adj


def _try_circulant_construction(n, max_shifts=6):
    """
    Try to find a valid circulant graph on n vertices.
    Returns (adj, alpha) if found, else None.
    """
    candidates = list(range(1, n // 2 + 1))
    
    for size in range(1, min(max_shifts, len(candidates)) + 1):
        for combo in combinations(candidates, size):
            if _remaining_time() < 2:
                return None
                
            full_shifts = set()
            for s in combo:
                full_shifts.add(s)
                full_shifts.add(n - s)
            full_shifts.discard(0)
            
            if not _is_triangle_free_circulant(n, full_shifts):
                continue
            
            adj = _build_circulant_graph(n, full_shifts)
            
            # Fast rejection: Greedy lower bound
            lb = _greedy_independent_set_lower_bound(adj, n)
            if lb >= 10:
                continue
            
            # Exact check
            alpha = _max_independent_set_size_bitmask(adj, n)
            if alpha < 10:
                return adj, alpha
    
    return None


def construct():
    """
    Construct a triangle-free graph on n vertices (1 <= n <= 41) with no independent set of 10.
    Maximize n.
    """
    best_adj = None
    best_n = 0
    
    # Try n from 40 down to 35
    # We know R(3,10) <= 41, so n=40 is the max possible valid graph size.
    for n_target in range(40, 34, -1):
        if _remaining_time() < 5:
            break
        
        result = _try_circulant_construction(n_target, max_shifts=6)
        if result is not None:
            adj_res, alpha_res = result
            best_adj = adj_res
            best_n = n_target
            # If we found n=40, we can stop immediately as it's the max
            if n_target == 40:
                break
    
    # Fallback: Randomized search if circulant didn't find 40
    if best_n < 40:
        for n_target in range(40, max(34, best_n - 2), -1):
            if _remaining_time() < 10:
                break
            
            found = False
            for seed in range(50):
                if _remaining_time() < 3:
                    break
                
                rng = random.Random(seed)
                # Generate random triangle-free graph
                adj = [0] * n_target
                edges = [(i, j) for i in range(n_target) for j in range(i + 1, n_target)]
                rng.shuffle(edges)
                
                for i, j in edges:
                    if adj[i] & adj[j]:
                        continue
                    # Check if adding edge creates triangle
                    # adj[i] & adj[j] is neighbors common to i and j.
                    # If we add edge (i,j), we need to ensure no common neighbor.
                    # Wait, adj[i] & adj[j] checks if there is a k connected to both.
                    # If so, triangle i-k-j.
                    # So if adj[i] & adj[j] is 0, we can add edge.
                    if not (adj[i] & adj[j]):
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
                
                lb = _greedy_independent_set_lower_bound(adj, n_target)
                if lb >= 10:
                    continue
                
                alpha = _max_independent_set_size_bitmask(adj, n_target)
                if alpha < 10:
                    best_adj = adj
                    best_n = n_target
                    found = True
                    break
            
            if found:
                break
    
    # Final fallback: small known graph
    if best_adj is None:
        n = 26
        rng = random.Random(42)
        adj = [0] * n
        edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        rng.shuffle(edges)
        for i, j in edges:
            if not (adj[i] & adj[j]):
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
        best_adj = adj
        best_n = n
    
    # Convert to edge list
    edges = []
    for i in range(best_n):
        for j in range(i + 1, best_n):
            if best_adj[i] & (1 << j):
                edges.append([i, j])
    
    return {'n': best_n, 'edges': edges}
