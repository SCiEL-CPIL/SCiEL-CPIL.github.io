"""Combined program: fast bitset C6 check + greedy + local search for C6-free graphs on 40 vertices."""
# EVOLVE-BLOCK-START
import time
import random
import sys

DEADLINE = 145.0  # safety margin before 150s kill

def _is_c6_free(adj_bits, n=40):
    """Check if graph (represented as list of int bitmasks) contains no C6.
    A C6: v0-v1-v2-v3-v4-v5-v0 with all distinct.
    For each edge (v0,v1), for each v5 in N(v0)\\{v1}, v2 in N(v1)\\{v0,v5},
    v4 in N(v5)\\{v0,v2}, check if N(v2) & N(v4) has a vertex v3 distinct from all.
    """
    for v0 in range(n):
        n0 = adj_bits[v0]
        # v1 is a neighbor of v0
        v1_mask = n0
        while v1_mask:
            v1 = (v1_mask & (-v1_mask)).bit_length() - 1
            v1_mask &= v1_mask - 1
            if v1 <= v0:
                continue
            # v5 in N(v0) \\ {v1}
            v5_mask = n0 & ~(1 << v1)
            while v5_mask:
                v5 = (v5_mask & (-v5_mask)).bit_length() - 1
                v5_mask &= v5_mask - 1
                # v2 in N(v1) \\ {v0, v5}
                v2_mask = adj_bits[v1] & ~(1 << v0) & ~(1 << v5)
                while v2_mask:
                    v2 = (v2_mask & (-v2_mask)).bit_length() - 1
                    v2_mask &= v2_mask - 1
                    # v4 in N(v5) \\ {v0, v2}
                    v4_mask = adj_bits[v5] & ~(1 << v0) & ~(1 << v2)
                    while v4_mask:
                        v4 = (v4_mask & (-v4_mask)).bit_length() - 1
                        v4_mask &= v4_mask - 1
                        # v3 must be in N(v2) & N(v4), distinct from v0,v1,v2,v4,v5
                        forbidden = (1 << v0) | (1 << v1) | (1 << v2) | (1 << v4) | (1 << v5)
                        v3_mask = adj_bits[v2] & adj_bits[v4] & ~forbidden
                        if v3_mask:
                            return False
    return True

def _count_edges(adj_bits, n=40):
    count = 0
    for i in range(n):
        count += bin(adj_bits[i]).count('1')
    return count // 2

def _get_edges(adj_bits, n=40):
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if adj_bits[i] & (1 << j):
                edges.append([i, j])
    return edges

def _random_greedy(n=40, seed=None, max_time=5.0):
    """Randomized greedy: try edges in random order, add if C6-free."""
    if seed is not None:
        random.seed(seed)
    start_time = time.time()
    adj = [0] * n
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    random.shuffle(all_edges)
    
    for u, v in all_edges:
        if time.time() - start_time > max_time:
            break
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        if not _is_c6_free(adj, n):
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
    return adj, _count_edges(adj, n)

def _local_search(adj, n=40, max_time=20.0):
    """Try to add non-edges one at a time."""
    start_time = time.time()
    current = _count_edges(adj, n)
    
    while time.time() - start_time < max_time:
        improved = False
        non_edges = [(i, j) for i in range(n) for j in range(i+1, n) if not (adj[i] & (1 << j))]
        random.shuffle(non_edges)
        
        for u, v in non_edges:
            if time.time() - start_time > max_time:
                return adj, current
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            if _is_c6_free(adj, n):
                current += 1
                improved = True
            else:
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
        
        if not improved:
            break
    return adj, current

def _swap_search(adj, n=40, max_time=15.0):
    """Edge swaps: remove one edge, try adding a non-edge."""
    start_time = time.time()
    current = _count_edges(adj, n)
    best_adj = adj[:]
    best_count = current
    
    while time.time() - start_time < max_time:
        improved = False
        cur_edges = [(i, j) for i in range(n) for j in range(i+1, n) if adj[i] & (1 << j)]
        non_edges = [(i, j) for i in range(n) for j in range(i+1, n) if not (adj[i] & (1 << j))]
        random.shuffle(cur_edges)
        random.shuffle(non_edges)
        
        for ru, rv in cur_edges:
            if time.time() - start_time > max_time:
                break
            adj[ru] &= ~(1 << rv)
            adj[rv] &= ~(1 << ru)
            
            for au, av in non_edges:
                if time.time() - start_time > max_time:
                    break
                if adj[au] & (1 << av):
                    continue
                adj[au] |= (1 << av)
                adj[av] |= (1 << au)
                if _is_c6_free(adj, n):
                    nc = _count_edges(adj, n)
                    if nc > best_count:
                        best_count = nc
                        best_adj = adj[:]
                        improved = True
                    # keep the swap (same count) as it may help
                else:
                    adj[au] &= ~(1 << av)
                    adj[av] &= ~(1 << au)
            
            adj[ru] |= (1 << rv)
            adj[rv] |= (1 << ru)
        
        if not improved:
            break
    
    if best_count > current:
        return best_adj, best_count
    return adj, current

def _known_good(n=40):
    """8 disjoint K5s = 100 edges, C6-free."""
    adj = [0] * n
    for k in range(8):
        base = 5 * k
        for a in range(5):
            for b in range(a+1, 5):
                adj[base+a] |= (1 << (base+b))
                adj[base+b] |= (1 << (base+a))
    return adj

def construct():
    start_time = time.time()
    n = 40
    best_adj = None
    best_edges = 0
    
    # Phase 1: Start from 8xK5
    adj = _known_good(n)
    best_adj = adj[:]
    best_edges = _count_edges(adj, n)
    
    # Phase 2: Local search from 8xK5
    if time.time() - start_time < DEADLINE - 30:
        adj_ls, count_ls = _local_search(adj, n, max_time=min(15.0, DEADLINE - (time.time() - start_time) - 15))
        if count_ls > best_edges:
            best_edges = count_ls
            best_adj = adj_ls
    
    # Phase 3: Multiple random greedy restarts
    for attempt in range(30):
        remaining = DEADLINE - (time.time() - start_time) - 10
        if remaining < 2:
            break
        mt = min(2.0, remaining / 2)
        adj_g, count_g = _random_greedy(n, seed=100 + attempt, max_time=mt)
        if count_g > best_edges:
            best_edges = count_g
            best_adj = adj_g[:]
        # Quick local search on good greedy results
        if count_g >= best_edges - 2 and time.time() - start_time < DEADLINE - 10:
            adj_l, count_l = _local_search(adj_g, n, max_time=min(3.0, DEADLINE - (time.time() - start_time) - 5))
            if count_l > best_edges:
                best_edges = count_l
                best_adj = adj_l[:]
    
    # Phase 4: Swap search on best
    if time.time() - start_time < DEADLINE - 5:
        adj_sw, count_sw = _swap_search(best_adj, n, max_time=min(10.0, DEADLINE - (time.time() - start_time) - 2))
        if count_sw > best_edges:
            best_edges = count_sw
            best_adj = adj_sw
    
    # Phase 5: Final local search
    if time.time() - start_time < DEADLINE - 2:
        adj_f, count_f = _local_search(best_adj, n, max_time=min(5.0, DEADLINE - (time.time() - start_time) - 1))
        if count_f > best_edges:
            best_edges = count_f
            best_adj = adj_f
    
    # Final verification
    if best_adj is None:
        best_adj = _known_good(n)
    
    if not _is_c6_free(best_adj, n):
        best_adj = _known_good(n)
    
    return _get_edges(best_adj, n)
# EVOLVE-BLOCK-END
