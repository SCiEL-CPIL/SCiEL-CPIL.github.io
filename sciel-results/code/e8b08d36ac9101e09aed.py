import random
import time
import itertools

N = 20
TOTAL_EDGES = N * (N - 1) // 2  # 190
DEADLINE = 140.0  # safety margin under 150s


def _edge_index(u, v):
    if u > v:
        u, v = v, u
    return u * N + v


def _init_adj():
    return [0] * N


def _add_edge(adj, u, v):
    adj[u] |= 1 << v
    adj[v] |= 1 << u


def _remove_edge(adj, u, v):
    adj[u] &= ~(1 << v)
    adj[v] &= ~(1 << u)


def _is_k33_free(adj):
    # Check: no three vertices have >=3 common neighbours
    for i in range(N):
        for j in range(i + 1, N):
            common = adj[i] & adj[j]
            # For each pair, if common has >=3 vertices that all share another common... 
            # Actually K3,3: need 3 vertices on one side, 3 on other, all 9 edges.
            # Equivalent: no set of 3 vertices with >=3 common neighbours.
            # So for each pair (i,j), count common neighbours; if >=3, then we need to check
            # whether any 3rd vertex k also has >=3 common neighbours with i and j.
            # Simpler: for each triple (i,j,k), check |N(i)&N(j)&N(k)| >= 3
            pass
    # Full triple check
    for i in range(N):
        for j in range(i + 1, N):
            ij = adj[i] & adj[j]
            for k in range(j + 1, N):
                if (ij & adj[k]).bit_count() >= 3:
                    return False
    return True


def _is_k33_free_fast(adj):
    """Check K3,3-free: no three vertices share >=3 common neighbours."""
    for i in range(N):
        ai = adj[i]
        for j in range(i + 1, N):
            ij = ai & adj[j]
            # iterate over common neighbours of i,j
            # if there are >=3 common neighbours, pick 3 and check if they share a 3rd...
            # Actually we need: exists k (k!=i,j) such that |N(i)&N(j)&N(k)| >= 3
            # For fixed i,j, we need to check all k. But that's O(n^3).
            # Optimization: if ij.bit_count() < 3, skip (can't have 3 common with any k)
            if ij.bit_count() < 3:
                continue
            for k in range(j + 1, N):
                if (ij & adj[k]).bit_count() >= 3:
                    return False
    return True


def _can_add(adj, u, v):
    """Check if adding edge (u,v) keeps graph K3,3-free."""
    # Adding edge (u,v) means u becomes a common neighbour of all pairs involving v, etc.
    # New common neighbours gained: 
    # For any pair (x,y) not involving u or v: common(x,y) unchanged.
    # Pairs involving u: (u, z) for z != v. New common = old common + {v} if v in N(z).
    #   i.e., common(u,z) gains v if v~z. So if v~z, then common(u,z) increases by 1.
    #   We need: for all z, |common(u,z)| < 3 after addition.
    #   Before: |common(u,z)|. After: |common(u,z)| + (1 if v~z else 0).
    #   So we need: for all z != v, if v~z then |common(u,z)| <= 1 (since +1 must keep <3, i.e., <=2 before... wait)
    #   Actually we need |common(u,z)| + [v~z] < 3 for all z != v.
    #   If v~z: need |common(u,z)| <= 1
    #   If not v~z: need |common(u,z)| <= 2 (unchanged, already satisfied if graph was valid)
    # Similarly for pairs (v, z).
    # Also pair (u,v) itself: common(u,v) = N(u)&N(v) before adding. After adding, u and v are neighbours of each other but that doesn't add to common neighbours of the pair (u,v) unless... wait, common neighbours of (u,v) are vertices w with w~u and w~v. Adding edge u-v doesn't change N(u)∩N(v) in terms of other vertices. So common(u,v) unchanged.
    
    au = adj[u]
    av = adj[v]
    
    # Check pairs (u, z) for z != v
    # For z where v~z (i.e., z in av), we need |N(u) & N(z)| <= 1
    # For z where not v~z, |N(u) & N(z)| <= 2 (already true)
    # So only need to check z in av (z != v, z != u)
    for z in range(N):
        if z == u or z == v:
            continue
        if (av >> z) & 1:  # v~z
            if (au & adj[z]).bit_count() > 1:
                return False
    
    # Check pairs (v, z) for z != u
    # For z where u~z (i.e., z in au), we need |N(v) & N(z)| <= 1
    for z in range(N):
        if z == u or z == v:
            continue
        if (au >> z) & 1:  # u~z
            if (av & adj[z]).bit_count() > 1:
                return False
    
    return True


def _local_search(adj, start_time, deadline):
    """Local search: try adding edges, then swaps."""
    n_edges = 0
    for i in range(N):
        n_edges += adj[i].bit_count()
    n_edges //= 2
    
    # Phase 1: greedy add
    improved = True
    while improved:
        improved = False
        for u in range(N):
            for v in range(u + 1, N):
                if time.time() > deadline:
                    return adj, n_edges
                if not ((adj[u] >> v) & 1):
                    if _can_add(adj, u, v):
                        _add_edge(adj, u, v)
                        n_edges += 1
                        improved = True
    
    # Phase 2: swap moves (remove one edge, add another)
    best_adj = [a for a in adj]
    best_edges = n_edges
    stall = 0
    while time.time() < deadline:
        improved = False
        # Try all single-edge additions again
        for u in range(N):
            for v in range(u + 1, N):
                if not ((adj[u] >> v) & 1):
                    if _can_add(adj, u, v):
                        _add_edge(adj, u, v)
                        n_edges += 1
                        improved = True
        if improved:
            stall = 0
            if n_edges > best_edges:
                best_adj = [a for a in adj]
                best_edges = n_edges
            continue
        
        # Try swaps: remove edge (a,b), add edge (c,d)
        found_swap = False
        for a in range(N):
            for b in range(a + 1, N):
                if not ((adj[a] >> b) & 1):
                    continue
                # Try removing (a,b)
                _remove_edge(adj, a, b)
                n_edges -= 1
                for c in range(N):
                    for d in range(c + 1, N):
                        if (c == a and d == b) or (c == b and d == a):
                            continue
                        if not ((adj[c] >> d) & 1):
                            if _can_add(adj, c, d):
                                _add_edge(adj, c, d)
                                n_edges += 1
                                # Check if this is valid overall (should be by construction)
                                found_swap = True
                                break
                        # Undo the add if we didn't take it
                        if not found_swap:
                            pass
                    if found_swap:
                        break
                if found_swap:
                    break
                # Undo the remove
                _add_edge(adj, a, b)
                n_edges += 1
            if found_swap:
                break
        if found_swap:
            stall = 0
            if n_edges > best_edges:
                best_adj = [a for a in adj]
                best_edges = n_edges
        else:
            stall += 1
            if stall > 5:
                break
    
    return best_adj, best_edges


def _greedy_construct():
    """Greedy construction with randomization."""
    adj = _init_adj()
    n_edges = 0
    edge_list = [(u, v) for u in range(N) for v in range(u + 1, N)]
    random.shuffle(edge_list)
    for u, v in edge_list:
        if _can_add(adj, u, v):
            _add_edge(adj, u, v)
            n_edges += 1
    return adj, n_edges


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    best_adj = None
    best_edges = 0
    
    # Multiple restarts with different random seeds
    for seed in range(10000):
        if time.time() > deadline:
            break
        random.seed(seed)
        adj, n_edges = _greedy_construct()
        
        # Local search
        adj, n_edges = _local_search(adj, start_time, deadline)
        
        if n_edges > best_edges:
            best_edges = n_edges
            best_adj = [a for a in adj]
        
        # Early stop if we've used too much time
        if time.time() > deadline - 2.0:
            break
    
    if best_adj is None:
        # Fallback: K2,18
        return [[a, b] for a in range(2) for b in range(2, 20)]
    
    # Convert to edge list
    edges = []
    for u in range(N):
        for v in range(u + 1, N):
            if (best_adj[u] >> v) & 1:
                edges.append([u, v])
    
    return edges
