import random
import time
import sys

sys.setrecursionlimit(10000)

def _time_up(deadline):
    return time.time() > deadline

def _check_triangle_free(adj, n):
    for i in range(n):
        for j in adj[i]:
            if j < i:
                continue
            if adj[i] & adj[j]:
                return False
    return True

def _exact_alpha(adj, n, deadline):
    """
    Returns the size of the maximum independent set.
    Uses branch and bound with bitmasks.
    """
    masks = [0] * n
    for i in range(n):
        m = 0
        for j in adj[i]:
            m |= (1 << j)
        masks[i] = m
        
    best = [0]
    
    # Precompute popcount for all possible masks? No, n=40 is too big for array.
    # Use bin(x).count('1') or int.bit_count() (Python 3.10+)
    # Let's assume Python 3.10+ for bit_count, otherwise fallback.
    try:
        (1).bit_count()
        def popcount(x):
            return x.bit_count()
    except AttributeError:
        def popcount(x):
            return bin(x).count('1')

    def dfs(candidates, current_size):
        if _time_up(deadline):
            return
        if not candidates:
            if current_size > best[0]:
                best[0] = current_size
            return
            
        # Upper bound
        if current_size + popcount(candidates) <= best[0]:
            return
            
        # Choose vertex with max degree in candidates
        best_v = -1
        max_deg = -1
        temp_cands = candidates
        while temp_cands:
            lsb = temp_cands & -temp_cands
            v = lsb.bit_length() - 1
            deg = popcount(candidates & masks[v])
            if deg > max_deg:
                max_deg = deg
                best_v = v
            temp_cands ^= lsb
            
        v = best_v
        if v == -1:
            return
            
        bit_v = 1 << v
        
        # Option 1: Include v
        new_cands_1 = candidates & ~masks[v] & ~bit_v
        dfs(new_cands_1, current_size + 1)
        
        # Option 2: Exclude v
        new_cands_2 = candidates & ~bit_v
        dfs(new_cands_2, current_size)
        
    all_mask = (1 << n) - 1
    dfs(all_mask, 0)
    
    return best[0]

def _make_triangle_free(adj, n, deadline):
    """
    Removes edges to ensure the graph is triangle-free.
    """
    changed = True
    while changed and not _time_up(deadline):
        changed = False
        for i in range(n):
            if _time_up(deadline):
                break
            # Iterate over a copy of the list since we modify the set
            for j in list(adj[i]):
                if j < i:
                    continue
                common = adj[i] & adj[j]
                if common:
                    adj[i].remove(j)
                    adj[j].remove(i)
                    changed = True
        if _time_up(deadline):
            break

def _make_maximal_triangle_free(adj, n, deadline):
    """
    Adds edges to make the graph maximal triangle-free.
    This tends to reduce the independence number.
    """
    added = True
    while added and not _time_up(deadline):
        added = False
        for i in range(n):
            if _time_up(deadline):
                break
            for j in range(i + 1, n):
                if j not in adj[i]:
                    # Check if adding (i,j) creates a triangle
                    if not (adj[i] & adj[j]):
                        adj[i].add(j)
                        adj[j].add(i)
                        added = True
        if _time_up(deadline):
            break

def construct():
    deadline = time.time() + 115.0 # Safety margin
    
    best_n = 0
    best_edges = []
    
    # Try to find a graph on 40 vertices first?
    # Then 39, 38, etc.
    for target_n in [40, 39, 38, 37, 36, 35, 34, 33, 32, 31, 30]:
        if _time_up(deadline):
            break
            
        found = False
        restart = 0
        while not _time_up(deadline):
            restart += 1
            n = target_n
            adj = [set() for _ in range(n)]
            
            # Strategy 1: Random Bipartite
            # Strategy 2: Random Graph + Triangle Removal
            # Alternate or pick randomly
            
            if restart % 2 == 0:
                # Random Bipartite
                mid = n // 2
                # Random split
                # Actually, just fix split for simplicity, or randomize
                # Let's fix split to 0..mid-1 and mid..n-1
                # Random edges between the two sets
                p = random.uniform(0.4, 0.8)
                for i in range(mid):
                    for j in range(mid, n):
                        if random.random() < p:
                            adj[i].add(j)
                            adj[j].add(i)
            else:
                # Random General Graph
                p = random.uniform(0.3, 0.6)
                for i in range(n):
                    for j in range(i + 1, n):
                        if random.random() < p:
                            adj[i].add(j)
                            adj[j].add(i)
                
                # Make triangle-free
                _make_triangle_free(adj, n, deadline)
                
                if _time_up(deadline):
                    break

            # Verify triangle-free (safety)
            if not _check_triangle_free(adj, n):
                continue

            # Make maximal triangle-free
            _make_maximal_triangle_free(adj, n, deadline)
            
            if _time_up(deadline):
                break

            # Check independence number
            # Heuristic check first to avoid expensive exact check on obvious failures
            # Greedy independent set
            best_heur = 0
            for _ in range(5):
                vertices = list(range(n))
                random.shuffle(vertices)
                ind_set = set()
                for v in vertices:
                    if not (ind_set & adj[v]):
                        ind_set.add(v)
                if len(ind_set) > best_heur:
                    best_heur = len(ind_set)
            
            if best_heur >= 10:
                # Likely invalid, but could be a bad heuristic.
                # If best_heur is very high (e.g., > 12), skip exact.
                # If close to 10, do exact.
                if best_heur > 12:
                    continue
                # else fall through to exact check

            # Exact check
            alpha = _exact_alpha(adj, n, deadline)
            
            if _time_up(deadline):
                break

            if alpha < 10:
                edges = []
                for i in range(n):
                    for j in adj[i]:
                        if i < j:
                            edges.append([i, j])
                if n > best_n:
                    best_n = n
                    best_edges = edges
                found = True
                break
                
        if found:
            break
            
    if best_n > 0:
        return {'n': best_n, 'edges': best_edges}
    else:
        # Fallback: Construct a known valid small graph
        # C5 blowup with k=4 gives n=20, alpha=8
        # Or just return a simple valid graph
        # Let's construct a C5 blowup with k=4
        n = 20
        edges = []
        # Vertices 0..4 are group 0, 5..9 group 1, etc.
        # C5 edges: (0,1), (1,2), (2,3), (3,4), (4,0)
        # Blowup: connect all vertices in group i to all in group j if (i,j) is edge in C5
        for i in range(5):
            for j in range(5):
                if (i == (j + 1) % 5) or (j == (i + 1) % 5):
                    for u in range(i * 4, (i + 1) * 4):
                        for v in range(j * 4, (j + 1) * 4):
                            edges.append([u, v])
        return {'n': n, 'edges': edges}
