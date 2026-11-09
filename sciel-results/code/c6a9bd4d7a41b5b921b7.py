import time
import random
import math
import sys

def _compute_max_book(adj, n=40):
    """Compute the maximum book size from adjacency bitmasks."""
    max_book = 0
    for u in range(n):
        for v in range(u + 1, n):
            if adj[u] & (1 << v):
                common = (adj[u] & adj[v]).bit_count()
                if common > max_book:
                    max_book = common
    return max_book

def _compute_max_book_edges(adj, n=40):
    """Compute the maximum book size and which edges achieve it."""
    max_book = 0
    for u in range(n):
        for v in range(u + 1, n):
            if adj[u] & (1 << v):
                common = (adj[u] & adj[v]).bit_count()
                if common > max_book:
                    max_book = common
    return max_book

def _local_search(adj, n, target_edges, time_limit, seed=42):
    """
    Local search: try edge swaps to reduce max book size.
    adj is a list of n bitmasks.
    Returns improved adj and score.
    """
    start_time = time.time()
    random.seed(seed)
    
    # Current score
    current_max = _compute_max_book(adj, n)
    best_max = current_max
    best_adj = adj[:]
    
    # Build edge set
    edge_set = set()
    for u in range(n):
        for v in range(u + 1, n):
            if adj[u] & (1 << v):
                edge_set.add((u, v))
    
    # All possible edges
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    # Temperature schedule
    T = 2.0
    alpha = 0.999
    Tend = 0.01
    
    while T > Tend and time.time() - start_time < time_limit:
        # Pick random edge to remove
        ru, rv = random.choice(list(edge_set))
        
        # Pick random non-edge to add
        ckey = None
        for _ in range(10):
            cu, cv = random.choice(all_edges)
            ckey = (min(cu, cv), max(cu, cv))
            if ckey not in edge_set:
                break
        if ckey is None or ckey in edge_set:
            continue
        cu, cv = ckey
        
        # Save affected edges
        # An edge (x,y) is affected if x or y is in {ru, rv, cu, cv}
        # and the edge exists (or will exist after swap)
        affected = set()
        affected.add((ru, rv))
        affected.add((cu, cv))
        
        # Edges incident to ru (other than rv)
        m = adj[ru] & ~(1 << rv)
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            affected.add((min(v, ru), max(v, ru)))
            m ^= lsb
        
        # Edges incident to rv (other than ru)
        m = adj[rv] & ~(1 << ru)
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            affected.add((min(v, rv), max(v, rv)))
            m ^= lsb
        
        # Edges incident to cu (other than cv)
        m = adj[cu] & ~(1 << cv)
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            affected.add((min(v, cu), max(v, cu)))
            m ^= lsb
        
        # Edges incident to cv (other than cu)
        m = adj[cv] & ~(1 << cu)
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            affected.add((min(v, cv), max(v, cv)))
            m ^= lsb
        
        # Save old book sizes for affected edges
        old_books = {}
        for k in affected:
            u, v = k
            if u < v and adj[u] & (1 << v):
                old_books[k] = (adj[u] & adj[v]).bit_count()
        
        # Apply swap
        adj[ru] &= ~(1 << rv)
        adj[rv] &= ~(1 << ru)
        edge_set.discard((ru, rv))
        
        adj[cu] |= (1 << cv)
        adj[cv] |= (1 << cu)
        edge_set.add((cu, cv))
        
        # Recompute max over affected edges
        new_max_affected = 0
        for k in affected:
            u, v = k
            if u < v and adj[u] & (1 << v):
                bs = (adj[u] & adj[v]).bit_count()
                if bs > new_max_affected:
                    new_max_affected = bs
        
        # Determine new global max
        # If the removed edge was the unique max, we need to recompute
        # Otherwise, new max = max(old_max, new_max_affected)
        # But if we removed a max edge and added a lower one, the max might drop
        # Simple approach: if old max > new_max_affected, check if any other edge has old max
        # For efficiency, just track: new global max = max(new_max_affected, max of unaffected)
        # We don't track max of unaffected, so we approximate:
        # If old max > new_max_affected, we assume the max stays (conservative)
        # If new_max_affected >= old max, new max = new_max_affected
        
        # Better: track a count of edges achieving the max
        # For now, use: new_max = max(old_max, new_max_affected) if we didn't remove a max edge
        # This is conservative (may miss improvements)
        
        # Let's be more precise: recompute max over all edges if the swap is promising
        # Actually, let's just compute new_max as max of all edge book sizes
        # But that's O(E) per iteration. With 401 edges, it's fine.
        
        # Actually, let's be smarter. The only edges whose book size changed are in 'affected'.
        # All other edges keep their old book size.
        # So new global max = max( max(old books of unaffected edges), new_max_affected )
        # We don't track max of unaffected edges explicitly.
        # Let's just recompute if new_max_affected < old_max (potential improvement)
        
        if new_max_affected < current_max:
            # Potential improvement: recompute global max
            new_global_max = 0
            for u in range(n):
                for v in range(u + 1, n):
                    if adj[u] & (1 << v):
                        bs = (adj[u] & adj[v]).bit_count()
                        if bs > new_global_max:
                            new_global_max = bs
            new_max = new_global_max
        else:
            new_max = max(current_max, new_max_affected)
        
        delta = new_max - current_max
        
        if delta <= 0 or random.random() < math.exp(-delta / T):
            current_max = new_max
            if new_max < best_max:
                best_max = new_max
                best_adj = adj[:]
        else:
            # Revert
            adj[cu] &= ~(1 << cv)
            adj[cv] &= ~(1 << cu)
            edge_set.discard((cu, cv))
            
            adj[ru] |= (1 << rv)
            adj[rv] |= (1 << ru)
            edge_set.add((ru, rv))
        
        T *= alpha
    
    return best_adj, best_max

def _construct_k2020_plus_one(n=40, target_edges=401, seed=42):
    """
    Construct K_{20,20} + 1 intra-partition edge, then local search.
    K_{20,20}: A={0..19}, B={20..39}, all cross edges.
    Add one edge within A, say (0,1).
    Then run local search to improve.
    """
    adj = [0] * n
    edge_set = set()
    
    # Add all K_{20,20} edges
    for a in range(20):
        for b in range(20, 40):
            adj[a] |= (1 << b)
            adj[b] |= (1 << a)
            edge_set.add((a, b))
    
    # Add one intra-A edge: (0, 1)
    adj[0] |= (1 << 1)
    adj[1] |= (1 << 0)
    edge_set.add((0, 1))
    
    # Run local search
    time_limit = 30.0
    adj, score = _local_search(adj, n, target_edges, time_limit, seed)
    
    # Convert back to edge list
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if adj[u] & (1 << v):
                edges.append([u, v])
    
    return edges, score

def _greedy_add_edges(adj, n, target_edges, seed=42):
    """
    Greedy: start from empty, add edges one by one minimizing max book increase.
    """
    random.seed(seed)
    edge_set = set()
    current_max = 0
    
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    random.shuffle(all_edges)
    
    for _ in range(target_edges):
        best_edge = None
        best_max_after = float('inf')
        
        # Sample candidates
        candidates = [e for e in all_edges if e not in edge_set]
        if len(candidates) > 300:
            candidates = random.sample(candidates, 300)
        
        for u, v in candidates:
            common = (adj[u] & adj[v]).bit_count()
            new_max = max(current_max, common)
            if new_max < best_max_after:
                best_max_after = new_max
                best_edge = (u, v)
                if best_max_after == current_max:
                    break
        
        if best_edge is None:
            best_edge = candidates[0]
        
        u, v = best_edge
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        edge_set.add((u, v))
        current_max = max(current_max, (adj[u] & adj[v]).bit_count())
    
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if adj[u] & (1 << v):
                edges.append([u, v])
    
    return edges, current_max

def construct():
    start_time = time.time()
    n = 40
    target_edges = 401
    deadline = start_time + 145.0  # Safety margin
    
    best_edges = None
    best_score = float('inf')
    
    # Strategy 1: K_{20,20} + 1 intra edge + local search
    # This is the most structured starting point
    if time.time() < deadline - 40.0:
        for seed in [42, 123, 456, 789, 1000, 2000, 3000, 4000, 5000, 6000]:
            if time.time() > deadline - 5.0:
                break
            time_left = deadline - time.time() - 5.0
            if time_left < 10.0:
                break
            
            edges, score = _construct_k2020_plus_one(n, target_edges, seed)
            if len(edges) != target_edges:
                continue
            if score < best_score:
                best_score = score
                best_edges = edges
    
    # Strategy 2: Greedy construction + local search
    if time.time() < deadline - 30.0:
        for seed in [42, 111, 222, 333, 444, 555]:
            if time.time() > deadline - 5.0:
                break
            time_left = deadline - time.time() - 5.0
            if time_left < 15.0:
                break
            
            adj = [0] * n
            edges, score = _greedy_add_edges(adj, n, target_edges, seed)
            if len(edges) != target_edges:
                continue
            
            # Run local search on this
            time_limit = min(10.0, deadline - time.time() - 5.0)
            if time_limit > 5.0:
                adj2, score2 = _local_search(adj, n, target_edges, time_limit, seed)
                if score2 < score:
                    score = score2
                    edges = []
                    for u in range(n):
                        for v in range(u + 1, n):
                            if adj2[u] & (1 << v):
                                edges.append([u, v])
            
            if score < best_score:
                best_score = score
                best_edges = edges
    
    # Strategy 3: Random + local search
    if time.time() < deadline - 20.0:
        for seed in [777, 888, 999, 1001, 1002]:
            if time.time() > deadline - 5.0:
                break
            time_left = deadline - time.time() - 5.0
            if time_left < 10.0:
                break
            
            adj = [0] * n
            random.seed(seed)
            all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
            random.shuffle(all_edges)
            for u, v in all_edges[:target_edges]:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
            
            time_limit = min(8.0, deadline - time.time() - 5.0)
            if time_limit > 3.0:
                adj2, score2 = _local_search(adj, n, target_edges, time_limit, seed)
                edges = []
                for u in range(n):
                    for v in range(u + 1, n):
                        if adj2[u] & (1 << v):
                            edges.append([u, v])
                
                if score2 < best_score:
                    best_score = score2
                    best_edges = edges
    
    if best_edges is None:
        # Fallback: K_{20,20} + (0,1)
        best_edges = []
        for a in range(20):
            for b in range(20, 40):
                best_edges.append([a, b])
        best_edges.append([0, 1])
    
    return best_edges
