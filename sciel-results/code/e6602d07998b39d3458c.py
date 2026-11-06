import time
import random
import sys

# Increase recursion limit for backtracking if needed, though iterative is better
sys.setrecursionlimit(10000)

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin
    
    # We aim for the known record of 35.
    # If we can't find 35, we might try 34, 33, etc., but 35 is the target.
    # The seed is 17. Any valid graph > 17 is an improvement.
    
    best_n = 0
    best_edges = []
    
    # Try to find a valid graph on n vertices
    # We prioritize larger n.
    # Start with 35. If that fails, try 34, etc.
    # However, 35 is the record, so it's the most likely to be found if at all.
    
    for target_n in [35, 34, 33, 32, 31, 30]:
        if time.time() > deadline - 10:
            break
            
        # Try multiple random restarts for this target_n
        found = False
        max_restarts = 5
        restart_time_budget = (deadline - time.time()) / max_restarts
        
        for r in range(max_restarts):
            if time.time() > deadline - 5:
                break
                
            adj = _random_graph(target_n, random.Random(r*1000 + 42))
            
            # WalkSAT search
            valid_graph = _walksat_search(adj, target_n, deadline - 5, random.Random(r*2000 + 99))
            
            if valid_graph:
                best_n = target_n
                best_edges = _edges_from_adj(valid_graph, target_n)
                found = True
                # If we found 35, we can stop early as it's the record
                if target_n == 35:
                    return {'n': best_n, 'edges': best_edges}
        
        # If we didn't find a graph for this n, we continue to smaller n
        # But we keep the best found so far (which might be from a previous loop iteration if we modified it, 
        # but here we only set best_n if found is True. Wait, if we fail 35, we try 34.
        # If we find 34, that's better than 17.
        # So we should keep the best valid graph found in any iteration.
        
    if best_n > 0:
        return {'n': best_n, 'edges': best_edges}
    
    # Fallback: Return a known valid small graph.
    # The seed returns 17. We can just return the seed's logic or a simpler valid graph.
    # A complete graph K3 is valid (n=3). 
    # But to ensure score >= 1.0, we should return the seed's result if possible.
    # The seed returns a 17-vertex graph.
    # Let's construct a simple valid graph, e.g., the Paley graph of order 17.
    return _paley17()

def _random_graph(n, rng):
    adj = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < 0.5:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    return adj

def _edges_from_adj(adj, n):
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if (adj[i] >> j) & 1:
                edges.append([i, j])
    return edges

def _paley17():
    squares = {x * x % 17 for x in range(1, 17)}
    return {'n': 17,
            'edges': [[i, j] for i in range(17) for j in range(i + 1, 17)
                      if (j - i) % 17 in squares]}

def _count_k4s(adj, n):
    """
    Count the number of K4 subgraphs.
    For each edge (u, v), let C be the set of common neighbors.
    The number of K4s containing (u, v) is the number of edges in the subgraph induced by C.
    Each K4 has 6 edges, so each K4 is counted 6 times.
    Total K4s = sum / 6.
    """
    total = 0
    for u in range(n):
        au = adj[u]
        # Iterate over neighbors v of u where v > u to avoid double counting edges?
        # No, iterating over all edges (u,v) with u < v is sufficient.
        for v in range(u + 1, n):
            if (au >> v) & 1:
                common = au & adj[v]
                # Count edges within common
                # We only need to count edges (x, y) in common where x < y
                # But since we are summing over all edges (u,v), we just count all edges in induced subgraph.
                # Let's just count pairs.
                c = common
                while c:
                    lsb = c & -c
                    x = lsb.bit_length() - 1
                    # Count neighbors of x within common, excluding x itself?
                    # Neighbors of x in common are (adj[x] & common) & ~(1<<x)
                    # But we need to count each edge once.
                    # If we iterate x, and count neighbors y in common with y > x, we count each edge once.
                    # However, `common` is a mask.
                    # Let's just iterate through bits of common.
                    others = common & ~(1 << x)
                    # Count bits in (adj[x] & others)
                    # But we must ensure we don't count edges twice?
                    # If we sum over all x in common, and for each x count neighbors y in common,
                    # we count each edge (x,y) twice (once for x, once for y).
                    # So total edges in induced subgraph = sum(popcount(adj[x] & common & ~(1<<x))) / 2.
                    # Wait, simpler:
                    # Just count all edges in the induced subgraph.
                    # We can do this by iterating x in common and counting neighbors in common with index > x.
                    pass
                
                # Efficient way to count edges in induced subgraph of mask `m`:
                # Sum over x in m of popcount(m & adj[x] & ~(1<<x)) / 2
                # Or simply:
                # For each x in m, let N_x = adj[x] & m.
                # Sum popcount(N_x) / 2.
                
                sum_deg = 0
                cm = common
                while cm:
                    lsb = cm & -cm
                    x = lsb.bit_length() - 1
                    sum_deg += bin(adj[x] & common).count('1')
                    cm ^= lsb
                edges_in_common = sum_deg // 2
                total += edges_in_common
    
    return total // 6

def _find_indep_set(adj, n, size):
    """
    Find an independent set of given size. Returns the set as a list of vertices, or None.
    Uses backtracking with heuristic ordering.
    """
    non_adj = [0] * n
    for i in range(n):
        non_adj[i] = ((1 << n) - 1) & ~(adj[i] | (1 << i))
    
    # Order vertices by degree in non_adj (ascending) to prune earlier
    # Actually, standard backtracking:
    
    def backtrack(candidates, current_set):
        if len(current_set) == size:
            return current_set[:]
        
        if not candidates:
            return None
            
        # Heuristic: pick vertex with fewest candidates in non_adj
        # To speed up, we can just pick the first one, but heuristic helps.
        # For n=35, size=6, simple backtracking is usually fast enough if graph is dense/sparse appropriately.
        
        # Let's pick the vertex with the smallest intersection with candidates
        best_v = -1
        min_cnt = n + 1
        c = candidates
        while c:
            lsb = c & -c
            v = lsb.bit_length() - 1
            cnt = bin(candidates & non_adj[v]).count('1')
            if cnt < min_cnt:
                min_cnt = cnt
                best_v = v
            c ^= lsb
        
        if best_v == -1:
            return None
            
        # Option 1: Include best_v
        new_candidates = candidates & non_adj[best_v]
        res = backtrack(new_candidates, current_set + [best_v])
        if res:
            return res
            
        # Option 2: Exclude best_v
        new_candidates = candidates & ~(1 << best_v)
        res = backtrack(new_candidates, current_set)
        if res:
            return res
            
        return None

    return backtrack((1 << n) - 1, [])

def _walksat_search(adj, n, deadline, rng):
    """
    Local search to remove K4s and Independent 6-sets.
    """
    steps = 0
    max_steps = 100000
    
    # Initial state
    k4_count = _count_k4s(adj, n)
    indep_set = _find_indep_set(adj, n, 6)
    
    # If already valid, return
    if k4_count == 0 and indep_set is None:
        return adj
        
    while time.time() < deadline and steps < max_steps:
        steps += 1
        
        # Determine move
        # If we have K4s, try to remove one.
        # If we have Indep 6, try to break one.
        
        # To find a specific K4, we need to locate one.
        # _count_k4s counts them. We can modify it to return one instance.
        
        k4_instance = _find_k4(adj, n)
        if k4_instance is None and indep_set is None:
            return adj
            
        # Decide which violation to fix
        # If both exist, pick randomly
        # If only one, pick that one
        
        if k4_instance and indep_set:
            if rng.random() < 0.5:
                target_type = 'k4'
            else:
                target_type = 'indep'
        elif k4_instance:
            target_type = 'k4'
        else:
            target_type = 'indep'
            
        if target_type == 'k4':
            # k4_instance is (u, v, w, x)
            # Pick a random edge in this K4 to remove
            u, v, w, x = k4_instance
            edges = [(u, v), (u, w), (u, x), (v, w), (v, x), (w, x)]
            a, b = rng.choice(edges)
            
            # Flip edge (a, b) - it must be present in K4
            adj[a] &= ~(1 << b)
            adj[b] &= ~(1 << a)
            
            # Check improvement
            new_k4_count = _count_k4s(adj, n)
            new_indep_set = _find_indep_set(adj, n, 6)
            
            # Acceptance criteria
            # We want to reduce total violations.
            # Violation metric: k4_count + (1 if indep_set else 0) * 10 (weight indep sets higher? or same?)
            # Let's use (k4_count, 1 if indep else 0)
            
            old_metric = (k4_count, 1 if indep_set else 0)
            new_metric = (new_k4_count, 1 if new_indep_set else 0)
            
            if new_metric < old_metric:
                k4_count = new_k4_count
                indep_set = new_indep_set
            else:
                # Revert
                adj[a] |= (1 << b)
                adj[b] |= (1 << a)
                # With small probability, accept neutral or slightly worse to escape local minima?
                # For now, strict improvement only, but add random kicks if stuck.
                if rng.random() < 0.05:
                    # Random kick: flip a random edge
                    a = rng.randrange(n)
                    b = rng.randrange(n)
                    if a == b: continue
                    if a > b: a, b = b, a
                    if (adj[a] >> b) & 1:
                        adj[a] &= ~(1 << b)
                        adj[b] &= ~(1 << a)
                    else:
                        adj[a] |= (1 << b)
                        adj[b] |= (1 << a)
                    k4_count = _count_k4s(adj, n)
                    indep_set = _find_indep_set(adj, n, 6)
        else:
            # target_type == 'indep'
            # indep_set is a list of 6 vertices
            # Pick a random pair to add an edge between
            u, v = rng.sample(indep_set, 2)
            
            # Add edge (u, v)
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            
            new_k4_count = _count_k4s(adj, n)
            new_indep_set = _find_indep_set(adj, n, 6)
            
            old_metric = (k4_count, 1 if indep_set else 0)
            new_metric = (new_k4_count, 1 if new_indep_set else 0)
            
            if new_metric < old_metric:
                k4_count = new_k4_count
                indep_set = new_indep_set
            else:
                # Revert
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
                if rng.random() < 0.05:
                    a = rng.randrange(n)
                    b = rng.randrange(n)
                    if a == b: continue
                    if a > b: a, b = b, a
                    if (adj[a] >> b) & 1:
                        adj[a] &= ~(1 << b)
                        adj[b] &= ~(1 << a)
                    else:
                        adj[a] |= (1 << b)
                        adj[b] |= (1 << a)
                    k4_count = _count_k4s(adj, n)
                    indep_set = _find_indep_set(adj, n, 6)
                    
    # Final check
    if _count_k4s(adj, n) == 0 and _find_indep_set(adj, n, 6) is None:
        return adj
    return None

def _find_k4(adj, n):
    """
    Find one K4. Returns (u, v, w, x) or None.
    """
    for u in range(n):
        au = adj[u]
        for v in range(u + 1, n):
            if (au >> v) & 1:
                common = au & adj[v]
                # Iterate over pairs in common
                c1 = common
                while c1:
                    lsb1 = c1 & -c1
                    w = lsb1.bit_length() - 1
                    c2 = common & ~(1 << w)
                    while c2:
                        lsb2 = c2 & -c2
                        x = lsb2.bit_length() - 1
                        # Check if w and x are connected
                        if (adj[w] >> x) & 1:
                            return (u, v, w, x)
                        c2 ^= lsb2
                    c1 ^= lsb1
    return None
