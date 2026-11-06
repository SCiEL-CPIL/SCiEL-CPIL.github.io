import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # 20s safety margin

    def has_k4(adj_bits, n):
        """Check for K4 using bitmasks. adj_bits[i] is an int bitmask of neighbors of i."""
        for i in range(n):
            for j in range(i + 1, n):
                if not (adj_bits[i] >> j) & 1:
                    continue
                # common neighbors of i and j
                common = adj_bits[i] & adj_bits[j]
                # iterate over common neighbors k > j
                k = (common >> (j + 1)) & ((1 << (n - j - 1)) - 1)
                while k:
                    low = k & (-k)
                    ki = low.bit_length() - 1
                    k_idx = j + 1 + ki
                    k &= k - 1
                    # common neighbors of i, j, k that are > k_idx
                    common_ijk = common & adj_bits[k_idx]
                    rem = (common_ijk >> (k_idx + 1)) & ((1 << (n - k_idx - 1)) - 1)
                    if rem:
                        return True
        return False

    def has_independent_6(adj_bits, n):
        """Check for independent set of size 6 using complement bitmasks."""
        # complement: non-neighbor bitmask
        comp = [0] * n
        for i in range(n):
            comp[i] = ((1 << n) - 1) ^ (1 << i) ^ adj_bits[i]
        
        # Find if there's a clique of size 6 in complement = independent set of 6 in G
        # Use recursive backtracking with pruning
        def find_clique_6(start, current, size):
            if size == 6:
                return True
            # Pruning: not enough remaining vertices
            if n - start < 6 - size:
                return False
            for v in range(start, n):
                if v in current:
                    continue
                # Check if v is connected to all in current in complement
                ok = True
                for u in current:
                    if not (comp[v] >> u) & 1:
                        ok = False
                        break
                if ok:
                    current.add(v)
                    if find_clique_6(v + 1, current, size + 1):
                        return True
                    current.remove(v)
            return False
        
        return find_clique_6(0, set(), 0)

    def is_valid(adj_bits, n):
        return not has_k4(adj_bits, n) and not has_independent_6(adj_bits, n)

    # Strategy: Build graph vertex by vertex using backtracking CSP
    # For each new vertex, its neighbor set must:
    # 1. Not contain a triangle (among existing vertices)
    # 2. Intersect every independent 5-set of existing graph
    
    # We'll try multiple random seeds and keep the best valid graph
    
    best_n = 1
    best_adj_bits = [0]
    best_edges = []
    
    def build_graph(target_n, seed):
        """Try to build a valid graph on target_n vertices."""
        random.seed(seed)
        adj_bits = [0] * target_n
        adj_sets = [set() for _ in range(target_n)]
        
        # Vertices are added in order 0, 1, 2, ...
        # For vertex i, choose its neighbors among {0, ..., i-1}
        
        def get_triangles(sets, n):
            """Get all triangles in graph on vertices 0..n-1."""
            tris = []
            for i in range(n):
                for j in sets[i]:
                    if j <= i:
                        continue
                    common = sets[i] & sets[j]
                    for k in common:
                        if k > j:
                            tris.append((i, j, k))
            return tris
        
        def get_indep_5(sets, n):
            """Get all independent sets of size 5 in graph on vertices 0..n-1."""
            result = []
            def gen(start, current):
                if len(current) == 5:
                    result.append(frozenset(current))
                    return
                remaining = n - start
                if len(current) + remaining < 5:
                    return
                for v in range(start, n):
                    ok = True
                    for u in current:
                        if v in sets[u]:
                            ok = False
                            break
                    if ok:
                        gen(v + 1, current + [v])
            gen(0, [])
            return result
        
        def can_add_vertex(i, neighbor_set, sets, n):
            """Check if adding vertex i with neighbors in neighbor_set keeps graph valid."""
            if i == 0:
                return True
            
            # Check 1: neighbor_set must be triangle-free
            # (no triangle entirely within neighbor_set)
            for tri in get_triangles(sets, i):
                if tri[0] in neighbor_set and tri[1] in neighbor_set and tri[2] in neighbor_set:
                    return False
            
            # Check 2: neighbor_set must intersect every independent 5-set
            # (so that adding vertex i doesn't create an independent 6-set)
            for i5 in get_indep_5(sets, i):
                if not (neighbor_set & i5):
                    return False
            
            # Check 3: Adding i with neighbor_set must not create a K4
            # A K4 involving i means: i connected to 3 vertices that form a triangle
            # i.e., there exist a,b,c in neighbor_set that form a triangle
            # This is the same as check 1! So check 1 covers K4.
            
            # Check 4: Adding i must not create independent 6-set not involving i
            # That's already guaranteed since we didn't change existing edges.
            
            return True
        
        # Build vertex by vertex
        for i in range(target_n):
            if time.time() > deadline - 5.0:
                return None
            
            if i == 0:
                continue
            
            # Generate candidate neighbor sets
            # Heuristic: neighbor set should be "large enough" to hit all indep-5-sets
            # but "small enough" to be triangle-free
            
            all_prev = set(range(i))
            
            # Get constraints
            triangles = get_triangles(adj_sets, i)
            indep_5s = get_indep_5(adj_sets, i)
            
            # Find all valid neighbor sets via backtracking
            # Start with all_prev and remove vertices to break triangles,
            # then verify hitting set property
            
            def find_valid_neighbor_set(i, all_prev, triangles, indep_5s, adj_sets):
                """Find a valid neighbor set for vertex i using backtracking."""
                # Start with all_prev
                S = set(all_prev)
                
                # Greedily remove vertices to break triangles
                for tri in triangles:
                    if tri[0] in S and tri[1] in S and tri[2] in S:
                        # Remove one random vertex from the triangle
                        v = random.choice(tri)
                        S.discard(v)
                
                # Now check hitting set property
                # If S doesn't hit some indep_5, we need to add vertices back
                # but adding might create triangles
                
                # Try: start with S, for each missed indep_5, try to add a vertex
                for i5 in indep_5s:
                    if not (S & i5):
                        # Need to add at least one from i5
                        added = False
                        for v in list(i5):
                            if v not in S:
                                # Check if adding v creates a triangle
                                creates_tri = False
                                for u in adj_sets[v]:
                                    if u in S:
                                        for w in adj_sets[u]:
                                            if w in S and w != v and v in adj_sets[w]:
                                                creates_tri = True
                                                break
                                        if creates_tri:
                                            break
                                if not creates_tri:
                                    S.add(v)
                                    added = True
                                    break
                        if not added:
                            return None
                
                # Final verification
                for tri in triangles:
                    if tri[0] in S and tri[1] in S and tri[2] in S:
                        return None
                for i5 in indep_5s:
                    if not (S & i5):
                        return None
                
                return S
            
            # Try multiple random attempts
            found = False
            for attempt in range(20):
                if time.time() > deadline - 5.0:
                    return None
                
                ns = find_valid_neighbor_set(i, all_prev, triangles, indep_5s, adj_sets)
                if ns is not None:
                    # Add vertex i with neighbor set ns
                    for v in ns:
                        adj_bits[i] |= (1 << v)
                        adj_bits[v] |= (1 << i)
                        adj_sets[i].add(v)
                        adj_sets[v].add(i)
                    found = True
                    break
            
            if not found:
                # Try a different heuristic: random subset of all_prev
                for attempt in range(50):
                    if time.time() > deadline - 5.0:
                        return None
                    # Random subset of size ~ i/2
                    size = random.randint(max(1, i // 3), min(i - 1, 2 * i // 3))
                    ns = random.sample(list(all_prev), size)
                    ns = set(ns)
                    
                    if can_add_vertex(i, ns, adj_sets, i):
                        for v in ns:
                            adj_bits[i] |= (1 << v)
                            adj_bits[v] |= (1 << i)
                            adj_sets[i].add(v)
                            adj_sets[v].add(i)
                        found = True
                        break
            
            if not found:
                return None
        
        # Verify the complete graph
        if is_valid(adj_bits, target_n):
            return adj_bits
        return None
    
    # Try to build graphs of increasing size
    for target in range(35, 28, -1):
        if time.time() > deadline - 10.0:
            break
        
        for seed in range(100):
            if time.time() > deadline - 10.0:
                break
            
            result = build_graph(target, seed)
            if result is not None:
                # Build edge list
                edges = []
                for i in range(target):
                    for j in range(i + 1, target):
                        if (result[i] >> j) & 1:
                            edges.append([i, j])
                
                if target > best_n:
                    best_n = target
                    best_adj_bits = result
                    best_edges = edges
                
                # Found a valid graph of this size, try larger
                break
    
    # Fallback: use Paley graph on 29 vertices if nothing better found
    if best_n < 29:
        q = 29
        n = q
        squares = set(x * x % q for x in range(1, q))
        adj_bits = [0] * n
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if (j - i) % q in squares:
                    adj_bits[i] |= (1 << j)
                    adj_bits[j] |= (1 << i)
                    edges.append([i, j])
        
        if is_valid(adj_bits, n):
            best_n = n
            best_adj_bits = adj_bits
            best_edges = edges
    
    # Final verification
    if not is_valid(best_adj_bits, best_n):
        # Shouldn't happen, but fallback to smallest valid
        best_n = 1
        best_adj_bits = [0]
        best_edges = []
    
    return {'n': best_n, 'edges': best_edges}
