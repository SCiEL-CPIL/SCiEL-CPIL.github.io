def construct():
    import sys
    from math import gcd
    import time
    
    N = 512
    # 1. Build conflict graph
    # conflict[i] is a set of j != i such that lcm(i, j) > N
    conflict = [set() for _ in range(N + 1)]
    
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            # Check if lcm(i, j) > N
            # lcm(i, j) = i * j // gcd(i, j)
            # To avoid overflow (not an issue in Python) and speed up:
            # i * j > N * gcd(i, j)
            g = gcd(i, j)
            if i * j > N * g:
                conflict[i].add(j)
                conflict[j].add(i)
                
    # 2. Solve Maximum Independent Set
    # Vertices are 1..N
    # We want a set S such that no two in S are in conflict.
    
    # Pre-sort vertices by degree in conflict graph (ascending)
    # Lower degree -> more compatible -> try earlier?
    # Actually, in MIS, we usually branch on vertices.
    # If we pick a vertex, we remove its neighbors.
    # Picking a vertex with high degree removes many candidates (good for pruning in some contexts?)
    # Or picking a vertex with low degree keeps many candidates?
    # Standard Bron-Kerbosch for Max Clique (which is MIS on complement) uses pivot.
    # Here, we are doing MIS directly.
    
    # Let's use a recursive search.
    # State: current_set, candidates (set of vertices compatible with current_set)
    
    best_set = set()
    best_size = 0
    
    # Initial candidates: all vertices
    # To help pruning, we can sort candidates.
    # We'll maintain candidates as a list or set.
    
    # Optimization: Use a global time limit
    start_time = time.time()
    time_limit = 55 # seconds
    
    def search(current_size, candidates):
        nonlocal best_set, best_size
        
        if time.time() - start_time > time_limit:
            return
        
        if not candidates:
            if current_size > best_size:
                best_size = current_size
                # We need to reconstruct the set, so we should pass the set
                # But passing sets by value is expensive.
                # Let's just track size for pruning, and store the set at the end?
                # No, we need the set.
                # We can pass a list of chosen vertices.
            return
            
        # Pruning
        # Upper bound: current_size + len(candidates)
        if current_size + len(candidates) <= best_size:
            return
            
        # Heuristic: Pick a vertex from candidates
        # Which one?
        # If we pick v, new_candidates = candidates - {v} - conflict[v]
        
        # Let's pick the vertex with the smallest degree within the current candidate set?
        # Or just the first one?
        # For better performance, we can try to pick a vertex that maximizes the reduction of candidates?
        # i.e., high degree in the induced subgraph on candidates.
        
        # To keep it simple and fast, let's just iterate through candidates.
        # But 512 is small.
        
        # We need to store the actual set to return it.
        # Let's modify search to take `path` (list of chosen nodes).
        
    # The above sketch is a bit messy regarding state management.
    # Let's implement a cleaner recursive function.
    
    # Vertices 1..N
    # conflict is a list of sets.
    
    # We will use a list `chosen` to track the current independent set.
    
    def mis_search(chosen, candidates):
        nonlocal best_set, best_size
        
        if time.time() - start_time > time_limit:
            return
        
        # Pruning
        if len(chosen) + len(candidates) <= best_size:
            return
        
        if not candidates:
            if len(chosen) > best_size:
                best_size = len(chosen)
                best_set = set(chosen)
            return
        
        # Pick a vertex to branch on
        # We can pick any vertex. 
        # Strategy: Pick a vertex v from candidates.
        # Branch 1: Include v.
        # Branch 2: Exclude v.
        
        # To make it efficient, we can use a "coloring" bound or similar, 
        # but simple pruning might be enough for N=512.
        
        # Let's pick the vertex with the highest degree in the induced subgraph?
        # Or just the first one?
        # Let's try picking the first one for simplicity, but order candidates by degree (desc) 
        # so we pick "conflicting" ones first? 
        # Actually, if we pick a vertex with many conflicts, we remove many from candidates in the "Include" branch.
        # This might lead to smaller subproblems.
        
        # Let's sort candidates by degree in the induced subgraph (descending) 
        # and pick the first one.
        
        # Calculate degrees in induced subgraph is O(|C|^2). 
        # Since |C| <= 512, this is fast.
        
        # However, doing this at every node is expensive.
        # Let's just pick the first vertex in the list for now, 
        # and rely on the initial ordering of the full graph.
        
        # Initial ordering: Sort 1..N by degree in conflict graph (descending).
        # High degree = many conflicts.
        
        # We'll maintain candidates as a list.
        
        v = candidates[0]
        rest = candidates[1:]
        
        # Branch 1: Include v
        # New candidates: those in rest that are NOT in conflict[v]
        new_candidates = [u for u in rest if u not in conflict[v]]
        chosen.append(v)
        mis_search(chosen, new_candidates)
        chosen.pop()
        
        # Branch 2: Exclude v
        # New candidates: rest
        mis_search(chosen, rest)
        
    # Initial candidates: all vertices 1..N
    # Order them by degree in conflict graph (descending)
    # High degree vertices are "bad" (conflict with many), so putting them first 
    # means if we include them, we prune a lot.
    # If we exclude them, we just move on.
    
    degrees = [len(conflict[i]) for i in range(1, N + 1)]
    vertices = list(range(1, N + 1))
    # Sort by degree descending
    vertices.sort(key=lambda x: degrees[x - 1], reverse=True)
    
    mis_search([], vertices)
    
    # If best_set is empty (shouldn't happen, at least {1} is valid), return [1]
    if not best_set:
        return [1]
        
    return list(best_set)
