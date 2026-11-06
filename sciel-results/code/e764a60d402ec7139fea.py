import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 35s buffer before 150s limit
    
    # Build Paley graph P_47
    # Edge (i, j) iff (i - j)^2 is a quadratic residue mod 47 (and i != j)
    # Quadratic residues mod 47: {x^2 mod 47 : x in 1..46}
    p = 47
    residues = set()
    for x in range(1, p):
        residues.add((x * x) % p)
    
    # Build adjacency for P_47
    paley_adj = [set() for _ in range(p)]
    for i in range(p):
        for j in range(i + 1, p):
            if (i - j) % p in residues or (j - i) % p in residues:
                paley_adj[i].add(j)
                paley_adj[j].add(i)
    
    # Verify P_47 is C4-free (quick check)
    # In a C4-free graph, any two vertices share at most 1 common neighbor
    # Actually for Paley graphs, the number of common neighbors is determined by character sums
    # For p=47, it should be C4-free. Let's trust this and verify in the final check.
    
    best_result = None
    
    # Try to find a good induced subgraph of P_47
    # Strategy: Start with all 47 vertices, greedily remove vertices that hurt min degree
    # Or: Start with a subset and grow it
    
    # Approach: Try all subsets of size n for n from 47 down to 40
    # For each n, find a subset S of {0..46} of size n such that
    # the induced subgraph P_47[S] has min degree >= n - 39
    
    # Greedy removal: start with all vertices, remove the vertex whose
    # removal maximizes the minimum degree of the remaining graph
    
    # Better approach: For each target n, try to find a good subset
    # Use greedy: start with all 47, iteratively remove worst vertex
    
    for target_n in range(47, 38, -1):
        if time.time() > deadline - 5:
            break
            
        min_deg_req = target_n - 39
        if min_deg_req < 1:
            min_deg_req = 1
        
        # Try greedy removal from full set
        # Start with all 47 vertices
        current = set(range(p))
        
        # We want to end up with target_n vertices
        # Iteratively remove the vertex that, when removed, gives the best
        # minimum degree in the resulting induced subgraph
        
        # Actually, let's think differently:
        # We want to KEEP target_n vertices such that min degree in induced subgraph >= min_deg_req
        # 
        # Greedy growth: start with a few vertices, add vertices that maintain high min degree
        # 
        # Alternative: greedy removal
        # Start with all 47. While |current| > target_n, remove the vertex v
        # such that removing v maximizes min_degree(induced_subgraph(current \ {v}))
        
        current = set(range(p))
        
        while len(current) > target_n:
            if time.time() > deadline - 5:
                break
            
            # For each candidate v to remove, compute min degree of induced subgraph on current \ {v}
            best_v = None
            best_min_deg = -1
            
            for v in current:
                # Compute min degree in induced subgraph on current \ {v}
                remaining = current - {v}
                min_deg = p  # large
                for u in remaining:
                    deg_u = len(paley_adj[u] & remaining)
                    if deg_u < min_deg:
                        min_deg = deg_u
                if min_deg > best_min_deg:
                    best_min_deg = min_deg
                    best_v = v
                elif min_deg == best_min_deg and best_v is not None:
                    # Tie-break: prefer removing vertex with lower degree in current
                    if len(paley_adj[v] & current) < len(paley_adj[best_v] & current):
                        best_v = v
            
            if best_v is not None:
                current.remove(best_v)
        
        # Now check if current has min degree >= min_deg_req
        if len(current) >= target_n:
            # Take exactly target_n vertices (current should be exactly target_n)
            if len(current) > target_n:
                # Trim to target_n
                current = set(list(current)[:target_n])
            
            # Check min degree
            min_deg = min(len(paley_adj[u] & current) for u in current)
            
            if min_deg >= min_deg_req:
                # Verify no C4
                has_c4 = False
                vertex_list = list(current)
                # Check: for any two vertices u, v in current, they share at most 1 common neighbor
                for idx, u in enumerate(vertex_list):
                    for v in vertex_list[idx+1:]:
                        common = paley_adj[u] & paley_adj[v] & current
                        if len(common) > 1:
                            has_c4 = True
                            break
                    if has_c4:
                        break
                
                if not has_c4:
                    # Build edges
                    edges = []
                    for u in vertex_list:
                        for v in vertex_list:
                            if u < v and v in paley_adj[u]:
                                edges.append([u, v])
                    
                    result = {'n': len(vertex_list), 'edges': edges}
                    if best_result is None or len(vertex_list) > best_result['n']:
                        best_result = result
                    
                    if len(vertex_list) == 47:
                        return best_result
                    # If we found 46, that's great
                    if len(vertex_list) >= 46:
                        # Do a few more random restarts to see if we can get 47
                        # But for now, return this
                        pass
        
        # If we didn't find a valid subset for this target_n, continue to smaller n
    
    # If greedy removal didn't work well, try random restarts
    if best_result is None or best_result['n'] < 46:
        import random
        random.seed(42)
        
        for target_n in range(47, 38, -1):
            if time.time() > deadline - 5:
                break
            
            min_deg_req = target_n - 39
            if min_deg_req < 1:
                min_deg_req = 1
            
            found = False
            for attempt in range(200):
                if time.time() > deadline - 5:
                    break
                
                # Random subset of size target_n from 0..46
                subset = set(random.sample(range(p), target_n))
                
                # Check min degree
                min_deg = min(len(paley_adj[u] & subset) for u in subset)
                
                if min_deg >= min_deg_req:
                    # Check C4
                    has_c4 = False
                    vertex_list = list(subset)
                    for idx, u in enumerate(vertex_list):
                        for v in vertex_list[idx+1:]:
                            common = paley_adj[u] & paley_adj[v] & subset
                            if len(common) > 1:
                                has_c4 = True
                                break
                        if has_c4:
                            break
                    
                    if not has_c4:
                        edges = []
                        for u in vertex_list:
                            for v in vertex_list:
                                if u < v and v in paley_adj[u]:
                                    edges.append([u, v])
                        
                        result = {'n': len(vertex_list), 'edges': edges}
                        if best_result is None or len(vertex_list) > best_result['n']:
                            best_result = result
                        
                        if len(vertex_list) == 47:
                            return best_result
                        found = True
                        break
            
            if found and target_n <= 46:
                # We found a valid graph for this n, but try to see if we can do better
                # Actually if we found for target_n, we should try larger n first
                # The loop goes from 47 down, so if we found 46, we can stop
                if target_n <= 46:
                    break
    
    # Final fallback: construct a simple C4-free graph
    # For small n, use a complete bipartite graph or similar
    if best_result is None or best_result['n'] < 39:
        # Fallback: n=39, empty graph (min degree 0 >= 0)
        # Actually min degree must be >= n-39 = 0, so empty graph works for n=39
        return {'n': 39, 'edges': []}
    
    return best_result
