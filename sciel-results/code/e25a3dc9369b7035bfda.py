import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 150s limit
    
    # Fallback: Paley graph on 17 vertices
    def get_paley_17():
        squares = {x * x % 17 for x in range(1, 17)}
        edges = []
        for i in range(17):
            for j in range(i + 1, 17):
                if (j - i) % 17 in squares:
                    edges.append([i, j])
        return {'n': 17, 'edges': edges}

    best_result = get_paley_17()
    
    # Target n values to try
    # We prioritize higher n but ensure we have a good fallback if time runs out.
    # The current program scores 1.1765 which corresponds to n=20 (17 is 1.0).
    # Let's try to push for 25, 30, 35.
    targets = [35, 34, 33, 32, 31, 30, 28, 25, 24, 23, 22, 21, 20]
    
    for n in targets:
        if time.time() > deadline:
            break
        
        # Try to find a valid graph for this n using SAT-based incremental construction
        # with local search repair as a fallback/refinement.
        result = try_find_graph_sat(n, deadline)
        
        if result:
            # Verify the result before accepting
            if is_valid(result):
                best_result = result
                # If we found a large graph, we can stop to ensure we return within time.
                # 35 is the record. 30+ is a significant improvement over 20.
                if n >= 30:
                    return best_result
                # If we found something better than our current best (which starts at 17),
                # we keep it, but continue trying for larger n if time permits.
                # However, if we found 20, we should probably try 21, 22, etc.
                # The loop is descending, so if we fail 35, we try 34, etc.
                # If we succeed at 25, we return 25? No, we should try 26, 27, 28...
                # But the list is descending. So if we succeed at 25, we have a valid 25.
                # Should we try 26? 26 is not in the list. We should insert it?
                # Or just rely on the list.
                # Let's just return if we get a "good" score, e.g., n >= 25.
                # If we get 20, we continue to 19? No, 19 is smaller.
                # The loop is [35, 34, ..., 20].
                # If we find 20, we return 20? That's the baseline.
                # If we find 25, we return 25.
                # If we find 30, we return 30.
                
                # Let's refine the stopping condition.
                # If we found n, and n is significantly larger than 17, it's a win.
                # But we want the MAXIMUM n.
                # Since the list is descending, the first success is the largest n in the list that worked.
                # So if we find 35, we return. If 35 fails, we try 34. If 34 works, we return 34.
                # This is correct for maximizing n among the tried targets.
                return best_result

    return best_result

def is_valid(graph):
    n = graph['n']
    edges = graph['edges']
    # Build adjacency matrix using numpy for speed
    A = np.zeros((n, n), dtype=np.int8)
    for u, v in edges:
        A[u, v] = 1
        A[v, u] = 1
    
    # Check K4
    # A K4 exists if there are 4 vertices all pairwise connected.
    # Iterate over triangles (i, j, k) and check if there is a 4th vertex l connected to all.
    # For n <= 40, this is fast enough with numpy bitsets or simple loops.
    # Simple loops with early exit.
    for i in range(n):
        for j in range(i+1, n):
            if A[i, j]:
                for k in range(j+1, n):
                    if A[i, k] and A[j, k]:
                        # i, j, k form a triangle. Check for l > k connected to all.
                        # Actually l can be any other vertex. But if we find one, it's a K4.
                        # To be efficient, we can just check if the common neighborhood of i,j,k is non-empty.
                        # Common neighbors: A[i] & A[j] & A[k]
                        # If any bit is set (excluding i,j,k themselves, though diagonal is 0), we have a K4.
                        # Actually, if l is i, j, or k, it's not a new vertex.
                        # So we check if (A[i] & A[j] & A[k]) has any 1.
                        # Since A[i,i]=0, the bits for i,j,k will be 0 in the intersection if they are not connected to themselves.
                        # Wait, A[i,k]=1, so bit k in A[i] is 1.
                        # A[k,i]=1, so bit i in A[k] is 1.
                        # The intersection will have bits for common neighbors.
                        # If there is any common neighbor l distinct from i,j,k, then K4.
                        # If the intersection has any 1, it means there is a vertex connected to i, j, and k.
                        # That vertex could be i, j, or k?
                        # A[i,i]=0, so bit i in A[i] is 0. So bit i in intersection is 0.
                        # Similarly for j and k.
                        # So any 1 in the intersection corresponds to a vertex l distinct from i,j,k.
                        # Thus, if (A[i] & A[j] & A[k]).sum() > 0, we have a K4.
                        if (A[i] & A[j] & A[k]).sum() > 0:
                            return False

    # Check Independent 6-set
    # Equivalent to K6 in complement.
    # Complement B = ~A (with diagonal 0).
    # We can check for K6 in B similarly.
    # But K6 check is heavier.
    # Alternative: Use the property that if alpha(G) >= 6, there is an independent set of size 6.
    # We can try to find a clique of size 6 in the complement.
    # Let's use a simple backtracking or just check all 6-subsets if n is small.
    # For n=35, C(35,6) ~ 1.6M. With numpy, checking 15 edges per subset might be slow in pure python.
    # Let's use a recursive clique finder on the complement.
    
    B = 1 - A
    np.fill_diagonal(B, 0)
    
    # Find clique of size 6 in B
    # Order vertices by degree in B descending
    degrees = B.sum(axis=1)
    order = np.argsort(degrees)[::-1]
    
    def find_clique_size_6(B, order, target_size=6):
        # Backtracking
        # candidates: list of vertices that can be added to the current clique
        # current: list of vertices in the current clique
        def backtrack(current, candidates):
            if len(current) == target_size:
                return current
            if not candidates:
                return None
            if len(candidates) + len(current) < target_size:
                return None
            
            # Try adding each candidate
            # To speed up, we can remove candidates that are not connected to the last added vertex?
            # Standard Bron-Kerbosch style or simple backtracking.
            
            for idx, v in enumerate(candidates):
                # Check if v is connected to all in current
                # In B, connected means B[u, v] == 1
                valid = True
                for u in current:
                    if B[u, v] == 0:
                        valid = False
                        break
                if valid:
                    # New candidates are those in candidates[idx+1:] that are connected to v in B
                    new_candidates = [w for w in candidates[idx+1:] if B[v, w] == 1]
                    res = backtrack(current + [v], new_candidates)
                    if res:
                        return res
            return None
        
        return backtrack([], list(order))
    
    if find_clique_size_6(B, order):
        return False
        
    return True

def try_find_graph_sat(n, deadline):
    """
    Uses a SAT-like approach: 
    1. Start with a random graph or extend a smaller valid graph.
    2. Use local search to fix violations (K4 or I6).
    3. The "SAT" aspect is in the constraint checking and targeted repair.
    """
    # We will use a local search that is aware of the constraints.
    # To improve over the previous version, we will:
    # 1. Use numpy for fast adjacency checks.
    # 2. Use a better initialization (e.g., start from a valid smaller graph if possible, or random with tuned p).
    # 3. Use a more efficient violation finder.
    
    # For now, let's stick to random initialization but with multiple restarts and numpy acceleration.
    
    max_restarts = 20
    max_iters_per_restart = 2000
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            return None
        
        # Initialize adjacency matrix
        # p = 0.5 is standard, but for K4-free, maybe lower density is better?
        # For alpha <= 5, higher density is better.
        # Let's try p=0.5.
        A = np.random.randint(0, 2, (n, n), dtype=np.int8)
        np.fill_diagonal(A, 0)
        # Symmetrize
        A = (A + A.T) % 2
        
        # Local search
        for it in range(max_iters_per_restart):
            if time.time() > deadline:
                return None
            
            # Find violation
            viol, vtype = find_violation_fast(A, n)
            
            if viol is None:
                # Valid graph found
                edges = []
                for i in range(n):
                    for j in range(i+1, n):
                        if A[i, j]:
                            edges.append([i, j])
                return {'n': n, 'edges': edges}
            
            # Flip an edge
            if vtype == 'K4':
                i, j, k, l = viol
                # Pick a random edge from the K4 to remove
                edges_to_flip = [(i,j), (i,k), (i,l), (j,k), (j,l), (k,l)]
            else: # I6
                i, j, k, l, m, p = viol
                # Pick a random non-edge from the I6 to add
                edges_to_flip = [(i,j), (i,k), (i,l), (i,m), (i,p),
                                 (j,k), (j,l), (j,m), (j,p),
                                 (k,l), (k,m), (k,p),
                                 (l,m), (l,p),
                                 (m,p)]
            
            u, v = random.choice(edges_to_flip)
            A[u, v] = 1 - A[u, v]
            A[v, u] = A[u, v]
            
    return None

def find_violation_fast(A, n):
    """
    Find a K4 or an Independent 6-set quickly.
    Returns (tuple_of_vertices, 'K4') or (tuple_of_vertices, 'I6') or (None, None).
    """
    # Check K4
    # Use the triangle intersection method
    for i in range(n):
        for j in range(i+1, n):
            if A[i, j]:
                # Common neighbors of i and j
                # common = A[i] & A[j]
                # We need a k > j in common? No, k can be any.
                # But to find a specific K4, we can iterate k.
                # Let's iterate k > j.
                for k in range(j+1, n):
                    if A[i, k] and A[j, k]:
                        # i, j, k form a triangle.
                        # Check if there is an l connected to i, j, k.
                        # l can be any vertex.
                        # If we find one, we have a K4.
                        # To return the vertices, we need l.
                        # Let's check if there is any l.
                        # If yes, find one.
                        common_ikj = A[i] & A[j] & A[k]
                        if common_ikj.sum() > 0:
                            # Find the first index
                            l = np.argmax(common_ikj)
                            return (i, j, k, l), 'K4'
    
    # Check I6
    # Find clique of size 6 in complement
    B = 1 - A
    np.fill_diagonal(B, 0)
    
    # Simple backtracking for clique of size 6
    # Order by degree
    degrees = B.sum(axis=1)
    order = np.argsort(degrees)[::-1]
    
    def find_clique_size_6(B, order, target_size=6):
        def backtrack(current, candidates):
            if len(current) == target_size:
                return current
            if not candidates:
                return None
            if len(candidates) + len(current) < target_size:
                return None
            
            for idx, v in enumerate(candidates):
                valid = True
                for u in current:
                    if B[u, v] == 0:
                        valid = False
                        break
                if valid:
                    new_candidates = [w for w in candidates[idx+1:] if B[v, w] == 1]
                    res = backtrack(current + [v], new_candidates)
                    if res:
                        return res
            return None
        
        return backtrack([], list(order))
    
    clique = find_clique_size_6(B, order)
    if clique:
        return tuple(clique), 'I6'
        
    return None, None
