import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 290  # Safety margin
    
    # 1. Initialize Paley graph on 37 vertices
    q = 37
    residues = {(x * x) % q for x in range(1, q)}
    n = q
    # adj[i] is a bitmask of neighbors of i
    adj = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if (j - i) % q in residues:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    
    # Helper to check if graph is (5,5)-free (for final verification)
    def is_5_5_free(adj, n):
        # Check for K5
        for v1 in range(n):
            n1 = adj[v1]
            # Iterate pairs in n1
            v1_bits = [v for v in range(n) if (n1 >> v) & 1]
            if len(v1_bits) < 4: continue
            for v2, v3, v4 in combinations(v1_bits, 3):
                # Check if v2,v3,v4 form a triangle and all connect to v1
                if ((adj[v2] >> v3) & 1) and ((adj[v2] >> v4) & 1) and ((adj[v3] >> v4) & 1):
                    # v1,v2,v3,v4 is a K4. Check if any neighbor of v1 in {v2,v3,v4} ... 
                    # Actually, we just need to check if there is a 5th vertex.
                    # This simple check is O(n^5). 
                    # For n=42, this is 42^5/120 ~ 1.5M * checks. Might be slow in Python if not optimized.
                    # Let's use a better check for the final verification.
                    pass
        # We will rely on the construction process to maintain validity.
        # But we must verify the final result.
        # A faster check:
        # For each 5-subset, check if clique or independent.
        # 42 choose 5 is 850k. 
        # 850k * 10 edge checks = 8.5M ops. 
        # In Python, 8.5M simple ops takes ~1-2 seconds. Acceptable.
        
        for subset in combinations(range(n), 5):
            # Check clique
            is_clique = True
            is_indep = True
            for i in range(5):
                for j in range(i+1, 5):
                    u, v = subset[i], subset[j]
                    if (adj[u] >> v) & 1:
                        is_indep = False
                    else:
                        is_clique = False
                    if not is_clique and not is_indep:
                        break
                if not is_clique and not is_indep:
                    break
            if is_clique or is_indep:
                return False
        return True

    # Helper to get list of K4s and I4s
    def get_k4s_i4s(adj, n):
        k4s = []
        i4s = []
        # Iterate all 4-subsets
        for subset in combinations(range(n), 4):
            is_clique = True
            is_indep = True
            for i in range(4):
                for j in range(i+1, 4):
                    u, v = subset[i], subset[j]
                    if (adj[u] >> v) & 1:
                        is_indep = False
                    else:
                        is_clique = False
                    if not is_clique and not is_indep:
                        break
                if not is_clique and not is_indep:
                    break
            if is_clique:
                k4s.append(frozenset(subset))
            if is_indep:
                i4s.append(frozenset(subset))
        return k4s, i4s

    k4s, i4s = get_k4s_i4s(adj, n)
    
    current_n = n
    
    # Function to try to add a vertex
    def try_add_vertex(adj, k4s, i4s, current_n):
        # We want a set S (neighbors of new node) such that:
        # 1. No K4 is contained in S
        # 2. No I4 is contained in V \ S (i.e., S intersects every I4)
        
        # Local search
        best_S = None
        for attempt in range(100): # Try 100 random starts
            # Initialize S randomly
            S = set()
            for v in range(current_n):
                if random.random() < 0.5:
                    S.add(v)
            
            # Local search to fix violations
            for _ in range(500): # Max iterations
                # Check violations
                bad_k4 = None
                for k in k4s:
                    if k.issubset(S):
                        bad_k4 = k
                        break
                
                bad_i4 = None
                if not bad_k4:
                    # Check I4s in V\S
                    V_minus_S = set(range(current_n)) - S
                    for i in i4s:
                        if i.issubset(V_minus_S):
                            bad_i4 = i
                            break
                
                if not bad_k4 and not bad_i4:
                    return S # Success
                
                if bad_k4:
                    # Remove a random vertex from bad_k4 that is in S
                    candidates = list(bad_k4 & S)
                    if candidates:
                        S.remove(random.choice(candidates))
                    else:
                        # Should not happen if bad_k4 is subset of S
                        break
                elif bad_i4:
                    # Add a random vertex from bad_i4 that is in V\S
                    candidates = list(bad_i4 - S)
                    if candidates:
                        S.add(random.choice(candidates))
                    else:
                        break
            
            # If we didn't find a solution, we could keep the best one found, 
            # but we just return None for this attempt.
        
        return None

    # Main loop to add vertices
    while current_n < 42 and time.time() < deadline - 10:
        S = try_add_vertex(adj, k4s, i4s, current_n)
        if S is None:
            # Try a few more times with different seeds or just break?
            # Maybe try 5 more times
            found = False
            for _ in range(5):
                S = try_add_vertex(adj, k4s, i4s, current_n)
                if S is not None:
                    found = True
                    break
            if not found:
                break
        
        # Add new vertex
        new_idx = current_n
        adj.append(0)
        for v in S:
            adj[v] |= (1 << new_idx)
            adj[new_idx] |= (1 << v)
        
        current_n += 1
        
        # Update K4s and I4s
        # New K4s:
        # 1. Old K4s are still K4s.
        # 2. New K4s involving new_idx: {new_idx} + Triangle in S
        # We need to find triangles in the subgraph induced by S.
        # And update I4s similarly.
        
        # Efficient update:
        # Just recompute K4s and I4s for the new graph?
        # For n=42, recomputing is ~ 100k checks. Fast enough.
        k4s, i4s = get_k4s_i4s(adj, current_n)
        
        # Check if we are still valid (should be by construction, but good to be sure)
        # Actually, the construction ensures no NEW K5/I5 are formed if S was valid.
        # But we must ensure the OLD graph was valid (it was).
        
    # Final verification
    if not is_5_5_free(adj, current_n):
        # Fallback to seed if somehow invalid
        return 37, [(i, j) for i in range(37) for j in range(i + 1, 37) if (j - i) % 37 in residues]
        
    edges = []
    for i in range(current_n):
        for j in range(i + 1, current_n):
            if (adj[i] >> j) & 1:
                edges.append((i, j))
                
    return current_n, edges
