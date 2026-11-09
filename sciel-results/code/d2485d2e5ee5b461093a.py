import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 5s margin for safety

    def count_violations(adj, n):
        """
        Counts the number of regular induced subgraphs of size >= 6.
        For speed, we primarily weight size 6, as larger sizes are rarer 
        but we must check all to be valid.
        Returns (total_violations, list_of_violations)
        """
        violations = []
        total = 0
        # We only need to check up to n. 
        # Optimization: If we find many size 6, we don't strictly need to count size 7+ 
        # for the *greedy choice* if size 6 dominates, but for *validity* we need 0 total.
        # However, for the greedy step, minimizing size 6 violations is the primary driver.
        # Let's count size 6 and 7 to be safe, as they are the most frequent.
        
        # For the greedy step, we only need a relative score. 
        # We will count size 6 violations. If 0, we check 7, etc.
        
        for k in range(6, n + 1):
            found_in_k = 0
            for subset in combinations(range(n), k):
                mask = 0
                for v in subset:
                    mask |= (1 << v)
                
                d0 = (adj[subset[0]] & mask).bit_count()
                is_reg = True
                for v in subset[1:]:
                    d = (adj[v] & mask).bit_count()
                    if d != d0:
                        is_reg = False
                        break
                if is_reg:
                    found_in_k += 1
                    # Store a few for debugging or tie-breaking? 
                    # For now, just count.
                    if found_in_k > 100: # Cap to save time if it's already bad
                        break
            total += found_in_k
            if total > 0:
                # If we have violations at size 6, we don't need to check larger sizes 
                # for the purpose of *ranking* candidates in the greedy step, 
                # because breaking size 6 is the priority. 
                # However, if total is 0, we must continue to check 7, 8...
                if k == 6:
                    # Continue checking larger k only if no size 6 violations found?
                    # Actually, if size 6 is 0, we MUST check 7. 
                    # If size 6 is > 0, minimizing size 6 is the main goal.
                    pass
            
            if found_in_k > 0 and k == 6:
                # If there are size 6 violations, larger ones are secondary.
                # But we must return 0 for validity. 
                # For the greedy score, we return the count.
                return total, violations
        
        return total, violations

    def is_valid_fast(adj, n):
        """
        Returns True if no regular induced subgraph of size >= 6 exists.
        """
        for k in range(6, n + 1):
            for subset in combinations(range(n), k):
                mask = 0
                for v in subset:
                    mask |= (1 << v)
                
                d0 = (adj[subset[0]] & mask).bit_count()
                is_reg = True
                for v in subset[1:]:
                    d = (adj[v] & mask).bit_count()
                    if d != d0:
                        is_reg = False
                        break
                if is_reg:
                    return False
        return True

    def greedy_build(target_n, seed):
        """
        Builds a graph on target_n vertices by adding vertices one by one.
        For each new vertex v (index i), we try all 2^i possible edge connections 
        to previous vertices 0..i-1. We select the one that minimizes the number 
        of regular induced subgraphs of size >= 6 in the resulting graph on i+1 vertices.
        """
        random.seed(seed)
        
        # Start with a small base that is likely valid. 
        # A random graph on 5 vertices is always valid (no subset of size 6).
        # We start adding from vertex 5 (index 5).
        
        # Initialize adjacency list as list of bitmasks
        adj = [0] * target_n
        edges = []
        
        # Vertices 0..4 are arbitrary. Let's pick a random graph for them.
        for i in range(5):
            for j in range(i+1, 5):
                if random.random() < 0.5:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
                    edges.append([i, j])
        
        # Now add vertices 5 to target_n-1
        for i in range(5, target_n):
            best_pattern = None
            best_score = float('inf')
            best_adj_copy = None
            best_edges_add = []
            
            # Number of previous vertices is i.
            # Possible patterns: 0 to 2^i - 1.
            # For i=5, 32 patterns.
            # For i=10, 1024 patterns.
            # For i=15, 32768 patterns.
            # Checking each pattern involves validating the graph of size i+1.
            # Validation cost: O(C(i+1, 6) * 6).
            # For i=10 (n=11): C(11,6)=462. 462*6 ~ 3k ops. 1024 patterns * 3k ~ 3M ops.
            # This is fast in Python if optimized? 
            # 3M simple bit ops might take ~0.1-0.2s.
            # For i=15 (n=16): C(16,6)=8008. 8008*6 ~ 50k ops. 32768 patterns * 50k ~ 1.6B ops.
            # This is TOO SLOW in pure Python.
            
            # Optimization: We only need to check if the NEW vertex creates violations.
            # A violation in graph G_{i+1} must include the new vertex i? 
            # Not necessarily. But the previous graph G_i was chosen to have minimal violations.
            # Actually, if G_i has 0 violations, G_{i+1} can only have violations involving i.
            # If G_i has violations, we are in trouble, but we aim for 0.
            
            # Let's assume we maintain a graph with 0 violations (or minimal).
            # When adding vertex i, we only need to check subsets that INCLUDE i.
            # Why? Because subsets NOT including i are already checked in G_i.
            # If G_i was valid (0 violations), then any violation in G_{i+1} MUST include i.
            # This reduces the check from C(i+1, k) to C(i, k-1).
            
            # So for adding vertex i, we check subsets S of {0..i-1} of size k-1.
            # The subset in G_{i+1} is S U {i}.
            
            # We need to check k from 6 to i+1.
            # For a given k, we check subsets S of size k-1 from the first i vertices.
            
            # Let's refine the scoring function for a candidate pattern P (bitmask of edges from i to 0..i-1):
            # 1. Construct the new adj for vertex i.
            # 2. Check all subsets S of {0..i-1} such that |S| >= 5 (so |S U {i}| >= 6).
            # 3. Check if S U {i} is regular.
            
            # We can precompute the degrees of vertices in S within G_i?
            # No, the degree of v in S U {i} is deg_Gi(v, S) + (1 if edge(v,i) else 0).
            
            # Let's implement a fast check for a specific pattern P.
            
            for p in range(1 << i):
                # Construct adj for vertex i
                adj_i = p # bitmask of neighbors among 0..i-1
                
                # Check for violations involving i
                valid = True
                # k goes from 6 to i+1
                # S size goes from 5 to i
                for k in range(6, i + 2):
                    s_size = k - 1
                    if s_size > i: continue
                    if s_size < 0: continue
                    
                    # Iterate subsets S of {0..i-1} of size s_size
                    for S in combinations(range(i), s_size):
                        # S is a tuple of vertices
                        # mask_S
                        mask_S = 0
                        for v in S:
                            mask_S |= (1 << v)
                        
                        # Degrees of v in S U {i}
                        # deg_new(v) = (adj[v] & mask_S).bit_count() + (1 if (adj_i >> v) & 1 else 0)
                        
                        # We need all deg_new(v) for v in S U {i} to be equal.
                        # Let d_i = popcount(adj_i) (degree of i in S U {i})
                        # Wait, degree of i in S U {i} is number of neighbors of i in S.
                        d_i = 0
                        for v in S:
                            if (p >> v) & 1:
                                d_i += 1
                        
                        # Now check if all v in S have degree d_i within S U {i}
                        # deg_S_U_i(v) = (adj[v] & mask_S).bit_count() + (1 if (p >> v) & 1 else 0)
                        
                        first_v = S[0]
                        d_target = (adj[first_v] & mask_S).bit_count() + (1 if (p >> first_v) & 1 else 0)
                        
                        if d_target != d_i:
                            continue # Not regular, skip this subset
            
                        # If d_target == d_i, check rest of S
                        is_reg = True
                        for v in S[1:]:
                            d_v = (adj[v] & mask_S).bit_count() + (1 if (p >> v) & 1 else 0)
                            if d_v != d_target:
                                is_reg = False
                                break
                        
                        if is_reg:
                            valid = False
                            break
                    if not valid:
                        break
                
                if valid:
                    # Found a valid pattern!
                    # We can pick this one. To add variety, we could pick randomly among valid ones,
                    # but the first one found is fine.
                    # However, for robustness, let's score by "margin" or just take the first.
                    # Taking the first valid pattern is fast.
                    
                    # Update adj and edges
                    adj[i] = p
                    for v in range(i):
                        if (p >> v) & 1:
                            adj[v] |= (1 << i)
                            edges.append([v, i])
                    
                    # Break out of pattern loop
                    break
                else:
                    # If we don't break, we continue to next pattern.
                    # If we finish all patterns without finding a valid one, we have a problem.
                    # In that case, we pick the pattern with the MINIMUM number of violations.
                    # For now, if no valid pattern found, we pick a random one and hope for the best,
                    # or we should track the best score.
                    
                    pass
            
            # If we didn't break (no valid pattern found), we need a fallback.
            # Let's check if we actually set adj[i].
            if adj[i] == 0 and i > 0:
                # It's possible we never found a valid pattern.
                # We should have tracked the best score. 
                # For simplicity in this first pass, if no valid pattern, pick random.
                # This might lead to invalid graph, but we validate at the end.
                # A better fallback: Pick the pattern with min violations.
                # Let's re-run the loop to find min violations if no valid one found.
                # To save code, we'll just pick random for now and rely on final validation.
                # Actually, let's just pick a random pattern if no valid one is found.
                p = random.randint(0, (1 << i) - 1)
                adj[i] = p
                for v in range(i):
                    if (p >> v) & 1:
                        adj[v] |= (1 << i)
                        edges.append([v, i])

        return edges

    # Main Logic
    best_result = {'n': 5, 'edges': []}
    
    # Try to build n=21
    # Greedy building is deterministic if we pick the FIRST valid pattern.
    # To explore different paths, we can randomize the order of patterns or the initial base.
    
    # Strategy:
    # 1. Try n=21 with several random seeds (affecting base graph and tie-breaking).
    # 2. If n=21 fails, try n=20.
    
    for n_target in [21, 20, 19]:
        if time.time() > deadline:
            break
            
        found = False
        # Try multiple seeds
        for seed in range(50):
            if time.time() > deadline:
                break
                
            edges = greedy_build(n_target, seed)
            
            # Validate
            adj = [0] * n_target
            for u, v in edges:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                
            if is_valid_fast(adj, n_target):
                best_result = {'n': n_target, 'edges': edges}
                found = True
                break
        
        if found:
            break
            
    return best_result
