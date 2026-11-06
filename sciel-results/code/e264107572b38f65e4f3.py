import itertools
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    # Helper to check if a graph has any regular induced subgraph of size >= 6
    def has_violation(adj, n):
        # adj is a list of integers (bitmasks)
        # Check all subsets of size k >= 6
        # Optimization: If we find one, return True immediately?
        # For local search, we need the COUNT to guide the search.
        # But counting all is expensive.
        # Maybe just return the count.
        
        count = 0
        for k in range(6, n + 1):
            for subset in itertools.combinations(range(n), k):
                mask = 0
                for v in subset:
                    mask |= (1 << v)
                
                # Check regularity
                # Get degrees of all vertices in subset within subset
                # If k is small, we can just loop
                degs = []
                for v in subset:
                    deg = (adj[v] & mask).bit_count()
                    degs.append(deg)
                
                # Check if all equal
                if all(d == degs[0] for d in degs):
                    count += 1
        return count

    # This is too slow for n=22. 4 million subsets * 6-22 iterations each.
    # We need a faster check or a smarter search.
    
    # Let's try a different approach:
    # Only check subsets of size 6, 7, 8. 
    # If we clear those, maybe larger ones clear too?
    # Or, use a heuristic: 
    # Random graphs have very few regular induced subgraphs.
    # We can just do random restarts and check.
    
    # To make it feasible, I will limit the check to size 6 and 7 initially.
    # If 0 violations of size 6 and 7, I will do a full check.
    
    def count_violations_fast(adj, n, max_k=8):
        count = 0
        for k in range(6, min(n, max_k) + 1):
            for subset in itertools.combinations(range(n), k):
                mask = 0
                for v in subset:
                    mask |= (1 << v)
                degs = [(adj[v] & mask).bit_count() for v in subset]
                if all(d == degs[0] for d in degs):
                    count += 1
        return count

    def count_violations_full(adj, n):
        return count_violations_fast(adj, n, max_k=n)

    # Try to find a graph for n=22
    # Start with Paley-like graph
    # Paley graph P_23: edge (u,v) if (u-v)^2 % 23 == 1
    # We use vertices 0..21
    
    def generate_paley_minus_one(n):
        # n should be 22
        adj = [0] * n
        # Original Paley on 23 vertices (0..22)
        # We remove vertex 22.
        # So vertices are 0..21.
        for u in range(n):
            for v in range(u + 1, n):
                diff = u - v
                if (diff * diff) % 23 == 1:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        return adj

    best_n = 0
    best_graph = None
    
    # Attempt n=22
    target_n = 22
    adj = generate_paley_minus_one(target_n)
    
    # Local search
    # Objective: minimize violations
    # Use simulated annealing or simple hill climbing with restarts
    
    # To speed up, we only update the count for subsets affected by the flip?
    # That's complex. Let's just re-evaluate.
    # But re-evaluating is slow.
    
    # Let's try a different strategy:
    # Random restarts with random graphs.
    # Check if valid.
    
    # If we can't find n=22, try n=21.
    
    # Given the time limit, I will try a few random restarts for n=22 and n=21.
    # And one structured start (Paley).
    
    # To make the check faster, I will only check size 6 and 7 in the inner loop.
    # If count is 0, I will do a full check.
    
    def is_valid_fast(adj, n):
        # Check size 6 and 7
        c = count_violations_fast(adj, n, max_k=7)
        if c > 0:
            return False
        # If fast check passes, do full check
        c_full = count_violations_full(adj, n)
        return c_full == 0

    # Search loop
    for n_try in [22, 21, 20]:
        if time.time() > deadline:
            break
            
        # Try Paley start for 22
        if n_try == 22:
            adj = generate_paley_minus_one(n_try)
            if is_valid_fast(adj, n_try):
                return {'n': n_try, 'edges': [[u, v] for u in range(n_try) for v in range(u+1, n_try) if adj[u] & (1 << v)]}
        
        # Random restarts
        for restart in range(20):
            if time.time() > deadline:
                break
                
            adj = [0] * n_try
            for u in range(n_try):
                for v in range(u + 1, n_try):
                    if random.random() < 0.5:
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                        
            # Local search: flip edges to reduce violations
            # We need a cost function. count_violations_fast is good for guidance.
            # But it's slow.
            # Let's just check if it's valid.
            if is_valid_fast(adj, n_try):
                return {'n': n_try, 'edges': [[u, v] for u in range(n_try) for v in range(u+1, n_try) if adj[u] & (1 << v)]}
                
            # If not valid, try to fix it?
            # For simplicity and speed, I'll just do more restarts.
            # If I find a valid one, I return.
            
    # If we didn't find 22, 21, or 20 via random/Paley, 
    # we should try to find a valid graph for n=20 using a more robust search.
    # Or just return a known valid graph for n=20 if we can generate one quickly.
    # Actually, random graphs on 20 vertices are very likely to be valid.
    # Let's do a dedicated search for n=20.
    
    if time.time() < deadline:
        n_try = 20
        for restart in range(50):
            if time.time() > deadline:
                break
            adj = [0] * n_try
            for u in range(n_try):
                for v in range(u + 1, n_try):
                    if random.random() < 0.5:
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
            if is_valid_fast(adj, n_try):
                return {'n': n_try, 'edges': [[u, v] for u in range(n_try) for v in range(u+1, n_try) if adj[u] & (1 << v)]}
                
    # Fallback: return n=5 (seed) or a small valid graph
    # A graph with no edges is valid for any n?
    # Induced subgraph of empty graph is empty. Degrees are 0.
    # So it IS regular!
    # Wait. "Regular subgraph" means all vertices have the same degree.
    # In an empty graph, all degrees are 0. So it is regular.
    # So the empty graph is INVALID for n >= 6.
    # The seed program returns n=5.
    # Why is n=5 valid?
    # "No set of 6 or more vertices".
    # If n=5, there are NO sets of 6 or more vertices.
    # So the condition is vacuously true.
    # So any graph on n < 6 is valid.
    # So I can return n=5, 4, etc.
    # But I want to maximize n.
    
    # If I failed to find n=20, I should return the best I found.
    # But I didn't track "best found" in the random loop, I just returned if valid.
    # If no valid graph found for 20, I should return a valid graph for n=5.
    
    return {'n': 5, 'edges': []}
