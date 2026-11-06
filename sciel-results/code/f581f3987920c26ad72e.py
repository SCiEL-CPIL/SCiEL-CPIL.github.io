import time
import random
import itertools

def construct():
    start_time = time.time()
    time_limit = 110.0  # Safety margin
    
    # We aim for n=20. If we can't find a valid one, we drop to 19, etc.
    # But n=20 is known to exist.
    
    def is_valid_graph(n, adj_bits):
        """
        Check if graph with n vertices and adjacency bitmasks adj_bits
        has no induced regular subgraph on >= 6 vertices.
        """
        # Precompute degrees for all subsets? No, too expensive.
        # Check subsets of size 6 to 20.
        # Optimization: If a subset of size k is regular, any superset is not necessarily regular.
        # We must check all subsets of size >= 6.
        
        # Fast check: 
        # A regular induced subgraph on k vertices means every vertex in the subset has the same degree within the subset.
        
        # We iterate by size k from 6 to n.
        # To speed up, we can check size 6 first. If we find a violation, we return False.
        # If we check all size 6 and find none, we must check size 7, etc.
        
        # This is still O(2^n * n^2) which is too slow for n=20 (2^20 is 1M, n^2 is 400 -> 400M ops).
        # We need a better verification or a search that *constructs* validity.
        
        # Actually, for n=20, 2^20 is ~1 million subsets.
        # Checking each subset takes O(k^2) or O(k * bits).
        # 1M * 400 ops is 400M ops. In Python, this is too slow (seconds to minutes).
        # We need C-level speed or a smarter algorithm.
        
        # Alternative: Use the fact that we are doing local search.
        # We only need to check if the *current* graph is valid.
        # If we are flipping edges, we can update the status of affected subsets.
        # But tracking all subsets is hard.
        
        # Let's stick to the "check all subsets" but optimize with NumPy or bit operations?
        # Or just limit the search to finding a graph and verify at the end?
        # If verification is too slow, we might timeout.
        
        # Let's try a different approach: 
        # Generate a random graph. 
        # Check for regular induced subgraphs of size 6.
        # If found, flip an edge in it.
        # Repeat.
        # We only check size 6? 
        # If a graph has no regular induced subgraph of size 6, does it imply no larger ones?
        # No. A 7-vertex regular graph might not contain a 6-vertex regular induced subgraph.
        # Example: K_7 is regular (deg 6). Any 6 vertices induce K_6 (deg 5). 
        # So if we only check size 6, we might miss a size 7 violation.
        # However, it is known that if a graph has no regular induced subgraph of size k, it might still have one of size k+1.
        # But in practice, for random graphs, small regular subgraphs are the main obstacle.
        # The task requires "No set of 6 or more".
        
        # Let's implement a verification that checks sizes 6, 7, 8... up to n.
        # To make it fast enough for Python, we must be efficient.
        # n=20. Number of subsets of size 6 is 38760.
        # Size 7 is 77520.
        # Size 8 is 125970.
        # Total subsets of size >= 6 is roughly 1M.
        # Checking each takes ~20-40 ops.
        # 1M * 40 = 40M ops.
        # Python can do ~10M-50M simple ops per second.
        # So one verification takes ~1 second.
        # We have 110 seconds. We can do ~100 verifications.
        # This is enough for a local search if the convergence is fast.
        
        return True # Placeholder, actual logic below
        
    # Better Verification Function
    def verify(n, adj_bits):
        # adj_bits[i] is an int with bits set for neighbors of i.
        # Check all subsets of size k from 6 to n.
        # We can prune: if a subset S is regular, its neighbors outside S don't matter for the "induced" property, 
        # but we are checking if S *is* the induced subgraph.
        
        # To speed up, we check smaller sizes first.
        # If we find a violation, return False.
        
        # We'll use itertools.combinations for clarity, but it's slow.
        # Bitmask iteration is faster.
        
        # Let's try a heuristic: Check only size 6 first.
        # If no size 6 violation, check size 7.
        # ...
        
        # For n=20, let's just do it.
        for k in range(6, n + 1):
            # Iterate through all combinations of k vertices
            # Using recursion or itertools
            # itertools.combinations(range(n), k)
            for subset in itertools.combinations(range(n), k):
                # Calculate degree of each vertex in subset
                degs = []
                for u in subset:
                    deg = 0
                    mask = adj_bits[u]
                    for v in subset:
                        if u != v and (mask >> v) & 1:
                            deg += 1
                    degs.append(deg)
                
                # Check if all degrees are equal
                first_deg = degs[0]
                regular = True
                for d in degs[1:]:
                    if d != first_deg:
                        regular = False
                        break
                
                if regular:
                    return False # Violation found
        
        return True # Valid
    
    # Local Search
    best_n = 0
    best_edges = []
    
    # Try n=20
    target_n = 20
    random.seed(42)
    
    # Initialize adj_bits
    adj_bits = [0] * target_n
    for i in range(target_n):
        for j in range(i + 1, target_n):
            if random.random() < 0.5:
                adj_bits[i] |= (1 << j)
                adj_bits[j] |= (1 << i)
                
    current_n = target_n
    valid = verify(current_n, adj_bits)
    
    if not valid:
        # Local search
        # We need to find a violation to fix.
        # The verify function above doesn't return the violation.
        # Let's modify verify to return the violating subset.
        
        def find_violation(n, adj_bits):
            for k in range(6, n + 1):
                for subset in itertools.combinations(range(n), k):
                    degs = []
                    for u in subset:
                        deg = 0
                        mask = adj_bits[u]
                        for v in subset:
                            if u != v and (mask >> v) & 1:
                                deg += 1
                        degs.append(deg)
                    
                    first_deg = degs[0]
                    regular = True
                    for d in degs[1:]:
                        if d != first_deg:
                            regular = False
                            break
                    
                    if regular:
                        return subset
            return None
        
        violation = find_violation(current_n, adj_bits)
        
        while violation is not None:
            if time.time() - start_time > time_limit:
                break
            
            # Pick two random vertices in the violation and flip their edge
            u, v = random.sample(violation, 2)
            # Flip edge u-v
            if (adj_bits[u] >> v) & 1:
                adj_bits[u] &= ~(1 << v)
                adj_bits[v] &= ~(1 << u)
            else:
                adj_bits[u] |= (1 << v)
                adj_bits[v] |= (1 << u)
            
            # Re-check for violation
            # Optimization: The previous violation might be fixed, but a new one might appear.
            # We must re-scan from the beginning or just re-run find_violation.
            violation = find_violation(current_n, adj_bits)
            
            if violation is None:
                valid = True
                break
        
        if valid:
            best_n = current_n
            best_edges = [[i, j] for i in range(current_n) for j in range(i + 1, current_n) if (adj_bits[i] >> j) & 1]
    
    if best_n == 0:
        # Fallback: Try smaller n? Or just return a trivial valid graph?
        # A graph with n=5 is always valid (no subset of size 6).
        # But we want to maximize n.
        # If we failed to find n=20, maybe try n=19?
        # Given time constraints, if n=20 fails, we likely won't find n=19 either quickly if the search is bad.
        # However, n=20 is known to exist. The search should find it.
        # If it doesn't, we return the best we have.
        
        # Let's try a quick search for n=19 if n=20 failed?
        # No, let's just return a valid graph on n=5 to avoid 0 score?
        # Wait, the task says "Maximize n". If we return n=5, score is low.
        # But returning an invalid graph scores 0.
        # So returning n=5 is better than 0.
        
        # Actually, let's try to construct a valid graph on n=20 more reliably.
        # The random start + local search is standard.
        # If it fails, we might have a bug or bad luck.
        
        # Let's add a fallback to a known valid structure if possible?
        # No, let's just return the best found.
        # If best_n is 0, we return n=5.
        return {'n': 5, 'edges': []}
        
    return {'n': best_n, 'edges': best_edges}
