import time
import random
from itertools import combinations
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin for 90s limit
    
    n = 14
    m = 35
    maximum = 10000
    
    # Generate candidate value sets
    # Small integers maximize collisions in sums and products
    candidates = []
    
    # 1. Sequential small integers
    for start in range(1, 10):
        vals = [start + i for i in range(n)]
        if max(vals) <= maximum and len(set(vals)) == n:
            candidates.append(vals)
            
    # 2. Random sets of small integers (1 to 20)
    # This is crucial: small numbers create many sum/product collisions
    random.seed(42)
    for _ in range(100):
        # Ensure distinct
        while True:
            vals = random.sample(range(1, 21), n)
            if len(set(vals)) == n:
                break
        candidates.append(sorted(vals))
        
    # 3. Random sets of small integers (1 to 30)
    for _ in range(50):
        while True:
            vals = random.sample(range(1, 31), n)
            if len(set(vals)) == n:
                break
        candidates.append(sorted(vals))
        
    # 4. Specific structured sets with known multiplicative relations
    # e.g., 1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 27, 32, 36
    # These have many a*b = c*d relations
    structured = [1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 27, 32, 36]
    if max(structured) <= maximum:
        candidates.append(structured)
        
    # Another structured set
    structured2 = [2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 27, 32, 36, 48]
    if max(structured2) <= maximum:
        candidates.append(structured2)

    # Deduplicate candidates
    seen = set()
    unique_candidates = []
    for v in candidates:
        key = tuple(v)
        if key not in seen:
            seen.add(key)
            unique_candidates.append(v)
            
    if not unique_candidates:
        unique_candidates = [list(range(1, 15))]
        
    all_possible_edges = [list(e) for e in combinations(range(n), 2)]
    num_edges_total = len(all_possible_edges)
    
    best_score = float('inf')
    best_edges = None
    best_vals = None
    
    def get_cost(edges_set, edge_sums, edge_prods):
        """Calculate the cost of a set of edges."""
        if not edges_set:
            return 0
        sums = set()
        prods = set()
        for e in edges_set:
            sums.add(edge_sums[e])
            prods.add(edge_prods[e])
        return max(len(sums), len(prods))
        
    def optimize_edges_for_vals(vals, time_limit):
        """
        Optimizes the set of edges for a specific value set.
        Uses Greedy Initialization + Simulated Annealing.
        """
        # Precompute sums and products for all edges
        edge_sums = {}
        edge_prods = {}
        for i, j in all_possible_edges:
            edge_sums[(i, j)] = vals[i] + vals[j]
            edge_prods[(i, j)] = vals[i] * vals[j]
            
        # Greedy Initialization
        # Start with a random set of m edges? Or build up?
        # Building up is better for finding dense clusters of sums/prods.
        # However, we need exactly m edges.
        # Let's start with a random set and refine.
        
        # Random Initialization
        current_edges = set(tuple(sorted(e)) for e in random.sample(all_possible_edges, m))
        
        # Calculate initial cost
        current_cost = get_cost(current_edges, edge_sums, edge_prods)
        best_cost = current_cost
        best_edges_state = set(current_edges)
        
        # Simulated Annealing
        T = 10.0  # Initial temperature
        cooling_rate = 0.9995
        min_T = 0.01
        
        start_sa = time.time()
        
        # Precompute edge lists for faster sampling
        # We need to sample an edge OUT and an edge IN
        # To avoid O(N) checks for "not in set", we can just sample and retry.
        # m=35, total=91. P(in set) ~ 0.38. Retry is fast.
        
        while T > min_T and time.time() - start_sa < time_limit:
            # Pick a random edge to remove (e_out)
            e_out = random.choice(list(current_edges))
            
            # Pick a random edge to add (e_in) not in current set
            while True:
                e_in = tuple(sorted(random.choice(all_possible_edges)))
                if e_in not in current_edges:
                    break
                    
            # Calculate cost change
            # We need to know if removing e_out eliminates a sum/prod, 
            # and if adding e_in introduces a new sum/prod.
            
            # Get current unique sums and prods counts
            # For efficiency, maintain counts
            # But for simplicity and correctness first, let's recompute or use a fast delta.
            # Recomputing 35 edges is very fast.
            
            # Let's try a faster delta approach.
            # Maintain sum_counts and prod_counts
            # This requires updating state on accept/reject.
            
            # Let's stick to recomputation for simplicity in this step, 
            # but optimize if it's too slow. 35 edges * 90s is plenty.
            
            new_edges = set(current_edges)
            new_edges.remove(e_out)
            new_edges.add(e_in)
            
            new_cost = get_cost(new_edges, edge_sums, edge_prods)
            
            delta = new_cost - current_cost
            
            if delta < 0 or random.random() < pow(2.71828, -delta / T):
                # Accept
                current_edges = new_edges
                current_cost = new_cost
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_edges_state = set(current_edges)
            else:
                # Reject (state unchanged)
                pass
                
            T *= cooling_rate
            
        return best_cost, [list(e) for e in best_edges_state]
        
    # Iterate over candidates
    num_candidates = len(unique_candidates)
    # Allocate time to each candidate
    # Prioritize the first few (sequential/small random) as they are likely best
    time_per_candidate = (deadline - time.time()) / num_candidates
    
    for idx, vals in enumerate(unique_candidates):
        if time.time() > deadline - 2.0:
            break
            
        # Ensure time_per_candidate is positive and reasonable
        if time_per_candidate < 0.2:
            time_per_candidate = 0.2
            
        # Run optimization
        # We might want to run multiple restarts for the best candidates
        # For now, one SA run per candidate with the allocated time
        
        cost, edges = optimize_edges_for_vals(vals, time_per_candidate)
        
        if cost < best_score:
            best_score = cost
            best_edges = edges
            best_vals = vals
            
        # If we found a very good score, we can stop early or use remaining time to polish
        if best_score <= 10: # Very good
             # Use remaining time to polish this specific solution
             remaining = deadline - time.time() - 2.0
             if remaining > 1.0:
                 cost, edges = optimize_edges_for_vals(best_vals, remaining)
                 if cost < best_score:
                     best_score = cost
                     best_edges = edges
             break

    # Fallback
    if best_edges is None:
        best_vals = unique_candidates[0]
        best_edges = all_possible_edges[:m]
        
    # Final validation
    assert len(best_vals) == n
    assert len(set(best_vals)) == n
    assert all(isinstance(v, int) and not isinstance(v, bool) for v in best_vals)
    assert all(1 <= v <= maximum for v in best_vals)
    assert len(best_edges) == m
    assert len(set(tuple(sorted(e)) for e in best_edges)) == m
    
    return {'values': list(best_vals), 'edges': [list(e) for e in best_edges]}
