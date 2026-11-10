import time
import random
from itertools import combinations
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin
    
    n = 14
    m = 35
    maximum = 10000
    
    # Generate diverse candidate value sets
    # Focus on sets with small numbers to maximize collisions
    candidates = []
    
    # 1. Sequential and near-sequential
    for start in range(1, 5):
        for step in range(1, 4):
            vals = [start + i * step for i in range(n)]
            if max(vals) <= maximum and len(set(vals)) == n:
                candidates.append(vals)
                
    # 2. Multiplicative structures (powers of 2, 3, etc.) - might have unique products, but let's see
    # Actually, powers of 2 have unique products. We want COLLISIONS.
    # So we want numbers that are highly composite or form arithmetic progressions.
    
    # 3. Random sets of small integers
    random.seed(42)
    for _ in range(50):
        # Pick 14 distinct random integers from 1 to 50
        vals = random.sample(range(1, 51), n)
        candidates.append(sorted(vals))
        
    # 4. Random sets with bias towards small numbers
    for _ in range(50):
        vals = random.sample(range(1, 100), n)
        candidates.append(sorted(vals))
        
    # 5. Specific known good structures?
    # E.g., sets that form many rectangles in multiplication table
    # a*b = c*d
    # Let's try to construct sets with many such relations
    # This is hard to do manually, so we rely on SA to find good edge subsets
    # for a variety of sets.
    
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
    
    # Simulated Annealing for edge selection
    def sa_optimize_edges(vals, initial_edges, time_limit):
        """
        Optimizes the set of edges to minimize max(distinct_sums, distinct_products).
        """
        current_edges = set(tuple(sorted(e)) for e in initial_edges)
        
        # Precompute sums and products for all edges
        edge_sums = {}
        edge_prods = {}
        for i, j in all_possible_edges:
            edge_sums[(i, j)] = vals[i] + vals[j]
            edge_prods[(i, j)] = vals[i] * vals[j]
            
        # Initial cost
        sum_counts = defaultdict(int)
        prod_counts = defaultdict(int)
        for e in current_edges:
            sum_counts[edge_sums[e]] += 1
            prod_counts[edge_prods[e]] += 1
            
        current_cost = max(len(sum_counts), len(prod_counts))
        best_cost = current_cost
        best_edges_state = set(current_edges)
        
        # SA Parameters
        T = 5.0  # Initial temperature
        cooling_rate = 0.999
        min_T = 0.01
        
        start_sa = time.time()
        
        while T > min_T and time.time() - start_sa < time_limit:
            # Pick a random edge to remove
            if not current_edges:
                break
            e_out = random.choice(list(current_edges))
            
            # Pick a random edge to add (not in current set)
            # We can sample from all edges and check
            # Or maintain a list of non-selected edges
            # For simplicity, sample from all and retry if in set
            # Since m=35 and total=91, probability of collision is ~38%. Acceptable.
            while True:
                e_in = tuple(sorted(random.choice(all_possible_edges)))
                if e_in not in current_edges:
                    break
                    
            # Calculate cost change
            # Remove e_out
            s_out = edge_sums[e_out]
            p_out = edge_prods[e_out]
            sum_counts[s_out] -= 1
            if sum_counts[s_out] == 0:
                del sum_counts[s_out]
            prod_counts[p_out] -= 1
            if prod_counts[p_out] == 0:
                del prod_counts[p_out]
                
            # Add e_in
            s_in = edge_sums[e_in]
            p_in = edge_prods[e_in]
            sum_counts[s_in] += 1
            prod_counts[p_in] += 1
            
            new_cost = max(len(sum_counts), len(prod_counts))
            
            delta = new_cost - current_cost
            
            if delta < 0 or random.random() < pow(2.71828, -delta / T):
                # Accept
                current_edges.remove(e_out)
                current_edges.add(e_in)
                current_cost = new_cost
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_edges_state = set(current_edges)
            else:
                # Reject, revert
                sum_counts[s_in] -= 1
                if sum_counts[s_in] == 0:
                    del sum_counts[s_in]
                prod_counts[p_in] -= 1
                if prod_counts[p_in] == 0:
                    del prod_counts[p_in]
                
                sum_counts[s_out] += 1
                prod_counts[p_out] += 1
                
            T *= cooling_rate
            
        return best_cost, [list(e) for e in best_edges_state]
    
    # Iterate over candidates
    # Time budget per candidate
    num_candidates = len(unique_candidates)
    time_per_candidate = (deadline - time.time()) / num_candidates - 1.0 # Reserve 1s for overhead
    
    for idx, vals in enumerate(unique_candidates):
        if time.time() > deadline - 5.0:
            break
            
        # Ensure time_per_candidate is positive
        if time_per_candidate < 0.5:
            time_per_candidate = 0.5
            
        # Initial edges: Greedy or Random
        # Random initialization is often better for SA
        random_edges = random.sample(all_possible_edges, m)
        
        # Run SA
        cost, edges = sa_optimize_edges(vals, random_edges, time_per_candidate)
        
        if cost < best_score:
            best_score = cost
            best_edges = edges
            best_vals = vals
            
        # Optional: Run a second SA with a different seed or initial state if time permits
        # But let's keep it simple first.
        
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
