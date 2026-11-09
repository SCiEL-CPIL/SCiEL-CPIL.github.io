"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
from itertools import combinations

def _score(values, edges):
    """Compute the raw value: max(distinct sums, distinct products)."""
    sums = set()
    products = set()
    for i, j in edges:
        s = values[i] + values[j]
        p = values[i] * values[j]
        sums.add(s)
        products.add(p)
    return max(len(sums), len(products))

def _evaluate_fast(values, edges):
    """Evaluate a configuration, returning score and detailed stats."""
    sums = set()
    products = set()
    for i, j in edges:
        s = values[i] + values[j]
        p = values[i] * values[j]
        sums.add(s)
        products.add(p)
    return max(len(sums), len(products)), len(sums), len(products)

def construct():
    """Construct values and edges minimizing max(distinct sums, distinct products)."""
    random.seed(42)
    
    n = 14
    num_edges = 35
    all_possible_edges = list(combinations(range(n), 2))
    
    # Strategy: Use values that are divisors of a highly composite number
    # to encourage product collisions.
    
    # Generate candidate value sets based on divisors of highly composite numbers
    base_numbers = [2520, 5040, 7560, 10080, 15120, 20160]
    candidate_value_sets = []
    
    for base in base_numbers:
        divisors = []
        for i in range(1, int(base**0.5) + 1):
            if base % i == 0:
                divisors.append(i)
                if i != base // i:
                    divisors.append(base // i)
        
        # Filter divisors <= 10000
        divisors = [d for d in divisors if d <= 10000]
        if len(divisors) < n:
            continue
            
        # Create multiple random subsets of size n from these divisors
        for _ in range(5):
            subset = random.sample(divisors, n)
            # Ensure distinctness (random.sample does this)
            if len(set(subset)) == n:
                candidate_value_sets.append(subset)
                
    # Add some manual "dense" sets if not enough generated
    if len(candidate_value_sets) < 20:
        candidate_value_sets.append([1, 2, 3, 4, 6, 8, 9, 12, 18, 24, 36, 48, 72, 144])
        candidate_value_sets.append([1, 2, 3, 4, 6, 8, 9, 12, 15, 20, 24, 30, 40, 60])
        candidate_value_sets.append([2, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256])

    best_score = float('inf')
    best_values = None
    best_edges = None
    
    # Time limit per value set evaluation
    time_limit_per_set = 5.0
    start_time = time.time()
    
    for values in candidate_value_sets:
        if time.time() - start_time > 50: # Leave 10s for final polish
            break
            
        # Pre-compute sums and products for all edges
        edge_props = {}
        for i, j in all_possible_edges:
            s = values[i] + values[j]
            p = values[i] * values[j]
            edge_props[(i, j)] = (s, p)
            
        # Initialize with a greedy selection
        selected_edges = []
        selected_sums = set()
        selected_products = set()
        remaining = set(all_possible_edges)
        
        # Greedy: Pick edges that add the fewest new distinct sums/products
        for _ in range(num_edges):
            best_edge = None
            best_cost = float('inf')
            
            for e in remaining:
                s, p = edge_props[e]
                new_s = 1 if s not in selected_sums else 0
                new_p = 1 if p not in selected_products else 0
                cost = new_s + new_p
                if cost < best_cost:
                    best_cost = cost
                    best_edge = e
                elif cost == best_cost:
                    # Tie-breaker: prefer edges with more "popular" sums/products?
                    # Just pick the first one found
                    pass
            
            if best_edge is None:
                break
                
            selected_edges.append(best_edge)
            s, p = edge_props[best_edge]
            selected_sums.add(s)
            selected_products.add(p)
            remaining.discard(best_edge)
            
        current_edges = selected_edges[:]
        current_score, _, _ = _evaluate_fast(values, current_edges)
        
        # Simulated Annealing / Local Search
        # We will try to improve by swapping edges
        # State: set of edges
        # Move: remove one edge, add one different edge
        
        t = 1.0
        t_min = 0.01
        alpha = 0.995
        
        set_start_time = time.time()
        
        while t > t_min and (time.time() - set_start_time) < time_limit_per_set:
            # Pick a random edge to remove
            out_idx = random.randint(0, len(current_edges) - 1)
            e_out = current_edges[out_idx]
            
            # Pick a random edge to add (not currently in set)
            current_set = set(current_edges)
            available = [e for e in all_possible_edges if e not in current_set]
            if not available:
                break
            e_in = random.choice(available)
            
            new_edges = current_edges[:]
            new_edges[out_idx] = e_in
            
            new_score, _, _ = _evaluate_fast(values, new_edges)
            
            if new_score < current_score:
                current_edges = new_edges
                current_score = new_score
            else:
                # Metropolis criterion
                delta = new_score - current_score
                if delta > 0:
                    prob = random.random() < 1.0 / (1.0 + delta * t)
                    if prob:
                        current_edges = new_edges
                        current_score = new_score
                else:
                    # Should not happen if new_score < current_score handled above
                    current_edges = new_edges
                    current_score = new_score
            
            t *= alpha
        
        if current_score < best_score:
            best_score = current_score
            best_values = values[:]
            best_edges = [list(e) for e in current_edges]
            
            # Early stop if we reach a very good score
            if best_score <= 20:
                break
                
    if best_values is None:
        best_values = list(range(1, 15))
        best_edges = [list(e) for e in all_possible_edges[:35]]
        
    # Final polish on the best found
    current_edges = [tuple(e) for e in best_edges]
    current_score, _, _ = _evaluate_fast(best_values, current_edges)
    
    # Quick local search again
    t = 0.5
    t_min = 0.01
    alpha = 0.99
    
    final_start_time = time.time()
    while t > t_min and (time.time() - final_start_time) < 5.0:
        out_idx = random.randint(0, len(current_edges) - 1)
        e_out = current_edges[out_idx]
        
        current_set = set(current_edges)
        available = [e for e in all_possible_edges if e not in current_set]
        if not available:
            break
        e_in = random.choice(available)
        
        new_edges = current_edges[:]
        new_edges[out_idx] = e_in
        
        new_score, _, _ = _evaluate_fast(best_values, new_edges)
        
        if new_score < current_score:
            current_edges = new_edges
            current_score = new_score
        else:
            delta = new_score - current_score
            if delta > 0:
                if random.random() < 1.0 / (1.0 + delta * t):
                    current_edges = new_edges
                    current_score = new_score
            else:
                current_edges = new_edges
                current_score = new_score
        
        t *= alpha
        
    best_edges = [list(e) for e in current_edges]
    best_score = current_score
    
    return {'values': best_values, 'edges': best_edges}
# EVOLVE-BLOCK-END
