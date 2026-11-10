"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin (90s limit)
    
    n = 14
    m = 35
    maximum = 10000
    
    # Helper to compute the objective
    def _raw_value(values, edges):
        sums = set()
        prods = set()
        for i, j in edges:
            v = values[i]
            w = values[j]
            sums.add(v + w)
            prods.add(v * w)
        return max(len(sums), len(prods))

    def score_pair(values, edges):
        return _raw_value(values, edges)

    # Strategy:
    # The current approach treats value selection and edge selection somewhat independently.
    # A materially different approach is to use a unified Simulated Annealing (SA) 
    # that simultaneously perturbs both the values and the edges.
    # 
    # Key Insight: 
    # To minimize distinct sums, we want values to form "sum-collisions" (e.g., arithmetic progressions).
    # To minimize distinct products, we want values to form "product-collisions" (e.g., harmonic-like structures or specific multiplicative relations).
    # These goals can conflict. 
    # 
    # For example, if values are close together (e.g., 1..14), sums are small and collide often.
    # Products: 1*14=14, 2*7=14, 4*? no. 1*12=12, 2*6=12, 3*4=12.
    # Let's check 1..14 with random edges.
    
    # Initial candidate: Powers of 2 (baseline 35)
    vals_p2 = [2**i for i in range(n)]
    # For powers of 2, every sum 2^i+2^j is unique, and every product 2^(i+j) is unique.
    # So any 35 edges give 35 distinct sums and 35 distinct products. Score = 35.
    
    # Let's try to find a configuration with score < 35.
    # If we can get score 34, that's an improvement.
    
    # Let's initialize with a "dense" set of values to encourage collisions.
    # Values: 1, 2, ..., 14.
    # Possible sums: 3 to 27. (25 distinct sums)
    # Possible products: many collisions.
    
    # We will use SA to optimize (values, edges) jointly.
    
    # State:
    # values: list of 14 distinct integers in [1, 10000]
    # edges: set of 35 (i, j) pairs
    
    # Start with a good heuristic:
    # Try small values first, as they allow more collisions in limited range.
    current_values = list(range(1, n+1))
    
    # Initial edges: pick 35 edges that maximize collisions?
    # Let's just start with random edges or the first 35.
    all_edges = list(combinations(range(n), 2))
    current_edges = set(random.sample(all_edges, m))
    
    current_score = _raw_value(current_values, current_edges)
    best_values = current_values.copy()
    best_edges = set(current_edges)
    best_score = current_score
    
    # Simulated Annealing
    T = 100.0
    cooling = 0.999
    min_T = 0.01
    
    # Precompute all edge sums and products for current values to speed up evaluation?
    # No, values change, so we recompute. But n=14, m=35 is small.
    
    # To speed up, we can maintain counts of sums and products.
    # But with value changes, we have to recompute all edges involving changed vertices.
    # Simpler: just recompute the whole score. 35 edges is very fast.
    
    while T > min_T and time.time() < deadline:
        # Perturbation
        # 1. Change one value
        # 2. Swap one edge
        # 3. Change two values
        
        r = random.random()
        
        new_values = current_values.copy()
        new_edges = set(current_edges)
        
        if r < 0.5:
            # Change one value
            idx = random.randint(0, n-1)
            # Pick a new random value, ensuring distinctness
            # To be efficient, just pick random and check
            new_val = random.randint(1, maximum)
            # Ensure distinctness
            if new_val not in new_values:
                new_values[idx] = new_val
            else:
                # Try a few times to find a valid distinct value
                for _ in range(10):
                    new_val = random.randint(1, maximum)
                    if new_val not in new_values:
                        new_values[idx] = new_val
                        break
        elif r < 0.8:
            # Swap one edge
            # Remove a random edge, add a random non-existent edge
            e_out = random.choice(list(current_edges))
            # Find a valid new edge
            while True:
                e_in = random.choice(all_edges)
                if e_in not in current_edges:
                    break
            new_edges.remove(e_out)
            new_edges.add(e_in)
        else:
            # Change two values
            idx1, idx2 = random.sample(range(n), 2)
            new_val1 = random.randint(1, maximum)
            new_val2 = random.randint(1, maximum)
            
            # Check distinctness
            temp_vals = new_values.copy()
            temp_vals[idx1] = new_val1
            temp_vals[idx2] = new_val2
            
            if len(set(temp_vals)) == n:
                new_values = temp_vals
            else:
                # Retry or skip
                pass

        # Evaluate
        new_score = _raw_value(new_values, new_edges)
        
        # Acceptance
        if new_score < current_score:
            current_values = new_values
            current_edges = new_edges
            current_score = new_score
            
            if current_score < best_score:
                best_score = current_score
                best_values = current_values.copy()
                best_edges = set(current_edges)
        else:
            # Accept with probability exp(-delta/T)
            delta = new_score - current_score
            if delta > 0:
                prob = pow(2.71828, -delta / T)
                if random.random() < prob:
                    current_values = new_values
                    current_edges = new_edges
                    current_score = new_score
        
        T *= cooling
        
        # If we hit the theoretical minimum? 
        # Minimum distinct sums for 35 edges?
        # Min distinct products for 35 edges?
        # If best_score <= 20, we might stop early? 
        # But let's keep going to be safe.
        
    # Final check: ensure types are correct
    # Values must be int, edges must be list of lists/tuples of int
    
    # One more local optimization pass on the best solution found
    # Try to improve best_edges for best_values
    # and best_values for best_edges
    
    # 1. Optimize edges for best_values
    # Greedy: sort all edges by "collision potential"?
    # Let's just do a quick random restart hill climb on edges for the best values found.
    if time.time() < deadline - 5:
        # Optimize edges
        best_edges_local = set(best_edges)
        current_edges_local = set(best_edges)
        current_score_local = _raw_value(best_values, current_edges_local)
        
        for _ in range(2000):
            if time.time() > deadline - 5:
                break
            e_out = random.choice(list(current_edges_local))
            while True:
                e_in = random.choice(all_edges)
                if e_in not in current_edges_local:
                    break
            new_edges_local = set(current_edges_local)
            new_edges_local.remove(e_out)
            new_edges_local.add(e_in)
            new_score_local = _raw_value(best_values, new_edges_local)
            
            if new_score_local < current_score_local:
                current_edges_local = new_edges_local
                current_score_local = new_score_local
                if current_score_local < best_score:
                    best_score = current_score_local
                    best_edges = set(current_edges_local)
        
        # Optimize values for best_edges
        # Random restart hill climb on values
        current_values_local = best_values.copy()
        current_score_local = _raw_value(current_values_local, best_edges)
        
        for _ in range(2000):
            if time.time() > deadline - 2:
                break
            idx = random.randint(0, n-1)
            new_val = random.randint(1, maximum)
            if new_val not in current_values_local:
                new_values_local = current_values_local.copy()
                new_values_local[idx] = new_val
                new_score_local = _raw_value(new_values_local, best_edges)
                
                if new_score_local < current_score_local:
                    current_values_local = new_values_local
                    current_score_local = new_score_local
                    if current_score_local < best_score:
                        best_score = current_score_local
                        best_values = current_values_local.copy()

    # Return best
    # Ensure edges are lists of ints
    edges_list = [list(e) for e in best_edges]
    # Sort edges for determinism? Not required but nice.
    edges_list.sort()
    
    return {
        'values': list(best_values),
        'edges': edges_list
    }
# EVOLVE-BLOCK-END
