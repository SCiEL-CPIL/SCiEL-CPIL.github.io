"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 90s limit
    
    n = 14
    m = 35
    max_val = 10000
    
    # Helper to evaluate a candidate (values, edges)
    def evaluate(values, edges):
        sums = set()
        prods = set()
        for u, v in edges:
            sums.add(values[u] + values[v])
            prods.add(values[u] * values[v])
        return max(len(sums), len(prods))
    
    # Precompute all possible edges
    all_edges = list(combinations(range(n), 2))
    num_edges = len(all_edges)
    
    # Map edge index to (u, v)
    edge_map = {i: (u, v) for i, (u, v) in enumerate(all_edges)}

    def generate_structured_values(base_strategy):
        """
        Generate values with high collision potential.
        Strategy 0: Multiples of a small number B. e.g., B * {1..14}
        Strategy 1: Multiples of B * {primes or small composites}
        Strategy 2: Random multiples of B
        """
        values = [0] * n
        if base_strategy == 0:
            # Pure multiples of a small base
            # To keep distinct, use consecutive integers
            B = random.randint(2, 50)
            # Ensure max value <= 10000
            max_mult = 10000 // B
            if max_mult < n:
                B = 10000 // n
                max_mult = n
            # Pick n distinct multipliers
            mults = random.sample(range(1, max_mult + 1), n)
            values = [B * m for m in mults]
        elif base_strategy == 1:
            # Mix of small base and larger numbers to create specific product matches
            # e.g., 10*1, 10*2, 10*3, 10*4, 10*5, 10*6, 10*7, 10*8, 10*9, 10*10, 10*11, 10*12, 10*13, 10*14
            B = random.randint(2, 20)
            mults = list(range(1, n + 1))
            # Shuffle to break simple sum patterns? No, keep order for now, edges will handle it
            values = [B * m for m in mults]
            # Randomly perturb some to break symmetry?
            for i in range(n):
                if random.random() < 0.2:
                    # Change to another multiple
                    new_m = random.randint(1, 10000 // B)
                    if new_m not in values:
                        values[i] = B * new_m
        else:
            # Random values biased towards small numbers
            values = sorted(random.sample(range(1, 100), n))
            
        # Ensure distinctness (just in case)
        seen = set()
        final_vals = []
        for v in values:
            if v not in seen:
                seen.add(v)
                final_vals.append(v)
        while len(final_vals) < n:
            new_v = random.randint(1, max_val)
            if new_v not in seen:
                seen.add(new_v)
                final_vals.append(new_v)
        
        return final_vals

    def optimize_edges_sa(values, initial_edges, max_time):
        """
        Simulated Annealing on the edge set.
        State: set of 35 edges.
        Move: Remove one edge, Add one different edge.
        Cost: max(distinct_sums, distinct_prods)
        """
        current_edges = set(initial_edges)
        current_score = evaluate(values, list(current_edges))
        
        # Precompute sums and prods for all possible edges
        edge_sums = [values[u] + values[v] for u, v in all_edges]
        edge_prods = [values[u] * values[v] for u, v in all_edges]
        
        # To speed up evaluation, we maintain counts of sums and prods
        # But for n=14, m=35, full evaluation is 35 iterations. Very fast.
        # Let's just use full evaluation for simplicity and correctness, 
        # but optimize the inner loop.
        
        best_edges = set(initial_edges)
        best_score = current_score
        
        T = 10.0
        cooling = 0.999
        min_T = 0.1
        
        while time.time() < max_time and T > min_T:
            # Pick a random edge to remove
            edges_list = list(current_edges)
            remove_idx = random.randint(0, len(edges_list) - 1)
            e_remove = edges_list[remove_idx]
            
            # Pick a random edge to add (not currently in set)
            # We can sample from all_edges until we find one not in current_edges
            # Or precompute available?
            # Since 91 edges, 35 in set, 56 available. Sampling is fine.
            while True:
                e_add = random.randint(0, num_edges - 1)
                if e_add not in current_edges:
                    break
            
            # Apply move
            current_edges.remove(e_remove)
            current_edges.add(e_add)
            
            new_score = evaluate(values, list(current_edges))
            
            delta = new_score - current_score
            if delta < 0 or random.random() < np.exp(-delta / T):
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_edges = set(current_edges)
            else:
                # Revert
                current_edges.remove(e_add)
                current_edges.add(e_remove)
            
            T *= cooling
            
            # Periodic restart if stuck?
            # If no improvement in last 1000 steps, maybe reheat?
            # For now, simple cooling.
            
        return list(best_edges), best_score

    best_global_score = float('inf')
    best_global_result = None
    
    # Try multiple value structures
    strategies = [0, 1, 2, 0, 1, 0, 1, 2]
    
    for strat in strategies:
        if time.time() > deadline:
            break
            
        values = generate_structured_values(strat)
        
        # Initial edges: Random selection
        initial_edges = random.sample(range(num_edges), m)
        
        # Optimize edges using SA
        opt_edges, opt_score = optimize_edges_sa(values, initial_edges, max_time=deadline - 5)
        
        if opt_score < best_global_score:
            best_global_score = opt_score
            best_global_result = {
                'values': values,
                'edges': [list(edge_map[e]) for e in opt_edges]
            }
            
            # If we get a very good score, try to refine values?
            # For now, stick to edge optimization on fixed structured values.
            # The structure of values is key to low distinct prods.
            
    # Fallback if no result found (should not happen)
    if best_global_result is None:
        values = list(range(1, n + 1))
        edges = [list(e) for e in all_edges[:m]]
        best_global_result = {'values': values, 'edges': edges}
        best_global_score = evaluate(values, edges)
        
    # Final Type Check and Cleanup
    res = {
        'values': [int(v) for v in best_global_result['values']],
        'edges': [[int(u), int(v)] for u, v in best_global_result['edges']]
    }
    
    return res

# EVOLVE-BLOCK-END
