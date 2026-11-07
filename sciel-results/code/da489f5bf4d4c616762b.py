"""Strategy: Joint optimization of value sets and edge selection via Simulated Annealing."""
# EVOLVE-BLOCK-START
import itertools
import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    n = 14
    m = 35
    
    # Initial values: Powers of 2 and 3 (from current best)
    base_vals = [1, 2, 4, 8, 3, 6, 12, 24, 16, 9, 18, 36, 72, 48]
    current_values = list(base_vals)
    
    # Initial edges: Greedy selection for current values
    def get_best_edges(values):
        all_edges = list(itertools.combinations(range(n), 2))
        sums = {}
        for u, v in all_edges:
            s = values[u] + values[v]
            if s not in sums:
                sums[s] = []
            sums[s].append((u, v))
        
        sorted_sums = sorted(sums.items(), key=lambda x: -len(x[1]))
        selected_edges = []
        used = set()
        
        for s, edges in sorted_sums:
            for e in edges:
                if e not in used:
                    selected_edges.append(e)
                    used.add(e)
                    if len(selected_edges) == m:
                        break
            if len(selected_edges) == m:
                break
        
        if len(selected_edges) < m:
            for e in all_edges:
                if e not in used:
                    selected_edges.append(e)
                    used.add(e)
                    if len(selected_edges) == m:
                        break
        
        return selected_edges[:m]
    
    current_edges = get_best_edges(current_values)
    
    def evaluate(values, edges):
        distinct_sums = set()
        distinct_prods = set()
        for u, v in edges:
            distinct_sums.add(values[u] + values[v])
            distinct_prods.add(values[u] * values[v])
        return max(len(distinct_sums), len(distinct_prods))
    
    current_score = evaluate(current_values, current_edges)
    best_values = list(current_values)
    best_edges = list(current_edges)
    best_score = current_score
    
    # Simulated Annealing parameters
    T = 100.0
    cooling_rate = 0.995
    min_T = 0.1
    
    # Precompute all possible edges
    all_edges = list(itertools.combinations(range(n), 2))
    
    while T > min_T and time.time() < deadline:
        # Generate neighbor
        new_values = list(current_values)
        new_edges = list(current_edges)
        
        # Randomly choose mutation type: values or edges
        if random.random() < 0.5:
            # Mutate values
            idx = random.randint(0, n - 1)
            # Try to replace with a random number, or adjust
            if random.random() < 0.5:
                # Random replacement
                new_val = random.randint(1, 10000)
                # Ensure distinctness
                if new_val in new_values:
                    # Try a few times
                    for _ in range(10):
                        new_val = random.randint(1, 10000)
                        if new_val not in new_values:
                            break
                    else:
                        continue # Skip this iteration if can't find distinct
                new_values[idx] = new_val
            else:
                # Small perturbation
                delta = random.randint(-50, 50)
                new_val = new_values[idx] + delta
                if new_val > 0 and new_val <= 10000 and new_val not in new_values:
                    new_values[idx] = new_val
                else:
                    continue
        else:
            # Mutate edges
            # Swap one edge for another
            if new_edges:
                idx = random.randint(0, len(new_edges) - 1)
                old_edge = new_edges[idx]
                # Pick a random new edge
                new_edge = random.choice(all_edges)
                if new_edge not in new_edges:
                    new_edges[idx] = new_edge
        
        # Check validity
        if len(set(new_values)) != n or any(v <= 0 or v > 10000 for v in new_values):
            continue
        if len(set(new_edges)) != m:
            continue
            
        new_score = evaluate(new_values, new_edges)
        
        # Acceptance probability
        delta_score = new_score - current_score
        if delta_score < 0:
            accept = True
        else:
            accept = random.random() < pow(2.71828, -delta_score / T)
            
        if accept:
            current_values = new_values
            current_edges = new_edges
            current_score = new_score
            
            if current_score < best_score:
                best_score = current_score
                best_values = list(current_values)
                best_edges = list(current_edges)
                
        T *= cooling_rate
        
    # Final check
    if evaluate(best_values, best_edges) > best_score:
        # Should not happen, but safety
        pass
        
    # Ensure types are correct
    vals = [int(x) for x in best_values]
    ed = [[int(u), int(v)] for u, v in best_edges]
    
    return {'values': vals, 'edges': ed}
# EVOLVE-BLOCK-END
