import random
import time
from itertools import combinations

def _score(values, edges):
    """Compute the raw value: max(distinct sums, distinct products)."""
    sums = set()
    products = set()
    for i, j in edges:
        sums.add(values[i] + values[j])
        products.add(values[i] * values[j])
    return max(len(sums), len(products))

def _greedy_edges(values, m=35):
    """
    Select m edges to minimize the number of distinct sums and products.
    Greedy strategy: iteratively pick the edge that adds the fewest new sums/products.
    """
    n = len(values)
    all_edges = list(combinations(range(n), 2))
    
    # Pre-compute sums and products for all edges
    edge_data = []
    for i, j in all_edges:
        s = values[i] + values[j]
        p = values[i] * values[j]
        edge_data.append((i, j, s, p))
        
    selected = []
    current_sums = set()
    current_prods = set()
    available = list(range(len(edge_data)))
    
    for _ in range(m):
        best_idx = -1
        best_score = float('inf')
        best_new_sums = 0
        best_new_prods = 0
        
        for idx in available:
            i, j, s, p = edge_data[idx]
            new_sums = 0 if s in current_sums else 1
            new_prods = 0 if p in current_prods else 1
            
            # We want to minimize the increase in distinct counts.
            # Since we take max at the end, we might want to balance them.
            # A simple heuristic: minimize new_sums + new_prods.
            # Or minimize max(len(current_sums)+new_sums, len(current_prods)+new_prods).
            # Let's try minimizing the potential final max.
            curr_max = max(len(current_sums), len(current_prods))
            # If we add this edge:
            next_sums = len(current_sums) + new_sums
            next_prods = len(current_prods) + new_prods
            next_max = max(next_sums, next_prods)
            
            # Prefer edges that keep next_max low.
            # Tie-breaker: prefer edges that add fewer total new items.
            score = next_max * 10 + (new_sums + new_prods)
            
            if score < best_score:
                best_score = score
                best_idx = idx
                best_new_sums = new_sums
                best_new_prods = new_prods
                
        if best_idx == -1:
            break
            
        i, j, s, p = edge_data[best_idx]
        selected.append([i, j])
        current_sums.add(s)
        current_prods.add(p)
        available.remove(best_idx)
        
    return selected

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s limit to be safe
    
    # Initial candidates
    candidates = [
        list(range(1, 15)),
        list(range(2, 16)),
        [2**i for i in range(14)],
        [3**i for i in range(14)],
        list(range(1, 20, 2))[:14], # 1, 3, 5...
        list(range(2, 29, 2))[:14], # 2, 4, 6...
    ]
    
    # Add some random small integers
    for _ in range(5):
        candidates.append(random.sample(range(1, 30), 14))
        
    best_values = None
    best_edges = None
    best_score = float('inf')
    
    # Evaluate initial candidates
    for vals in candidates:
        if len(set(vals)) != 14:
            continue
        if any(v < 1 or v > 10000 for v in vals):
            continue
            
        edges = _greedy_edges(vals)
        score = _score(vals, edges)
        
        if score < best_score:
            best_score = score
            best_values = vals[:]
            best_edges = edges[:]
            
    # Local Search on Values
    # We will try to mutate the best values found so far
    current_values = best_values[:]
    current_edges = best_edges[:]
    current_score = best_score
    
    iterations = 0
    while time.time() < deadline:
        iterations += 1
        
        # Create a new candidate by mutating one value
        new_values = current_values[:]
        idx = random.randint(0, 13)
        old_val = new_values[idx]
        
        # Strategy: try small perturbations or random replacements
        if random.random() < 0.5:
            # Perturbation
            delta = random.randint(-5, 5)
            if delta == 0:
                delta = 1
            new_val = old_val + delta
            if new_val < 1 or new_val > 10000:
                continue
            if new_val in new_values:
                # Check if it collides with another value
                # If so, skip or swap? Skipping is safer for distinctness check
                # Actually, we need distinct values.
                # Let's just check if it's in the set of other values
                others = set(new_values)
                others.remove(old_val)
                if new_val in others:
                    continue
        else:
            # Random replacement from a small range to keep structure
            new_val = random.randint(1, 50)
            if new_val == old_val:
                continue
            others = set(new_values)
            others.remove(old_val)
            if new_val in others:
                continue
            
        new_values[idx] = new_val
        
        # Ensure distinctness and bounds
        if len(set(new_values)) != 14:
            continue
        if any(v < 1 or v > 10000 for v in new_values):
            continue
            
        # Find best edges for this new value set
        new_edges = _greedy_edges(new_values)
        new_score = _score(new_values, new_edges)
        
        if new_score < current_score:
            current_score = new_score
            current_values = new_values[:]
            current_edges = new_edges[:]
            
            # Update global best
            if current_score < best_score:
                best_score = current_score
                best_values = current_values[:]
                best_edges = current_edges[:]
        
        # Occasionally restart from a random candidate to escape local minima
        if iterations % 100 == 0:
             rand_vals = random.sample(range(1, 100), 14)
             if len(set(rand_vals)) == 14:
                 rand_edges = _greedy_edges(rand_vals)
                 rand_score = _score(rand_vals, rand_edges)
                 if rand_score < current_score:
                     current_values = rand_vals[:]
                     current_edges = rand_edges[:]
                     current_score = rand_score
                     if current_score < best_score:
                         best_score = current_score
                         best_values = current_values[:]
                         best_edges = current_edges[:]

    # Final verification
    # Ensure types are correct
    final_values = [int(v) for v in best_values]
    final_edges = [[int(i), int(j)] for i, j in best_edges]
    
    return {'values': final_values, 'edges': final_edges}
