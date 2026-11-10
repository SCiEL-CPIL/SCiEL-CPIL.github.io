import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin
    
    n = 14
    m = 35
    
    def evaluate_edge_subset(vals, edge_indices):
        """
        vals: list of 14 values.
        edge_indices: list of indices (0..90) representing edges in K14.
        Returns max(distinct_sums, distinct_products).
        """
        sums = set()
        prods = set()
        for idx in edge_indices:
            # Convert linear index to (i, j)
            # 0: (0,1), 1: (0,2), ..., 12: (0,13), 13: (1,2), ...
            # i = (sqrt(8*idx + 1) - 1) / 2
            # j = idx - i*(i+1)/2 + i + 1
            # Faster: precomputed mapping
            i, j = edge_map[idx]
            sums.add(vals[i] + vals[j])
            prods.add(vals[i] * vals[j])
        return max(len(sums), len(prods))

    # Precompute edge mapping for K14
    edge_map = []
    for i in range(n):
        for j in range(i+1, n):
            edge_map.append((i, j))
    # edge_map[k] = (i, j)
    
    def optimize_edges_for_values(vals):
        """
        Given 14 values, find the best 35 edges.
        Uses local search / simulated annealing.
        """
        # Precompute sums and products for all 91 edges
        all_sums = []
        all_prods = []
        for k in range(91):
            i, j = edge_map[k]
            all_sums.append(vals[i] + vals[j])
            all_prods.append(vals[i] * vals[j])
        
        # Initial selection: random 35 edges
        current_edges = set(random.sample(range(91), m))
        current_score = evaluate_edge_subset(vals, current_edges)
        
        best_score = current_score
        best_edges = current_edges.copy()
        
        # Local Search
        # We can swap an edge in current_edges with one not in current_edges
        # To speed up, we can track counts of sums and products
        
        # Let's use a simpler, robust local search with restarts
        max_iters = 2000
        for _ in range(max_iters):
            if time.time() > deadline - 10:
                break
            
            # Pick a random edge to remove
            e_out = random.choice(list(current_edges))
            # Pick a random edge to add
            e_in = random.choice([k for k in range(91) if k not in current_edges])
            
            new_edges = current_edges - {e_out} | {e_in}
            
            # Evaluate new edges
            # Optimization: We can update sums/prods sets incrementally?
            # Removing e_out: if sum/prod becomes unique, count decreases.
            # Adding e_in: if sum/prod is new, count increases.
            
            # For simplicity and correctness, just re-evaluate.
            # 35 edges is small.
            new_score = evaluate_edge_subset(vals, new_edges)
            
            if new_score < current_score:
                current_edges = new_edges
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_edges = current_edges.copy()
            # Plateau move?
            elif new_score == current_score and random.random() < 0.1:
                current_edges = new_edges
                current_score = new_score
                
        return best_score, best_edges

    best_global_score = float('inf')
    best_global_vals = None
    best_global_edges = None
    
    # Generate candidate value sets
    candidates = []
    
    # 1. Powers of 2
    candidates.append([2**i for i in range(14)])
    
    # 2. Arithmetic Progressions
    for start in [1, 2, 3]:
        for d in [1, 2, 3, 5]:
            vals = [start + i*d for i in range(14)]
            if max(vals) <= 10000:
                candidates.append(vals)
                
    # 3. Multiples of a base
    for base in [2, 3, 5, 10]:
        for start in range(1, 10):
            vals = [start * base + i * base for i in range(14)]
            if max(vals) <= 10000:
                candidates.append(vals)
                
    # 4. Random sets with structure (divisors)
    random.seed(42)
    for _ in range(20):
        # Pick a random base and multipliers
        base = random.randint(1, 10)
        multipliers = random.sample(range(1, 20), 14)
        vals = sorted([m * base for m in multipliers])
        if len(set(vals)) == 14 and max(vals) <= 10000:
            candidates.append(vals)
            
    # 5. Random distinct integers
    for _ in range(20):
        vals = random.sample(range(1, 100), 14)
        candidates.append(vals)
        
    # Evaluate each candidate
    for vals in candidates:
        if time.time() > deadline - 20:
            break
        score, edges = optimize_edges_for_values(vals)
        if score < best_global_score:
            best_global_score = score
            best_global_vals = vals[:]
            best_global_edges = edges.copy()
            
    # Fallback
    if best_global_vals is None:
        best_global_vals = list(range(1, 15))
        best_global_edges = set(random.sample(range(91), 35))
        best_global_score = evaluate_edge_subset(best_global_vals, best_global_edges)
        
    # Construct final result
    final_values = [int(v) for v in best_global_vals]
    final_edges = []
    for idx in best_global_edges:
        i, j = edge_map[idx]
        final_edges.append([int(i), int(j)])
        
    return {'values': final_values, 'edges': final_edges}
