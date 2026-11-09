"""
Strategy: Fix the graph structure (edges) to a deterministic subset (first 35 edges of K14) 
and optimize the vertex values using Simulated Annealing. 
This changes the search space from edge subsets (C(91,35) ~ 10^22) to value sets 
(10000^14) but allows for more direct control over the collision structure 
via value manipulation (e.g., arithmetic progressions for sums, geometric/multiplicative 
structures for products).
"""
# EVOLVE-BLOCK-START
import time
import random
import math
from itertools import combinations

def _raw_score(values, edges):
    """Compute the raw score: max(distinct sums, distinct products)."""
    sums = set()
    prods = set()
    for u, v in edges:
        su = values[u]
        sv = values[v]
        sums.add(su + sv)
        prods.add(su * sv)
    return max(len(sums), len(prods))

def _evaluate_state(values, edges):
    """
    Fast evaluation.
    values: list of 14 ints
    edges: list of 35 tuples (u, v)
    """
    sums = set()
    prods = set()
    for u, v in edges:
        su = values[u]
        sv = values[v]
        sums.add(su + sv)
        prods.add(su * sv)
    return max(len(sums), len(prods))

def _generate_initial_values():
    """
    Generate diverse initial value sets.
    We want sets that encourage collisions.
    1. Arithmetic Progression: Many sum collisions (a_i + a_j = a_k + a_l if i+j=k+l).
    2. Geometric Progression: Many product collisions (a_i * a_j = a_k * a_l if i+j=k+l).
    3. Small integers: High density of small sums/products.
    4. Multiples of a base: High density.
    """
    candidates = []
    
    # 1. Arithmetic Progression: 1, 2, ..., 14
    candidates.append(list(range(1, 15)))
    
    # 2. Arithmetic Progression with larger step: 1, 3, 5, ... (Odd numbers)
    candidates.append([2*i + 1 for i in range(14)])
    
    # 3. Geometric Progression: 1, 2, 4, ..., 2^13 (Might exceed 10000? 2^13=8192, ok)
    # But distinctness is required. 1, 2, 4... are distinct.
    candidates.append([2**i for i in range(14)])
    
    # 4. Small random integers
    random.seed(42)
    vals = random.sample(range(1, 50), 14)
    candidates.append(vals)
    
    # 5. Highly composite numbers (many factors -> product collisions?)
    # 2, 3, 4, 6, 8, 9, 10, 12, 14, 15, 16, 18, 20, 24
    candidates.append([2, 3, 4, 6, 8, 9, 10, 12, 14, 15, 16, 18, 20, 24])
    
    # 6. Powers of small primes?
    # 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 3 (mix)
    # Let's stick to simple APs and randoms for variety.
    
    return candidates

def _simulated_annealing_values(initial_values, edges, time_limit):
    """
    Optimize values using Simulated Annealing.
    Moves:
    1. Replace a random value with a new random value in [1, 10000].
    2. Perturb a random value by +/- delta.
    3. Swap two values.
    Constraint: All 14 values must be distinct positive integers <= 10000.
    """
    n = 14
    current_values = list(initial_values)
    # Ensure distinctness and validity for start
    if len(set(current_values)) != n or any(v < 1 or v > 10000 for v in current_values):
        # Fallback to random valid set
        current_values = random.sample(range(1, 10001), n)
        
    current_score = _evaluate_state(current_values, edges)
    best_values = list(current_values)
    best_score = current_score
    
    start_temp = 500.0
    end_temp = 1.0
    cooling_rate = 0.9995
    
    temp = start_temp
    start_time = time.time()
    
    while temp > end_temp:
        if time.time() - start_time > time_limit:
            break
            
        # Choose a move
        move_type = random.random()
        
        new_values = list(current_values)
        
        if move_type < 0.4:
            # 1. Replace random value
            idx = random.randint(0, n - 1)
            new_val = random.randint(1, 10000)
            # Check distinctness
            if new_val in new_values:
                # Try a few times to find a distinct one
                for _ in range(10):
                    new_val = random.randint(1, 10000)
                    if new_val not in new_values:
                        break
                else:
                    continue # Skip if can't find distinct
            new_values[idx] = new_val
            
        elif move_type < 0.8:
            # 2. Perturb random value
            idx = random.randint(0, n - 1)
            delta = random.randint(-50, 50)
            new_val = new_values[idx] + delta
            if 1 <= new_val <= 10000 and new_val not in new_values:
                new_values[idx] = new_val
            else:
                # Try smaller delta or different idx
                continue
                
        else:
            # 3. Swap two values
            i, j = random.sample(range(n), 2)
            new_values[i], new_values[j] = new_values[j], new_values[i]
            # Swap always preserves distinctness
            
        new_score = _evaluate_state(new_values, edges)
        
        delta_score = new_score - current_score
        if delta_score < 0 or random.random() < math.exp(-delta_score / temp):
            current_values = new_values
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_values = list(new_values)
                
        temp *= cooling_rate
        
    return best_values, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 85.0 # Safety margin
    
    # Fixed edge set: First 35 edges of K14
    # K14 has 91 edges.
    all_edges = list(combinations(range(14), 2))
    # To ensure we cover a "typical" graph structure, we take the first 35.
    # Note: The choice of edges might matter, but since we are optimizing values,
    # we assume a "generic" dense subgraph is fine. 
    # Actually, let's use a slightly more structured set if possible, 
    # but the first 35 is deterministic and valid.
    edges = [list(e) for e in all_edges[:35]]
    
    best_global_score = float('inf')
    best_global_values = None
    
    value_sets = _generate_initial_values()
    
    # We have ~85 seconds. 
    # Let's allocate time to multiple restarts/initializations.
    # 5 initial sets * 2 restarts each? 
    # Or just run SA for each initial set for a fixed time.
    
    num_restarts = 10
    time_per_run = (deadline - time.time()) / num_restarts
    
    for i in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Cycle through initial sets
        initial_values = value_sets[i % len(value_sets)]
        
        # Add some random noise to initial values for restarts > len(value_sets)
        if i >= len(value_sets):
            # Perturb the base set
            perturbed = list(initial_values)
            for _ in range(3):
                idx = random.randint(0, 13)
                delta = random.randint(-10, 10)
                new_val = perturbed[idx] + delta
                if 1 <= new_val <= 10000 and new_val not in perturbed:
                    perturbed[idx] = new_val
            initial_values = perturbed
            
        values, score = _simulated_annealing_values(initial_values, edges, time_per_run)
        
        # Validate
        if len(set(values)) != 14:
            continue
        if any(v < 1 or v > 10000 for v in values):
            continue
        if not all(isinstance(v, int) for v in values):
            continue
            
        if score < best_global_score:
            best_global_score = score
            best_global_values = values
            
            # Early exit if we hit a very low score (theoretical min is likely > 10)
            # With 35 edges, max distinct sums is 35. If we get < 20, that's good.
            if best_global_score <= 15:
                break
                
    if best_global_values is None:
        # Fallback
        best_global_values = list(range(1, 15))
        
    # Ensure types are Python int
    final_values = [int(v) for v in best_global_values]
    
    return {
        'values': final_values,
        'edges': edges
    }
# EVOLVE-BLOCK-END
