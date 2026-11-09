"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations
import math

_DEADLINE = time.time() + 85  # 5s safety margin before 90s kill

def _score(values, edges):
    """Compute raw score: max(distinct sums, distinct products)."""
    sums = set()
    prods = set()
    for u, v in edges:
        a = values[u]
        b = values[v]
        sums.add(a + b)
        prods.add(a * b)
    return max(len(sums), len(prods))

def _local_search_edges(values, n, m, time_limit=5.0):
    """
    Local search to find best m-edge subset minimizing score.
    Uses efficient state tracking for sum/product counts.
    """
    all_edges = list(combinations(range(n), 2))
    edge_indices = {e: i for i, e in enumerate(all_edges)}
    
    # Precompute sums and products for all edges
    edge_sums = {}
    edge_prods = {}
    for u, v in all_edges:
        a = values[u]
        b = values[v]
        edge_sums[(u, v)] = a + b
        edge_prods[(u, v)] = a * b

    # Initialize with a random set of m edges
    current_edges = set(random.sample(all_edges, m))
    
    # Initialize counts
    sum_counts = {}
    prod_counts = {}
    distinct_sums = 0
    distinct_prods = 0
    
    for e in current_edges:
        s = edge_sums[e]
        p = edge_prods[e]
        if s not in sum_counts:
            sum_counts[s] = 0
            distinct_sums += 1
        sum_counts[s] += 1
        
        if p not in prod_counts:
            prod_counts[p] = 0
            distinct_prods += 1
        prod_counts[p] += 1
            
    current_score = max(distinct_sums, distinct_prods)
    best_score = current_score
    best_edges = set(current_edges)
    
    start_time = time.time()
    no_improve_count = 0
    max_no_improve = 2000
    
    while time.time() - start_time < time_limit:
        if time.time() > _DEADLINE:
            break
            
        # Pick a random edge to remove
        e_remove = random.choice(list(current_edges))
        s_rem = edge_sums[e_remove]
        p_rem = edge_prods[e_remove]
        
        # Pick a random edge to add (not in current_edges)
        # To speed up, we can pick from a pre-generated list of candidates or just random sample
        # Generating a list of non-edges every iteration is O(N^2). 
        # Instead, pick a random edge from all_edges, if it's in current_edges, pick another.
        # Or maintain a list of non-edges.
        # Let's just try random sampling from all_edges until we find one not in current_edges.
        e_add = None
        for _ in range(10): # Max 10 tries to avoid infinite loop if m is close to total
            candidate = random.choice(all_edges)
            if candidate not in current_edges:
                e_add = candidate
                break
        if e_add is None:
            # Should not happen if m < len(all_edges)
            continue
            
        s_add = edge_sums[e_add]
        p_add = edge_prods[e_add]
        
        # Calculate change in distinct counts
        # Remove e_remove
        delta_sums = 0
        delta_prods = 0
        
        if sum_counts[s_rem] == 1:
            delta_sums -= 1
        if prod_counts[p_rem] == 1:
            delta_prods -= 1
            
        # Add e_add
        if s_add not in sum_counts or sum_counts[s_add] == 0:
            delta_sums += 1
        if p_add not in prod_counts or prod_counts[p_add] == 0:
            delta_prods += 1
            
        new_distinct_sums = distinct_sums + delta_sums
        new_distinct_prods = distinct_prods + delta_prods
        new_score = max(new_distinct_sums, new_distinct_prods)
        
        if new_score < current_score:
            # Accept improvement
            current_edges.discard(e_remove)
            current_edges.add(e_add)
            
            # Update state
            sum_counts[s_rem] -= 1
            if sum_counts[s_rem] == 0:
                del sum_counts[s_rem]
            prod_counts[p_rem] -= 1
            if prod_counts[p_rem] == 0:
                del prod_counts[p_rem]
                
            if s_add not in sum_counts:
                sum_counts[s_add] = 0
            sum_counts[s_add] += 1
            if p_add not in prod_counts:
                prod_counts[p_add] = 0
            prod_counts[p_add] += 1
            
            distinct_sums = new_distinct_sums
            distinct_prods = new_distinct_prods
            current_score = new_score
            
            if current_score < best_score:
                best_score = current_score
                best_edges = set(current_edges)
                no_improve_count = 0
            else:
                no_improve_count += 1
        elif new_score == current_score:
            # Accept equal score to explore (with some probability to avoid cycles)
            if random.random() < 0.5:
                current_edges.discard(e_remove)
                current_edges.add(e_add)
                
                sum_counts[s_rem] -= 1
                if sum_counts[s_rem] == 0:
                    del sum_counts[s_rem]
                prod_counts[p_rem] -= 1
                if prod_counts[p_rem] == 0:
                    del prod_counts[p_rem]
                    
                if s_add not in sum_counts:
                    sum_counts[s_add] = 0
                sum_counts[s_add] += 1
                if p_add not in prod_counts:
                    prod_counts[p_add] = 0
                prod_counts[p_add] += 1
                
                distinct_sums = new_distinct_sums
                distinct_prods = new_distinct_prods
                # current_score remains same
                no_improve_count += 1
        else:
            no_improve_count += 1
            
        if no_improve_count > max_no_improve:
            # Restart with new random initialization if stuck
            current_edges = set(random.sample(all_edges, m))
            sum_counts = {}
            prod_counts = {}
            distinct_sums = 0
            distinct_prods = 0
            for e in current_edges:
                s = edge_sums[e]
                p = edge_prods[e]
                if s not in sum_counts:
                    sum_counts[s] = 0
                    distinct_sums += 1
                sum_counts[s] += 1
                if p not in prod_counts:
                    prod_counts[p] = 0
                    distinct_prods += 1
                prod_counts[p] += 1
            current_score = max(distinct_sums, distinct_prods)
            no_improve_count = 0

    return best_score, [list(e) for e in best_edges]

def construct():
    n = 14
    m = 35
    max_val = 10000
    
    best_score = m  # worst case: all distinct
    best_result = None
    
    # Candidate value sets to try
    candidates = []
    
    # 1. Small consecutive integers
    candidates.append(list(range(1, n + 1)))
    
    # 2. Small even numbers
    candidates.append(list(range(2, 2 * n + 1, 2)))
    
    # 3. Highly composite-like small numbers (divisors of 2520)
    # Divisors of 2520: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 15, 18, 20, 21, 24, 28, 30, 35, 36, 40, 42, 45, 56, 60, 63, 70, 72, 84, 90, 105, 120, 126, 140, 168, 180, 210, 252, 280, 315, 360, 420, 504, 630, 840, 1260, 2520
    divisors_2520 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 15, 18, 20, 21, 24, 28, 30, 35, 36, 40, 42, 45, 56, 60, 63, 70, 72, 84, 90, 105, 120, 126, 140, 168, 180, 210, 252, 280, 315, 360, 420, 504, 630, 840, 1260, 2520]
    # Pick 14 smallest distinct divisors
    candidates.append(sorted(divisors_2520)[:14])
    
    # 4. Divisors of 5040 (more composite)
    # 5040 = 2^4 * 3^2 * 5 * 7
    # Smallest divisors: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 15, 16...
    candidates.append([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 15, 16])
    
    # 5. Mixed small values
    candidates.append([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
    
    # 6. Random small value sets
    random.seed(42)
    for _ in range(10):
        vals = sorted(random.sample(range(1, 100), n))
        candidates.append(vals)
    
    # 7. Arithmetic progressions
    for start in range(1, 6):
        for step in range(1, 5):
            vals = [start + i * step for i in range(n)]
            if max(vals) <= max_val and min(vals) >= 1:
                candidates.append(vals)

    # Filter valid candidates
    valid_candidates = []
    for vals in candidates:
        if len(set(vals)) != n:
            continue
        if min(vals) < 1 or max(vals) > max_val:
            continue
        if not all(isinstance(v, int) and not isinstance(v, bool) for v in vals):
            continue
        valid_candidates.append(vals)

    # Run local search for each candidate
    # Allocate time based on number of candidates
    time_per_candidate = max(1.0, (80.0 / len(valid_candidates)))
    
    for vals in valid_candidates:
        if time.time() > _DEADLINE:
            break
            
        score, edges = _local_search_edges(vals, n, m, time_limit=time_per_candidate)
        
        if score < best_score:
            best_score = score
            best_result = (vals, edges)
            if best_score <= 2:
                break
    
    if best_result is None:
        # Fallback
        values = list(range(1, n + 1))
        edges = [list(e) for e in list(combinations(range(n), 2))[:m]]
    else:
        values, edges = best_result
    
    # Ensure types are correct
    values = [int(v) for v in values]
    edges = [[int(u), int(v)] for u, v in edges]
    
    return {'values': values, 'edges': edges}
# EVOLVE-BLOCK-END
