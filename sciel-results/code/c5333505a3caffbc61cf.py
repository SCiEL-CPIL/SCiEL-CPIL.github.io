"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
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

def _evaluate_state(values, edges_set):
    """
    Fast evaluation using pre-computed lookup or direct calculation.
    """
    sums = set()
    prods = set()
    for u, v in edges_set:
        sums.add(values[u] + values[v])
        prods.add(values[u] * values[v])
    return max(len(sums), len(prods))

def _generate_value_structures():
    """
    Generate diverse sets of 14 distinct positive integers <=10000.
    Focus on structures that create high collision rates for sums and products.
    """
    candidates = []
    
    # 1. Arithmetic Progressions (High additive collision)
    # e.g., a, a+d, a+2d...
    for a in range(1, 10):
        for d in range(1, 10):
            vals = [a + i*d for i in range(14)]
            if max(vals) <= 10000 and len(set(vals)) == 14:
                candidates.append(vals)
                
    # 2. Geometric Progressions (High multiplicative collision)
    # e.g., a, a*r, a*r^2...
    for a in range(1, 5):
        for r in range(2, 10):
            vals = [a * (r**i) for i in range(14)]
            if max(vals) <= 10000 and len(set(vals)) == 14:
                candidates.append(vals)
            else:
                break # r too large

    # 3. Multiples of a highly composite number
    # Highly composite numbers have many divisors, leading to product overlaps
    for base in [6, 12, 24, 36, 48, 60, 72, 84, 96, 120]:
        # Select 14 distinct multiples
        # To keep them small, use consecutive multiples? Or sparse?
        # Consecutive multiples: k*base, (k+1)*base...
        # Sum collisions: (k1+k2)*base. Product collisions: (k1*k2)*base^2.
        # This reduces the problem to finding k's.
        # Let's try k=1..14
        vals = [i * base for i in range(1, 15)]
        if max(vals) <= 10000 and len(set(vals)) == 14:
            candidates.append(vals)
            
        # Try sparse multiples to increase range? No, small values usually better for collisions.
        
    # 4. Powers of 2 (Multiplicative structure)
    candidates.append([2**i for i in range(1, 15)])
    candidates.append([2**i + 1 for i in range(1, 15)]) # Mersenne-like

    # 5. Small integers with specific structure
    # 1-14
    candidates.append(list(range(1, 15)))
    # 2-15
    candidates.append(list(range(2, 16)))
    # Even numbers
    candidates.append([2*i for i in range(1, 15)])
    # Odd numbers
    candidates.append([2*i+1 for i in range(1, 15)])
    
    # 6. Random sets with bias towards small numbers
    random.seed(42)
    for _ in range(20):
        # Sample from 1..100 with replacement, then unique
        vals = []
        while len(vals) < 14:
            v = random.randint(1, 100)
            if v not in vals:
                vals.append(v)
        candidates.append(sorted(vals))

    # 7. Random sets with bias towards highly composite
    random.seed(43)
    hc = [1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 20, 24, 30, 36, 40, 42, 48, 56, 60, 72, 84, 90, 96, 108, 120, 144, 168, 180, 240, 360, 720, 840, 1260, 1680, 2520, 5040, 7560]
    for _ in range(20):
        vals = random.sample(hc, 14)
        if max(vals) <= 10000:
            candidates.append(sorted(vals))

    # Deduplicate candidates
    unique_candidates = []
    seen = set()
    for c in candidates:
        t = tuple(sorted(c))
        if t not in seen:
            seen.add(t)
            unique_candidates.append(list(t))
            
    return unique_candidates

def _simulated_annealing(values, initial_edges, time_limit, seed=0):
    """
    Optimize the edge set using Simulated Annealing.
    Moves: Swap an existing edge with a non-existing edge.
    Objective: Minimize max(distinct_sums, distinct_prods).
    """
    rng = random.Random(seed)
    n = len(values)
    all_edges = list(combinations(range(n), 2))
    
    current_edges = set(initial_edges)
    
    # Precompute sums and prods for current state
    sums = set()
    prods = set()
    for u, v in current_edges:
        sums.add(values[u] + values[v])
        prods.add(values[u] * values[v])
    
    current_score = max(len(sums), len(prods))
    best_score = current_score
    best_edges = set(current_edges)
    
    start_time = time.time()
    temp = 10.0
    cooling_rate = 0.995
    
    while time.time() - start_time < time_limit:
        # Generate a move: remove a random edge, add a random edge
        if not current_edges:
            break
        remove_edge = rng.choice(tuple(current_edges))
        
        # Find a non-edge to add
        # To speed up, pick a random candidate from all_edges and check if it's in current_edges
        # Or maintain a list of non-edges? 91 edges total, 35 in set. 56 non-edges.
        # Simple approach: pick random from all_edges, if in current_edges, retry.
        # Or just pick random from all_edges not in current_edges.
        # Creating a list of non-edges each time is O(E^2) or O(E*E_set).
        # Let's just pick random from all_edges and check membership.
        add_edge = None
        while add_edge is None or add_edge in current_edges:
            add_edge = rng.choice(all_edges)
            
        # Apply move
        current_edges.remove(remove_edge)
        current_edges.add(add_edge)
        
        # Recalculate score
        new_sums = set()
        new_prods = set()
        for u, v in current_edges:
            new_sums.add(values[u] + values[v])
            new_prods.add(values[u] * values[v])
        
        new_score = max(len(new_sums), len(prods)) # BUG: should be len(new_prods)
        new_score = max(len(new_sums), len(new_prods))
        
        delta = new_score - current_score
        
        if delta < 0 or rng.random() < math.exp(-delta / temp):
            current_score = new_score
            sums = new_sums
            prods = new_prods
            if new_score < best_score:
                best_score = new_score
                best_edges = set(current_edges)
        else:
            # Revert
            current_edges.remove(add_edge)
            current_edges.add(remove_edge)
            
        temp *= cooling_rate
        if temp < 0.01:
            # Reheat or break? Reheat allows escaping local minima.
            temp = 5.0
            
    return [list(e) for e in best_edges], best_score

def _greedy_edge_selection(values, n=14, m=35):
    """
    Greedy initialization: Add edges that minimize the increase in distinct sums/products.
    """
    all_edges = list(combinations(range(n), 2))
    random.shuffle(all_edges)
    
    current_edges = set()
    sums = set()
    prods = set()
    
    # We want exactly 35 edges.
    # Greedy: pick the edge that adds the least new distinct sums/prods.
    # If tie, pick randomly.
    
    remaining_edges = list(all_edges)
    
    for _ in range(m):
        best_edge = None
        best_cost = float('inf')
        
        for e in remaining_edges:
            u, v = e
            s = values[u] + values[v]
            p = values[u] * values[v]
            
            new_sums = 0 if s in sums else 1
            new_prods = 0 if p in prods else 1
            
            cost = new_sums + new_prods
            
            if cost < best_cost:
                best_cost = cost
                best_edge = e
        
        if best_edge:
            current_edges.add(best_edge)
            remaining_edges.remove(best_edge)
            sums.add(values[best_edge[0]] + values[best_edge[1]])
            prods.add(values[best_edge[0]] * values[best_edge[1]])
            
    return [list(e) for e in current_edges]

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    best_global_score = float('inf')
    best_global_result = None
    
    value_sets = _generate_value_structures()
    
    # Shuffle value sets to ensure variety in time allocation
    random.shuffle(value_sets)
    
    for values in value_sets:
        if time.time() > deadline:
            break
            
        # Ensure types are correct
        values = [int(v) for v in values]
        
        # Check constraints
        if len(set(values)) != 14:
            continue
        if any(v < 1 or v > 10000 for v in values):
            continue
            
        # 1. Greedy Initialization
        greedy_edges = _greedy_edge_selection(values)
        
        # 2. Simulated Annealing Refinement
        # Allocate time based on remaining budget
        time_left = deadline - time.time()
        # Reserve some time for other value sets?
        # If we have many value sets, we can't spend too much on each.
        # Let's spend up to 5s per value set, but stop if time runs out.
        sa_time = min(5.0, time_left / 2.0)
        
        if sa_time < 1.0:
            # Not enough time for meaningful SA, just use greedy
            score = _raw_score(values, greedy_edges)
            if score < best_global_score:
                best_global_score = score
                best_global_result = {
                    'values': values,
                    'edges': greedy_edges
                }
        else:
            refined_edges, refined_score = _simulated_annealing(values, greedy_edges, sa_time)
            
            if refined_score < best_global_score:
                best_global_score = refined_score
                best_global_result = {
                    'values': values,
                    'edges': refined_edges
                }
                
        # Early exit if we hit a very low score
        if best_global_score <= 10:
            break

    if best_global_result is None:
        # Fallback
        return {
            'values': list(range(1, 15)),
            'edges': [list(e) for e in list(combinations(range(14), 2))[:35]]
        }
        
    # Ensure types are exact
    final_values = [int(v) for v in best_global_result['values']]
    final_edges = [[int(u), int(v)] for u, v in best_global_result['edges']]
    
    return {
        'values': final_values,
        'edges': final_edges
    }
# EVOLVE-BLOCK-END
