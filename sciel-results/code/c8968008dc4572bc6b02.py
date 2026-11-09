import time
import random
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

def _generate_divisor_value_sets():
    """
    Generate value sets designed to maximize collisions in products and sums.
    Strategy: Use divisors of highly composite numbers.
    If values are divisors of N, then a*b = c*d is common if N has many factorizations.
    Also, small ranges help sums collide.
    """
    candidates = []
    
    # 1. Divisors of 840 (highly composite: 2^3 * 3 * 5 * 7)
    # Divisors: 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 15, 20, 21, 24, 28, 30, 35, 40, 42, 56, 60, 70, 84, 105, 120, 140, 168, 210, 280, 420, 840
    # We need 14 distinct. Let's pick the smallest 14 to keep sums small.
    divs_840 = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 15, 20, 21]
    candidates.append(divs_840)
    
    # 2. Divisors of 720 (2^4 * 3^2 * 5)
    # Smallest 14: 1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 18, 20
    divs_720 = [1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 18, 20]
    candidates.append(divs_720)
    
    # 3. Divisors of 360 (2^3 * 3^2 * 5)
    # Smallest 14: 1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 18, 20, 24
    divs_360 = [1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 18, 20, 24]
    candidates.append(divs_360)
    
    # 4. Multiples of a common base, scaled down? No, distinct positive ints.
    # Try a set with many small numbers.
    candidates.append(list(range(1, 15)))
    
    # 5. A mix of small primes and composites to balance sum/product collision.
    candidates.append([2, 3, 4, 5, 6, 8, 9, 10, 12, 14, 15, 16, 18, 20])
    
    # 6. Another highly composite structure: Divisors of 2520 (2^3*3^2*5*7)
    # Smallest 14: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 15, 18
    divs_2520 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 15, 18]
    candidates.append(divs_2520)
    
    # 7. Random search for a good set? 
    # Let's add a few random sets of small numbers, seeded for reproducibility.
    random.seed(42)
    for _ in range(5):
        vals = random.sample(range(1, 50), 14)
        candidates.append(sorted(vals))
        
    return candidates

def _find_optimal_edges(values, time_limit):
    """
    Exhaustively search for the best 35 edges out of 91.
    C(91, 35) is approx 1.5 * 10^22. This is TOO MANY for full enumeration.
    Wait, C(91, 35) is huge. 
    Let's re-evaluate the "exhaustive" claim.
    C(91, 35) is indeed massive. I cannot enumerate all subsets.
    
    However, the previous strategy used greedy. 
    Is there a smarter exact method?
    No, this is a combinatorial optimization problem (Minimize distinct values in sets).
    It's NP-hard in general.
    
    BUT, looking at the inspirations, they used Simulated Annealing or Joint Heuristics.
    The current program uses Greedy.
    
    Let's try a **Randomized Local Search** with a much larger number of iterations 
    and a better initial state, or **Simulated Annealing** as suggested by Inspiration 1.
    
    Let's implement Simulated Annealing for edge selection.
    State: Set of 35 edges.
    Move: Swap an edge in the set with an edge not in the set.
    Cost: max(distinct_sums, distinct_prods).
    """
    n = len(values)
    all_edges = list(combinations(range(n), 2))
    edge_indices = {e: i for i, e in enumerate(all_edges)}
    
    # Precompute sums and products for all edges
    edge_sums = [values[u] + values[v] for u, v in all_edges]
    edge_prods = [values[u] * values[v] for u, v in all_edges]
    
    def calc_cost(edge_set):
        sums = set()
        prods = set()
        for e in edge_set:
            idx = edge_indices[e]
            sums.add(edge_sums[idx])
            prods.add(edge_prods[idx])
        return max(len(sums), len(prods))
    
    # Initial state: Random 35 edges
    # Or Greedy initial state to be safe
    def greedy_init():
        sel = set()
        c_sums = set()
        c_prods = set()
        for _ in range(35):
            best_e = None
            best_score = float('inf')
            for e in all_edges:
                if e in sel:
                    continue
                idx = edge_indices[e]
                s = edge_sums[idx]
                p = edge_prods[idx]
                ns = len(c_sums) + (0 if s in c_sums else 1)
                np = len(c_prods) + (0 if p in c_prods else 1)
                score = max(ns, np)
                if score < best_score:
                    best_score = score
                    best_e = e
            if best_e is None:
                break
            sel.add(best_e)
            idx = edge_indices[best_e]
            c_sums.add(edge_sums[idx])
            c_prods.add(edge_prods[idx])
        return sel

    current_edges = greedy_init()
    current_cost = calc_cost(current_edges)
    
    best_edges = set(current_edges)
    best_cost = current_cost
    
    # Simulated Annealing
    start_temp = 1.0
    end_temp = 0.01
    cooling = 0.99
    temp = start_temp
    
    start_time = time.time()
    
    # We will run SA until time limit
    while time.time() - start_time < time_limit:
        if temp <= end_temp:
            break
            
        # Generate a neighbor: swap one edge in with one edge out
        in_edges = list(current_edges)
        out_edges = [e for e in all_edges if e not in current_edges]
        
        if not in_edges or not out_edges:
            break
            
        e_in = random.choice(in_edges)
        e_out = random.choice(out_edges)
        
        # Try the swap
        new_edges = set(current_edges)
        new_edges.remove(e_in)
        new_edges.add(e_out)
        
        new_cost = calc_cost(new_edges)
        delta = new_cost - current_cost
        
        if delta < 0 or random.random() < pow(2.71828, -delta / temp):
            current_edges = new_edges
            current_cost = new_cost
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_edges = set(current_edges)
                
        temp *= cooling
        
    return [list(e) for e in best_edges], best_cost

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    best_global_score = float('inf')
    best_global_result = None
    
    value_sets = _generate_divisor_value_sets()
    
    # Shuffle to ensure variety in order
    random.seed(100)
    random.shuffle(value_sets)
    
    for values in value_sets:
        # Ensure types are correct
        values = [int(v) for v in values]
        
        # Check constraints
        if len(set(values)) != 14:
            continue
        if any(v < 1 or v > 10000 for v in values):
            continue
            
        time_left = deadline - time.time()
        if time_left < 1.0:
            break
            
        # Allocate time for edge optimization
        # Give each value set a reasonable chunk, but allow more if we have time
        # Since we have ~55s and ~10 value sets, we can give ~5s each.
        edge_time = min(5.0, time_left / 2.0)
        
        edges, score = _find_optimal_edges(values, edge_time)
        
        if score < best_global_score:
            best_global_score = score
            best_global_result = {
                'values': values,
                'edges': edges
            }
            
        # Early exit if we hit a very low score
        if best_global_score <= 10:
            break

    if best_global_result is None:
        # Fallback
        values = list(range(1, 15))
        edges = [list(e) for e in list(combinations(range(14), 2))[:35]]
        return {
            'values': values,
            'edges': edges
        }
        
    return best_global_result
