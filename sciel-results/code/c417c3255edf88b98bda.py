"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
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

def _generate_value_candidates():
    """
    Generate diverse sets of 14 distinct positive integers <= 10000.
    Focus on numbers that create additive and multiplicative collisions.
    """
    candidates = []
    
    # 1. Small integers 1-14
    candidates.append(list(range(1, 15)))
    
    # 2. Multiples of 2
    candidates.append([i * 2 for i in range(1, 15)])
    
    # 3. Multiples of 3
    candidates.append([i * 3 for i in range(1, 15)])
    
    # 4. Multiples of 4
    candidates.append([i * 4 for i in range(1, 15)])
    
    # 5. Multiples of 5
    candidates.append([i * 5 for i in range(1, 15)])
    
    # 6. Multiples of 6
    candidates.append([i * 6 for i in range(1, 15)])
    
    # 7. Mix of small primes and composites
    candidates.append([2, 3, 4, 5, 6, 8, 9, 10, 12, 14, 15, 16, 18, 20])
    
    # 8. Numbers with high factor overlap (multiples of 6, shifted)
    candidates.append([6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84])
    
    # 9. Random small numbers (seeded for reproducibility)
    random.seed(123)
    base = list(range(1, 50))
    random.shuffle(base)
    candidates.append(base[:14])
    
    # 10. Another random set
    random.seed(456)
    base = list(range(1, 100))
    random.shuffle(base)
    candidates.append(base[:14])
    
    # 11. Powers of 2 (distinct products guaranteed, sums might collide? no, sums of distinct powers are unique)
    # Actually powers of 2 are bad for products (all unique) and sums (all unique).
    # But let's try a mix.
    
    # 12. Arithmetic progression with large step
    candidates.append([100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400])
    
    # 13. Geometric-ish
    candidates.append([1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192])
    
    # 14. Random set with high collision potential (small numbers)
    random.seed(789)
    base = list(range(1, 30))
    random.shuffle(base)
    candidates.append(base[:14])
    
    return candidates

def _greedy_edge_selection(values, n, m):
    """
    Greedy selection of edges.
    1. Precompute sum and product for all possible edges.
    2. Sort edges by a heuristic: prefer edges that create fewer new distinct sums/products.
       Initially, all edges are "new". 
       Heuristic: 
       - If sum exists in current sums, it's a "collision" (good).
       - If product exists in current prods, it's a "collision" (good).
       - We want to minimize the growth of distinct sets.
       
       Strategy:
       - Start with empty sets.
       - At each step, for every candidate edge, calculate the "cost" of adding it.
         Cost = 1 if sum is new, 0 if sum is old.
         Cost += 1 if product is new, 0 if product is old.
       - Pick the edge with minimum cost. If tie, pick randomly or by some secondary metric.
       - Add it to the graph, update sets.
       - Repeat until m edges are selected.
    """
    all_edges = list(combinations(range(n), 2))
    
    # Precompute sums and products for all edges
    edge_props = []
    for u, v in all_edges:
        s = values[u] + values[v]
        p = values[u] * values[v]
        edge_props.append((u, v, s, p))
        
    # Sort edges initially? No, we need dynamic evaluation.
    # But we can sort by "potential" to speed up? 
    # Actually, n=14, m=35. 91 edges total.
    # Greedy loop: 35 iterations. In each, check 91 edges. 35*91 ~ 3000 ops. Very fast.
    
    selected_edges = []
    current_sums = set()
    current_prods = set()
    
    available_edges = set(range(len(all_edges)))
    
    for _ in range(m):
        best_idx = -1
        best_cost = float('inf')
        best_u = -1
        best_v = -1
        best_s = -1
        best_p = -1
        
        # To handle ties, we can collect all min-cost edges and pick one randomly
        min_cost_edges = []
        
        for idx in available_edges:
            u, v, s, p = edge_props[idx]
            
            cost = 0
            if s not in current_sums:
                cost += 1
            if p not in current_prods:
                cost += 1
                
            if cost < best_cost:
                best_cost = cost
                min_cost_edges = [(idx, u, v, s, p)]
            elif cost == best_cost:
                min_cost_edges.append((idx, u, v, s, p))
                
        # Pick one from min_cost_edges
        # Random selection among ties adds diversity
        if min_cost_edges:
            choice = random.choice(min_cost_edges)
            idx, u, v, s, p = choice
            selected_edges.append([u, v])
            current_sums.add(s)
            current_prods.add(p)
            available_edges.remove(idx)
            
    return selected_edges, _evaluate_state(values, set(tuple(sorted(e)) for e in selected_edges))

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    best_global_score = float('inf')
    best_global_result = None
    
    value_sets = _generate_value_candidates()
    
    # Shuffle value sets to avoid bias? Or keep deterministic order?
    # Let's keep deterministic order but maybe interleave with random restarts?
    # Greedy is deterministic if tie-breaking is deterministic.
    # If tie-breaking is random, we get different results.
    
    for i, values in enumerate(value_sets):
        # Ensure types are correct
        values = [int(v) for v in values]
        
        # Check constraints
        if len(set(values)) != 14:
            continue
        if any(v < 1 or v > 10000 for v in values):
            continue
            
        # Run Greedy with multiple restarts (due to random tie-breaking)
        # How many restarts? 
        # Greedy is very fast. We can do many.
        # Let's say 100 restarts per value set if time permits.
        
        for restart in range(100):
            if time.time() > deadline:
                break
                
            # Randomize the initial state? 
            # Greedy doesn't have an initial state, it builds from scratch.
            # But the tie-breaking is random.
            # We can also shuffle the edge order if we used a different greedy variant,
            # but here we evaluate all edges each step.
            
            edges, score = _greedy_edge_selection(values, 14, 35)
            
            # Validate
            if len(edges) != 35:
                continue
            edge_tuples = set(tuple(sorted(e)) for e in edges)
            if len(edge_tuples) != 35:
                continue
            if any(u == v for u, v in edges):
                continue
                
            if score < best_global_score:
                best_global_score = score
                best_global_result = {
                    'values': values,
                    'edges': edges
                }
                
                # Early exit if perfect
                if best_global_score <= 10:
                    break
                    
        if time.time() > deadline:
            break
            
    if best_global_result is None:
        # Fallback
        return {
            'values': list(range(1, 15)),
            'edges': [list(e) for e in list(combinations(range(14), 2))[:35]]
        }
        
    return best_global_result
# EVOLVE-BLOCK-END
