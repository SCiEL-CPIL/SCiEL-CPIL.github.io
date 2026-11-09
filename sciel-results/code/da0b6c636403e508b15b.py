"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
from itertools import combinations
from collections import defaultdict

_DEADLINE = time.time() + 85  # safety margin before 90s kill

def _score_fast(values, edges):
    """Compute raw score: max(distinct sums, distinct products)."""
    sums = set()
    prods = set()
    for u, v in edges:
        a = values[u]
        b = values[v]
        sums.add(a + b)
        prods.add(a * b)
    return max(len(sums), len(prods))

def _score_detailed(values, edges):
    """Compute distinct sums and products separately."""
    sums = set()
    prods = set()
    for u, v in edges:
        a = values[u]
        b = values[v]
        sums.add(a + b)
        prods.add(a * b)
    return len(sums), len(prods)

def _greedy_edge_selection_structured(values, n, m):
    """
    Structurally-aware greedy edge selection:
    1. Precompute all C(n,2) edge (sum, prod) pairs.
    2. Count global frequency of each sum and product.
    3. Greedily select edges that minimize marginal distinct growth,
       with tie-breaking favoring high-collision edges.
    4. Use a conflict-aware scoring: edges whose (sum,prod) pair
       is rare are penalized more.
    """
    all_edges = list(combinations(range(n), 2))
    
    # Precompute sum and product for all edges
    edge_info = {}
    for i, j in all_edges:
        s = values[i] + values[j]
        p = values[i] * values[j]
        edge_info[(i, j)] = (s, p)
    
    # Count global frequency of each sum and product
    sum_freq = defaultdict(int)
    prod_freq = defaultdict(int)
    pair_freq = defaultdict(int)
    for (i, j), (s, p) in edge_info.items():
        sum_freq[s] += 1
        prod_freq[p] += 1
        pair_freq[(s, p)] += 1
    
    selected_edges = []
    selected_set = set()
    current_sums = set()
    current_prods = set()
    
    # Sort edges by initial "quality" (collision potential)
    edge_order = sorted(all_edges, key=lambda e: (
        -sum_freq[edge_info[e][0]],
        -prod_freq[edge_info[e][1]],
        -pair_freq[edge_info[e]]
    ))
    
    for _ in range(m):
        best_edge = None
        best_score = float('inf')
        
        for (i, j) in edge_order:
            if (i, j) in selected_set:
                continue
            
            s, p = edge_info[(i, j)]
            
            # Marginal cost: how many new distinct values does this add?
            new_sums = 0 if s in current_sums else 1
            new_prods = 0 if p in current_prods else 1
            
            # Combined score: minimize marginal, maximize collision frequency
            marginal = new_sums + new_prods
            collision_bonus = sum_freq[s] * 2 + prod_freq[p] * 2 + pair_freq[(s, p)]
            
            # Primary: minimize marginal; secondary: maximize collision
            score = marginal * 10000 - collision_bonus
            
            if score < best_score:
                best_score = score
                best_edge = (i, j)
        
        if best_edge is None:
            break
        
        selected_edges.append(best_edge)
        selected_set.add(best_edge)
        s, p = edge_info[best_edge]
        current_sums.add(s)
        current_prods.add(p)
    
    return selected_edges, current_sums, current_prods

def _local_search_edges(values, edges, n, m, max_iter=5000):
    """
    Local search to improve edge selection: try swapping edges
    to reduce max(distinct_sums, distinct_products).
    """
    cur_edges = set(edges)
    cur_sums_set = set()
    cur_prods_set = set()
    for u, v in cur_edges:
        cur_sums_set.add(values[u] + values[v])
        cur_prods_set.add(values[u] * values[v])
    cur_score = max(len(cur_sums_set), len(cur_prods_set))
    
    best_edges = set(cur_edges)
    best_score = cur_score
    
    all_edges = list(combinations(range(n), 2))
    non_selected = [e for e in all_edges if e not in cur_edges]
    
    for iteration in range(max_iter):
        if time.time() > _DEADLINE:
            break
        if cur_score <= 2:
            break
        
        # Try random edge swaps
        improved = False
        for _ in range(20):
            if time.time() > _DEADLINE:
                break
            
            if not non_selected:
                break
            
            # Pick a random edge to remove and a random edge to add
            edge_to_remove = random.choice(list(cur_edges))
            edge_to_add = random.choice(non_selected)
            
            if edge_to_add in cur_edges:
                continue
            
            new_edges = cur_edges - {edge_to_remove} | {edge_to_add}
            
            # Compute new score incrementally
            new_sums = set(cur_sums_set)
            new_prods = set(cur_prods_set)
            
            # Remove contributions from removed edge (need to recount if it was unique)
            # Simpler: recompute from scratch for correctness
            new_sums = set()
            new_prods = set()
            for u, v in new_edges:
                a = values[u]
                b = values[v]
                new_sums.add(a + b)
                new_prods.add(a * b)
            
            new_score = max(len(new_sums), len(new_prods))
            
            if new_score < cur_score:
                cur_edges = new_edges
                cur_sums_set = new_sums
                cur_prods_set = new_prods
                cur_score = new_score
                improved = True
                
                if cur_score < best_score:
                    best_score = cur_score
                    best_edges = set(cur_edges)
                
                # Update non_selected
                non_selected = [e for e in all_edges if e not in cur_edges]
                break
        
        if not improved and iteration % 100 == 0:
            # Occasionally try a 2-edge swap
            if len(non_selected) >= 2:
                e1_remove = random.choice(list(cur_edges))
                e2_remove = random.choice([e for e in cur_edges if e != e1_remove])
                e1_add = random.choice(non_selected)
                e2_add = random.choice([e for e in non_selected if e != e1_add and e not in cur_edges])
                
                if e1_add not in cur_edges and e2_add not in cur_edges and e1_add != e2_add:
                    new_edges = cur_edges - {e1_remove, e2_remove} | {e1_add, e2_add}
                    new_sums = set()
                    new_prods = set()
                    for u, v in new_edges:
                        a = values[u]
                        b = values[v]
                        new_sums.add(a + b)
                        new_prods.add(a * b)
                    new_score = max(len(new_sums), len(new_prods))
                    
                    if new_score < cur_score:
                        cur_edges = new_edges
                        cur_sums_set = new_sums
                        cur_prods_set = new_prods
                        cur_score = new_score
                        if cur_score < best_score:
                            best_score = cur_score
                            best_edges = set(cur_edges)
                        non_selected = [e for e in all_edges if e not in cur_edges]
    
    return best_score, list(best_edges)

def _polish_values(values, edges, n, m, max_iter=2000):
    """
    Polish values: try small perturbations to reduce distinct sums/products.
    """
    cur_values = list(values)
    cur_score = _score_fast(cur_values, edges)
    
    best_values = list(cur_values)
    best_score = cur_score
    
    for iteration in range(max_iter):
        if time.time() > _DEADLINE:
            break
        if cur_score <= 2:
            break
        
        # Try mutating one value
        idx = random.randint(0, n - 1)
        old_val = cur_values[idx]
        
        # Try several candidate values
        candidates = set()
        for delta in range(-5, 6):
            candidates.add(old_val + delta)
        candidates.add(random.randint(1, 200))
        candidates.add(random.randint(1, 50))
        
        for new_val in candidates:
            if new_val < 1 or new_val > 10000:
                continue
            if new_val == old_val:
                continue
            
            new_values = list(cur_values)
            new_values[idx] = new_val
            
            # Check distinctness
            if len(set(new_values)) != n:
                continue
            
            new_score = _score_fast(new_values, edges)
            
            if new_score < cur_score:
                cur_values = new_values
                cur_score = new_score
                
                if cur_score < best_score:
                    best_score = cur_score
                    best_values = list(cur_values)
                
                break
    
    return best_score, best_values

def _final_polish(values, edges, n, m):
    """
    Final high-precision polish: alternate between value tweaks and edge swaps
    with tighter tolerance.
    """
    cur_values = list(values)
    cur_edges = set(edges)
    cur_score = _score_fast(cur_values, cur_edges)
    
    best_values = list(cur_values)
    best_edges = set(cur_edges)
    best_score = cur_score
    
    all_edges = list(combinations(range(n), 2))
    
    max_rounds = 200
    for rnd in range(max_rounds):
        if time.time() > _DEADLINE:
            break
        if cur_score <= 2:
            break
        
        improved = False
        
        # Try value mutations
        for idx in range(n):
            if time.time() > _DEADLINE:
                break
            old_val = cur_values[idx]
            for delta in range(-3, 4):
                new_val = old_val + delta
                if new_val < 1 or new_val > 10000:
                    continue
                if new_val == old_val:
                    continue
                
                new_values = list(cur_values)
                new_values[idx] = new_val
                if len(set(new_values)) != n:
                    continue
                
                new_score = _score_fast(new_values, cur_edges)
                if new_score < cur_score:
                    cur_values = new_values
                    cur_score = new_score
                    if cur_score < best_score:
                        best_score = cur_score
                        best_values = list(cur_values)
                        best_edges = set(cur_edges)
                    improved = True
                    break
        
        # Try edge swaps
        non_selected = [e for e in all_edges if e not in cur_edges]
        if non_selected:
            for _ in range(10):
                if time.time() > _DEADLINE:
                    break
                e_rem = random.choice(list(cur_edges))
                e_add = random.choice(non_selected)
                if e_add in cur_edges:
                    continue
                
                new_edges = cur_edges - {e_rem} | {e_add}
                new_score = _score_fast(cur_values, new_edges)
                if new_score < cur_score:
                    cur_edges = new_edges
                    cur_score = new_score
                    if cur_score < best_score:
                        best_score = cur_score
                        best_values = list(cur_values)
                        best_edges = set(cur_edges)
                    non_selected = [e for e in all_edges if e not in cur_edges]
                    improved = True
                    break
        
        if not improved:
            break
    
    return best_score, best_values, list(best_edges)

def _generate_structured_values(n, max_val=10000):
    """Generate diverse structured value sets that promote sum/product collisions."""
    candidates = []
    
    # 1. Consecutive small integers
    candidates.append(list(range(1, n + 1)))
    
    # 2. Even numbers
    candidates.append(list(range(2, 2 * n + 1, 2)))
    
    # 3. Arithmetic progressions
    for start in range(1, 10):
        for step in range(1, 6):
            vals = [start + i * step for i in range(n)]
            if max(vals) <= max_val and min(vals) >= 1:
                candidates.append(vals)
    
    # 4. Values with additive structure (many pairs sum to same value)
    for S in range(2 * n, 2 * n + 20):
        vals = []
        for i in range(n):
            if i < n // 2:
                vals.append(i + 1)
            else:
                vals.append(S - (i - n // 2 + 1))
        if len(set(vals)) == n and min(vals) >= 1 and max(vals) <= max_val:
            candidates.append(sorted(vals))
    
    # 5. Multiplicative structure (powers of 2)
    candidates.append([1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192])
    
    # 6. Random with bias toward small values
    random.seed(42)
    for _ in range(20):
        vals = sorted(random.sample(range(1, 50), n))
        candidates.append(vals)
    
    # 7. Fibonacci-like
    candidates.append([1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610])
    
    # 8. Narrow range
    for lo in range(1, 15):
        hi = lo + 8
        if hi - lo + 1 >= n:
            vals = random.sample(range(lo, hi + 1), n)
            if len(vals) == n:
                candidates.append(sorted(vals))
    
    # 9. Pairs that multiply to same value
    base_vals = [1, 2, 3, 4, 6, 8, 12, 24, 48, 96, 192, 384, 768, 1536]
    candidates.append(base_vals)
    
    # 10. Mixed small and medium
    for _ in range(10):
        small = random.sample(range(1, 20), n // 2)
        medium = random.sample(range(20, 100), n - n // 2)
        vals = sorted(small + medium)
        if len(set(vals)) == n:
            candidates.append(vals)
    
    # 11. Divisor-rich numbers (many factor pairs)
    divisor_rich = [1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 18, 20]
    candidates.append(divisor_rich)
    
    # 12. Numbers with many divisors
    high_div = [1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 14, 15, 16, 18]
    candidates.append(high_div)
    
    # 13. Geometric progression with small ratio
    for ratio in [2, 3]:
        base = 1
        vals = [base * (ratio ** i) for i in range(n)]
        if max(vals) <= max_val:
            candidates.append(vals)
    
    # 14. Mix of 1s and 2s patterns (but must be distinct)
    # Use values like 1,2,3,4,5,6,7,8,9,10,11,12,13,14 with some swaps
    for perm_seed in range(5):
        rng = random.Random(perm_seed)
        base = list(range(1, n + 1))
        rng.shuffle(base)
        candidates.append(sorted(base))
    
    return candidates

def construct():
    n = 14
    m = 35
    max_val = 10000
    
    best_score = m + 1
    best_result = None
    
    candidates = _generate_structured_values(n, max_val)
    
    random.seed(12345)
    
    for vals in candidates:
        if time.time() > _DEADLINE:
            break
        
        # Validate
        if len(set(vals)) != n or min(vals) < 1 or max(vals) > max_val:
            continue
        if not all(isinstance(v, int) and not isinstance(v, bool) for v in vals):
            continue
        
        remaining = _DEADLINE - time.time()
        if remaining < 3:
            break
        
        # Step 1: Structured greedy edge selection
        edges, _, _ = _greedy_edge_selection_structured(vals, n, m)
        
        # Step 2: Local search on edges
        edge_score, improved_edges = _local_search_edges(vals, edges, n, m, max_iter=3000)
        
        # Step 3: Polish values
        final_score, polished_values = _polish_values(vals, improved_edges, n, m, max_iter=1500)
        
        # Step 4: Final high-precision polish (alternate values and edges)
        final_score, polished_values, final_edges = _final_polish(polished_values, improved_edges, n, m)
        
        # Recompute with final values and edges
        final_score = _score_fast(polished_values, final_edges)
        
        if final_score < best_score:
            best_score = final_score
            best_result = (polished_values, final_edges)
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
