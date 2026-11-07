"""Strategy: Exhaustive evaluation of structured collision-rich value sets with exact greedy edge selection for maximum bucket coverage."""
# EVOLVE-BLOCK-START
import itertools
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    n = 14
    m = 35
    
    # 1. Generate a diverse pool of candidate value sets.
    # We focus on sets that inherently have many additive and multiplicative collisions.
    # - Geometric progressions: a, ar, ar^2... 
    #   Sums: a(1+r^k). Products: a^2 r^(i+k).
    #   If we pick edges such that i+k is constant for many edges, products collide.
    #   If we pick edges such that k is constant (e.g., i, i+1), sums are a(1+r) * r^i? No, a r^i + a r^(i+1) = a r^i (1+r).
    #   This creates a geometric progression of sums. Distinct sums = number of distinct i.
    #   If we pick edges (i, i+1) for i=0..12, we have 13 edges. Sums are distinct.
    #   However, if we pick edges (i, j) such that i+j is constant, products collide.
    #   Sums a r^i + a r^j. If i+j is constant, say S, then j=S-i.
    #   Sum = a r^i + a r^(S-i). This is symmetric. Number of distinct sums is roughly S/2.
    
    # - Small integers: 1..14
    #   Sums: 3..27. Many collisions.
    #   Products: 2..182. Many collisions.
    
    candidates = []
    
    # Set A: 1 to 14
    candidates.append(list(range(1, 15)))
    
    # Set B: Powers of 2 (1, 2, 4, ..., 4096) - Bad for products (unique), bad for sums (unique binary).
    # Skip powers of 2 as they maximize distinctness.
    
    # Set C: Geometric Progression with small ratio
    # r=2: 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192
    # Products: 2^k. Unique per sum of exponents.
    # Sums: 2^i + 2^j. Unique per pair (i,j).
    # This is the WORST case for distinct counts. We want collisions.
    
    # Set D: Arithmetic Progression
    # 1, 2, 3, ..., 14 (already in A)
    # 2, 4, 6, ..., 28
    candidates.append([2*i for i in range(1, 15)])
    
    # Set E: Mixed small numbers with high collision potential
    # Numbers with many factors encourage product collisions.
    # e.g., 1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 27, 32, 36
    candidates.append([1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 27, 32, 36])
    
    # Set F: Repetitive structure via small base
    # 1, 1, 1... not allowed (distinct values required).
    
    # Set G: Randomized small integers (seeded for reproducibility in logic, but we want best)
    # We will stick to deterministic structured sets first.
    
    # Set H: "Almost" geometric
    # 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610 (Fibonacci)
    # Sums and products tend to be distinct.
    
    # Let's add a few random small sets to the candidate pool to ensure diversity.
    import random
    random.seed(42)
    for _ in range(5):
        vals = sorted(random.sample(range(1, 100), n))
        candidates.append(vals)
        
    # Deduplicate
    unique_candidates = []
    seen = set()
    for c in candidates:
        key = tuple(c)
        if key not in seen:
            seen.add(key)
            unique_candidates.append(c)

    best_raw = float('inf')
    best_result = None
    all_pairs = list(itertools.combinations(range(n), 2))
    
    # For each candidate value set, we solve the edge selection problem.
    # Goal: Select m=35 edges to minimize max(distinct_sums, distinct_prods).
    
    # Strategy for Edge Selection:
    # 1. Compute all pair sums and products.
    # 2. Identify the "core" sums and products that appear most frequently in the full graph.
    # 3. Use a greedy set cover approach: 
    #    - We want to pick edges that "hit" existing sums/prods as much as possible.
    #    - Initially, pick the edge with the most frequent sum AND product?
    #    - Better: Iteratively pick the edge that minimizes the increase in distinct counts.
    #    - Since m=35 is small relative to 91 pairs, we can use a simple greedy:
    #      Start with empty set.
    #      At each step, for each candidate edge, calculate new distinct_sums and new distinct_prods.
    #      Pick the edge that minimizes max(new_distinct_sums, new_distinct_prods).
    #      Tie-breaker: minimize sum of new distincts.
    
    for vals in unique_candidates:
        if time.time() > deadline - 5:
            break
            
        # Pre-compute
        pair_sums = {}
        pair_prods = {}
        for u, v in all_pairs:
            pair_sums[(u, v)] = vals[u] + vals[v]
            pair_prods[(u, v)] = vals[u] * vals[v]
            
        # Greedy Edge Selection
        selected_edges = set()
        current_sums = set()
        current_prods = set()
        
        # To make it faster, we can pre-compute the frequency of each sum and product in the full graph.
        # This helps in tie-breaking or initialization, but the greedy step is local.
        
        for _ in range(m):
            if time.time() > deadline - 1:
                break
                
            best_edge = None
            best_metric = float('inf')
            
            # Optimization: Only consider edges not already selected.
            # In the first few steps, all 91 are candidates. Later, 91 - k.
            # 91 * 35 * (set ops) is fast enough.
            
            for e in all_pairs:
                if e in selected_edges:
                    continue
                
                s = pair_sums[e]
                p = pair_prods[e]
                
                # Check if adding this edge increases distinct counts
                # new_sums_count = len(current_sums) + (0 if s in current_sums else 1)
                # new_prods_count = len(current_prods) + (0 if p in current_prods else 1)
                
                add_s = 0 if s in current_sums else 1
                add_p = 0 if p in current_prods else 1
                
                new_ds = len(current_sums) + add_s
                new_dp = len(current_prods) + add_p
                
                metric = max(new_ds, new_dp)
                
                # Tie-breaker: Prefer edges that add to BOTH existing sets (add_s=0, add_p=0)
                # Then prefer those that add to one.
                # Secondary metric: sum of new counts
                tie_breaker = new_ds + new_dp
                
                if best_edge is None:
                    best_edge = e
                    best_metric = metric
                    best_tie = tie_breaker
                else:
                    if metric < best_metric:
                        best_edge = e
                        best_metric = metric
                        best_tie = tie_breaker
                    elif metric == best_metric:
                        if tie_breaker < best_tie:
                            best_edge = e
                            best_tie = tie_breaker
            
            if best_edge is None:
                break
                
            selected_edges.add(best_edge)
            current_sums.add(pair_sums[best_edge])
            current_prods.add(pair_prods[best_edge])
            
        # Evaluate final result
        final_raw = max(len(current_sums), len(current_prods))
        
        if final_raw < best_raw:
            best_raw = final_raw
            edges_list = [[int(u), int(v)] for u, v in selected_edges]
            edges_list.sort()
            best_result = {
                'values': [int(x) for x in vals],
                'edges': edges_list
            }

    # Fallback if something went wrong
    if best_result is None:
        vals = list(range(1, 15))
        # Simple random edges
        edges = [list(e) for e in random.sample(all_pairs, m)]
        edges.sort()
        best_result = {
            'values': vals,
            'edges': edges
        }

    return best_result
# EVOLVE-BLOCK-END
