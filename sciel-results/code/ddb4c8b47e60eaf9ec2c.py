import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    timeout = 55.0  # Safety margin
    
    n = 14
    m = 35
    max_val = 10000
    
    # Initial set: 1 to 14
    best_values = list(range(1, n + 1))
    best_score = float('inf')
    best_edges = []
    
    # Precompute all possible edges
    all_edges = list(combinations(range(n), 2))
    
    def evaluate(values, edges):
        sums = set()
        prods = set()
        for u, v in edges:
            sums.add(values[u] + values[v])
            prods.add(values[u] * values[v])
        return max(len(sums), len(prods))
    
    def greedy_select_edges(values):
        # Select m edges to minimize max(distinct sums, distinct products)
        selected = []
        current_sums = set()
        current_prods = set()
        
        # Precompute sums and products for all pairs
        pair_data = []
        for u, v in all_edges:
            s = values[u] + values[v]
            p = values[u] * values[v]
            pair_data.append((u, v, s, p))
        
        # Greedy selection
        # We want to pick edges that cause the least increase in distinct counts.
        # Score of an edge (u,v) given current sets:
        # new_max = max(len(current_sums) + (0 if s in current_sums else 1), 
        #               len(current_prods) + (0 if p in current_prods else 1))
        # We pick the edge with the minimum new_max. If tie, maybe prefer one that adds to both? 
        # Actually, minimizing the max is the goal.
        
        for _ in range(m):
            best_edge = None
            best_score_edge = float('inf')
            
            for u, v, s, p in pair_data:
                if (u, v) in selected or (v, u) in selected: # selected is list of tuples
                    continue
                # Check if already selected
                if any(e == (u,v) or e == (v,u) for e in selected):
                    continue
                
                add_s = 1 if s not in current_sums else 0
                add_p = 1 if p not in current_prods else 0
                
                new_num_sums = len(current_sums) + add_s
                new_num_prods = len(current_prods) + add_p
                score = max(new_num_sums, new_num_prods)
                
                if score < best_score_edge:
                    best_score_edge = score
                    best_edge = (u, v)
                    best_s = s
                    best_p = p
            
            if best_edge:
                selected.append(best_edge)
                current_sums.add(best_s)
                current_prods.add(best_p)
                
        return selected

    # Initial evaluation
    best_edges = greedy_select_edges(best_values)
    best_score = evaluate(best_values, best_edges)
    
    # Local Search
    # We will try to perturb values and see if we can get a better score.
    # Random seed for reproducibility in this context, but we want diversity if run multiple times?
    # The task says "construct()", implying a single call.
    
    random.seed(42)
    
    while time.time() - start_time < timeout:
        # Pick a random vertex to change
        idx = random.randint(0, n - 1)
        old_val = best_values[idx]
        
        # Try a new value
        # Search in a neighborhood or random range
        # Small integers are generally better for keeping sums/products small and overlapping.
        # Let's try values in range 1..50 initially, then widen if stuck?
        # Or just random 1..100.
        
        new_val = random.randint(1, 100)
        if new_val == old_val:
            continue
            
        # Ensure distinctness
        if new_val in best_values:
            # Swap or skip? Skip is safer for simple logic.
            # Or find a unique value.
            # Let's just pick another random one.
            continue
            
        new_values = best_values.copy()
        new_values[idx] = new_val
        
        # Re-optimize edges for this new set
        new_edges = greedy_select_edges(new_values)
        new_score = evaluate(new_values, new_edges)
        
        if new_score < best_score:
            best_score = new_score
            best_values = new_values
            best_edges = new_edges
            
            # If we hit a very low score, we might stop early?
            # Theoretical minimum?
            # With 35 edges, we have 35 sums and 35 products.
            # If all sums are equal, distinct sums = 1.
            # If all products are equal, distinct products = 1.
            # Can we have a graph where all edge sums are equal?
            # u+v = C. This is a matching. Max matching size is 7. We need 35 edges. Impossible.
            # So distinct sums must be > 1.
            # What is the lower bound?
            # For a complete graph K_n, sums range from min+2nd_min to max+2nd_max.
            # With n=14, range is roughly 2*max - 2*min.
            # If values are 1..14, range is 3..27 (25 values).
            # Can we do better?
            # If values are clustered, e.g., 10,11,12...
            # Sums: 21..29 (9 values).
            # Products: 100..121 (22 values).
            # Max is 22. This is better than 25.
            # Let's try a cluster.
            
            # If we use values 10..23 (14 values).
            # Sums: 21..45 (25 values).
            # Wait, 10+11=21, 22+23=45. Range 25.
            # Products: 110..506.
            
            # What if we use very small values?
            # 1, 1, 1... not distinct.
            # 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14.
            # Sums: 3..27 (25).
            # Products: 2..182.
            
            # Let's try 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23.
            # Sums: 21..45 (25).
            # Products: 110..506.
            
            # It seems the spread of values determines the spread of sums.
            # To minimize distinct sums, we want the range of values to be small.
            # But they must be distinct integers.
            # Min range for 14 distinct ints is 13 (e.g., 1..14 or 10..23).
            # Sum range is roughly 2 * max_val - 2 * min_val.
            # For 1..14: 28-2=26 range (25 values).
            # For 10..23: 46-20=26 range (25 values).
            # So distinct sums are likely around 25-30 no matter what, unless we pick non-consecutive values that overlap sums?
            # E.g., 1, 2, 4, 8... sums are unique. Bad.
            # Consecutive integers maximize sum collisions.
            
            # So the bottleneck might be the products.
            # For 1..14, products are 2..182.
            # How many distinct products in K_14 on 1..14?
            # 1*k = k (13 distinct)
            # 2*k = 2,4..28 (13 distinct, some overlap with 1*k)
            # ...
            # Let's estimate.
            # If we can reduce distinct products below 25, we win.
            # Or reduce sums below 25?
            # Sums of 1..14: 3,4,5...27. All present. 25 distinct.
            # Can we get fewer than 25 distinct sums with 35 edges?
            # We don't need ALL edges. We pick 35.
            # We can pick edges that only produce sums in a small subset.
            # E.g., only pick edges where sum is in {3,4,5,...,10}.
            # How many such edges exist in 1..14?
            # Sum=3: (1,2) -> 1 edge
            # Sum=4: (1,3) -> 1 edge
            # ...
            # Sum=10: (1,9), (2,8), (3,7), (4,6) -> 4 edges
            # Total edges with sum <= 10:
            # 1+1+2+3+4+5+6+7+8+9+10... 
            # Triangular numbers.
            # Sum=k has k-1 edges? No.
            # Pairs (i,j) with i<j and i+j=k.
            # i can go from 1 to floor((k-1)/2).
            # k=3: 1 pair (1,2)
            # k=4: 1 pair (1,3)
            # k=5: 2 pairs (1,4), (2,3)
            # k=6: 2 pairs (1,5), (2,4)
            # k=7: 3 pairs
            # k=8: 3 pairs
            # k=9: 4 pairs
            # k=10: 4 pairs
            # Total: 1+1+2+2+3+3+4+4 = 20 edges.
            # We need 35. So we must go up to sum=12 or 13.
            # k=11: 5 pairs
            # k=12: 5 pairs
            # 20 + 5 + 5 = 30 edges.
            # k=13: 6 pairs.
            # 30 + 6 = 36 edges.
            # So we can pick 35 edges with sums in {3, ..., 13}.
            # Distinct sums: 3..13 is 11 values.
            # This is much better than 25!
            
            # Now check products for these specific edges.
            # The edges are those with small sums.
            # (1,2), (1,3), (1,4), (2,3), (1,5), (2,4), (1,6), (2,5), (3,4), (1,7), (2,6), (3,5), (1,8), (2,7), (3,6), (4,5), (1,9), (2,8), (3,7), (4,6), (1,10), (2,9), (3,8), (4,7), (5,6), (1,11), (2,10), (3,9), (4,8), (5,7), (1,12), (2,11), (3,10), (4,9), (5,8), (6,7)...
            # Let's list the products for the first 35 edges (sums 3 to 13).
            # This specific set of edges is fixed by the "small sum" constraint.
            # We can calculate the distinct products for this set.
            
        # If we find a score < best_score, we update.
        
    # Final check and return
    # Ensure types are correct
    return {
        'values': [int(v) for v in best_values],
        'edges': [list(e) for e in best_edges]
    }
