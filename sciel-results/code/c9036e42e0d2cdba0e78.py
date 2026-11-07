"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    n = 14
    m = 35
    max_val = 10000
    
    # Precompute all possible edges
    all_edges = list(combinations(range(n), 2))
    num_possible_edges = len(all_edges)
    
    # Helper to evaluate a candidate (values, edges) using NumPy for speed
    def evaluate(values, edges):
        if not edges:
            return 0
        v_arr = np.array(values, dtype=np.int64)
        e_arr = np.array(edges, dtype=np.int32)
        
        # Get values for u and v
        u_vals = v_arr[e_arr[:, 0]]
        v_vals = v_arr[e_arr[:, 1]]
        
        sums = u_vals + v_vals
        prods = u_vals * v_vals
        
        # Count distinct
        distinct_sums = len(np.unique(sums))
        distinct_prods = len(np.unique(prods))
        
        return max(distinct_sums, distinct_prods)

    # Helper to get random distinct values
    def random_values():
        # Random sample of 14 distinct integers from 1 to 10000
        return random.sample(range(1, max_val + 1), n)
    
    # Greedy edge selection:
    # Start with empty edge set.
    # At each step, pick the edge from remaining candidates that minimizes the 
    # increase in max(distinct_sums, distinct_prods).
    # To make this efficient, we can approximate or just brute force the remaining 
    # candidates against the current set.
    # Since m=35 and total=91, we do 35 iterations.
    # In each iteration, we test up to 91 candidates.
    # Evaluation is O(m) per candidate if we just add it and check.
    # Total cost: 35 * 91 * O(35) ~ 100k ops. Very fast.
    
    def greedy_edges_for_values(values):
        v_arr = np.array(values, dtype=np.int64)
        selected_edges = []
        selected_set = set()
        
        # Precompute sums and products for ALL edges to speed up checking
        # But we need to check distinctness against the *current* selected set.
        
        current_sums = set()
        current_prods = set()
        
        # We want to minimize the final max(distinct_sums, distinct_prods).
        # Greedy heuristic: pick edge that adds the fewest NEW distinct sums/prods.
        # Tie-break: prefer edges that add 0 new distincts.
        
        # To make it faster, we can just iterate through all possible edges and pick the best one each time.
        # 91 edges * 35 steps = 3185 evaluations.
        # Each evaluation: check if sum/prod is in current sets. O(1) avg.
        
        for _ in range(m):
            best_edge = None
            best_score = float('inf')
            best_new_sums = 0
            best_new_prods = 0
            
            # Shuffle to break ties randomly
            candidates = all_edges[:]
            random.shuffle(candidates)
            
            for u, v in candidates:
                if (u, v) in selected_set or (v, u) in selected_set:
                    continue
                
                su = values[u]
                sv = values[v]
                s = su + sv
                p = su * sv
                
                new_s = 0 if s in current_sums else 1
                new_p = 0 if p in current_prods else 1
                
                # Heuristic score: prioritize minimizing the max size growth
                # If we add this edge, new distinct sums = len(current_sums) + new_s
                # new distinct prods = len(current_prods) + new_p
                # We want to minimize max(len(current_sums) + new_s, len(current_prods) + new_p)
                # Since len(current_sums) and len(current_prods) are fixed for this step,
                # we just want to minimize max(new_s, new_p)? 
                # Not exactly, because if current_sums is already much larger, adding to prods is better.
                
                target_sums = len(current_sums) + new_s
                target_prods = len(current_prods) + new_p
                target_max = max(target_sums, target_prods)
                
                # Secondary objective: minimize total new distincts (sums+prods)
                total_new = new_s + new_p
                
                score = (target_max, total_new)
                
                if score < best_score:
                    best_score = score
                    best_edge = (u, v)
                    best_new_sums = new_s
                    best_new_prods = new_p
            
            if best_edge is None:
                break
                
            selected_edges.append([best_edge[0], best_edge[1]])
            selected_set.add(best_edge)
            
            su = values[best_edge[0]]
            sv = values[best_edge[1]]
            current_sums.add(su + sv)
            current_prods.add(su * sv)
            
        return selected_edges

    # Local search on values
    def optimize_values(values, edges):
        current_values = values[:]
        current_score = evaluate(current_values, edges)
        best_values = current_values[:]
        best_score = current_score
        
        # Try to find values that create more collisions.
        # Strategy: 
        # 1. Random restarts of values? No, we are optimizing given edges.
        # 2. Perturb values to see if it reduces distinct sums/prods.
        
        T = 10.0
        T_min = 0.1
        cooling = 0.95
        
        while time.time() < deadline - 5.0 and T > T_min:
            # Perturb 1 value
            idx = random.randint(0, n - 1)
            old_val = current_values[idx]
            # Try a new value. 
            # To encourage collisions, maybe try values close to existing ones?
            # Or random.
            new_val = random.randint(1, max_val)
            while new_val in current_values:
                new_val = random.randint(1, max_val)
            
            current_values[idx] = new_val
            new_score = evaluate(current_values, edges)
            
            delta = new_score - current_score
            if delta < 0 or random.random() < np.exp(-delta / T):
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_values = current_values[:]
            else:
                current_values[idx] = old_val
            
            T *= cooling
            
        return best_values, best_score

    # Main Strategy:
    # 1. Generate a random set of values.
    # 2. Use greedy edge selection to find good edges for those values.
    # 3. Optimize the values for those fixed edges.
    # 4. Repeat with different random initial values.
    
    best_overall_values = None
    best_overall_edges = None
    best_overall_score = float('inf')
    
    num_restarts = 20
    
    for r in range(num_restarts):
        if time.time() > deadline - 5.0:
            break
            
        # Initialize values. 
        # Sometimes try structured values (e.g., small integers) to see if they help.
        if r % 3 == 0:
            # Small integers 1..14
            vals = list(range(1, n + 1))
            random.shuffle(vals)
        elif r % 3 == 1:
            # Random small
            vals = random.sample(range(1, 100), n)
        else:
            # Random large
            vals = random.sample(range(1, max_val + 1), n)
            
        # Step 1: Greedy Edges
        edges = greedy_edges_for_values(vals)
        
        # Step 2: Optimize Values for these edges
        opt_vals, score = optimize_values(vals, edges)
        
        if score < best_overall_score:
            best_overall_score = score
            best_overall_values = opt_vals
            best_overall_edges = edges
            
            # Optional: Re-run greedy edges on the optimized values?
            # The edges were chosen for the *initial* values.
            # The optimized values might have different collision structures.
            # Let's try re-selecting edges for the optimized values.
            new_edges = greedy_edges_for_values(opt_vals)
            new_score = evaluate(opt_vals, new_edges)
            
            if new_score < best_overall_score:
                best_overall_score = new_score
                best_overall_edges = new_edges
                
                # And maybe optimize values again for these new edges?
                # This could be a loop. Let's do one more pass of value optimization.
                opt_vals2, score2 = optimize_values(opt_vals, new_edges)
                if score2 < best_overall_score:
                    best_overall_score = score2
                    best_overall_values = opt_vals2
                    
    # Final sanity check and type conversion
    if best_overall_values is None:
        # Fallback
        best_overall_values = random_values()
        best_overall_edges = random.sample(all_edges, m)
        
    res = {
        'values': [int(v) for v in best_overall_values],
        'edges': [[int(u), int(v)] for u, v in best_overall_edges]
    }
    
    return res

# EVOLVE-BLOCK-END
