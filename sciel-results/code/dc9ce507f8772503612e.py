"""Strategy: Joint optimization of value sets and edge selection via Simulated Annealing with log-sum-exp smooth surrogate."""
# EVOLVE-BLOCK-START
import itertools
import time
import math
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    n = 14
    m = 35
    
    def raw_score(values, edges):
        """Compute the actual integer objective: max(distinct sums, distinct products)."""
        sums = set()
        prods = set()
        for u, v in edges:
            sums.add(values[u] + values[v])
            prods.add(values[u] * values[v])
        return max(len(sums), len(prods))

    def smooth_score(values, edges, beta=10.0):
        """
        Smooth surrogate for the objective.
        Uses log-sum-exp to approximate max(count_sums, count_prods) continuously.
        This helps SA navigate the landscape more effectively than the hard step function.
        """
        sums = []
        prods = []
        for u, v in edges:
            sums.append(values[u] + values[v])
            prods.append(values[u] * values[v])
        
        # Count frequencies
        from collections import Counter
        sum_counts = Counter(sums)
        prod_counts = Counter(prods)
        
        # We want to minimize the number of distinct values.
        # Let S be the set of distinct sums, P be the set of distinct prods.
        # Objective: max(|S|, |P|).
        # Smooth approximation: log(sum(exp(beta * 1_{s in S})) + sum(exp(beta * 1_{p in P}))) / beta ?
        # Actually, simpler: just use the hard count but add a small noise or use it directly?
        # The prompt suggests smooth surrogate.
        # Let's approximate max(A, B) where A=|S|, B=|P|.
        # max(A,B) ~ (1/beta) * log(exp(beta*A) + exp(beta*B))
        a = len(sum_counts)
        b = len(prod_counts)
        
        # To avoid overflow, use the standard log-sum-exp trick
        max_val = max(a, b)
        if max_val == 0:
            return 0.0
        return (max_val + math.log(math.exp(beta*(a-max_val)) + math.exp(beta*(b-max_val)))) / beta

    def random_value_set():
        """Generate a random distinct value set in [1, 10000]."""
        # Mix of strategies: sometimes smooth numbers, sometimes random
        r = random.random()
        if r < 0.3:
            # Smooth numbers
            primes = [2, 3, 5, 7, 11, 13]
            vals = set()
            while len(vals) < n:
                v = 1
                for _ in range(random.randint(1, 6)):
                    v *= random.choice(primes)
                if 1 <= v <= 10000:
                    vals.add(v)
            return list(vals)
        elif r < 0.6:
            # Arithmetic progression
            start = random.randint(1, 50)
            step = random.randint(1, 20)
            vals = [start + i * step for i in range(n)]
            if max(vals) <= 10000 and len(set(vals)) == n:
                return vals
            else:
                # Fallback to random
                return random.sample(range(1, 10001), n)
        else:
            # Random
            return random.sample(range(1, 10001), n)

    def random_edge_set():
        """Generate a random set of 35 edges."""
        all_edges = list(itertools.combinations(range(n), 2))
        return random.sample(all_edges, m)

    # Initial states
    best_state = None
    best_raw = float('inf')
    best_smooth = float('inf')
    
    # Generate a few initial candidates to pick a good start
    candidates = []
    for i in range(10):
        vals = random_value_set()
        edges = random_edge_set()
        rs = raw_score(vals, edges)
        ss = smooth_score(vals, edges)
        candidates.append((rs, ss, vals, edges))
        
    # Sort by raw score
    candidates.sort(key=lambda x: x[0])
    curr_raw, curr_smooth, curr_vals, curr_edges = candidates[0]
    best_raw, best_smooth, best_vals, best_edges = curr_raw, curr_smooth, curr_vals, curr_edges

    # SA Parameters
    t_start = 10.0
    t_end = 0.01
    max_iters = 50000  # We will check time in loop
    
    # Precompute all edges for fast access
    all_edges = list(itertools.combinations(range(n), 2))
    
    # State representation for SA:
    # values: list of ints
    # edges: set of tuples (for O(1) check)
    
    curr_edges_set = set(curr_edges)
    
    # Helper to update score efficiently?
    # For n=14, m=35, recalculating is fast enough.
    # 35 edges * (add+mul) * 2 sets = ~140 ops. 50k iters = 7M ops. Should be fine in Python if optimized.
    
    for i in range(max_iters):
        if time.time() > deadline - 5.0:
            break
            
        # Temperature
        progress = i / max_iters
        t = t_start * (t_end / t_start) ** progress
        
        # Generate Neighbor
        r = random.random()
        
        if r < 0.5:
            # Perturb Values
            new_vals = list(curr_vals)
            idx = random.randint(0, n - 1)
            # Small change
            change = random.randint(-10, 10)
            if change == 0:
                change = random.choice([-1, 1])
            new_val = new_vals[idx] + change
            if 1 <= new_val <= 10000 and new_val not in new_vals:
                # Replace
                new_vals[idx] = new_val
            else:
                # Try random replacement
                new_val = random.randint(1, 10000)
                if new_val not in new_vals:
                    new_vals[idx] = new_val
                else:
                    continue # Skip if can't make a valid move
        else:
            # Perturb Edges
            new_edges_set = set(curr_edges_set)
            # Remove one edge
            edge_to_remove = random.choice(list(new_edges_set))
            new_edges_set.remove(edge_to_remove)
            # Add one edge
            new_edge = random.choice(all_edges)
            if new_edge not in new_edges_set:
                new_edges_set.add(new_edge)
            else:
                continue
            
            new_vals = curr_vals
            new_edges_list = list(new_edges_set)

        # Evaluate
        if r < 0.5:
            new_edges_list = list(curr_edges_set)
            
        new_raw = raw_score(new_vals, new_edges_list)
        new_smooth = smooth_score(new_vals, new_edges_list)
        
        # Acceptance Criterion
        # We primarily care about raw score, but smooth helps navigation.
        # If raw improves, always accept.
        # If raw is same, accept with prob exp(-(new_smooth - curr_smooth)/t)
        # If raw worsens, accept with prob exp(-(new_raw - curr_raw)/t) * exp(-(new_smooth - curr_smooth)/t)
        
        delta_raw = new_raw - curr_raw
        delta_smooth = new_smooth - curr_smooth
        
        if delta_raw < 0:
            accept = True
        elif delta_raw == 0:
            if delta_smooth < 0:
                accept = True
            else:
                accept = random.random() < math.exp(-delta_smooth / max(t, 1e-9))
        else:
            # Penalize raw increase heavily
            accept = random.random() < math.exp(-(delta_raw * 5.0 + delta_smooth) / max(t, 1e-9))
            
        if accept:
            curr_vals = new_vals
            curr_edges_set = set(new_edges_list)
            curr_raw = new_raw
            curr_smooth = new_smooth
            
            if curr_raw < best_raw:
                best_raw = curr_raw
                best_smooth = curr_smooth
                best_vals = list(curr_vals)
                best_edges = list(curr_edges_set)
                
                # If we reach a very good score, tighten search?
                # For now, just keep going.

    # Final Polish: Local Search on the best found state
    if time.time() < deadline - 1.0 and best_vals:
        # Try all single edge swaps
        best_edges_set = set(best_edges)
        improved = True
        while improved and time.time() < deadline - 0.5:
            improved = False
            current_best_raw = raw_score(best_vals, list(best_edges_set))
            
            # Iterate over all possible edges to remove
            for e_remove in best_edges_set:
                if time.time() > deadline - 0.5:
                    break
                best_edges_set.remove(e_remove)
                
                # Try adding all possible edges
                best_add = None
                best_add_raw = current_best_raw
                
                for e_add in all_edges:
                    if e_add in best_edges_set:
                        continue
                    
                    best_edges_set.add(e_add)
                    new_edges_list = list(best_edges_set)
                    new_raw = raw_score(best_vals, new_edges_list)
                    
                    if new_raw < best_add_raw:
                        best_add_raw = new_raw
                        best_add = e_add
                    best_edges_set.remove(e_add)
                
                if best_add is not None:
                    best_edges_set.add(best_add)
                    improved = True
                    break # Restart loop with new state
                
                best_edges_set.add(e_remove) # Revert removal if no improvement found for this removal

    # Ensure types are correct
    final_vals = [int(v) for v in best_vals]
    final_edges = [[int(u), int(v)] for u, v in best_edges]
    
    # Validation
    if len(set(final_vals)) != n or any(v < 1 or v > 10000 for v in final_vals):
        # Fallback
        final_vals = list(range(1, 15))
        final_edges = list(itertools.combinations(range(n), 2))[:m]
        final_edges = [[int(u), int(v)] for u, v in final_edges]
        
    if len(final_edges) != m:
        # Fallback
        final_edges = list(itertools.combinations(range(n), 2))[:m]
        final_edges = [[int(u), int(v)] for u, v in final_edges]

    return {'values': final_vals, 'edges': final_edges}
# EVOLVE-BLOCK-END
