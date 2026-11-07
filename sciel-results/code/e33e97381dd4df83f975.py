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
    
    # Helper to evaluate a candidate (values, edges)
    def evaluate(values, edges):
        if not edges:
            return 0
        v_arr = np.array(values, dtype=np.int64)
        e_arr = np.array(edges, dtype=np.int32)
        
        u_vals = v_arr[e_arr[:, 0]]
        v_vals = v_arr[e_arr[:, 1]]
        
        sums = u_vals + v_vals
        prods = u_vals * v_vals
        
        distinct_sums = len(np.unique(sums))
        distinct_prods = len(np.unique(prods))
        
        return max(distinct_sums, distinct_prods)

    # Strategy: 
    # 1. Generate candidate value sets that are "collision-prone".
    #    Small integers or integers with many factors tend to create more product collisions.
    #    Also, having duplicate-like structures (e.g., pairs) helps sums.
    # 2. For each value set, greedily build the 35 edges.
    #    At each step, pick the edge (u,v) not yet added that minimizes the increase in 
    #    max(distinct_sums, distinct_prods). If tie, prefer one that creates more collisions.
    
    def greedy_build_edges(values):
        """
        Greedily select m edges to minimize max(distinct_sums, distinct_prods).
        """
        current_edges = []
        edge_set = set()
        
        # Pre-compute sums and prods for all possible edges to speed up lookup?
        # Or just maintain sets and check membership.
        # With n=14, there are 91 edges. We need 35.
        # 35 steps * 91 candidates * O(1) check = ~3000 ops. Very fast.
        
        # To be efficient, we can pre-compute the sum and prod for every possible edge
        # and store them in a list of (edge, sum, prod).
        edge_data = []
        for u, v in all_edges:
            s = values[u] + values[v]
            p = values[u] * values[v]
            edge_data.append((u, v, s, p))
        
        sums_set = set()
        prods_set = set()
        
        # We want to minimize the final max(len(sums_set), len(prods_set)).
        # Greedy heuristic: Pick edge that adds to the smaller set, or creates a collision.
        # Let's try a few different greedy strategies and keep the best.
        
        best_edges = None
        best_score = float('inf')
        
        # Strategy A: Always pick edge that results in min(new_len_sums, new_len_prods) increase
        # Strategy B: Pick edge that maximizes collisions (sum in sums_set OR prod in prods_set)
        
        # Let's implement a generic greedy loop with a scoring function for the next edge
        # Score of an edge (s, p) given current sets:
        # delta_s = 1 if s not in sums_set else 0
        # delta_p = 1 if p not in prods_set else 0
        # We want to minimize max(len(sums)+delta_s, len(prods)+delta_p)
        
        for attempt in range(20): # Try a few randomizations if greedy gets stuck?
             # Actually, pure greedy is deterministic. 
             # Let's just run the deterministic greedy.
             if attempt > 0:
                 break # Just do it once for now, maybe add random tie-breaking later if needed
                
             cur_sums = set()
             cur_prods = set()
             used_edges = set()
             selected_edges = []
             
             for _ in range(m):
                 best_next_edge = None
                 best_next_score = float('inf')
                 
                 for u, v, s, p in edge_data:
                     if (u, v) in used_edges:
                         continue
                     
                     new_sums_size = len(cur_sums) + (0 if s in cur_sums else 1)
                     new_prods_size = len(cur_prods) + (0 if p in cur_prods else 1)
                     
                     # Objective: Minimize the max of the two sizes
                     # Secondary objective: Prefer collisions (lower individual sizes)
                     # Let's use a tuple: (max_size, new_sums_size, new_prods_size)
                     # Actually, just max_size is the primary goal.
                     score = max(new_sums_size, new_prods_size)
                     
                     # Tie-breaker: prefer edges that create collisions (lower sum/prod sizes)
                     # If scores are equal, pick the one with smaller sum of sizes
                     tie_breaker = new_sums_size + new_prods_size
                     
                     if (score, tie_breaker) < (best_next_score, float('inf')):
                         best_next_score = score
                         best_next_edge = (u, v, s, p)
                         
                         # If we found a perfect collision (score didn't increase), we could break early?
                         # No, because we need to fill 35 edges.
                         
                 if best_next_edge is None:
                     break # Should not happen
                    
                 u, v, s, p = best_next_edge
                 selected_edges.append([u, v])
                 used_edges.add((u, v))
                 cur_sums.add(s)
                 cur_prods.add(p)
                 
             if len(selected_edges) == m:
                 score = evaluate(values, selected_edges)
                 if score < best_score:
                     best_score = score
                     best_edges = selected_edges
                     
        return best_edges, best_score

    def generate_collision_prone_values():
        """
        Generate values that are likely to produce collisions.
        1. Small integers (1-50): High density of sums/prods.
        2. Multiples of a common number.
        3. Random but biased towards small numbers.
        """
        strategy = random.randint(0, 3)
        
        if strategy == 0:
            # Small random integers
            return random.sample(range(1, 60), n)
        elif strategy == 1:
            # Multiples of a random small number
            base = random.randint(2, 10)
            candidates = [base * i for i in range(1, 100)]
            # Ensure distinct
            return random.sample(candidates, n)
        elif strategy == 2:
            # Mix of very small and some larger
            smalls = random.sample(range(1, 20), 8)
            larges = random.sample(range(20, 100), 6)
            vals = smalls + larges
            random.shuffle(vals)
            return vals
        else:
            # Geometric progression (often distinct products, but let's see)
            base = random.uniform(1.1, 1.5)
            start = random.randint(1, 10)
            vals = [int(start * (base ** i)) for i in range(n)]
            # Ensure distinct and <= 10000
            if len(set(vals)) < n or max(vals) > 10000:
                return random.sample(range(1, 100), n)
            return vals

    best_overall_score = float('inf')
    best_overall_values = None
    best_overall_edges = None
    
    # Time budget for multiple restarts
    max_restarts = 50
    
    for i in range(max_restarts):
        if time.time() > deadline - 2.0:
            break
            
        values = generate_collision_prone_values()
        # Ensure distinct
        if len(set(values)) != n:
            # Fix duplicates by adding random offsets
            seen = set()
            new_vals = []
            for v in values:
                while v in seen:
                    v += random.randint(1, 5)
                seen.add(v)
                new_vals.append(v)
            values = new_vals
            if max(values) > 10000:
                values = random.sample(range(1, 100), n)
        
        edges, score = greedy_build_edges(values)
        
        if edges and score < best_overall_score:
            best_overall_score = score
            best_overall_values = values.copy()
            best_overall_edges = [list(e) for e in edges]
            
            # If we hit a very low score, we can stop early?
            # Theoretical minimum:
            # Sums: min 14-1+1 = 14? No, 14 vertices, min sum 1+2=3, max...
            # If we can get max(distinct) to ~20-25, that's good.
            # Current best is ~40-50.
            if best_overall_score <= 20:
                break

    # Fallback if something went wrong
    if best_overall_edges is None:
        values = random.sample(range(1, 100), n)
        edges = random.sample(all_edges, m)
        best_overall_values = values
        best_overall_edges = [list(e) for e in edges]
        best_overall_score = evaluate(values, edges)

    res = {
        'values': [int(v) for v in best_overall_values],
        'edges': [[int(u), int(v)] for u, v in best_overall_edges]
    }
    
    return res

# EVOLVE-BLOCK-END
