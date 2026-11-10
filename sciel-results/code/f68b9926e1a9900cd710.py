"""Rewritten program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s
    
    # We'll try to find a valid coloring for as large n as possible.
    # Strategy: use simulated annealing with vectorized violation counting.
    
    best_n = 0
    best_colors = None
    
    # Try increasing n from small to large, or start with a reasonable guess
    # The published construction has 34 points, so aim for that range.
    
    # First, let's try a few candidate n values in descending order
    # to maximize the chance of finding a large valid coloring.
    
    candidate_ns = [34, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20]
    
    def count_violations(colors, n):
        """Count red K4^(3) and blue K5^(3) violations using vectorized ops."""
        # colors is a numpy array of 0s and 1s, length C(n,3)
        # Map from triple index to (a,b,c)
        # We need to check all 4-subsets for red K4 and all 5-subsets for blue K5.
        
        # Precompute: for each 4-subset {a,b,c,d}, the 4 triples are:
        # (a,b,c), (a,b,d), (a,c,d), (b,c,d)
        # For each 5-subset {a,b,c,d,e}, the 10 triples are all C(5,3)=10 triples.
        
        # This is expensive to do naively. Let's use a smarter approach.
        
        # For red K4^(3): a set of 4 vertices has all 4 triples red.
        # For blue K5^(3): a set of 5 vertices has all 10 triples blue.
        
        # We can precompute the triple indices for each 4-subset and 5-subset.
        
        # Build a mapping from (a,b,c) to index in the colors array
        triple_to_idx = {}
        idx = 0
        for a in range(n):
            for b in range(a+1, n):
                for c in range(b+1, n):
                    triple_to_idx[(a,b,c)] = idx
                    idx += 1
        
        # Count red K4 violations
        red_viol = 0
        for quad in combinations(range(n), 4):
            a, b, c, d = quad
            triples = [(a,b,c), (a,b,d), (a,c,d), (b,c,d)]
            if all(colors[triple_to_idx[t]] == 0 for t in triples):
                red_viol += 1
        
        # Count blue K5 violations
        blue_viol = 0
        for quint in combinations(range(n), 5):
            triples = list(combinations(quint, 3))
            if all(colors[triple_to_idx[t]] == 1 for t in triples):
                blue_viol += 1
        
        return red_viol, blue_viol
    
    # For efficiency, we need a faster violation counting method.
    # Let's precompute the list of triples for each 4-set and 5-set.
    
    def build_violation_lookup(n):
        """Precompute which triple indices belong to each 4-set and 5-set."""
        triple_to_idx = {}
        idx = 0
        triples_list = []
        for a in range(n):
            for b in range(a+1, n):
                for c in range(b+1, n):
                    triple_to_idx[(a,b,c)] = idx
                    triples_list.append((a,b,c))
                    idx += 1
        
        # For each 4-set, store the 4 triple indices
        quad_triples = {}
        for quad in combinations(range(n), 4):
            a, b, c, d = quad
            triples = [(a,b,c), (a,b,d), (a,c,d), (b,c,d)]
            quad_triples[quad] = [triple_to_idx[t] for t in triples]
        
        # For each 5-set, store the 10 triple indices
        quint_triples = {}
        for quint in combinations(range(n), 5):
            triples = list(combinations(quint, 3))
            quint_triples[quint] = [triple_to_idx[t] for t in triples]
        
        return triple_to_idx, quad_triples, quint_triples, triples_list
    
    # For larger n, this lookup table becomes very large. Let's be careful.
    # C(34,4) = 46376, C(34,5) = 278256. That's manageable.
    
    for n in candidate_ns:
        if time.time() > deadline - 30:
            break
        
        num_triples = n * (n-1) * (n-2) // 6
        if num_triples > 20000:
            # Too many triples, might be slow
            continue
        
        triple_to_idx, quad_triples, quint_triples, triples_list = build_violation_lookup(n)
        
        # Simulated annealing
        def fast_violation_count(colors):
            """Fast violation count using precomputed lookups."""
            red_viol = 0
            for quad, t_indices in quad_triples.items():
                if all(colors[i] == 0 for i in t_indices):
                    red_viol += 1
            
            blue_viol = 0
            for quint, t_indices in quint_triples.items():
                if all(colors[i] == 1 for i in t_indices):
                    blue_viol += 1
            
            return red_viol, blue_viol
        
        # Initialize with random coloring
        colors = np.array([random.choice([0, 1]) for _ in range(num_triples)], dtype=np.int8)
        
        red_viol, blue_viol = fast_violation_count(colors)
        current_cost = red_viol * 10 + blue_viol  # weight blue violations more heavily
        
        best_colors_sa = colors.copy()
        best_cost = current_cost
        
        if best_cost == 0:
            return {'n': n, 'colors': colors.tolist()}
        
        # Simulated annealing
        T = 10.0
        T_min = 0.1
        cooling = 0.999
        max_iters = 50000
        
        for it in range(max_iters):
            if time.time() > deadline - 10:
                break
            
            # Pick a random triple to flip
            i = random.randint(0, num_triples - 1)
            old_color = colors[i]
            new_color = 1 - old_color
            colors[i] = new_color
            
            # Recompute cost (this is expensive; we could optimize with incremental updates)
            red_viol_new, blue_viol_new = fast_violation_count(colors)
            new_cost = red_viol_new * 10 + blue_viol_new
            
            delta = new_cost - current_cost
            
            if delta < 0 or random.random() < np.exp(-delta / T):
                current_cost = new_cost
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_colors_sa = colors.copy()
                    if best_cost == 0:
                        break
            else:
                colors[i] = old_color  # revert
            
            T *= cooling
            if T < T_min:
                T = T_min
        
        if best_cost == 0:
            return {'n': n, 'colors': best_colors_sa.tolist()}
        
        # Track the best result so far
        if best_cost < 100:  # if we're close to valid
            if n > best_n:
                best_n = n
                best_colors = best_colors_sa.tolist()
    
    # If we couldn't find a valid coloring, return the best we have or a fallback
    if best_colors is not None and best_n > 0:
        # Verify the best coloring
        # Re-verify
        n = best_n
        num_triples = n * (n-1) * (n-2) // 6
        triple_to_idx, quad_triples, quint_triples, _ = build_violation_lookup(n)
        colors_arr = np.array(best_colors, dtype=np.int8)
        
        red_viol = 0
        for quad, t_indices in quad_triples.items():
            if all(colors_arr[i] == 0 for i in t_indices):
                red_viol += 1
        
        blue_viol = 0
        for quint, t_indices in quint_triples.items():
            if all(colors_arr[i] == 1 for i in t_indices):
                blue_viol += 1
        
        if red_viol == 0 and blue_viol == 0:
            return {'n': n, 'colors': best_colors}
    
    # Fallback: return a small valid coloring
    # For n=3, any coloring works (no K4 or K5 possible)
    n = 3
    colors = [0, 0, 0]  # all red, no K4 since we only have 3 vertices
    return {'n': n, 'colors': colors}

# EVOLVE-BLOCK-END
