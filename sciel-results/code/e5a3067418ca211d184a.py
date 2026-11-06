import time
import numpy as np
from itertools import combinations
from math import comb

def construct():
    start_time = time.time()
    deadline = start_time + 112  # safety margin before 150s kill

    def search_for_n(n, time_limit, seed=0):
        M = comb(n, 3)
        triples = list(combinations(range(n), 3))
        tri_to_idx = {tri: i for i, tri in enumerate(triples)}

        # Precompute incidence
        quad_triples = []
        for quad in combinations(range(n), 4):
            idxs = [tri_to_idx[tuple(x for j, x in enumerate(quad) if j != i)] for i in range(4)]
            quad_triples.append(idxs)
        quint_triples = []
        for quint in combinations(range(n), 5):
            idxs = [tri_to_idx[t] for t in combinations(quint, 3)]
            quint_triples.append(idxs)

        n_quads = len(quad_triples)
        n_quints = len(quint_triples)

        # Inverse incidence: triple -> list of quads/quints it belongs to
        triple_to_quads = [[] for _ in range(M)]
        for qidx, tidxs in enumerate(quad_triples):
            for t in tidxs:
                triple_to_quads[t].append(qidx)
        
        triple_to_quints = [[] for _ in range(M)]
        for qidx, tidxs in enumerate(quint_triples):
            for t in tidxs:
                triple_to_quints[t].append(qidx)

        rng = np.random.RandomState(seed)

        # Better initialization: use GF(2^k) trace-based algebraic seed
        # For n up to 34, we can use a structured approach
        # Color triple (a,b,c) based on linear algebra over GF(2)
        # This provides a low-violation structured starting point
        
        # Use a simple but effective heuristic: 
        # Red if (a+b+c) % 3 == 0, with some perturbation for larger n
        colors = np.zeros(M, dtype=np.int8)
        for i, (a, b, c) in enumerate(triples):
            # Use a more sophisticated pattern
            s = a + b + c
            if n >= 20:
                # Add some structure based on vertex positions
                colors[i] = 0 if (s % 3 == 0 or (s % 5 == 2 and (a ^ b) % 2 == 1)) else 1
            else:
                colors[i] = 0 if s % 3 == 0 else 1

        # Initialize counts
        red_counts = np.zeros(n_quads, dtype=np.int32)
        blue_counts = np.zeros(n_quints, dtype=np.int32)
        
        # Populate counts
        for t in range(M):
            if colors[t] == 0: # Red
                for q in triple_to_quads[t]:
                    red_counts[q] += 1
            else: # Blue
                for k in triple_to_quints[t]:
                    blue_counts[k] += 1
        
        # Count initial violations
        total_violations = int(np.sum(red_counts == 4) + np.sum(blue_counts == 10))

        def flip(t):
            nonlocal total_violations
            old_color = colors[t]
            
            if old_color == 0: # Red -> Blue
                # Update Red counts (decrease)
                for q in triple_to_quads[t]:
                    old_c = red_counts[q]
                    if old_c == 4:
                        total_violations -= 1
                    red_counts[q] = old_c - 1
                
                # Update Blue counts (increase)
                for k in triple_to_quints[t]:
                    old_c = blue_counts[k]
                    blue_counts[k] = old_c + 1
                    if old_c + 1 == 10:
                        total_violations += 1
                
                colors[t] = 1

            else: # Blue -> Red
                # Update Blue counts (decrease)
                for k in triple_to_quints[t]:
                    old_c = blue_counts[k]
                    if old_c == 10:
                        total_violations -= 1
                    blue_counts[k] = old_c - 1
                
                # Update Red counts (increase)
                for q in triple_to_quads[t]:
                    old_c = red_counts[q]
                    red_counts[q] = old_c + 1
                    if old_c + 1 == 4:
                        total_violations += 1
                
                colors[t] = 0

        def compute_delta(t):
            """Compute the change in total_violations if we flip triple t."""
            old_color = colors[t]
            delta = 0
            if old_color == 0:
                # Red -> Blue
                for q in triple_to_quads[t]:
                    if red_counts[q] == 4: 
                        delta -= 1
                for k in triple_to_quints[t]:
                    if blue_counts[k] == 9: 
                        delta += 1
            else:
                # Blue -> Red
                for k in triple_to_quints[t]:
                    if blue_counts[k] == 10: 
                        delta -= 1
                for q in triple_to_quads[t]:
                    if red_counts[q] == 3: 
                        delta += 1
            return delta

        # Conflict-driven local search: focus on violating structures
        # Strategy: 
        # 1. Identify all currently violating quads and quints
        # 2. For each violating structure, consider flipping one of its triples
        # 3. Choose the flip that breaks the most violations (or creates the fewest)
        
        max_iterations = 50000
        stagnant_count = 0
        best_violations = total_violations
        
        for it in range(max_iterations):
            if time.time() > start_time + time_limit:
                break
            if total_violations == 0:
                return colors.tolist()
            
            # Find violating structures
            violating_quads = [i for i in range(n_quads) if red_counts[i] == 4]
            violating_quints = [i for i in range(n_quints) if blue_counts[i] == 10]
            
            if not violating_quads and not violating_quints:
                return colors.tolist()
            
            # Collect candidate flips from violating structures
            candidates = set()
            
            # From violating quads: flip any of the 4 red triples
            for qidx in violating_quads:
                for t in quad_triples[qidx]:
                    if colors[t] == 0:  # Only consider flipping red triples
                        candidates.add(t)
            
            # From violating quints: flip any of the 10 blue triples
            for qidx in violating_quints:
                for t in quint_triples[qidx]:
                    if colors[t] == 1:  # Only consider flipping blue triples
                        candidates.add(t)
            
            # If no candidates (shouldn't happen), pick random
            if not candidates:
                t = rng.randint(M)
                delta = compute_delta(t)
                if delta <= 0:
                    flip(t)
                else:
                    # Accept with some probability (simulated annealing style)
                    if rng.random() < 0.1:
                        flip(t)
                continue
            
            # Evaluate all candidates and pick the best
            best_delta = 0
            best_t = -1
            candidate_list = list(candidates)
            
            # If too many candidates, sample
            if len(candidate_list) > 200:
                idxs = rng.choice(len(candidate_list), size=200, replace=False)
                candidate_list = [candidate_list[i] for i in idxs]
            
            for t in candidate_list:
                delta = compute_delta(t)
                if delta < best_delta:
                    best_delta = delta
                    best_t = t
            
            if best_t != -1:
                flip(best_t)
                if total_violations < best_violations:
                    best_violations = total_violations
                    stagnant_count = 0
                else:
                    stagnant_count += 1
            else:
                # No improving move found; do a random flip to escape local minimum
                t = rng.randint(M)
                flip(t)
                stagnant_count += 1
            
            # If stuck for too long, do a more aggressive restart
            if stagnant_count > 500:
                # Perturb: flip 5-10 random triples
                for _ in range(rng.randint(5, 11)):
                    t = rng.randint(M)
                    flip(t)
                stagnant_count = 0
        
        return None

    best_n = 1
    best_colors = [0]

    # Try to find the maximum n that can be solved
    # Start from high n and work down if needed
    for n in [34, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]:
        if time.time() > deadline - 5:
            break
        
        # Give more time for larger n
        if n >= 30:
            t_limit = 30.0
        elif n >= 25:
            t_limit = 25.0
        elif n >= 20:
            t_limit = 20.0
        else:
            t_limit = 15.0
        
        res = search_for_n(n, t_limit, seed=n)
        if res is not None:
            best_n = n
            best_colors = res
            break  # Found a valid coloring, stop
    
    # If we didn't find anything (shouldn't happen), return a trivial valid coloring
    if best_n == 1:
        # For n=1, there are no triples, so it's trivially valid
        best_colors = []
    
    return {'n': best_n, 'colors': best_colors}
