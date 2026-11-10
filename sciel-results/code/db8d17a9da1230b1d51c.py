"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    def time_left():
        return deadline - time.time()
    
    def solve_for_n(n, max_time):
        """
        Attempts to find a valid coloring for n vertices within max_time seconds.
        Returns (colors, n) if found, else (None, None).
        """
        if time_left() < 5.0:
            return None, None
            
        # Precomputation
        triples = list(combinations(range(n), 3))
        num_triples = len(triples)
        triple_to_idx = {t: i for i, t in enumerate(triples)}
        
        four_sets = list(combinations(range(n), 4))
        num_k4 = len(four_sets)
        
        five_sets = list(combinations(range(n), 5))
        num_k5 = len(five_sets)
        
        # Map K4s to their constituent triples
        k4_triples = []
        for s4 in four_sets:
            a, b, c, d = s4
            k4_triples.append([
                triple_to_idx[(a, b, c)],
                triple_to_idx[(a, b, d)],
                triple_to_idx[(a, c, d)],
                triple_to_idx[(b, c, d)]
            ])
        
        # Map K5s to their constituent triples
        k5_triples = []
        for s5 in five_sets:
            a, b, c, d, e = s5
            k5_triples.append([
                triple_to_idx[(a, b, c)],
                triple_to_idx[(a, b, d)],
                triple_to_idx[(a, b, e)],
                triple_to_idx[(a, c, d)],
                triple_to_idx[(a, c, e)],
                triple_to_idx[(a, d, e)],
                triple_to_idx[(b, c, d)],
                triple_to_idx[(b, c, e)],
                triple_to_idx[(b, d, e)],
                triple_to_idx[(c, d, e)]
            ])
            
        # Reverse maps: triple -> list of K4 indices, triple -> list of K5 indices
        triple_to_k4 = [[] for _ in range(num_triples)]
        for i, tris in enumerate(k4_triples):
            for t in tris:
                triple_to_k4[t].append(i)
                
        triple_to_k5 = [[] for _ in range(num_triples)]
        for i, tris in enumerate(k5_triples):
            for t in tris:
                triple_to_k5[t].append(i)

        def evaluate_and_init():
            colors = np.random.randint(0, 2, size=num_triples, dtype=np.int8)
            red_counts = np.zeros(num_k4, dtype=np.int32)
            blue_counts = np.zeros(num_k5, dtype=np.int32)
            
            # Calculate initial counts
            for i, tris in enumerate(k4_triples):
                red_counts[i] = np.sum(colors[tris] == 0)
                
            for i, tris in enumerate(k5_triples):
                blue_counts[i] = np.sum(colors[tris] == 1)
                
            total_violations = np.sum(red_counts == 4) + np.sum(blue_counts == 10)
            return colors, red_counts, blue_counts, total_violations

        def flip_triple(t_idx, colors, red_counts, blue_counts, total_violations):
            """
            Flips the color of triple t_idx.
            Returns new total_violations.
            Modifies colors, red_counts, blue_counts in place.
            """
            old_color = colors[t_idx]
            new_color = 1 - old_color
            colors[t_idx] = new_color
            
            if old_color == 0: # Red -> Blue
                # Update K4s (Red count decreases)
                for k4_idx in triple_to_k4[t_idx]:
                    old_rc = red_counts[k4_idx]
                    new_rc = old_rc - 1
                    red_counts[k4_idx] = new_rc
                    if old_rc == 4:
                        total_violations -= 1
                    # If new_rc == 4, it wasn't 4 before (was 5? impossible for K4)
                    
                # Update K5s (Blue count increases)
                for k5_idx in triple_to_k5[t_idx]:
                    old_bc = blue_counts[k5_idx]
                    new_bc = old_bc + 1
                    blue_counts[k5_idx] = new_bc
                    if new_bc == 10:
                        total_violations += 1
            else: # Blue -> Red
                # Update K4s (Red count increases)
                for k4_idx in triple_to_k4[t_idx]:
                    old_rc = red_counts[k4_idx]
                    new_rc = old_rc + 1
                    red_counts[k4_idx] = new_rc
                    if new_rc == 4:
                        total_violations += 1
                    
                # Update K5s (Blue count decreases)
                for k5_idx in triple_to_k5[t_idx]:
                    old_bc = blue_counts[k5_idx]
                    new_bc = old_bc - 1
                    blue_counts[k5_idx] = new_bc
                    if old_bc == 10:
                        total_violations -= 1
                        
            return total_violations

        # Simulated Annealing
        best_violations = float('inf')
        best_colors = None
        
        # Number of restarts based on time
        # Each restart: init + SA
        # We estimate time per step. 
        # For n=34, num_triples ~ 6000.
        # Update cost is proportional to degree of triple in K4 and K5 graphs.
        # Avg K4s per triple: 4 * C(n-3, 1) / 3 ? No.
        # A triple (a,b,c) is in K4s formed by adding one vertex. C(n-3, 1) = n-3 K4s.
        # A triple (a,b,c) is in K5s formed by adding two vertices. C(n-3, 2) K5s.
        # For n=34: K4s per triple = 31. K5s per triple = C(31,2) = 465.
        # So each step touches ~31 K4s and ~465 K5s. ~500 ops.
        # 500 ops is fast. 100,000 steps ~ 50M ops ~ 0.1s? Maybe 0.5s in Python.
        # So we can do many steps.
        
        attempts = 0
        while time_left() > 10.0:
            colors, red_counts, blue_counts, total_violations = evaluate_and_init()
            
            if total_violations == 0:
                return colors, n
                
            if total_violations < best_violations:
                best_violations = total_violations
                best_colors = colors.copy()
                
            temp = 1.0
            cooling = 0.9999
            steps_in_this_run = 0
            max_steps_per_run = 200000
            
            while steps_in_this_run < max_steps_per_run and time_left() > 10.0:
                t_idx = np.random.randint(0, num_triples)
                old_violations = total_violations
                
                new_violations = flip_triple(t_idx, colors, red_counts, blue_counts, total_violations)
                
                delta = new_violations - old_violations
                
                if delta <= 0 or (temp > 0.01 and np.random.random() < np.exp(-delta / temp)):
                    total_violations = new_violations
                    if new_violations < best_violations:
                        best_violations = new_violations
                        best_colors = colors.copy()
                        if best_violations == 0:
                            return colors, n
                else:
                    # Revert
                    flip_triple(t_idx, colors, red_counts, blue_counts, total_violations)
                    
                temp *= cooling
                steps_in_this_run += 1
                
            attempts += 1
            
        if best_violations == 0:
            return best_colors, n
        return None, None

    # Try n=34 first
    best_n = 0
    best_colors = None
    
    # Target n=34
    if time_left() > 30.0:
        res_colors, res_n = solve_for_n(34, time_left() - 30.0)
        if res_colors is not None:
            best_n = 34
            best_colors = res_colors
        else:
            # Try n=33
            if time_left() > 20.0:
                res_colors, res_n = solve_for_n(33, time_left() - 20.0)
                if res_colors is not None:
                    best_n = 33
                    best_colors = res_colors
            else:
                # Fallback to smaller n if time is tight
                for n_try in [32, 31, 30, 25, 20]:
                    if time_left() < 10.0: break
                    res_colors, res_n = solve_for_n(n_try, time_left() - 5.0)
                    if res_colors is not None:
                        best_n = n_try
                        best_colors = res_colors
                        break

    # If we still don't have a good n, try a quick random search for small n to ensure we return something valid
    if best_n == 0:
        # Fallback to a known valid small n, e.g., n=9 from seed logic or just random n=20
        # The seed program returned n=9 (score 1.0) or similar. 
        # Let's just use a simple valid coloring for n=10 or so to be safe, or rely on the solver for n=20.
        # Actually, solve_for_n(20) should be very fast.
        if time_left() > 5.0:
             res_colors, res_n = solve_for_n(20, 5.0)
             if res_colors is not None:
                 best_n = 20
                 best_colors = res_colors
        if best_n == 0:
             # Absolute fallback: n=1, colors=[]
             best_n = 1
             best_colors = []

    # Convert to list
    if best_colors is not None:
        color_list = best_colors.tolist()
    else:
        color_list = []
        
    return {'n': best_n, 'colors': color_list}

# EVOLVE-BLOCK-END
