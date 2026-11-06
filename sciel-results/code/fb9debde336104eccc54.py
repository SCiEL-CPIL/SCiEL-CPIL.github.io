import numpy as np
import time
import itertools
import random

def construct():
    # Target n to maximize
    # Known bound is 34. Let's try 34, then 33, etc.
    
    deadline = time.time() + 110  # 10s safety margin
    
    def verify(colors, n):
        """
        Verify if a coloring is valid.
        colors: list of 0/1 for triples in lex order.
        """
        triples = list(itertools.combinations(range(n), 3))
        triple_to_idx = {t: i for i, t in enumerate(triples)}
        
        # Check Red K4
        for s in itertools.combinations(range(n), 4):
            red_count = 0
            for t in itertools.combinations(s, 3):
                if colors[triple_to_idx[t]] == 0:
                    red_count += 1
            if red_count == 4:
                return False
        
        # Check Blue K5
        for s in itertools.combinations(range(n), 5):
            blue_count = 0
            for t in itertools.combinations(s, 3):
                if colors[triple_to_idx[t]] == 1:
                    blue_count += 1
            if blue_count == 10:
                return False
                
        return True

    def try_solve_walksat(n, time_budget):
        """
        WalkSAT-style local search.
        """
        triples = list(itertools.combinations(range(n), 3))
        num_triples = len(triples)
        triple_to_idx = {t: i for i, t in enumerate(triples)}
        
        # Precompute constraints
        # Red K4s: list of lists of triple indices
        red_k4s = []
        for s in itertools.combinations(range(n), 4):
            idxs = [triple_to_idx[t] for t in itertools.combinations(s, 3)]
            red_k4s.append(idxs)
            
        # Blue K5s: list of lists of triple indices
        blue_k5s = []
        for s in itertools.combinations(range(n), 5):
            idxs = [triple_to_idx[t] for t in itertools.combinations(s, 3)]
            blue_k5s.append(idxs)
            
        num_red_k4s = len(red_k4s)
        num_blue_k5s = len(blue_k5s)
        
        # Initialize random
        colors = np.random.randint(0, 2, num_triples)
        
        # Helper to get violation count
        def get_violation_counts():
            red_viol = 0
            for idxs in red_k4s:
                if np.all(colors[idxs] == 0):
                    red_viol += 1
            blue_viol = 0
            for idxs in blue_k5s:
                if np.all(colors[idxs] == 1):
                    blue_viol += 1
            return red_viol, blue_viol

        red_viol, blue_viol = get_violation_counts()
        total_viol = red_viol + blue_viol
        
        if total_viol == 0:
            return colors.tolist()
            
        # WalkSAT Loop
        # We pick a random violated constraint and flip a random variable in it.
        # To be smarter, we can pick the variable that minimizes total violations (Hill Climbing) 
        # but WalkSAT with randomization is robust against local minima.
        
        while time.time() < time_budget:
            if total_viol == 0:
                return colors.tolist()
                
            # Pick a random violated constraint
            if red_viol > 0 and (blue_viol == 0 or np.random.random() < red_viol / (red_viol + blue_viol)):
                # Pick random red K4
                k4_idx = np.random.randint(num_red_k4s)
                # We need to ensure it's actually violated. 
                # Since we track counts, we don't have a list of *currently* violated ones readily available without scanning.
                # Scanning is O(N^4) or O(N^5) which is too slow per step.
                # Instead, let's just pick a random triple and see if it helps? No, that's the previous method.
                
                # Better: Maintain a list of violated constraints? 
                # Or just pick a random triple from the whole space and evaluate delta?
                # The previous method did delta evaluation. It was just slow due to Python loops in SA.
                # Let's stick to the delta evaluation method but optimize it.
                
                # Actually, for n=34, num_triples = 5984.
                # num_red_k4s = C(34,4) = 46376.
                # num_blue_k5s = C(34,5) = 278256.
                # Checking all constraints is expensive.
                
                # Let's go back to the delta method but make it faster.
                # The previous code had a bug or was just inefficient.
                # Let's use a simpler heuristic: Random flip, accept if it reduces violations.
                # If stuck, random flip.
                
                pass # This branch is complex to implement efficiently in pure Python without precomputed reverse maps.
            
            # Let's use a different strategy: 
            # 1. Start with a "structured" random initialization.
            # 2. Use local search: flip one triple, check delta.
            # To speed up delta check, we precompute for each triple which K4s and K5s it belongs to.
            
            # This was done in the original code. Why was it slow?
            # Probably the `while` loop with `time.time()` checks and Python overhead.
            # Let's try a batch of flips.
            
            # Reset to the optimized delta method from the original, but fix the logic and speed.
            # I will rewrite the whole solver inside this function to be self-contained and optimized.
            
            break # Exit this complex branch, use the optimized delta method below.

        # Optimized Delta Method
        # Precompute reverse maps
        sets_4_containing = [[] for _ in range(num_triples)]
        for i_4, idxs in enumerate(red_k4s):
            for t_idx in idxs:
                sets_4_containing[t_idx].append(i_4)
                
        sets_5_containing = [[] for _ in range(num_triples)]
        for i_5, idxs in enumerate(blue_k5s):
            for t_idx in idxs:
                sets_5_containing[t_idx].append(i_5)

        # Initialize counts
        red_k4_count = np.zeros(num_red_k4s, dtype=np.int8)
        blue_k5_count = np.zeros(num_blue_k5s, dtype=np.int8)
        
        for i_4, idxs in enumerate(red_k4s):
            red_k4_count[i_4] = np.sum(colors[idxs] == 0)
            
        for i_5, idxs in enumerate(blue_k5s):
            blue_k5_count[i_5] = np.sum(colors[idxs] == 1)
            
        current_violations = np.sum(red_k4_count == 4) + np.sum(blue_k5_count == 10)
        
        if current_violations == 0:
            return colors.tolist()
            
        T = 1.0
        cooling = 0.9995
        min_T = 0.01
        
        while time.time() < time_budget:
            if current_violations == 0:
                return colors.tolist()
                
            t_idx = np.random.randint(num_triples)
            c_old = colors[t_idx]
            c_new = 1 - c_old
            
            delta_violations = 0
            
            # Update 4-sets
            for i_4 in sets_4_containing[t_idx]:
                old_red = red_k4_count[i_4]
                if c_old == 0:
                    new_red = old_red - 1
                else:
                    new_red = old_red + 1
                v_old = 1 if old_red == 4 else 0
                v_new = 1 if new_red == 4 else 0
                delta_violations += (v_new - v_old)
                
            # Update 5-sets
            for i_5 in sets_5_containing[t_idx]:
                old_blue = blue_k5_count[i_5]
                if c_old == 1:
                    new_blue = old_blue - 1
                else:
                    new_blue = old_blue + 1
                v_old = 1 if old_blue == 10 else 0
                v_new = 1 if new_blue == 10 else 0
                delta_violations += (v_new - v_old)
                
            # Accept
            if delta_violations < 0 or (T > min_T and np.random.random() < np.exp(-delta_violations / T)):
                colors[t_idx] = c_new
                for i_4 in sets_4_containing[t_idx]:
                    if c_old == 0:
                        red_k4_count[i_4] -= 1
                    else:
                        red_k4_count[i_4] += 1
                for i_5 in sets_5_containing[t_idx]:
                    if c_old == 1:
                        blue_k5_count[i_5] -= 1
                    else:
                        blue_k5_count[i_5] += 1
                current_violations += delta_violations
                T *= cooling
            else:
                T *= cooling # Cool anyway to escape plateaus? Or not. Standard SA cools on accept. 
                # Let's cool on every step to ensure we eventually stop exploring.
                
        return None

    best_n = 0
    best_colors = None
    
    # Try n=34 with 80s budget
    t_start = time.time()
    colors_34 = try_solve_walksat(34, time.time() + 80)
    if colors_34 is not None:
        # Double check
        if verify(colors_34, 34):
            best_n = 34
            best_colors = colors_34
        else:
            # Internal error in solver, ignore
            pass

    if best_n == 0:
        # Try n=33 with remaining time (approx 30s)
        t_start = time.time()
        colors_33 = try_solve_walksat(33, time.time() + 30)
        if colors_33 is not None:
            if verify(colors_33, 33):
                best_n = 33
                best_colors = colors_33

    if best_n == 0:
        # Try n=32 with remaining time (approx 10s)
        t_start = time.time()
        colors_32 = try_solve_walksat(32, time.time() + 10)
        if colors_32 is not None:
            if verify(colors_32, 32):
                best_n = 32
                best_colors = colors_32

    if best_n == 0:
        # Fallback: Find a valid coloring for n=10 quickly using random restarts
        # n=10 is small enough that random search with verification will find one instantly.
        n = 10
        found = False
        for _ in range(1000):
            colors = [int(x) for x in np.random.randint(0, 2, len(list(itertools.combinations(range(n), 3))))]
            if verify(colors, n):
                best_n = n
                best_colors = colors
                found = True
                break
        if not found:
            # This should never happen for n=10, but just in case, return a trivially small one
            # n=3: 1 triple. No K4, no K5. Always valid.
            n = 3
            colors = [0]
            best_n = n
            best_colors = colors

    return {'n': best_n, 'colors': best_colors}
