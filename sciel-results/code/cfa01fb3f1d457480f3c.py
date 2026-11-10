import numpy as np
import time
import itertools
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    # Precomputed structures for n up to 35
    # We will focus on n=34, 33, 32...
    
    def get_structures(n):
        """
        Returns structures needed for the repair algorithm.
        """
        triples = list(itertools.combinations(range(n), 3))
        triple_index = {t: i for i, t in enumerate(triples)}
        M = len(triples)
        
        # Generate K4s
        k4s = list(itertools.combinations(range(n), 4))
        N_k4 = len(k4s)
        # For each K4, store the 4 triple indices
        M4 = np.zeros((N_k4, 4), dtype=np.int32)
        # For each triple, which K4s does it belong to?
        triple_to_k4s = [[] for _ in range(M)]
        
        for i, k4 in enumerate(k4s):
            t1 = triple_index[(k4[0], k4[1], k4[2])]
            t2 = triple_index[(k4[0], k4[1], k4[3])]
            t3 = triple_index[(k4[0], k4[2], k4[3])]
            t4 = triple_index[(k4[1], k4[2], k4[3])]
            M4[i] = [t1, t2, t3, t4]
            triple_to_k4s[t1].append(i)
            triple_to_k4s[t2].append(i)
            triple_to_k4s[t3].append(i)
            triple_to_k4s[t4].append(i)
            
        # Generate K5s
        k5s = list(itertools.combinations(range(n), 5))
        N_k5 = len(k5s)
        # For each K5, store the 10 triple indices
        M5 = np.zeros((N_k5, 10), dtype=np.int32)
        triple_to_k5s = [[] for _ in range(M)]
        
        for i, k5 in enumerate(k5s):
            idx = 0
            for t in itertools.combinations(k5, 3):
                t_idx = triple_index[t]
                M5[i, idx] = t_idx
                triple_to_k5s[t_idx].append(i)
                idx += 1
                
        return M4, M5, triple_to_k4s, triple_to_k5s, M

    def solve_for_n(n, time_limit):
        """
        Uses a repair-based local search (WalkSAT style) to find a valid coloring.
        """
        M4, M5, triple_to_k4s, triple_to_k5s, M = get_structures(n)
        N_k4 = len(M4)
        N_k5 = len(M5)
        
        # Initial random coloring
        colors = np.random.randint(0, 2, M, dtype=np.int8)
        
        # Initialize counts
        # red_count_k4[i] = number of red triples in K4 i
        # blue_count_k5[i] = number of blue triples in K5 i
        red_counts = np.sum(colors[M4] == 0, axis=1)
        blue_counts = np.sum(colors[M5] == 1, axis=1)
        
        # Identify violations
        # Violated K4: red_count == 4
        # Violated K5: blue_count == 10
        v4_mask = red_counts == 4
        v5_mask = blue_counts == 10
        
        num_v4 = np.sum(v4_mask)
        num_v5 = np.sum(v5_mask)
        total_violations = num_v4 + num_v5
        
        if total_violations == 0:
            return colors
        
        # Lists of violated indices for fast random selection
        # We will maintain these lists dynamically or just pick from full set and check?
        # Given N_k4 ~ 46k and N_k5 ~ 278k for n=34, picking randomly from full set
        # and checking if it's violated is O(1) amortized if violation rate is high.
        # But if violation rate drops, it becomes inefficient.
        # Let's maintain a list of violated indices.
        
        violated_k4s = np.where(v4_mask)[0]
        violated_k5s = np.where(v5_mask)[0]
        
        best_colors = colors.copy()
        best_violations = total_violations
        
        current_time = time.time()
        
        # Parameters for randomization
        # We can add a "random walk" component: with small prob, flip a random edge 
        # not involved in a violation to escape local minima.
        
        while time.time() < current_time + time_limit:
            if total_violations == 0:
                return colors
            
            # Pick a violation
            # Weighted by type? K5 violations are "harder" (10 edges) vs K4 (4 edges).
            # But flipping 1 edge in a K5 only reduces blue count by 1.
            # Let's just pick randomly from all violations.
            
            if total_violations > 0:
                # Randomly choose between K4 and K5 violation
                # Probability proportional to count? Or 50/50?
                # Let's try 50/50 if both exist, else the one that exists.
                if num_v4 > 0 and num_v5 > 0:
                    if random.random() < 0.5:
                        # Pick K4
                        k4_idx = violated_k4s[random.randint(0, len(violated_k4s)-1)]
                        # Pick a random triple from this K4
                        t_idx = M4[k4_idx, random.randint(0, 3)]
                    else:
                        # Pick K5
                        k5_idx = violated_k5s[random.randint(0, len(violated_k5s)-1)]
                        # Pick a random triple from this K5
                        t_idx = M5[k5_idx, random.randint(0, 9)]
                elif num_v4 > 0:
                    k4_idx = violated_k4s[random.randint(0, len(violated_k4s)-1)]
                    t_idx = M4[k4_idx, random.randint(0, 3)]
                else:
                    k5_idx = violated_k5s[random.randint(0, len(violated_k5s)-1)]
                    t_idx = M5[k5_idx, random.randint(0, 9)]
                
                # Flip the triple
                old_c = colors[t_idx]
                new_c = 1 - old_c
                colors[t_idx] = new_c
                
                # Update counts for affected K4s
                # If old_c was 0 (red), red_count decreases by 1.
                # If old_c was 1 (blue), red_count increases by 1 (since blue->red is false, wait.
                # red_count is count of 0s.
                # If 0 -> 1: red_count - 1
                # If 1 -> 0: red_count + 1
                delta_red = 1 if old_c == 1 else -1
                delta_blue = -delta_red # 1 if 0->1, -1 if 1->0. Wait.
                # blue_count is count of 1s.
                # If 0 -> 1: blue_count + 1
                # If 1 -> 0: blue_count - 1
                # So delta_blue = 1 if old_c == 0 else -1
                # And delta_red = -delta_blue
                
                for k4_i in triple_to_k4s[t_idx]:
                    old_rc = red_counts[k4_i]
                    new_rc = old_rc + delta_red
                    red_counts[k4_i] = new_rc
                    
                    old_v = 1 if old_rc == 4 else 0
                    new_v = 1 if new_rc == 4 else 0
                    if old_v != new_v:
                        if new_v:
                            # Became violated
                            violated_k4s = np.append(violated_k4s, k4_i)
                            num_v4 += 1
                        else:
                            # No longer violated
                            # Remove from list (inefficient if list is large, but N is small enough)
                            # Or mark as invalid and filter periodically?
                            # For simplicity and speed, let's just update counts and 
                            # occasionally rebuild the violated lists.
                            # Actually, removing from np array is O(N).
                            # Let's use a list of lists or just a mask?
                            # With N_k4 ~ 46k, np.append is okay-ish but not great.
                            # Let's use a Python list for violated indices and convert to np array?
                            # Or just keep the mask and pick randomly from full range, checking mask?
                            # If violation rate is low, this is bad.
                            # Let's stick to the list but handle removal carefully or rebuild periodically.
                            pass # Logic below handles rebuild
                    
                    total_violations += (new_v - old_v)
                    
                for k5_i in triple_to_k5s[t_idx]:
                    old_bc = blue_counts[k5_i]
                    new_bc = old_bc + delta_blue
                    blue_counts[k5_i] = new_bc
                    
                    old_v = 1 if old_bc == 10 else 0
                    new_v = 1 if new_bc == 10 else 0
                    if old_v != new_v:
                        if new_v:
                            violated_k5s = np.append(violated_k5s, k5_i)
                            num_v5 += 1
                        else:
                            pass
                            
                    total_violations += (new_v - old_v)
                
                # Periodic rebuild of violated lists to remove stale entries
                # Every 1000 steps or so?
                # Or if len(violated_k4s) > 2 * num_v4 (indicating many stale entries)
                # Let's just rebuild every 5000 steps to keep it simple and fast.
                # Actually, for n=34, the search space is huge. 
                # A simpler approach: 
                # Instead of maintaining dynamic lists, just pick a random index from [0, N_k4) 
                # and check if it's violated. 
                # If violation rate is ~10%, we need ~10 tries.
                # If violation rate is ~1%, we need ~100 tries.
                # Given we are trying to reach 0, the rate will drop.
                # So dynamic lists are better.
                
                # Optimization: If total_violations < best_violations, save best
                if total_violations < best_violations:
                    best_violations = total_violations
                    best_colors = colors.copy()
                    if total_violations == 0:
                        return colors
                
                # Simple restart/randomization if stuck?
                # If we don't improve for a long time, maybe re-randomize?
                # For now, let the repair run.
                
                # To avoid infinite loops in low-violation states, add small random noise?
                # No, repair is greedy. If stuck, it's a local min.
                # We can implement a "kick" phase: if no improvement in X steps, 
                # flip K random edges.
                
        return best_colors

    # Main loop
    # Try n=34 first, then 33, etc.
    # Allocate time wisely.
    
    target_ns = [34, 33, 32, 31, 30]
    
    # Give more time to higher n? Or equal time?
    # n=34 is hardest. Let's give it 60s.
    # n=33: 30s.
    # n=32: 20s.
    # Remaining: 10s for fallback.
    
    time_allocs = {
        34: 60.0,
        33: 30.0,
        32: 20.0,
        31: 15.0,
        30: 10.0
    }
    
    for n in target_ns:
        if time.time() > deadline - 5:
            break
            
        t_alloc = time_allocs.get(n, 5.0)
        if time.time() + t_alloc > deadline - 2:
            t_alloc = deadline - time.time() - 2.0
            if t_alloc < 1.0:
                continue
                
        # Run multiple restarts within the time limit
        # Each restart is a fresh random init + repair search
        # Let's run repair search for the whole allocated time, 
        # but with internal restarts if stuck?
        # The current solve_for_n runs for the whole time_limit.
        # It doesn't restart. 
        # Let's modify solve_for_n to accept a time limit and just run.
        # But if it gets stuck, it wastes time.
        # Let's add a simple restart mechanism inside solve_for_n?
        # Or just call it multiple times with smaller chunks?
        
        # Let's call it once with the full time. 
        # If it finds a solution, great.
        # If not, it returns the best found.
        
        sol = solve_for_n(n, t_alloc)
        
        # Verify solution
        M4, M5, _, _, _ = get_structures(n)
        # Check for red K4
        red_counts = np.sum(sol[M4] == 0, axis=1)
        if np.any(red_counts == 4):
            continue # Invalid
            
        # Check for blue K5
        blue_counts = np.sum(sol[M5] == 1, axis=1)
        if np.any(blue_counts == 10):
            continue # Invalid
            
        # Valid!
        return {'n': n, 'colors': sol.tolist()}
        
    # Fallback: return a known small valid solution
    # n=9 is valid with the modular coloring.
    n = 9
    return {'n': n, 'colors': [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in itertools.combinations(range(n), 3)]}

