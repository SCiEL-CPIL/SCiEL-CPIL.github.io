import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    def try_n(n, time_budget):
        """Try to find a valid coloring for n vertices within time_budget seconds."""
        if n < 4:
            # Trivially valid
            triples = list(combinations(range(n), 3))
            return {'n': n, 'colors': [0] * len(triples)}, True
        
        triples = list(combinations(range(n), 3))
        num_triples = len(triples)
        triple_to_idx = {t: i for i, t in enumerate(triples)}
        
        # Precompute K4 and K5 structures
        k4_list = list(combinations(range(n), 4))
        num_k4 = len(k4_list)
        k5_list = list(combinations(range(n), 5))
        num_k5 = len(k5_list)
        
        k4_data = []
        triple_to_k4s = [[] for _ in range(num_triples)]
        for i, k4 in enumerate(k4_list):
            ts = []
            for j in range(4):
                t = tuple(sorted(k4[:j] + k4[j+1:]))
                idx = triple_to_idx[t]
                ts.append(idx)
                triple_to_k4s[idx].append(i)
            k4_data.append(ts)
            
        k5_data = []
        triple_to_k5s = [[] for _ in range(num_triples)]
        for i, k5 in enumerate(k5_list):
            ts = []
            for combo in combinations(range(5), 3):
                t = tuple(sorted([k5[j] for j in combo]))
                idx = triple_to_idx[t]
                ts.append(idx)
                triple_to_k5s[idx].append(i)
            k5_data.append(ts)
        
        k4_arr = np.array(k4_data, dtype=np.int32)
        k5_arr = np.array(k5_data, dtype=np.int32)
        
        # Strategy: Use a structured initialization based on Z_n arithmetic if possible,
        # then refine with Simulated Annealing.
        
        # Heuristic initialization:
        # Color 0 (Red) if (a+b+c) % n < n//2 ? Or some other pattern.
        # A common pattern for Ramsey colorings is based on quadratic residues or similar.
        # Let's try: color 0 if (a*b + b*c + c*a) % 2 == 0 ?
        # Or simply: color 0 if (a+b+c) % 3 == 0.
        
        # Let's try a few different initializations and pick the best one quickly,
        # or just use random with SA.
        # Given the time budget, we will run multiple SA restarts.
        
        best_violations = float('inf')
        best_colors = None
        
        end_time = time.time() + time_budget
        
        while time.time() < end_time:
            # Random initialization
            colors = np.random.randint(0, 2, size=num_triples).astype(np.int8)
            
            # Calculate initial violations
            counts_k4 = np.sum(colors[k4_arr], axis=1)
            counts_k5 = np.sum(colors[k5_arr], axis=1)
            violations = np.sum(counts_k4 == 4) + np.sum(counts_k5 == 10)
            
            if violations == 0:
                return {'n': n, 'colors': colors.tolist()}, True
                
            if violations < best_violations:
                best_violations = violations
                best_colors = colors.copy()
                
            # Simulated Annealing
            T = 10.0
            T_min = 0.01
            cooling = 0.9995
            no_improve = 0
            max_no_improve = 20000
            
            # Time limit for this SA run
            sa_end = min(end_time, time.time() + 5.0)
            
            while time.time() < sa_end and no_improve < max_no_improve:
                idx = random.randint(0, num_triples - 1)
                old_val = colors[idx]
                new_val = 1 - old_val
                delta = new_val - old_val
                
                # Update counts and violations
                # We need to update counts_k4 and counts_k5 for the affected K4s and K5s
                # and track the change in total violations.
                
                old_viol = 0
                new_viol = 0
                
                # Update K4s
                for k4_idx in triple_to_k4s[idx]:
                    old_count = counts_k4[k4_idx]
                    new_count = old_count + delta
                    counts_k4[k4_idx] = new_count
                    old_viol += 1 if old_count == 4 else 0
                    new_viol += 1 if new_count == 4 else 0
                    
                # Update K5s
                for k5_idx in triple_to_k5s[idx]:
                    old_count = counts_k5[k5_idx]
                    new_count = old_count + delta
                    counts_k5[k5_idx] = new_count
                    old_viol += 1 if old_count == 10 else 0
                    new_viol += 1 if new_count == 10 else 0
                
                delta_v = new_viol - old_viol
                
                if delta_v <= 0:
                    # Accept
                    colors[idx] = new_val
                    violations += delta_v
                    no_improve = 0
                    if violations < best_violations:
                        best_violations = violations
                        best_colors = colors.copy()
                        if best_violations == 0:
                            return {'n': n, 'colors': colors.tolist()}, True
                else:
                    # Accept with probability exp(-delta_v / T)
                    if T > T_min:
                        prob = np.exp(-delta_v / T)
                        if random.random() < prob:
                            colors[idx] = new_val
                            violations += delta_v
                            no_improve += 1
                        else:
                            no_improve += 1
                    else:
                        no_improve += 1
                        
                T *= cooling
                
            # End of SA run
            
        # If we didn't find a solution in the allocated time
        if best_violations == 0:
            return {'n': n, 'colors': best_colors.tolist()}, True
        else:
            # Return the best attempt, but mark as invalid?
            # The caller needs to know if it's valid.
            # We return None or a flag.
            return None, False

    # Try n=34 first
    # We allocate most of the time to n=34.
    # If it fails, try n=33, etc.
    
    # Estimate time for n=34.
    # num_triples = C(34,3) = 5984
    # num_k4 = C(34,4) = 46376
    # num_k5 = C(34,5) = 278256
    
    # We have ~145s total.
    # Let's try n=34 with 130s.
    res, valid = try_n(34, 130.0)
    if valid:
        return res
        
    # If 34 fails, try 33
    if time.time() < deadline - 10:
        res, valid = try_n(33, 10.0)
        if valid:
            return res
            
    # If 33 fails, try smaller n as fallback
    # n=20 is likely easier
    if time.time() < deadline - 5:
        res, valid = try_n(20, 5.0)
        if valid:
            return res
            
    # Final fallback: small n
    n_fallback = 5
    triples = list(combinations(range(n_fallback), 3))
    # For n=5, we need no Red K4 (impossible as n<4? No, K4 needs 4 vertices, so n=5 has K4s)
    # Wait, K4 needs 4 vertices. If n=5, there are C(5,4)=5 K4s.
    # We must avoid Red K4.
    # We must avoid Blue K5. There is only 1 K5 (all 5 vertices).
    # So we need to color the 10 triples of K5 such that not all are Blue, and no 4-subset is all Red.
    # A simple valid coloring for n=5:
    # Make one triple Blue, rest Red?
    # If (0,1,2) is Blue, others Red.
    # Check K4s:
    # K4 {0,1,2,3}: Triples (0,1,2)[B], (0,1,3)[R], (0,2,3)[R], (1,2,3)[R]. Not all Red. OK.
    # K4 {0,1,2,4}: Similar. OK.
    # K4 {0,1,3,4}: Triples (0,1,3)[R], (0,1,4)[R], (0,3,4)[R], (1,3,4)[R]. ALL RED! VIOLATION.
    # So that doesn't work.
    
    # Let's use a known valid small construction or just random search for small n which is fast.
    # Or simply return n=4?
    # For n=4, there is 1 K4 and 0 K5s.
    # We just need the 4 triples not to be all Red.
    # So color one Blue, rest Red.
    # Colors: [1, 0, 0, 0] (index 0 is (0,1,2), etc.)
    # (0,1,2)=1, (0,1,3)=0, (0,2,3)=0, (1,2,3)=0.
    # K4 {0,1,2,3} has 1 Blue, 3 Red. Not all Red. Valid.
    
    n_fallback = 4
    triples = list(combinations(range(n_fallback), 3))
    colors = [1, 0, 0, 0]
    return {'n': n_fallback, 'colors': colors}

# EVOLVE-BLOCK-END
