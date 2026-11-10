import numpy as np
import time
import itertools
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin

    def get_structures(n):
        triples = list(itertools.combinations(range(n), 3))
        triple_index = {t: i for i, t in enumerate(triples)}
        M = len(triples)
        
        k4s = list(itertools.combinations(range(n), 4))
        N_k4 = len(k4s)
        M4 = np.zeros((N_k4, 4), dtype=np.int32)
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
            
        k5s = list(itertools.combinations(range(n), 5))
        N_k5 = len(k5s)
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
        M4, M5, triple_to_k4s, triple_to_k5s, M = get_structures(n)
        N_k4 = len(M4)
        N_k5 = len(M5)
        
        # Random init
        colors = np.random.randint(0, 2, M, dtype=np.int8)
        
        # Init counts
        red_counts = np.sum(colors[M4] == 0, axis=1).astype(np.int32)
        blue_counts = np.sum(colors[M5] == 1, axis=1).astype(np.int32)
        
        # Violation masks
        v4_mask = red_counts == 4
        v5_mask = blue_counts == 10
        
        num_v4 = np.sum(v4_mask)
        num_v5 = np.sum(v5_mask)
        total_violations = num_v4 + num_v5
        
        if total_violations == 0:
            return colors
        
        # Lists of violated indices
        violated_k4s = list(np.where(v4_mask)[0])
        violated_k5s = list(np.where(v5_mask)[0])
        
        best_colors = colors.copy()
        best_violations = total_violations
        
        current_time = time.time()
        no_improve_count = 0
        
        while time.time() < current_time + time_limit:
            if total_violations == 0:
                return colors
            
            # Pick a violation
            if total_violations > 0:
                if num_v4 > 0 and num_v5 > 0:
                    if random.random() < 0.5:
                        if not violated_k4s:
                            violated_k4s = list(np.where(v4_mask)[0])
                            num_v4 = len(violated_k4s)
                        if violated_k4s:
                            k4_idx = violated_k4s[random.randint(0, len(violated_k4s)-1)]
                            t_idx = M4[k4_idx, random.randint(0, 3)]
                        else:
                            if not violated_k5s:
                                violated_k5s = list(np.where(v5_mask)[0])
                                num_v5 = len(violated_k5s)
                            k5_idx = violated_k5s[random.randint(0, len(violated_k5s)-1)]
                            t_idx = M5[k5_idx, random.randint(0, 9)]
                    else:
                        if not violated_k5s:
                            violated_k5s = list(np.where(v5_mask)[0])
                            num_v5 = len(violated_k5s)
                        if violated_k5s:
                            k5_idx = violated_k5s[random.randint(0, len(violated_k5s)-1)]
                            t_idx = M5[k5_idx, random.randint(0, 9)]
                        else:
                            if not violated_k4s:
                                violated_k4s = list(np.where(v4_mask)[0])
                                num_v4 = len(violated_k4s)
                            k4_idx = violated_k4s[random.randint(0, len(violated_k4s)-1)]
                            t_idx = M4[k4_idx, random.randint(0, 3)]
                elif num_v4 > 0:
                    if not violated_k4s:
                        violated_k4s = list(np.where(v4_mask)[0])
                        num_v4 = len(violated_k4s)
                    k4_idx = violated_k4s[random.randint(0, len(violated_k4s)-1)]
                    t_idx = M4[k4_idx, random.randint(0, 3)]
                else:
                    if not violated_k5s:
                        violated_k5s = list(np.where(v5_mask)[0])
                        num_v5 = len(violated_k5s)
                    k5_idx = violated_k5s[random.randint(0, len(violated_k5s)-1)]
                    t_idx = M5[k5_idx, random.randint(0, 9)]
                
                # Flip
                old_c = colors[t_idx]
                new_c = 1 - old_c
                colors[t_idx] = new_c
                
                delta_red = 1 if old_c == 1 else -1
                delta_blue = -delta_red
                
                # Update K4s
                for k4_i in triple_to_k4s[t_idx]:
                    old_rc = red_counts[k4_i]
                    new_rc = old_rc + delta_red
                    red_counts[k4_i] = new_rc
                    
                    old_v = (old_rc == 4)
                    new_v = (new_rc == 4)
                    if old_v != new_v:
                        if new_v:
                            violated_k4s.append(k4_i)
                            num_v4 += 1
                        else:
                            pass
                        total_violations += 1 if new_v else -1
                        
                # Update K5s
                for k5_i in triple_to_k5s[t_idx]:
                    old_bc = blue_counts[k5_i]
                    new_bc = old_bc + delta_blue
                    blue_counts[k5_i] = new_bc
                    
                    old_v = (old_bc == 10)
                    new_v = (new_bc == 10)
                    if old_v != new_v:
                        if new_v:
                            violated_k5s.append(k5_i)
                            num_v5 += 1
                        else:
                            pass
                        total_violations += 1 if new_v else -1
                        
                # Check improvement
                if total_violations < best_violations:
                    best_violations = total_violations
                    best_colors = colors.copy()
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                    
                # Kick if stuck
                if no_improve_count > 5000:
                    for _ in range(10):
                        t_kick = random.randint(0, M-1)
                        old_ck = colors[t_kick]
                        new_ck = 1 - old_ck
                        colors[t_kick] = new_ck
                        
                        delta_reak = 1 if old_ck == 1 else -1
                        delta_bluek = -delta_reak
                        
                        for k4_i in triple_to_k4s[t_kick]:
                            old_rc = red_counts[k4_i]
                            new_rc = old_rc + delta_reak
                            red_counts[k4_i] = new_rc
                            old_v = (old_rc == 4)
                            new_v = (new_rc == 4)
                            if old_v != new_v:
                                if new_v:
                                    violated_k4s.append(k4_i)
                                    num_v4 += 1
                                else:
                                    pass
                                total_violations += 1 if new_v else -1
                                
                        for k5_i in triple_to_k5s[t_kick]:
                            old_bc = blue_counts[k5_i]
                            new_bc = old_bc + delta_bluek
                            blue_counts[k5_i] = new_bc
                            old_v = (old_bc == 10)
                            new_v = (new_bc == 10)
                            if old_v != new_v:
                                if new_v:
                                    violated_k5s.append(k5_i)
                                    num_v5 += 1
                                else:
                                    pass
                                total_violations += 1 if new_v else -1
                    no_improve_count = 0
                    violated_k4s = [i for i in set(violated_k4s) if red_counts[i] == 4]
                    num_v4 = len(violated_k4s)
                    violated_k5s = [i for i in set(violated_k5s) if blue_counts[i] == 10]
                    num_v5 = len(violated_k5s)
                    total_violations = num_v4 + num_v5

        return best_colors

    # Try n=34, then 33, etc.
    for n in [34, 33, 32, 31, 30]:
        if time.time() > deadline - 10:
            break
            
        t_alloc = deadline - time.time() - 5
        if t_alloc < 1.0:
            continue
            
        sol = solve_for_n(n, t_alloc)
        
        # Verify
        M4, M5, _, _, _ = get_structures(n)
        red_counts = np.sum(sol[M4] == 0, axis=1)
        if np.any(red_counts == 4):
            continue
            
        blue_counts = np.sum(sol[M5] == 1, axis=1)
        if np.any(blue_counts == 10):
            continue
            
        return {'n': n, 'colors': sol.tolist()}
        
    # Fallback n=5
    n = 5
    triples = list(itertools.combinations(range(n), 3))
    colors = [1] * len(triples)
    colors[0] = 0
    return {'n': n, 'colors': colors}
