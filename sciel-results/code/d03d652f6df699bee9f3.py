import time
from math import log, floor
import itertools
import numpy as np

def score_set(U):
    U = sorted(set(U))
    if 0 not in U:
        return -1
    n = len(U)
    if n < 2:
        return -1
        
    # Use NumPy for faster computation
    U_arr = np.array(U, dtype=np.int64)
    
    # Compute all pairwise sums
    sums_arr = U_arr[:, None] + U_arr[None, :]
    unique_sums = np.unique(sums_arr)
    
    # Compute all pairwise differences
    diffs_arr = U_arr[:, None] - U_arr[None, :]
    unique_diffs = np.unique(diffs_arr)
    
    ratio = len(unique_diffs) / len(unique_sums)
    if ratio <= 1:
        return -1
        
    max_u = U_arr[-1]
    D = 2 * max_u + 1
    if D <= 1:
        return -1
        
    return 1 + log(ratio) / log(D)

def construct():
    deadline = time.time() + 110  # Safety margin
    
    best_score = -1
    best_set = [0, 1, 4, 9, 11]
    
    # Update best if initial is good
    s = score_set(best_set)
    if s > best_score:
        best_score = s
    
    # Phase 1: Exhaustive search for small sets with moderate max_val
    max_val = 30
    
    for n in range(3, 6): # Sizes 3 to 5 (faster)
        if time.time() > deadline:
            break
            
        for combo in itertools.combinations(range(1, max_val + 1), n - 1):
            if time.time() > deadline:
                break
                
            U = (0,) + combo
            s = score_set(U)
            if s > best_score:
                best_score = s
                best_set = list(U)
    
    # Phase 2: Interval-based construction with exponentially growing gaps
    # Model: U = {0} ∪ {a_i} where a_i are centers of intervals
    # Intervals have exponentially growing gaps: gap_i = gap_0 * alpha^i
    # We tune: number of intervals (k), initial gap (g0), growth ratio (alpha)
    
    if time.time() < deadline - 30:
        # Search over number of intervals k and parameters g0, alpha
        for k in range(2, 15):  # Number of intervals/points
            if time.time() > deadline:
                break
            
            # For each k, optimize g0 and alpha
            # g0: initial gap (first point after 0)
            # alpha: growth ratio of gaps
            
            # Coarse grid search
            g0_values = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144]
            alpha_values = [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.2, 2.5, 3.0]
            
            best_g0 = 1
            best_alpha = 1.5
            best_s_coarse = -1
            
            for g0 in g0_values:
                if time.time() > deadline:
                    break
                for alpha in alpha_values:
                    if time.time() > deadline:
                        break
                    
                    # Generate set: 0, then points at cumulative gaps
                    # Point i is at sum of gaps 0 to i-1
                    # gap_i = g0 * alpha^i
                    U = [0]
                    current_pos = 0
                    for i in range(k):
                        gap = g0 * (alpha ** i)
                        current_pos += int(round(gap))
                        U.append(current_pos)
                    
                    U = sorted(set(U))
                    if len(U) < 2:
                        continue
                    
                    s = score_set(U)
                    if s > best_s_coarse:
                        best_s_coarse = s
                        best_g0 = g0
                        best_alpha = alpha
            
            # Fine search around best coarse solution
            if best_s_coarse > -1 and time.time() < deadline - 10:
                # Ternary search on alpha for fixed best_g0
                alpha_lo = max(1.01, best_alpha * 0.8)
                alpha_hi = min(5.0, best_alpha * 1.2)
                
                for _ in range(25):
                    if time.time() > deadline:
                        break
                    m1 = alpha_lo + (alpha_hi - alpha_lo) / 3
                    m2 = alpha_hi - (alpha_hi - alpha_lo) / 3
                    
                    # Evaluate at m1
                    U1 = [0]
                    current_pos1 = 0
                    for i in range(k):
                        gap = best_g0 * (m1 ** i)
                        current_pos1 += int(round(gap))
                        U1.append(current_pos1)
                    U1 = sorted(set(U1))
                    s1 = score_set(U1) if len(U1) >= 2 else -1
                    
                    # Evaluate at m2
                    U2 = [0]
                    current_pos2 = 0
                    for i in range(k):
                        gap = best_g0 * (m2 ** i)
                        current_pos2 += int(round(gap))
                        U2.append(current_pos2)
                    U2 = sorted(set(U2))
                    s2 = score_set(U2) if len(U2) >= 2 else -1
                    
                    if s1 < s2:
                        alpha_lo = m1
                    else:
                        alpha_hi = m2
                
                alpha_opt = (alpha_lo + alpha_hi) / 2
                
                # Now ternary search on g0 for fixed alpha_opt
                g0_lo = max(1, best_g0 * 0.5)
                g0_hi = min(200, best_g0 * 2.0)
                
                for _ in range(25):
                    if time.time() > deadline:
                        break
                    m1 = g0_lo + (g0_hi - g0_lo) / 3
                    m2 = g0_hi - (g0_hi - g0_lo) / 3
                    
                    # Evaluate at m1
                    U1 = [0]
                    current_pos1 = 0
                    for i in range(k):
                        gap = m1 * (alpha_opt ** i)
                        current_pos1 += int(round(gap))
                        U1.append(current_pos1)
                    U1 = sorted(set(U1))
                    s1 = score_set(U1) if len(U1) >= 2 else -1
                    
                    # Evaluate at m2
                    U2 = [0]
                    current_pos2 = 0
                    for i in range(k):
                        gap = m2 * (alpha_opt ** i)
                        current_pos2 += int(round(gap))
                        U2.append(current_pos2)
                    U2 = sorted(set(U2))
                    s2 = score_set(U2) if len(U2) >= 2 else -1
                    
                    if s1 < s2:
                        g0_lo = m1
                    else:
                        g0_hi = m2
                
                g0_opt = (g0_lo + g0_hi) / 2
                
                # Final evaluation
                U_final = [0]
                current_pos = 0
                for i in range(k):
                    gap = g0_opt * (alpha_opt ** i)
                    current_pos += int(round(gap))
                    U_final.append(current_pos)
                U_final = sorted(set(U_final))
                
                if len(U_final) >= 2:
                    s = score_set(U_final)
                    if s > best_score:
                        best_score = s
                        best_set = list(U_final)
    
    # Phase 3: Local search to refine the best set found
    if time.time() < deadline - 10:
        current_set = set(best_set)
        
        # Try adding elements
        max_current = max(best_set)
        for new_elem in range(1, max_current + 50):
            if time.time() > deadline:
                break
            if new_elem in current_set:
                continue
            new_set = current_set | {new_elem}
            s = score_set(list(new_set))
            if s > best_score:
                best_score = s
                best_set = list(new_set)
                current_set = new_set
                max_current = max(best_set)
        
        # Try removing elements (keep at least 2 elements)
        if len(best_set) > 2:
            for elem in list(best_set):
                if time.time() > deadline:
                    break
                if elem == 0:
                    continue
                new_set = set(best_set) - {elem}
                if len(new_set) < 2:
                    continue
                s = score_set(list(new_set))
                if s > best_score:
                    best_score = s
                    best_set = list(new_set)
    
    return best_set
