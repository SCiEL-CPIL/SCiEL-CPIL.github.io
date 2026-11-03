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
    
    # Compute all pairwise sums (i <= j to avoid duplicates, but we want unique sums)
    # Actually, we need all unique sums from i,j where 0<=i<=j<n
    # But for the sumset, we just need all unique values of u_i + u_j
    # Using outer sum and taking unique
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
    max_val = 40
    
    for n in range(3, 7): # Sizes 3 to 6
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
    
    # Phase 2: Geometric progression search
    if time.time() < deadline - 20:
        # Try geometric progressions U = {floor(b^k) | k=0,...,m}
        for m in range(2, 12):  # Number of terms
            if time.time() > deadline:
                break
            
            # For each m, try different bases b
            # b should be > 1 for increasing sequence
            # Try b from 1.1 to 10
            b_low = 1.01
            b_high = 10.0
            
            # Coarse search first
            for _ in range(50):
                if time.time() > deadline:
                    break
                b = b_low + (b_high - b_low) * _ / 49.0
                U = [floor(b**k) for k in range(m + 1)]
                # Ensure 0 is in U (it should be since b^0 = 1, but we need 0)
                # Actually, we want nonnegative integers containing 0
                # Let's use U = {0} ∪ {floor(b^k) | k=1,...,m}
                U = [0] + [floor(b**k) for k in range(1, m + 1)]
                U = sorted(set(U))
                if len(U) < 2:
                    continue
                s = score_set(U)
                if s > best_score:
                    best_score = s
                    best_set = list(U)
            
            # Fine search around best b found
            # We'll do a ternary search on b for this m
            if time.time() < deadline - 5:
                # Find best b in coarse grid
                best_b = None
                best_s_coarse = -1
                for _ in range(100):
                    if time.time() > deadline:
                        break
                    b = b_low + (b_high - b_low) * _ / 99.0
                    U = [0] + [floor(b**k) for k in range(1, m + 1)]
                    U = sorted(set(U))
                    if len(U) < 2:
                        continue
                    s = score_set(U)
                    if s > best_s_coarse:
                        best_s_coarse = s
                        best_b = b
                
                if best_b is not None and time.time() < deadline - 5:
                    # Ternary search around best_b
                    lo = max(b_low, best_b * 0.8)
                    hi = min(b_high, best_b * 1.2)
                    for _ in range(30):
                        if time.time() > deadline:
                            break
                        m1 = lo + (hi - lo) / 3
                        m2 = hi - (hi - lo) / 3
                        
                        U1 = [0] + [floor(m1**k) for k in range(1, m + 1)]
                        U1 = sorted(set(U1))
                        s1 = score_set(U1) if len(U1) >= 2 else -1
                        
                        U2 = [0] + [floor(m2**k) for k in range(1, m + 1)]
                        U2 = sorted(set(U2))
                        s2 = score_set(U2) if len(U2) >= 2 else -1
                        
                        if s1 < s2:
                            lo = m1
                        else:
                            hi = m2
                    
                    # Evaluate at optimal point
                    b_opt = (lo + hi) / 2
                    U = [0] + [floor(b_opt**k) for k in range(1, m + 1)]
                    U = sorted(set(U))
                    if len(U) >= 2:
                        s = score_set(U)
                        if s > best_score:
                            best_score = s
                            best_set = list(U)
    
    # Phase 3: Local search to refine the best set found
    if time.time() < deadline - 10:
        # Try adding elements
        current_set = set(best_set)
        max_current = max(best_set)
        
        # Try adding elements up to 200
        for new_elem in range(1, 200):
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
                    continue  # Must keep 0
                new_set = set(best_set) - {elem}
                if len(new_set) < 2:
                    continue
                s = score_set(list(new_set))
                if s > best_score:
                    best_score = s
                    best_set = list(new_set)
    
    return best_set
