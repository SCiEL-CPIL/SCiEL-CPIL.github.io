import time
from math import log, floor, ceil
import itertools
import numpy as np

def score_set(U):
    U = sorted(set(U))
    if 0 not in U:
        return -1
    n = len(U)
    if n < 2:
        return -1
        
    U_arr = np.array(U, dtype=np.int64)
    sums_arr = U_arr[:, None] + U_arr[None, :]
    unique_sums = np.unique(sums_arr)
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

def score_set_fast(U):
    """Faster scoring using set operations instead of numpy for small sets."""
    U = sorted(set(U))
    if 0 not in U:
        return -1
    n = len(U)
    if n < 2:
        return -1
    
    # Use Python sets for small n, numpy for larger
    if n <= 30:
        sums = set()
        diffs = set()
        for a in U:
            for b in U:
                sums.add(a + b)
                diffs.add(a - b)
    else:
        U_arr = np.array(U, dtype=np.int64)
        sums_arr = U_arr[:, None] + U_arr[None, :]
        sums = set(np.unique(sums_arr).tolist())
        diffs_arr = U_arr[:, None] - U_arr[None, :]
        diffs = set(np.unique(diffs_arr).tolist())
    
    ratio = len(diffs) / len(sums)
    if ratio <= 1:
        return -1
        
    max_u = U[-1]
    D = 2 * max_u + 1
    if D <= 1:
        return -1
        
    return 1 + log(ratio) / log(D)

def construct():
    deadline = time.time() + 145  # Safety margin for 150s limit
    
    best_score = -1
    best_set = [0, 1, 4, 9, 11]
    
    s = score_set(best_set)
    if s > best_score:
        best_score = s
    
    # Phase 1: Exhaustive search for small sets with moderate max_val
    max_val = 40
    
    for n in range(3, 6):
        if time.time() > deadline - 40:
            break
        for combo in itertools.combinations(range(1, max_val + 1), n - 1):
            if time.time() > deadline - 40:
                break
            U = (0,) + combo
            s = score_set_fast(U)
            if s > best_score:
                best_score = s
                best_set = list(U)
    
    # Phase 2: Power-of-2 and structured constructions
    if time.time() < deadline - 20:
        # Power of 2
        for m in range(3, 12):
            U = [0] + [2**k for k in range(1, m + 1)]
            s = score_set_fast(U)
            if s > best_score:
                best_score = s
                best_set = list(U)
        
        # Mixed: powers of 2 with offsets
        for m in range(3, 10):
            for offset in range(0, 4):
                U = [0] + [(2**k + offset) for k in range(1, m + 1)]
                U = sorted(set(U))
                if len(U) < 2:
                    continue
                s = score_set_fast(U)
                if s > best_score:
                    best_score = s
                    best_set = list(U)
        
        # Fibonacci-like
        for m in range(3, 15):
            fib = [0, 1]
            for _ in range(m - 1):
                fib.append(fib[-1] + fib[-2])
            U = fib[:m + 1]
            s = score_set_fast(U)
            if s > best_score:
                best_score = s
                best_set = list(U)
    
    # Phase 3: Geometric progression search with ternary refinement
    if time.time() < deadline - 15:
        for m in range(2, 12):
            if time.time() > deadline - 15:
                break
            
            b_low = 1.01
            b_high = 20.0
            
            # Coarse grid search
            best_b = None
            best_s_coarse = -1
            for i in range(60):
                if time.time() > deadline - 15:
                    break
                b = b_low + (b_high - b_low) * i / 59.0
                U = [0] + [floor(b**k) for k in range(1, m + 1)]
                U = sorted(set(U))
                if len(U) < 2:
                    continue
                s = score_set_fast(U)
                if s > best_s_coarse:
                    best_s_coarse = s
                    best_b = b
            
            if best_b is not None and time.time() < deadline - 15:
                # Ternary search around best_b
                lo = max(b_low, best_b * 0.8)
                hi = min(b_high, best_b * 1.2)
                for _ in range(30):
                    if time.time() > deadline - 15:
                        break
                    m1 = lo + (hi - lo) / 3
                    m2 = hi - (hi - lo) / 3
                    
                    U1 = [0] + [floor(m1**k) for k in range(1, m + 1)]
                    U1 = sorted(set(U1))
                    s1 = score_set_fast(U1) if len(U1) >= 2 else -1
                    
                    U2 = [0] + [floor(m2**k) for k in range(1, m + 1)]
                    U2 = sorted(set(U2))
                    s2 = score_set_fast(U2) if len(U2) >= 2 else -1
                    
                    if s1 < s2:
                        lo = m1
                    else:
                        hi = m2
                
                b_opt = (lo + hi) / 2
                U = [0] + [floor(b_opt**k) for k in range(1, m + 1)]
                U = sorted(set(U))
                if len(U) >= 2:
                    s = score_set_fast(U)
                    if s > best_score:
                        best_score = s
                        best_set = list(U)
    
    # Phase 4: Local perturbation search around best set - expanded
    if time.time() < deadline - 5:
        current_set = set(best_set)
        
        # Try adding elements (wider range)
        for new_elem in range(1, 300):
            if time.time() > deadline - 5:
                break
            if new_elem in current_set:
                continue
            new_set = current_set | {new_elem}
            s = score_set_fast(list(new_set))
            if s > best_score:
                best_score = s
                best_set = list(new_set)
                current_set = new_set
        
        # Try removing elements
        if len(best_set) > 2:
            for elem in list(best_set):
                if time.time() > deadline - 5:
                    break
                if elem == 0:
                    continue
                new_set = set(best_set) - {elem}
                if len(new_set) < 2:
                    continue
                s = score_set_fast(list(new_set))
                if s > best_score:
                    best_score = s
                    best_set = list(new_set)
        
        # Try shifting elements by larger amounts
        for _ in range(5):
            if time.time() > deadline - 5:
                break
            improved = False
            for i, elem in enumerate(list(best_set)):
                if time.time() > deadline - 5:
                    break
                if elem == 0:
                    continue
                for delta in range(-10, 11):
                    if delta == 0:
                        continue
                    new_elem = elem + delta
                    if new_elem < 0 or new_elem in best_set:
                        continue
                    new_set = list(best_set)
                    new_set[i] = new_elem
                    new_set = sorted(set(new_set))
                    if 0 not in new_set or len(new_set) < 2:
                        continue
                    s = score_set_fast(new_set)
                    if s > best_score:
                        best_score = s
                        best_set = new_set
                        improved = True
                        break
            if not improved:
                break
        
        # Scale the set (multiply by small integers)
        for scale in [2, 3, 5]:
            if time.time() > deadline - 5:
                break
            scaled = [0] + [x * scale for x in best_set if x > 0]
            s = score_set_fast(scaled)
            if s > best_score:
                best_score = s
                best_set = scaled
    
    return best_set
