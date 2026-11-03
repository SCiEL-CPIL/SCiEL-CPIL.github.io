import time
from math import log, floor
import itertools
import random
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
    
    # Phase 2: Geometric progression search with ternary refinement
    if time.time() < deadline - 20:
        # Try geometric progressions U = {0} ∪ {floor(b^k) | k=1,...,m}
        for m in range(2, 12):  # Number of terms
            if time.time() > deadline:
                break
            
            # For each m, try different bases b
            b_low = 1.01
            b_high = 10.0
            
            # Coarse search first
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
                
                if s > best_score:
                    best_score = s
                    best_set = list(U)
            
            # Fine search around best b found using ternary search
            if best_b is not None and time.time() < deadline - 5:
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
        current_set = set(best_set)
        
        # Try adding elements
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
    
    # Phase 4: Multi-start random local search (inspired by inspiration 2)
    if time.time() < deadline - 5:
        random.seed(42)
        
        # Known good sets as seeds
        candidates = [
            [0, 1, 3],
            [0, 1, 4, 9, 11],
            [0, 1, 3, 7, 12],
            [0, 1, 4, 10, 12, 17],
            [0, 1, 4, 10, 18, 23, 31],
            [0, 1, 4, 10, 12, 17, 25],
            [0, 1, 2, 4, 8],
            [0, 1, 2, 5, 11],
            [0, 1, 3, 7, 14],
            [0, 1, 3, 8, 13],
            [0, 2, 3, 7, 12],
            [0, 1, 4, 11, 13],
            [0, 1, 4, 9, 13],
        ]
        
        for c in candidates:
            s = score_set(c)
            if s > best_score:
                best_score = s
                best_set = c
        
        # Random local search from multiple starting points
        while time.time() < deadline - 3:
            # Start from a random valid set
            n = random.randint(4, 8)
            max_val = random.randint(15, 60)
            
            other_elements = random.sample(range(1, max_val), n - 1)
            current = sorted([0] + other_elements)
            
            current_score = score_set(current)
            if current_score < 0:
                continue
            
            # Local search: try moving one element
            improved = True
            while improved and time.time() < deadline - 3:
                improved = False
                for i in range(len(current)):
                    if i == 0:
                        continue  # Keep 0
                    old_val = current[i]
                    # Try changing to a nearby value
                    for delta in [-3, -2, -1, 1, 2, 3]:
                        new_val = old_val + delta
                        if new_val <= 0 or new_val >= max_val + 10:
                            continue
                        new_set = sorted(set([0] + [current[j] if j != i else new_val for j in range(len(current))]))
                        if len(new_set) != len(current):
                            continue
                        new_score = score_set(new_set)
                        if new_score > current_score:
                            current = new_set
                            current_score = new_score
                            improved = True
                            if current_score > best_score:
                                best_score = current_score
                                best_set = current[:]
                            break
                    if improved:
                        break
                
                # Also try adding/removing elements
                if time.time() < deadline - 3:
                    # Try adding an element
                    if len(current) < 10:
                        max_cur = max(current)
                        for new_val in range(max_cur + 1, max_cur + 20):
                            if new_val not in current:
                                new_set = sorted(current + [new_val])
                                new_score = score_set(new_set)
                                if new_score > current_score:
                                    current = new_set
                                    current_score = new_score
                                    if current_score > best_score:
                                        best_score = current_score
                                        best_set = current[:]
                                    break
                    
                    # Try removing an element (not 0)
                    if len(current) > 4:
                        for i in range(1, len(current)):
                            new_set = [current[j] for j in range(len(current)) if j != i]
                            new_score = score_set(new_set)
                            if new_score > current_score:
                                current = new_set
                                current_score = new_score
                                if current_score > best_score:
                                    best_score = current_score
                                    best_set = current[:]
                                break
    
    return best_set
