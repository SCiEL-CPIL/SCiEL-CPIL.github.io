import time
import random
import copy
import numpy as np

def _square_nonzero_count(coeffs):
    """Count nonzero terms in the exact square of a polynomial given as a dict {exp: coeff}."""
    exps = list(coeffs.keys())
    square = {}
    for i in range(len(exps)):
        for j in range(i, len(exps)):
            deg = exps[i] + exps[j]
            if i == j:
                square[deg] = square.get(deg, 0) + coeffs[exps[i]] * coeffs[exps[j]]
            else:
                square[deg] = square.get(deg, 0) + 2 * coeffs[exps[i]] * coeffs[exps[j]]
    return sum(1 for v in square.values() if v != 0)

def _square_nonzero_count_fast(coeffs_dict):
    """Fast count using numpy convolution for dense-ish exponent sets."""
    exps = sorted(coeffs_dict.keys())
    if not exps:
        return 0
    max_exp = exps[-1]
    if max_exp > 80:
        return _square_nonzero_count(coeffs_dict)
    poly = np.zeros(2 * max_exp + 1, dtype=np.int64)
    for e, c in coeffs_dict.items():
        poly[e] = c
    sq = np.convolve(poly, poly)
    return int(np.sum(sq != 0))

def _square_coeffs(coeffs_dict):
    """Return the square as a dict {degree: coefficient}."""
    exps = sorted(coeffs_dict.keys())
    square = {}
    for i in range(len(exps)):
        for j in range(i, len(exps)):
            deg = exps[i] + exps[j]
            if i == j:
                square[deg] = square.get(deg, 0) + coeffs_dict[exps[i]] * coeffs_dict[exps[j]]
            else:
                square[deg] = square.get(deg, 0) + 2 * coeffs_dict[exps[i]] * coeffs_dict[exps[j]]
    return square

def _to_list(coeffs_dict, max_deg=40):
    """Convert dict to coefficient list in ascending degree."""
    lst = [0] * (max_deg + 1)
    for exp, coeff in coeffs_dict.items():
        if 0 <= exp <= max_deg:
            lst[exp] = coeff
    # Trim trailing zeros but keep at least 1
    while len(lst) > 1 and lst[-1] == 0:
        lst.pop()
    return lst

def _validate(lst):
    """Check if the coefficient list is valid."""
    if len(lst) > 41:
        return False
    if len(lst) == 0 or lst[-1] == 0:
        return False
    if any(abs(c) > 64 for c in lst):
        return False
    if sum(1 for c in lst if c != 0) != 12:
        return False
    return True

def construct():
    start_time = time.time()
    deadline = 85.0  # safety margin before 90s kill
    
    best_score = float('inf')
    best_result = None
    
    random.seed(42)
    
    # Diverse initial structures
    base_exponents_list = [
        list(range(12)),                    # 0..11
        list(range(0, 24, 2)),              # 0,2,4,...,22
        list(range(0, 36, 3)),              # 0,3,6,...,33
        list(range(1, 25, 2)),              # 1,3,5,...,23
        [0,1,3,4,6,7,9,10,12,13,15,16],     # clustered
        [0,2,4,6,8,10,12,14,16,18,20,22],   # even
        [0,1,2,3,4,5,6,7,8,9,10,11],        # sequential
        [0,1,2,4,5,6,8,9,10,12,13,14],      # paired clusters
        [0,2,3,5,6,8,9,11,12,14,15,17],     # zigzag
        [0,1,3,5,7,9,11,13,15,17,19,21],    # odd+even mix
    ]
    
    valid_base_exps = []
    for ex in base_exponents_list:
        if max(ex) <= 40 and len(ex) == 12:
            valid_base_exps.append(ex)
    
    for base_exps in valid_base_exps:
        if time.time() - start_time > deadline:
            break
        
        # Multiple random restarts per base structure
        for restart in range(3):
            if time.time() - start_time > deadline:
                break
            
            # Initialize with random coefficients
            coeffs_dict = {e: random.randint(1, 10) for e in base_exps}
            
            # Local search
            current_score = _square_nonzero_count_fast(coeffs_dict)
            
            # Phase 1: Broad exploration
            for iteration in range(100000):
                if time.time() - start_time > deadline:
                    break
                
                idx = random.randint(0, 11)
                exp = base_exps[idx]
                old_val = coeffs_dict[exp]
                
                if random.random() < 0.25:
                    new_val = old_val + random.randint(-2, 2)
                elif random.random() < 0.5:
                    new_val = random.randint(-20, 20)
                else:
                    new_val = random.randint(-64, 64)
                
                if new_val == 0 or abs(new_val) > 64:
                    continue
                
                coeffs_dict[exp] = new_val
                new_score = _square_nonzero_count_fast(coeffs_dict)
                
                if new_score < current_score:
                    current_score = new_score
                else:
                    coeffs_dict[exp] = old_val
                
                if current_score < best_score:
                    best_score = current_score
                    lst = [0] * 41
                    for e, c in coeffs_dict.items():
                        if 0 <= e <= 40:
                            lst[e] = c
                    while len(lst) > 1 and lst[-1] == 0:
                        lst.pop()
                    if _validate(lst):
                        best_result = lst[:]
            
            # Phase 2: Targeted refinement - try sign flips and small adjustments
            for _ in range(30000):
                if time.time() - start_time > deadline:
                    break
                
                idx = random.randint(0, 11)
                exp = base_exps[idx]
                old_val = coeffs_dict[exp]
                
                # Try negating
                new_val = -old_val
                if new_val == 0 or abs(new_val) > 64:
                    continue
                coeffs_dict[exp] = new_val
                new_score = _square_nonzero_count_fast(coeffs_dict)
                if new_score < current_score:
                    current_score = new_score
                else:
                    coeffs_dict[exp] = old_val
                
                if current_score < best_score:
                    best_score = current_score
                    lst = [0] * 41
                    for e, c in coeffs_dict.items():
                        if 0 <= e <= 40:
                            lst[e] = c
                    while len(lst) > 1 and lst[-1] == 0:
                        lst.pop()
                    if _validate(lst):
                        best_result = lst[:]
    
    # Fallback
    if best_result is None:
        return [1] * 12
    
    return best_result
