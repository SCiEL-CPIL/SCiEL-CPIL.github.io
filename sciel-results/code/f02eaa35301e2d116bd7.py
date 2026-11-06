import time
import random
import itertools
import numpy as np

MAX_N = 20
MAX_VAL = 10000
TIME_LIMIT = 85  # safety margin before 90s kill

def _objective(A):
    """Compute max(|A+A|, |A*A|) for a set A of distinct positive integers.
    Uses set-based counting for speed."""
    arr = sorted(A)
    n = len(arr)
    # Upper-triangular pairs including diagonal
    sums = set()
    prods = set()
    for i in range(n):
        ai = arr[i]
        for j in range(i, n):
            aj = arr[j]
            sums.add(ai + aj)
            prods.add(ai * aj)
    return max(len(sums), len(prods))

def _is_valid(A):
    if len(A) != 20:
        return False
    if len(set(A)) != 20:
        return False
    for x in A:
        if not isinstance(x, int) or x < 1 or x > 10000:
            return False
    return True

def _smooth_numbers(limit, count):
    """Generate the 'count' smallest smooth numbers <= limit."""
    smooths = set()
    i_max = int(limit.bit_length())
    j_max = int(limit.bit_length())
    k_max = int(limit.bit_length())
    
    for i in range(i_max + 1):
        pow2 = 1 << i
        if pow2 > limit: break
        for j in range(j_max + 1):
            pow3 = pow2 * (3 ** j)
            if pow3 > limit: break
            for k in range(k_max + 1):
                val = pow3 * (5 ** k)
                if val > limit: break
                smooths.add(val)
    
    sorted_smooths = sorted(smooths)
    return sorted_smooths[:count]

def _geometric_start(start, ratio_num, ratio_den):
    """Generate a geometric progression with rational ratio p/q."""
    A = []
    val = start
    for _ in range(20):
        if val > MAX_VAL:
            break
        if val in A:
            while val in A:
                val += 1
            if val > MAX_VAL:
                break
        A.append(val)
        val = (val * ratio_num + ratio_den // 2) // ratio_den
    if len(A) == 20 and _is_valid(A):
        return A
    return None

def _logarithmic_set(base, count=20):
    """Generate numbers approximately at logarithmic spacing."""
    A = []
    for i in range(count):
        val = int(base * (1.3 ** i))
        if val > MAX_VAL:
            break
        if val not in A and val >= 1:
            A.append(val)
    while len(A) < count:
        A.append(len(A) + 1)
    if len(A) == 20 and _is_valid(A):
        return A
    return None

def _construct_candidates():
    """Generate a list of candidate starting sets."""
    candidates = []
    
    # 1. Smooth numbers
    try:
        smooths = _smooth_numbers(10000, 20)
        if len(smooths) == 20:
            candidates.append(smooths)
    except:
        pass
        
    # 2. Geometric Progressions
    rational_ratios = [
        (6, 5), (7, 6), (8, 7), (5, 4), (4, 3),
        (9, 8), (11, 10), (3, 2), (5, 3), (7, 5),
        (10, 9), (12, 11), (13, 12), (14, 13), (15, 14),
    ]
    for ratio_num, ratio_den in rational_ratios:
        for start in [1, 2, 3, 4, 5]:
            A = _geometric_start(start, ratio_num, ratio_den)
            if A is not None:
                candidates.append(A)
                
    # 3. Arithmetic Progressions
    for start in [1, 2, 5, 10]:
        for step in [1, 2, 3, 5]:
            A = [start + i * step for i in range(20)]
            if max(A) <= 10000 and _is_valid(A):
                candidates.append(A)
    
    # 4. Logarithmic sets
    for base in [1, 2, 3, 5]:
        A = _logarithmic_set(base)
        if A is not None:
            candidates.append(A)
    
    # 5. Random with different ranges
    rng = random.Random(123)
    for _ in range(3):
        candidates.append(rng.sample(range(1, 10001), 20))
    for _ in range(3):
        candidates.append(rng.sample(range(1, 1001), 20))
    for _ in range(2):
        candidates.append(rng.sample(range(1, 200), 20))
    
    return candidates

def _local_search(A, time_limit, rng):
    """Local search with simulated annealing and paired swaps."""
    A = sorted(A)
    current_obj = _objective(A)
    best_A = list(A)
    best_obj = current_obj
    
    start_time = time.time()
    temperature = 30.0
    cooling_rate = 0.999
    
    while temperature > 0.01:
        if time.time() - start_time > time_limit:
            break
        
        # Choose move type
        move_type = rng.random()
        
        if move_type < 0.5:
            # Single element change
            idx = rng.randint(0, 19)
            old_val = A[idx]
            
            r = rng.random()
            if r < 0.4:
                new_val = old_val + rng.randint(-5, 5)
            elif r < 0.7:
                new_val = old_val + rng.randint(-50, 50)
            elif r < 0.9:
                new_val = rng.randint(1, 10000)
            else:
                # Try to find a value that creates collisions
                new_val = rng.randint(1, 10000)
            
            if new_val < 1 or new_val > 10000 or new_val in A:
                continue
            
            A[idx] = new_val
            new_obj = _objective(A)
            A[idx] = old_val
            
            delta = new_obj - current_obj
            if delta <= 0 or rng.random() < min(1.0, max(0.0, -delta / (temperature + 1e-10))):
                A[idx] = new_val
                current_obj = new_obj
                if new_obj < best_obj:
                    best_obj = new_obj
                    best_A = list(A)
        
        else:
            # Paired swap: change two elements simultaneously
            idx1, idx2 = rng.sample(range(20), 2)
            old1, old2 = A[idx1], A[idx2]
            
            new1 = old1 + rng.randint(-20, 20)
            new2 = old2 + rng.randint(-20, 20)
            
            if new1 < 1 or new1 > 10000 or new2 < 1 or new2 > 10000:
                continue
            if new1 == new2 or new1 in A or new2 in A:
                continue
            
            A[idx1] = new1
            A[idx2] = new2
            new_obj = _objective(A)
            A[idx1] = old1
            A[idx2] = old2
            
            delta = new_obj - current_obj
            if delta <= 0 or rng.random() < min(1.0, max(0.0, -delta / (temperature + 1e-10))):
                A[idx1] = new1
                A[idx2] = new2
                current_obj = new_obj
                if new_obj < best_obj:
                    best_obj = new_obj
                    best_A = list(A)
        
        temperature *= cooling_rate
    
    # Final polish: try all single-element replacements with nearby values
    if time.time() - start_time < time_limit - 1:
        A_polish = list(best_A)
        obj_polish = best_obj
        for i in range(20):
            if time.time() - start_time > time_limit - 1:
                break
            old = A_polish[i]
            for delta_val in range(-10, 11):
                if delta_val == 0:
                    continue
                nv = old + delta_val
                if nv < 1 or nv > 10000 or nv in A_polish:
                    continue
                A_polish[i] = nv
                o = _objective(A_polish)
                if o < obj_polish:
                    obj_polish = o
                    best_A = list(A_polish)
                    best_obj = obj_polish
                A_polish[i] = old
    
    return best_A, best_obj

def construct():
    start_time = time.time()
    time_budget = TIME_LIMIT - 5
    
    candidates = _construct_candidates()
    
    best_A = None
    best_obj = float('inf')
    
    rng = random.Random(42)
    
    # Phase 1: Quick evaluation of all candidates
    candidate_objs = []
    for A in candidates:
        if not _is_valid(A):
            continue
        obj = _objective(A)
        candidate_objs.append((obj, A))
    
    candidate_objs.sort(key=lambda x: x[0])
    
    n_cands = len(candidate_objs)
    if n_cands == 0:
        best_A = list(range(1, 21))
        best_obj = _objective(best_A)
    else:
        # Try top candidates with local search
        top_n = min(8, n_cands)
        for obj_init, A in candidate_objs[:top_n]:
            elapsed = time.time() - start_time
            remaining = time_budget - elapsed
            if remaining < 2:
                break
            refined_A, obj = _local_search(A, remaining, rng)
            if obj < best_obj:
                best_obj = obj
                best_A = refined_A
        
        # Try more candidates if time permits
        for obj_init, A in candidate_objs[top_n:]:
            elapsed = time.time() - start_time
            remaining = time_budget - elapsed
            if remaining < 3:
                break
            if obj_init >= best_obj:
                continue
            refined_A, obj = _local_search(A, min(remaining, 5.0), rng)
            if obj < best_obj:
                best_obj = obj
                best_A = refined_A
    
    if best_A is None:
        best_A = list(range(1, 21))
        best_obj = _objective(best_A)
        
    # Final refinement
    remaining = TIME_LIMIT - 5 - (time.time() - start_time)
    if remaining > 2:
        try:
            refined_A, refined_obj = _local_search(best_A, remaining, rng)
            if refined_obj < best_obj:
                best_A = refined_A
        except Exception:
            pass

    if not _is_valid(best_A):
        best_A = list(range(1, 21))
        
    return sorted(best_A)
