import time
import random
import itertools
import numpy as np

MAX_N = 20
MAX_VAL = 10000
TIME_LIMIT = 85  # safety margin before 90s kill

def _objective(A):
    """Compute max(|A+A|, |A*A|) for a set A of distinct positive integers.
    Uses NumPy vectorization for speed."""
    arr = np.array(sorted(A), dtype=np.int64)
    n = len(arr)
    # Upper-triangular indices including diagonal
    iu = np.triu_indices(n, k=0)
    sums = arr[iu[0]] + arr[iu[1]]
    prods = arr[iu[0]] * arr[iu[1]]
    n_sums = len(np.unique(sums))
    n_prods = len(np.unique(prods))
    return max(n_sums, n_prods)

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
    # Generate all numbers of form 2^i * 3^j * 5^k <= limit
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
    """Generate a geometric progression with rational ratio p/q.
    Uses exact integer arithmetic to avoid floating point issues."""
    A = []
    val = start
    for _ in range(20):
        if val > MAX_VAL:
            break
        if val in A:
            # Nudge to next unused value
            while val in A:
                val += 1
            if val > MAX_VAL:
                break
        A.append(val)
        # Next term: val * ratio_num / ratio_den (rounded)
        val = (val * ratio_num + ratio_den // 2) // ratio_den
    if len(A) == 20 and _is_valid(A):
        return A
    return None

def _construct_candidates():
    """Generate a list of candidate starting sets."""
    candidates = []
    
    # 1. Smooth numbers (powers of 2, 3, 5)
    try:
        smooths = _smooth_numbers(10000, 20)
        if len(smooths) == 20:
            candidates.append(smooths)
    except:
        pass
        
    # 2. Geometric Progressions with rational ratios
    # Ratios chosen to create multiplicative collisions
    rational_ratios = [
        (6, 5),   # 1.2
        (7, 6),   # ~1.167
        (8, 7),   # ~1.143
        (5, 4),   # 1.25
        (4, 3),   # ~1.333
        (9, 8),   # 1.125
        (11, 10), # 1.1
        (3, 2),   # 1.5
        (5, 3),   # ~1.667
        (7, 5),   # 1.4
    ]
    for ratio_num, ratio_den in rational_ratios:
        for start in [1, 2, 3, 4]:
            A = _geometric_start(start, ratio_num, ratio_den)
            if A is not None:
                candidates.append(A)
                
    # 3. Arithmetic Progressions
    for start in [1, 2, 5]:
        for step in [1, 2, 3]:
            A = [start + i * step for i in range(20)]
            if max(A) <= 10000 and _is_valid(A):
                candidates.append(A)
                
    # 4. Random
    candidates.append(random.sample(range(1, 10001), 20))
    candidates.append(random.sample(range(1, 1001), 20))
    
    return candidates

def _local_search(A, time_limit, rng):
    """Local search with simulated annealing, using NumPy-accelerated objective."""
    A = sorted(A)
    current_obj = _objective(A)
    best_A = list(A)
    best_obj = current_obj
    
    start_time = time.time()
    temperature = 50.0
    cooling_rate = 0.998
    
    while temperature > 0.05:
        if time.time() - start_time > time_limit:
            break
        
        idx = rng.randint(0, 19)
        old_val = A[idx]
        
        for _ in range(30):
            if time.time() - start_time > time_limit:
                break
            
            # Generate new value: mix of local and global moves
            r = rng.random()
            if r < 0.4:
                # Small local move
                new_val = old_val + rng.randint(-5, 5)
            elif r < 0.7:
                # Medium local move
                new_val = old_val + rng.randint(-50, 50)
            else:
                # Global move
                new_val = rng.randint(1, 10000)
            
            if new_val < 1 or new_val > 10000:
                continue
            if new_val in A:
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
                break
                
        temperature *= cooling_rate
    
    # Final polish: try all single-element replacements with nearby values
    if time.time() - start_time < time_limit - 1:
        A_polish = list(best_A)
        obj_polish = best_obj
        for i in range(20):
            if time.time() - start_time > time_limit - 1:
                break
            old = A_polish[i]
            for delta_val in range(-3, 4):
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
    
    # Phase 1: Quick evaluation of all candidates to find good starting points
    candidate_objs = []
    for A in candidates:
        if not _is_valid(A):
            continue
        obj = _objective(A)
        candidate_objs.append((obj, A))
    
    # Sort by objective, try best ones first
    candidate_objs.sort(key=lambda x: x[0])
    
    # Allocate time: try top candidates with more time each
    n_cands = len(candidate_objs)
    if n_cands == 0:
        best_A = list(range(1, 21))
        best_obj = _objective(best_A)
    else:
        # Try top 5 candidates with full local search
        top_n = min(5, n_cands)
        for obj_init, A in candidate_objs[:top_n]:
            elapsed = time.time() - start_time
            remaining = time_budget - elapsed
            if remaining < 2:
                break
            refined_A, obj = _local_search(A, remaining, rng)
            if obj < best_obj:
                best_obj = obj
                best_A = refined_A
        
        # If we have time, try a few more with shorter budgets
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
        
    # Final refinement with remaining time
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
