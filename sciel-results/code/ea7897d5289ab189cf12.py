import time
import random
import numpy as np

MAX_N = 20
MAX_VAL = 10000
TIME_LIMIT = 85  # safety margin before 90s kill


def _objective(A):
    """Compute max(|A+A|, |A*A|) for a set A of distinct positive integers."""
    A_arr = np.array(sorted(A), dtype=np.int64)
    n = len(A_arr)
    i_idx, j_idx = np.triu_indices(n)
    sums = A_arr[i_idx] + A_arr[j_idx]
    products = A_arr[i_idx] * A_arr[j_idx]
    return max(len(np.unique(sums)), len(np.unique(products)))


def _is_valid(A):
    if len(A) != 20:
        return False
    if len(set(A)) != 20:
        return False
    for x in A:
        if not isinstance(x, int) or x < 1 or x > 10000:
            return False
    return True


def _two_gp_candidate(base1, base2, n1, n2, start1=1, start2=1):
    """Generate a candidate set as union of two geometric progressions."""
    A = []
    # First GP: start1 * base1^i for i in range(n1)
    val = start1
    for _ in range(n1):
        if val in A or val > MAX_VAL:
            continue
        A.append(val)
        val *= base1
    # Second GP: start2 * base2^j for j in range(n2)
    val = start2
    for _ in range(n2):
        if val in A or val > MAX_VAL:
            continue
        A.append(val)
        val *= base2
    return A


def _construct_candidates():
    """Generate candidate starting sets with strong multiplicative structure."""
    candidates = []
    
    # Two-GP combinations: powers of 2 and powers of 3
    for n1 in range(10, 15):
        n2 = 20 - n1
        # Pure powers: 2^0..2^(n1-1) and 3^0..3^(n2-1)
        A = _two_gp_candidate(2, 3, n1, n2, 1, 1)
        if len(A) == 20 and _is_valid(A):
            candidates.append(A)
    
    # Offset GPs to avoid overlap
    for n1 in [10, 11, 12, 13]:
        n2 = 20 - n1
        for s1 in [1, 2, 3, 4]:
            for s2 in [1, 2, 3, 5, 7]:
                A = _two_gp_candidate(2, 3, n1, n2, s1, s2)
                if len(A) == 20 and _is_valid(A):
                    candidates.append(A)
                    break
            else:
                continue
            break
    
    # Powers of 2 and 5
    for n1 in [10, 11, 12]:
        n2 = 20 - n1
        A = _two_gp_candidate(2, 5, n1, n2, 1, 1)
        if len(A) == 20 and _is_valid(A):
            candidates.append(A)
    
    # Powers of 3 and 5
    for n1 in [8, 9, 10]:
        n2 = 20 - n1
        A = _two_gp_candidate(3, 5, n1, n2, 1, 1)
        if len(A) == 20 and _is_valid(A):
            candidates.append(A)
    
    # Smooth numbers (2,3,5-smooth)
    smooths = set()
    for i in range(15):
        p2 = 1 << i
        if p2 > MAX_VAL:
            break
        for j in range(12):
            p3 = p2 * (3 ** j)
            if p3 > MAX_VAL:
                break
            for k in range(8):
                v = p3 * (5 ** k)
                if v > MAX_VAL:
                    break
                smooths.add(v)
    sorted_smooths = sorted(smooths)
    if len(sorted_smooths) >= 20:
        candidates.append(sorted_smooths[:20])
    
    # Geometric progressions with various ratios
    for ratio in [1.5, 2.0, 2.5, 3.0]:
        for start in [1, 2]:
            A = []
            current = float(start)
            for _ in range(20):
                val = int(round(current))
                if val in A:
                    val += 1
                    while val in A:
                        val += 1
                if val > MAX_VAL:
                    break
                A.append(val)
                current *= ratio
            if len(A) == 20 and _is_valid(A):
                candidates.append(A)
    
    # Random for diversity
    rng = random.Random(42)
    candidates.append(rng.sample(range(1, 10001), 20))
    
    return candidates


def _local_search(A, time_limit, rng):
    """Simulated annealing local search with swap and replace moves."""
    A = sorted(A)
    current_obj = _objective(A)
    best_A = list(A)
    best_obj = current_obj
    
    start_time = time.time()
    temperature = 40.0
    cooling_rate = 0.996
    
    while temperature > 0.05:
        if time.time() - start_time > time_limit:
            break
        
        move_type = rng.random()
        accepted = False
        
        if move_type < 0.4:
            # Swap two elements
            i, j = rng.sample(range(20), 2)
            old_i, old_j = A[i], A[j]
            # Try swapping with nearby values
            for _ in range(15):
                if time.time() - start_time > time_limit:
                    break
                new_i = max(1, min(MAX_VAL, old_i + rng.randint(-10, 10)))
                new_j = max(1, min(MAX_VAL, old_j + rng.randint(-10, 10)))
                if new_i == new_j or new_i in A or new_j in A:
                    continue
                A[i], A[j] = new_i, new_j
                new_obj = _objective(A)
                A[i], A[j] = old_i, old_j
                
                delta = new_obj - current_obj
                if delta <= 0 or rng.random() < min(1.0, max(0.0, -delta / (temperature + 1e-10))):
                    A[i], A[j] = new_i, new_j
                    current_obj = new_obj
                    if new_obj < best_obj:
                        best_obj = new_obj
                        best_A = list(A)
                    accepted = True
                    break
        elif move_type < 0.8:
            # Replace one element
            idx = rng.randint(0, 19)
            old_val = A[idx]
            for _ in range(20):
                if time.time() - start_time > time_limit:
                    break
                r = rng.random()
                if r < 0.3:
                    new_val = max(1, min(MAX_VAL, old_val + rng.randint(-5, 5)))
                elif r < 0.6:
                    new_val = max(1, min(MAX_VAL, old_val + rng.randint(-30, 30)))
                elif r < 0.85:
                    new_val = rng.randint(1, max(2, old_val * 3))
                else:
                    new_val = rng.randint(1, MAX_VAL)
                
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
                    accepted = True
                    break
        else:
            # Mutate: pick two elements and shift them toward each other
            i, j = rng.sample(range(20), 2)
            old_i, old_j = A[i], A[j]
            for _ in range(10):
                if time.time() - start_time > time_limit:
                    break
                shift = rng.randint(-15, 15)
                new_i = max(1, min(MAX_VAL, old_i + shift))
                new_j = max(1, min(MAX_VAL, old_j - shift))
                if new_i == new_j or new_i in A or new_j in A:
                    continue
                A[i], A[j] = new_i, new_j
                new_obj = _objective(A)
                A[i], A[j] = old_i, old_j
                
                delta = new_obj - current_obj
                if delta <= 0 or rng.random() < min(1.0, max(0.0, -delta / (temperature + 1e-10))):
                    A[i], A[j] = new_i, new_j
                    current_obj = new_obj
                    if new_obj < best_obj:
                        best_obj = new_obj
                        best_A = list(A)
                    accepted = True
                    break
        
        temperature *= cooling_rate
    
    return best_A, best_obj


def construct():
    start_time = time.time()
    time_budget = TIME_LIMIT - 5
    
    candidates = _construct_candidates()
    
    best_A = None
    best_obj = float('inf')
    
    rng = random.Random(42)
    
    # Evaluate candidates and run local search on the best ones
    scored_candidates = []
    for A in candidates:
        if not _is_valid(A):
            continue
        obj = _objective(A)
        scored_candidates.append((obj, A))
    
    scored_candidates.sort(key=lambda x: x[0])
    
    # Run local search on top candidates
    for obj, A in scored_candidates[:8]:
        elapsed = time.time() - start_time
        remaining = time_budget - elapsed
        if remaining < 1.0:
            break
        refined_A, refined_obj = _local_search(A, remaining * 0.5, rng)
        if refined_obj < best_obj:
            best_obj = refined_obj
            best_A = refined_A
    
    if best_A is None:
        best_A = list(range(1, 21))
        best_obj = _objective(best_A)
    
    # Final intensive refinement
    remaining = TIME_LIMIT - 5 - (time.time() - start_time)
    if remaining > 2:
        try:
            rng2 = random.Random(123)
            refined_A, refined_obj = _local_search(best_A, remaining * 0.7, rng2)
            if refined_obj < best_obj:
                best_A = refined_A
                best_obj = refined_obj
        except Exception:
            pass
    
    # Final polish with tighter parameters
    remaining = TIME_LIMIT - 5 - (time.time() - start_time)
    if remaining > 1:
        try:
            rng3 = random.Random(999)
            refined_A, refined_obj = _local_search(best_A, remaining, rng3)
            if refined_obj < best_obj:
                best_A = refined_A
                best_obj = refined_obj
        except Exception:
            pass
    
    if not _is_valid(best_A):
        best_A = list(range(1, 21))
    
    return sorted(best_A)
