"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random
import numpy as np

_DEADLINE = None

def _get_deadline():
    global _DEADLINE
    if _DEADLINE is None:
        _DEADLINE = time.time() + 140
    return _DEADLINE

def _time_remaining():
    return max(0, _get_deadline() - time.time())

def _score_set_numpy(U):
    """Fast exact scoring using numpy broadcasting for small/medium sets."""
    U_arr = np.array(sorted(set(U)), dtype=np.int64)
    n = len(U_arr)
    if n == 0:
        return 0.0
    max_u = int(U_arr[-1])
    if max_u == 0:
        return 1.0
    
    # Compute sumset and difference set
    sums = U_arr[:, None] + U_arr[None, :]
    diffs = U_arr[:, None] - U_arr[None, :]
    
    sumset_size = len(np.unique(sums))
    diffset_size = len(np.unique(diffs))
    
    if sumset_size == 0:
        return 0.0
    
    ratio = diffset_size / sumset_size
    if ratio <= 0:
        return 0.0
    
    log_ratio = math.log(ratio)
    log_denom = math.log(2 * max_u + 1)
    if log_denom == 0:
        return 1.0
    return 1.0 + log_ratio / log_denom

def _score_set_python(U):
    """Pure Python scoring for very small sets or fallback."""
    U = sorted(set(U))
    n = len(U)
    if n == 0:
        return 0.0
    max_u = max(U)
    if max_u == 0:
        return 1.0
    
    sumset = set()
    diffset = set()
    for i in range(n):
        ui = U[i]
        for j in range(n):
            sumset.add(ui + U[j])
            diffset.add(ui - U[j])
    
    ratio = len(diffset) / len(sumset)
    if ratio <= 0:
        return 0.0
    log_ratio = math.log(ratio)
    log_denom = math.log(2 * max_u + 1)
    if log_denom == 0:
        return 1.0
    return 1.0 + log_ratio / log_denom

def _score_set(U):
    """Dispatch to best scoring method."""
    U = sorted(set(U))
    n = len(U)
    if n == 0:
        return 0.0
    max_u = max(U)
    if max_u == 0:
        return 1.0
    if n <= 60:
        return _score_set_numpy(U)
    else:
        return _score_set_python(U)

# Known optimal Golomb rulers
_KNOWN_GR = {
    1: [0],
    2: [0, 1],
    3: [0, 1, 3],
    4: [0, 1, 4, 6],
    5: [0, 1, 4, 10, 11],
    6: [0, 1, 4, 10, 12, 17],
    7: [0, 1, 4, 10, 12, 17, 22],
    8: [0, 1, 4, 10, 12, 17, 22, 29],
    9: [0, 1, 4, 10, 12, 17, 22, 29, 36],
    10: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44],
    11: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52],
    12: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52, 60],
    13: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52, 60, 68],
    14: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52, 60, 68, 77],
    15: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52, 60, 68, 77, 86],
    16: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52, 60, 68, 77, 86, 95],
    17: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52, 60, 68, 77, 86, 95, 104],
    18: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52, 60, 68, 77, 86, 95, 104, 113],
    19: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52, 60, 68, 77, 86, 95, 104, 113, 122],
    20: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52, 60, 68, 77, 86, 95, 104, 113, 122, 131],
    21: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52, 60, 68, 77, 86, 95, 104, 113, 122, 131, 140],
    22: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52, 60, 68, 77, 86, 95, 104, 113, 122, 131, 140, 149],
    23: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52, 60, 68, 77, 86, 95, 104, 113, 122, 131, 140, 149, 158],
    24: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52, 60, 68, 77, 86, 95, 104, 113, 122, 131, 140, 149, 158, 167],
    25: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52, 60, 68, 77, 86, 95, 104, 113, 122, 131, 140, 149, 158, 167, 176],
}

def _is_golomb(ruler):
    """Check if a ruler is a valid Golomb ruler."""
    diffs = set()
    for i in range(len(ruler)):
        for j in range(i + 1, len(ruler)):
            d = ruler[j] - ruler[i]
            if d in diffs:
                return False
            diffs.add(d)
    return True

def _greedy_golomb(n, max_len=2000):
    """Generate a Golomb ruler of order n using greedy with backtracking."""
    if n in _KNOWN_GR:
        return _KNOWN_GR[n]
    if n <= 1:
        return [0]
    
    ruler = [0]
    for i in range(1, n):
        placed = False
        pos = 1
        while pos < max_len:
            candidate = ruler + [pos]
            if _is_golomb(candidate):
                ruler.append(pos)
                placed = True
                break
            pos += 1
        if not placed:
            break
    return ruler

def _twin_prime_ladder(start_prime, num_pairs):
    """Construct a set using cumulative sums of twin prime pairs."""
    def is_prime(x):
        if x < 2:
            return False
        if x < 4:
            return True
        if x % 2 == 0 or x % 3 == 0:
            return False
        i = 5
        while i * i <= x:
            if x % i == 0 or x % (i + 2) == 0:
                return False
            i += 6
        return True
    
    # Find twin primes starting from start_prime
    twin_primes = []
    p = start_prime
    while len(twin_primes) < num_pairs and p < 100000:
        if is_prime(p) and is_prime(p + 2):
            twin_primes.append((p, p + 2))
        p += 1
    
    if not twin_primes:
        return [0]
    
    # Build U as cumulative sums of the smaller primes in each twin pair
    U = [0]
    current = 0
    for p1, p2 in twin_primes:
        current += p1
        U.append(current)
    
    return sorted(set(U))

def _local_search(U, max_iters=20000, deadline=None):
    """Improve set U via local search with simulated annealing."""
    U = sorted(set(U))
    best_score = _score_set(U)
    best_U = U[:]
    current_score = best_score
    current_U = U[:]
    
    max_u = max(U)
    upper_bound = max_u + 50
    
    T = 0.02
    cooling_rate = 0.999
    min_T = 0.0001
    no_improve = 0
    max_no_improve = 60
    
    for it in range(max_iters):
        if deadline and time.time() > deadline:
            break
        if _time_remaining() < 5:
            break
        if no_improve >= max_no_improve and T < min_T:
            break
        
        improved = False
        n = len(U)
        
        # Try moving one element
        indices_to_try = list(range(1, n))
        random.shuffle(indices_to_try)
        
        for i in indices_to_try:
            old_val = U[i]
            # Try nearby values
            for offset in range(-15, 16):
                nv = old_val + offset
                if nv <= 0:
                    continue
                new_U = U[:]
                new_U[i] = nv
                if len(set(new_U)) != len(new_U):
                    continue
                s = _score_set(new_U)
                delta_s = s - current_score
                if delta_s > 1e-10:
                    current_score = s
                    current_U = new_U[:]
                    U = new_U[:]
                    if s > best_score:
                        best_score = s
                        best_U = new_U[:]
                    improved = True
                    no_improve = 0
                    break
                elif T > min_T and random.random() < math.exp(delta_s / T):
                    current_score = s
                    current_U = new_U[:]
                    U = new_U[:]
                    improved = True
                    break
            if improved:
                break
        
        if not improved:
            no_improve += 1
            
            # Try adding an element
            if n < 45:
                for _ in range(20):
                    nv = random.randint(1, upper_bound)
                    if nv in U:
                        continue
                    new_U = U + [nv]
                    s = _score_set(new_U)
                    delta_s = s - current_score
                    if delta_s > 1e-10 or (T > min_T and random.random() < math.exp(delta_s / T)):
                        current_score = s
                        current_U = new_U[:]
                        U = new_U[:]
                        if s > best_score:
                            best_score = s
                            best_U = new_U[:]
                        improved = True
                        no_improve = 0
                        break
            
            # Try removing an element
            if not improved and n > 4:
                for i in range(1, n):
                    new_U = U[:i] + U[i+1:]
                    s = _score_set(new_U)
                    delta_s = s - current_score
                    if delta_s > 1e-10 or (T > min_T and random.random() < math.exp(delta_s / T)):
                        current_score = s
                        current_U = new_U[:]
                        U = new_U[:]
                        if s > best_score:
                            best_score = s
                            best_U = new_U[:]
                        improved = True
                        no_improve = 0
                        break
        
        if not improved:
            # Try larger perturbation
            if random.random() < 0.1 and n >= 4:
                new_U = U[:]
                i1, i2 = random.sample(range(1, n), 2)
                new_U[i1] = random.randint(1, upper_bound)
                new_U[i2] = random.randint(1, upper_bound)
                if len(set(new_U)) == len(new_U):
                    s = _score_set(new_U)
                    delta_s = s - current_score
                    if delta_s > 1e-10 or (T > min_T and random.random() < math.exp(delta_s / T)):
                        current_score = s
                        current_U = new_U[:]
                        U = new_U[:]
                        if s > best_score:
                            best_score = s
                            best_U = new_U[:]
                        improved = True
                        no_improve = 0
        
        T = max(min_T, T * cooling_rate)
    
    return sorted(set(best_U)), best_score

def _polish(U, deadline=None):
    """High-precision polish pass."""
    U = sorted(set(U))
    n = len(U)
    if n > 50:
        return _local_search(U, max_iters=5000, deadline=deadline)
    
    best_score = _score_set_numpy(U)
    best_U = U[:]
    max_u = max(U)
    upper_bound = max_u + 30
    
    no_improve = 0
    max_no_improve = 40
    
    for it in range(15000):
        if deadline and time.time() > deadline:
            break
        if _time_remaining() < 3:
            break
        if no_improve >= max_no_improve:
            break
        
        improved = False
        indices_to_try = list(range(1, len(U)))
        random.shuffle(indices_to_try)
        
        for i in indices_to_try:
            old_val = U[i]
            for offset in range(-20, 21):
                nv = old_val + offset
                if nv <= 0:
                    continue
                new_U = U[:]
                new_U[i] = nv
                if len(set(new_U)) != len(new_U):
                    continue
                s = _score_set_numpy(new_U)
                if s > best_score + 1e-12:
                    best_score = s
                    best_U = new_U[:]
                    U = new_U[:]
                    improved = True
                    no_improve = 0
                    break
            if improved:
                break
        
        if not improved:
            no_improve += 1
            if len(U) < 60:
                for _ in range(30):
                    nv = random.randint(1, upper_bound)
                    if nv in U:
                        continue
                    new_U = U + [nv]
                    s = _score_set_numpy(new_U)
                    if s > best_score + 1e-12:
                        best_score = s
                        best_U = new_U[:]
                        U = new_U[:]
                        improved = True
                        no_improve = 0
                        break
            if not improved and len(U) > 4:
                for i in range(1, len(U)):
                    new_U = U[:i] + U[i+1:]
                    s = _score_set_numpy(new_U)
                    if s > best_score + 1e-12:
                        best_score = s
                        best_U = new_U[:]
                        U = new_U[:]
                        improved = True
                        no_improve = 0
                        break
    
    return sorted(set(best_U)), best_score

def construct():
    global _DEADLINE
    start_time = time.time()
    _DEADLINE = start_time + 140
    deadline = _DEADLINE
    
    best_U = [0, 1, 3]
    best_score = _score_set(best_U)
    
    random.seed(42)
    candidates = []
    
    # 1. Golomb rulers
    for n in range(3, 26):
        if time.time() > deadline - 80:
            break
        if n in _KNOWN_GR:
            gr = _KNOWN_GR[n]
        else:
            gr = _greedy_golomb(n, max_len=500)
        candidates.append(gr)
    
    # 2. Twin prime ladders
    for start_p in [3, 5, 11, 17, 29, 41, 59, 71, 101, 137]:
        if time.time() > deadline - 80:
            break
        for num_pairs in range(3, 25):
            if time.time() > deadline - 80:
                break
            tp = _twin_prime_ladder(start_p, num_pairs)
            if len(tp) >= 3:
                candidates.append(tp)
    
    # 3. Random sets
    for trial in range(40):
        if time.time() > deadline - 80:
            break
        n = random.randint(3, 40)
        max_val = random.randint(30, 500)
        U = [0] + random.sample(range(1, max_val + 1), n - 1)
        U = sorted(U)
        candidates.append(U)
    
    # 4. Sparse random sets
    for trial in range(20):
        if time.time() > deadline - 80:
            break
        n = random.randint(5, 30)
        max_val = random.randint(100, 2000)
        U = [0] + random.sample(range(1, max_val + 1), n - 1)
        U = sorted(U)
        candidates.append(U)
    
    # 5. Structured sets
    for base in [2, 3, 5, 7, 11]:
        if time.time() > deadline - 80:
            break
        U = [0]
        for i in range(1, 25):
            U.append(i * base)
        candidates.append(U[:25])
    
    # 6. Powers-based
    for p in [2, 3, 4]:
        if time.time() > deadline - 80:
            break
        U = [0]
        for i in range(1, 15):
            val = p ** i
            if val > 10000:
                break
            U.append(val)
        candidates.append(U)
    
    # Evaluate candidates
    scored = []
    for U in candidates:
        if time.time() > deadline - 60:
            break
        s = _score_set(U)
        scored.append((s, U))
        if s > best_score:
            best_score = s
            best_U = U[:]
    
    scored.sort(key=lambda x: -x[0])
    
    # Local search on best
    if time.time() < deadline - 40:
        best_U, best_score = _local_search(best_U, max_iters=20000, deadline=deadline - 40)
    
    # Local search on top candidates
    for s, U in scored[:15]:
        if time.time() > deadline - 40:
            break
        if U == best_U:
            continue
        if s > best_score - 0.03:
            improved_U, improved_score = _local_search(U, max_iters=8000, deadline=deadline - 40)
            if improved_score > best_score:
                best_score = improved_score
                best_U = improved_U
    
    # Final polish
    if time.time() < deadline - 10:
        best_U, best_score = _polish(best_U, deadline=deadline - 5)
    
    return sorted(set(best_U))
# EVOLVE-BLOCK-END
