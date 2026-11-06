import time
import random
import math

N = 20
MAX_VAL = 10000
TIME_LIMIT = 85.0  # safety margin before 90s kill


def _objective(A):
    """Compute max(|A+A|, |A*A|) for a set A of N distinct positive integers."""
    n = len(A)
    # Sumset: all a_i + a_j for i <= j
    sums = set()
    for i in range(n):
        ai = A[i]
        for j in range(i, n):
            sums.add(ai + A[j])
    
    # Product set: all a_i * a_j for i <= j
    prods = set()
    for i in range(n):
        ai = A[i]
        for j in range(i, n):
            prods.add(ai * A[j])
    
    return max(len(sums), len(prods))


def _is_valid(A):
    """Check if A is a valid set: N distinct positive integers <= MAX_VAL."""
    if len(A) != N:
        return False
    if len(set(A)) != N:
        return False
    for x in A:
        if x < 1 or x > MAX_VAL:
            return False
    return True


def _geometric_start():
    """Generate a geometric progression starting set."""
    best = None
    best_score = float('inf')
    for base in range(2, 12):
        for start in range(1, 100):
            A = []
            val = start
            for _ in range(N):
                if val > MAX_VAL:
                    break
                A.append(val)
                val = val * base
            if len(A) != N:
                continue
            if not _is_valid(A):
                continue
            score = _objective(A)
            if score < best_score:
                best_score = score
                best = list(A)
    return best, best_score


def _arithmetic_start():
    """Generate an arithmetic progression starting set."""
    best = None
    best_score = float('inf')
    for a in range(1, 100):
        for d in range(1, 100):
            A = [a + i * d for i in range(N)]
            if A[-1] > MAX_VAL:
                continue
            if not _is_valid(A):
                continue
            score = _objective(A)
            if score < best_score:
                best_score = score
                best = list(A)
    return best, best_score


def _random_start(seed=42):
    """Generate a random starting set."""
    rng = random.Random(seed)
    while True:
        A = rng.sample(range(1, MAX_VAL + 1), N)
        if _is_valid(A):
            return A, _objective(A)


def _local_search(A, max_iter=5000, seed=42, time_limit=80.0):
    """Local search: try replacing one element at a time."""
    rng = random.Random(seed)
    A = list(A)
    current_score = _objective(A)
    best_A = list(A)
    best_score = current_score
    start_time = time.time()
    
    for _ in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        # Pick a random element to replace
        idx = rng.randint(0, N - 1)
        old_val = A[idx]
        
        # Try a few random replacements
        for _ in range(3):
            new_val = rng.randint(1, MAX_VAL)
            if new_val == old_val:
                continue
            if new_val in A:
                continue
            
            A[idx] = new_val
            score = _objective(A)
            if score < best_score:
                best_score = score
                best_A = list(A)
            A[idx] = old_val
        
        # Also try nearby values
        for delta in [-1, 1, -2, 2, -3, 3, -5, 5, -10, 10, -20, 20]:
            new_val = old_val + delta
            if new_val < 1 or new_val > MAX_VAL:
                continue
            if new_val == old_val:
                continue
            if new_val in A:
                continue
            
            A[idx] = new_val
            score = _objective(A)
            if score < best_score:
                best_score = score
                best_A = list(A)
            A[idx] = old_val
    
    return best_A, best_score


def _simulated_annealing(A, max_iter=10000, seed=42, time_limit=80.0):
    """Simulated annealing for better exploration."""
    rng = random.Random(seed)
    A = list(A)
    current_score = _objective(A)
    best_A = list(A)
    best_score = current_score
    
    T = 10.0
    cooling = 0.999
    start_time = time.time()
    
    for _ in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        idx = rng.randint(0, N - 1)
        old_val = A[idx]
        
        # Generate a neighbor
        if rng.random() < 0.5:
            new_val = rng.randint(1, MAX_VAL)
        else:
            new_val = old_val + rng.randint(-30, 30)
        
        if new_val < 1 or new_val > MAX_VAL:
            continue
        if new_val == old_val:
            continue
        if new_val in A:
            continue
        
        A[idx] = new_val
        new_score = _objective(A)
        delta = new_score - current_score
        
        if delta < 0 or rng.random() < math.exp(-delta / T):
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_A = list(A)
        else:
            A[idx] = old_val
        
        T *= cooling
        if T < 0.1:
            T = 0.1
    
    return best_A, best_score


def _multi_start_search(time_budget):
    """Multi-start search with diverse initializations."""
    start_time = time.time()
    best_A = None
    best_score = float('inf')
    
    # Phase 1: Try structured constructions
    candidates = []
    
    # Geometric progressions
    A_geo, score_geo = _geometric_start()
    if A_geo is not None:
        candidates.append((A_geo, score_geo))
    
    # Arithmetic progressions
    A_arith, score_arith = _arithmetic_start()
    if A_arith is not None:
        candidates.append((A_arith, score_arith))
    
    # Random starts with different seeds
    for seed in [42, 123, 456, 789, 1111]:
        if time.time() - start_time > time_budget - 10:
            break
        A_rand, score_rand = _random_start(seed=seed)
        candidates.append((A_rand, score_rand))
    
    # Sort by score
    candidates.sort(key=lambda x: x[1])
    
    if not candidates:
        # Fallback: simple range
        A_range = list(range(1, N + 1))
        score_range = _objective(A_range)
        return A_range, score_range
    
    best_A, best_score = candidates[0]
    
    # Phase 2: Local search from best candidates
    for A_init, score_init in candidates[:3]:
        if time.time() - start_time > time_budget - 5:
            break
        A_ls, score_ls = _local_search(A_init, max_iter=3000, seed=123, time_limit=time_budget - 5)
        if score_ls < best_score:
            best_score = score_ls
            best_A = A_ls
    
    # Phase 3: Simulated annealing from best found
    if time.time() - start_time < time_budget - 3:
        A_sa, score_sa = _simulated_annealing(best_A, max_iter=5000, seed=456, time_limit=time_budget - 3)
        if score_sa < best_score:
            best_score = score_sa
            best_A = A_sa
    
    # Phase 4: One more round of local search if time permits
    if time.time() - start_time < time_budget - 2:
        A_final, score_final = _local_search(best_A, max_iter=2000, seed=789, time_limit=time_budget - 2)
        if score_final < best_score:
            best_score = score_final
            best_A = A_final
    
    return best_A, best_score


def construct():
    start_time = time.time()
    
    # Run multi-start search with time budget
    best_A, best_score = _multi_start_search(time_budget=TIME_LIMIT - 5)
    
    # Final check
    if time.time() - start_time > TIME_LIMIT - 3:
        return best_A
    
    return best_A
