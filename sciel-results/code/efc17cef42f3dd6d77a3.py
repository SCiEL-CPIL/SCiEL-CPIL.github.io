"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import random
import numpy as np

_start_time = time.time()
_DEADLINE = 105  # safety margin before 150s limit

def _elapsed():
    return time.time() - _start_time

def _score_set(U):
    """Compute score = 1 + log(|U-U|/|U+U|)/log(2*max(U)+1)."""
    if not U or 0 not in U:
        return -1.0
    n = len(U)
    max_u = max(U)
    if max_u == 0:
        return -1.0
    arr = np.array(U, dtype=np.int64)
    # Use FFT for larger sets to speed up computation
    if n > 50:
        # For large sets, use a more efficient approach
        sums = arr[:, None] + arr[None, :]
        diffs = arr[:, None] - arr[None, :]
        num_sums = len(np.unique(sums))
        num_diffs = len(np.unique(diffs))
    else:
        sums = arr[:, None] + arr[None, :]
        diffs = arr[:, None] - arr[None, :]
        num_sums = len(np.unique(sums))
        num_diffs = len(np.unique(diffs))
    if num_sums == 0:
        return -1.0
    ratio = num_diffs / num_sums
    if ratio <= 1:
        return -1.0
    log_denom = math.log(2 * max_u + 1)
    if log_denom <= 0:
        return -1.0
    return 1.0 + math.log(ratio) / log_denom

def _golomb_seed(n, max_val):
    """Greedy Golomb ruler: pick n elements with distinct pairwise differences."""
    U = [0]
    diffs = set()
    for _ in range(1, n):
        best_v = None
        for v in range(1, max_val + 1):
            if v in U:
                continue
            ok = True
            for a in U:
                d = abs(v - a)
                if d in diffs:
                    ok = False
                    break
            if ok:
                best_v = v
                break
        if best_v is None:
            break
        U.append(best_v)
        for a in U:
            diffs.add(abs(best_v - a))
    return U

def _prime_step_construction(max_primes=40, max_val=50000):
    """Greedy construction using prime-step gaps to exploit additive independence."""
    # Generate primes up to max_val
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n+1, i):
                    is_prime[j] = False
        return [i for i, p in enumerate(is_prime) if p]
    
    primes = sieve(max_val)
    if len(primes) < 2:
        return [0]
    
    U = [0]
    current_sum = 0
    best_score = _score_set(U)
    
    # Use larger prime steps to create more spread
    # Try different strategies: consecutive primes, every other prime, etc.
    for strategy in [0, 1, 2]:  # 0=consecutive, 1=every other, 2=every third
        if _elapsed() > _DEADLINE - 20:
            break
        U_temp = [0]
        current_sum_temp = 0
        score_temp = best_score
        primes_to_use = primes[1::(strategy+1)]
        
        for p in primes_to_use:
            if _elapsed() > _DEADLINE - 20:
                break
            new_val = current_sum_temp + p
            if new_val > max_val:
                continue
            candidate = U_temp + [new_val]
            s = _score_set(candidate)
            if s > score_temp + 1e-10:
                U_temp = candidate
                current_sum_temp = new_val
                score_temp = s
        
        if score_temp > best_score:
            best_score = score_temp
            U = U_temp
    
    return U

def _local_search(initial, max_steps=2000):
    """Local search: add/remove/swap elements to improve score."""
    best_U = list(initial)
    best_score = _score_set(best_U)
    current_U = list(best_U)
    current_score = best_score
    stall_count = 0

    for step in range(max_steps):
        if _elapsed() > _DEADLINE:
            break
        max_val = max(current_U) if current_U else 0
        improved = False
        cur_set = set(current_U)

        # Try adding a new element - focus on values that create new difference patterns
        candidates = []
        
        # Values that are far apart to maximize difference set
        for multiplier in [2, 3, 5, 7, 11]:
            for delta in range(1, 20):
                candidates.append(max_val * multiplier + delta)
        
        # Prime-like steps from existing elements
        def is_prime(n):
            if n < 2:
                return False
            for i in range(2, int(n**0.5) + 1):
                if n % i == 0:
                    return False
            return True
        
        for a in current_U:
            for p in [3, 5, 7, 11, 13, 17, 19, 23, 29, 31]:
                if is_prime(p):
                    candidates.append(a + p)
                    candidates.append(a + p * 2)
        
        # Some interior values (sparse sampling)
        if max_val > 10:
            step_size = max(1, max_val // 50)
            for v in range(1, max_val + 1, step_size):
                if v not in cur_set:
                    candidates.append(v)
        
        # Random exploration
        for _ in range(30):
            candidates.append(random.randint(1, max_val * 5))
        
        # Remove duplicates and invalid values
        candidates = list(set([v for v in candidates if v > 0 and v not in cur_set]))
        random.shuffle(candidates)

        for new_val in candidates[:30]:
            if new_val <= 0 or new_val in cur_set:
                continue
            new_U = list(current_U) + [new_val]
            new_U.sort()
            new_score = _score_set(new_U)
            if new_score > current_score + 1e-10:
                current_U = new_U
                current_score = new_score
                cur_set = set(current_U)
                if current_score > best_score:
                    best_U = list(current_U)
                    best_score = current_score
                improved = True
                stall_count = 0
                break

        if not improved:
            # Try removing an element
            if len(current_U) > 2:
                remove_candidates = list(range(1, len(current_U)))
                random.shuffle(remove_candidates)
                for idx in remove_candidates[:15]:
                    new_U = [x for i, x in enumerate(current_U) if i != idx]
                    new_score = _score_set(new_U)
                    if new_score > current_score + 1e-10:
                        current_U = new_U
                        current_score = new_score
                        cur_set = set(current_U)
                        if current_score > best_score:
                            best_U = list(current_U)
                            best_score = current_score
                        improved = True
                        stall_count = 0
                        break

        if not improved:
            stall_count += 1
            if stall_count > 40:
                # Perturb: replace a random element with a new value
                if len(current_U) > 2:
                    idx = random.randint(1, len(current_U) - 1)
                    max_val = max(current_U)
                    new_val = random.randint(1, max_val * 5)
                    if new_val not in cur_set:
                        current_U[idx] = new_val
                        current_U.sort()
                        cur_set = set(current_U)
                        current_score = _score_set(current_U)
                        stall_count = 0
                else:
                    break
            elif step > 150:
                break

    return best_U, best_score

def construct():
    global _start_time
    _start_time = time.time()

    best_U = [0, 1, 3]
    best_score = _score_set(best_U)

    # Try Golomb rulers of various sizes
    # Greedy Golomb is a good seed for maximizing |U-U|/|U+U|
    for n in range(3, 31):
        if _elapsed() > _DEADLINE - 5:
            break
        # Max val for greedy Golomb is approx n^2
        max_val = n * n * 3
        U = _golomb_seed(n, max_val)
        if len(U) == n:
            s = _score_set(U)
            if s > best_score:
                best_score = s
                best_U = U[:]

    # Local search on the best Golomb ruler found
    # We restrict the search to swaps that keep the set valid and improve score
    # This avoids the degradation caused by adding/removing elements in the previous local search
    if _elapsed() < _DEADLINE - 2:
        current_U = list(best_U)
        current_score = best_score
        improved = True
        while improved and _elapsed() < _DEADLINE - 2:
            improved = False
            for i in range(1, len(current_U)):
                orig_val = current_U[i]
                # Try values in range [orig_val - 10, orig_val + 10]
                # We allow some collisions in differences if it significantly reduces sumset size
                # But usually we want to keep differences distinct to keep |U-U| high
                for delta in range(-10, 11):
                    new_val = orig_val + delta
                    if new_val <= 0:
                        continue
                    if new_val in current_U:
                        continue
                    new_U = current_U.copy()
                    new_U[i] = new_val
                    new_U.sort()
                    s = _score_set(new_U)
                    if s > current_score + 1e-9:
                        current_U = new_U
                        current_score = s
                        improved = True
                        break
                if improved:
                    break
            
        if current_score > best_score:
            best_score = current_score
            best_U = current_U

    return sorted(best_U)
