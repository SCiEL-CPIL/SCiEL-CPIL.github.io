"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
from itertools import combinations

# EVOLVE-BLOCK-START
def _compute_score(U):
    """Compute the score for a set U of nonnegative integers containing 0."""
    if not U or 0 not in U:
        return 0.0
    U_set = set(U)
    n = len(U_set)
    if n <= 1:
        return 0.0
    max_u = max(U_set)
    if max_u == 0:
        return 0.0
    # Compute sumset U+U
    sumset = set()
    for a in U_set:
        for b in U_set:
            sumset.add(a + b)
    # Compute difference set U-U
    diffset = set()
    for a in U_set:
        for b in U_set:
            diffset.add(a - b)
    # Score: 1 + log(|U-U|/|U+U|)/log(2*max(U)+1)
    ratio = len(diffset) / len(sumset)
    if ratio <= 0:
        return 0.0
    denom = math.log(2 * max_u + 1)
    if denom <= 0:
        return 0.0
    score = 1.0 + math.log(ratio) / denom
    return score

def _is_prime(n):
    """Check if n is prime."""
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

def _generate_primes(count, start=2):
    """Generate the first `count` primes starting from `start`."""
    primes = []
    candidate = start
    while len(primes) < count:
        if _is_prime(candidate):
            primes.append(candidate)
        candidate += 1
    return primes

def construct():
    deadline = time.time() + 110  # safety margin before 120s limit
    
    # Start with a good initial set
    best_set = {0, 1, 3}
    best_score = _compute_score(best_set)
    
    # Try some analytical constructions first
    # Construction 1: {0, 1, 3, 9, 27, ...} powers of 3
    for k in range(2, 15):
        U = {0}
        val = 1
        for i in range(k):
            U.add(val)
            val *= 3
        s = _compute_score(U)
        if s > best_score:
            best_score = s
            best_set = set(U)
    
    # Construction 2: Squares
    for k in range(2, 20):
        U = {i * i for i in range(k)}
        s = _compute_score(U)
        if s > best_score:
            best_score = s
            best_set = set(U)
    
    # Construction 3: Binary-like sets
    for k in range(2, 12):
        U = {1 << i for i in range(k)}
        s = _compute_score(U)
        if s > best_score:
            best_score = s
            best_set = set(U)
    
    # Construction 4: Prime-step arithmetic progressions
    # U = {0, p1, p1+p2, p1+p2+p3, ...} where pi are distinct primes
    # This exploits additive independence of primes to minimize sumset collisions
    primes = _generate_primes(20, start=2)
    for k in range(2, 18):
        if k > len(primes):
            break
        U = {0}
        current = 0
        for i in range(k):
            current += primes[i]
            U.add(current)
        s = _compute_score(U)
        if s > best_score:
            best_score = s
            best_set = set(U)
    
    # Construction 5: Mix of prime steps with varying densities
    # Try different prime starting points
    for start_p in [2, 3, 5, 7, 11]:
        primes_sub = _generate_primes(15, start=start_p)
        for k in range(2, 15):
            if k > len(primes_sub):
                break
            U = {0}
            current = 0
            for i in range(k):
                current += primes_sub[i]
                U.add(current)
            s = _compute_score(U)
            if s > best_score:
                best_score = s
                best_set = set(U)
    
    # Local search: try adding/removing elements
    U_list = list(best_set)
    
    # Try adding elements
    max_val = max(U_list) if U_list else 0
    search_range = max_val * 3 + 10
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try adding new elements
        for new_elem in range(1, search_range + 1):
            if new_elem in best_set:
                continue
            candidate = best_set | {new_elem}
            s = _compute_score(candidate)
            if s > best_score + 1e-12:
                best_score = s
                best_set = candidate
                U_list = list(best_set)
                max_val = max(U_list)
                search_range = max_val * 3 + 10
                improved = True
                if time.time() > deadline:
                    break
        # Try removing elements (keep 0)
        if time.time() > deadline:
            break
        for elem in list(best_set):
            if elem == 0:
                continue
            candidate = best_set - {elem}
            if len(candidate) < 2:
                continue
            s = _compute_score(candidate)
            if s > best_score + 1e-12:
                best_score = s
                best_set = candidate
                U_list = list(best_set)
                max_val = max(U_list) if U_list else 0
                search_range = max_val * 3 + 10
                improved = True
                if time.time() > deadline:
                    break
    
    return sorted(best_set)
# EVOLVE-BLOCK-END
