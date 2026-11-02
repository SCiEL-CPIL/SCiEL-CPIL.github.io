"""Optimized construction for sumset/difference-set exponent lower bound."""
import time
import math
import random

# EVOLVE-BLOCK-START

def _compute_score(U):
    """Compute the score: 1 + log(|U-U|/|U+U|)/log(2*max(U)+1)."""
    if not U or len(U) < 2:
        return 0.0
    U = sorted(set(U))
    max_u = U[-1]
    if max_u == 0:
        return 0.0
    
    n = len(U)
    sum_set = set()
    diff_set = set()
    
    for i in range(n):
        ui = U[i]
        for j in range(i, n):
            sum_set.add(ui + U[j])
        for j in range(n):
            diff_set.add(ui - U[j])
    
    if len(sum_set) == 0:
        return 0.0
    
    ratio = len(diff_set) / len(sum_set)
    if ratio <= 0:
        return 0.0
    
    denom = math.log(2 * max_u + 1)
    if denom <= 0:
        return 0.0
    
    return 1.0 + math.log(ratio) / denom

def _compute_score_fast(U):
    """Fast score computation using sorted lists and two-pointer technique."""
    if not U or len(U) < 2:
        return 0.0
    U = sorted(set(U))
    max_u = U[-1]
    if max_u == 0:
        return 0.0
    
    n = len(U)
    
    # Compute sumset size using two pointers on sorted array
    # U+U where we consider i<=j to avoid double counting
    sum_count = 0
    last_sum = -1
    for i in range(n):
        for j in range(i, n):
            s = U[i] + U[j]
            if s != last_sum:
                sum_count += 1
                last_sum = s
    
    # Compute diffset size
    # U-U includes all differences, can be negative
    # Use a set for simplicity but with early termination if too slow
    diff_set = set()
    for i in range(n):
        ui = U[i]
        for j in range(n):
            diff_set.add(ui - U[j])
    
    if sum_count == 0:
        return 0.0
    
    ratio = len(diff_set) / sum_count
    if ratio <= 0:
        return 0.0
    
    denom = math.log(2 * max_u + 1)
    if denom <= 0:
        return 0.0
    
    return 1.0 + math.log(ratio) / denom

def _greedy_construct(initial_gaps=None, max_elements=50, deadline=None):
    """
    Greedy construction: start with {0} and add elements one by one,
    choosing the next element to maximize the score.
    """
    if deadline is None:
        deadline = time.time() + 100
    
    U = [0]
    best_score = 0.0
    
    # If initial gaps provided, use them
    if initial_gaps:
        for gap in initial_gaps:
            if time.time() > deadline - 30:
                break
            new_val = U[-1] + gap
            U.append(new_val)
        U = sorted(set(U))
        best_score = _compute_score(U)
    else:
        # Start with small primes as gaps
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199]
        for p in primes:
            if time.time() > deadline - 30:
                break
            if len(U) >= max_elements:
                break
            candidate = U + [U[-1] + p]
            s = _compute_score(candidate)
            if s > best_score or len(candidate) <= 2:
                best_score = s
                U = candidate
    
    return U

def _local_search(U, deadline, max_iter=30000, rng_seed=42):
    """
    Local search: try modifying elements to improve score.
    Uses a smarter neighborhood: try replacing an element with values
    that create more unique differences while keeping sums controlled.
    """
    U = sorted(set(U))
    if len(U) < 2:
        return U
    
    best_U = U[:]
    best_score = _compute_score(best_U)
    
    start_time = time.time()
    rng = random.Random(rng_seed)
    
    # Precompute some statistics
    n = len(best_U)
    
    for it in range(max_iter):
        if time.time() > deadline - 20:
            break
        
        # Pick a random index to modify (not 0)
        idx = rng.randint(1, len(best_U) - 1)
        old_val = best_U[idx]
        
        # Generate candidate values around the current value
        # Focus on values that might create new unique differences
        candidates = set()
        
        # Nearby values
        for delta in range(-30, 31):
            if delta != 0:
                v = old_val + delta
                if v > 0:
                    candidates.add(v)
        
        # Values based on other elements (to create structured differences)
        for other in best_U:
            if other == old_val:
                continue
            # Try values that are sums/differences with other elements
            for delta in range(-5, 6):
                v = other + delta
                if v > 0 and v != old_val:
                    candidates.add(v)
                v = abs(other - delta)
                if v > 0 and v != old_val:
                    candidates.add(v)
        
        # Try some random values in the range
        max_u = max(best_U)
        for _ in range(20):
            v = rng.randint(1, max_u + 20)
            candidates.add(v)
        
        candidates = sorted(candidates)
        
        improved = False
        for v in candidates:
            if v in best_U:
                continue
            
            new_U = best_U[:]
            new_U[idx] = v
            new_U = sorted(set(new_U))
            
            if 0 not in new_U:
                continue
            
            s = _compute_score(new_U)
            if s > best_score + 1e-12:
                best_score = s
                best_U = new_U
                improved = True
                break
        
        if not improved and it % 500 == 0:
            # Try larger perturbations occasionally
            idx = rng.randint(1, len(best_U) - 1)
            old_val = best_U[idx]
            for _ in range(30):
                delta = rng.randint(-100, 100)
                new_val = old_val + delta
                if new_val < 0:
                    continue
                new_U = best_U[:]
                new_U[idx] = new_val
                new_U = sorted(set(new_U))
                if 0 not in new_U:
                    continue
                s = _compute_score(new_U)
                if s > best_score + 1e-12:
                    best_score = s
                    best_U = new_U
                    break
    
    return best_U

def _add_element(U, deadline):
    """Try adding a new element to U to improve score."""
    U = sorted(set(U))
    best_U = U[:]
    best_score = _compute_score(best_U)
    
    max_u = U[-1] if U else 0
    
    # Try adding elements in strategic positions
    candidates = set()
    
    # Near existing elements
    for u in U:
        for delta in range(-10, 11):
            v = u + delta
            if v > 0:
                candidates.add(v)
    
    # In gaps between elements
    for i in range(len(U) - 1):
        gap = U[i+1] - U[i]
        if gap > 2:
            mid = U[i] + gap // 2
            for delta in range(-5, 6):
                v = mid + delta
                if v > 0 and v < U[i+1]:
                    candidates.add(v)
    
    # Beyond max
    for delta in range(1, 50):
        candidates.add(max_u + delta)
    
    # Random values
    rng = random.Random(123)
    for _ in range(50):
        v = rng.randint(1, max_u + 30)
        candidates.add(v)
    
    candidates = sorted(candidates)
    
    for v in candidates:
        if v in U:
            continue
        if time.time() > deadline - 25:
            break
        new_U = sorted(U + [v])
        s = _compute_score(new_U)
        if s > best_score + 1e-12:
            best_score = s
            best_U = new_U
    
    return best_U

def _remove_element(U, deadline):
    """Try removing an element from U to improve score."""
    U = sorted(set(U))
    if len(U) <= 2:
        return U
    
    best_U = U[:]
    best_score = _compute_score(best_U)
    
    for i in range(1, len(U)):  # Don't remove 0
        if time.time() > deadline - 30:
            break
        new_U = U[:i] + U[i+1:]
        s = _compute_score(new_U)
        if s > best_score + 1e-12:
            best_score = s
            best_U = new_U
            # Continue trying more removals
            U = best_U
            i = 0
    
    return best_U

def _multi_start(deadline):
    """Try multiple different starting configurations and pick the best."""
    starts = []
    
    # Prime-based starts
    for init_primes in [[2, 3, 5, 7, 11, 13], [3, 5, 7, 11, 13, 17], [5, 7, 11, 13, 17, 19]]:
        U = [0]
        for p in init_primes:
            U.append(U[-1] + p)
        starts.append(U)
    
    # Superincreasing-like
    U = [0, 1]
    for i in range(1, 8):
        U.append(U[-1] * 2 + 1)
    starts.append(U)
    
    # Fibonacci-like
    U = [0, 1, 2]
    for i in range(1, 10):
        U.append(U[-1] + U[-2])
    starts.append(U)
    
    # Squares
    U = [i*i for i in range(10)]
    starts.append(U)
    
    # Cubes
    U = [i**3 for i in range(8)]
    starts.append(U)
    
    best_U = None
    best_score = -1
    
    for U in starts:
        if time.time() > deadline - 60:
            break
        s = _compute_score(U)
        if s > best_score:
            best_score = s
            best_U = U
    
    return best_U if best_U else [0, 1, 3]

def construct():
    deadline = time.time() + 130  # Safety margin from 150s
    
    # Multi-start to find a good initial set
    U = _multi_start(deadline)
    
    # Greedy extension
    if time.time() < deadline - 40:
        U = _greedy_construct(deadline=deadline)
    
    # Local search
    if time.time() < deadline - 30:
        U = _local_search(U, deadline, max_iter=25000)
    
    # Try adding elements
    if time.time() < deadline - 20:
        U = _add_element(U, deadline)
    
    # Try removing elements
    if time.time() < deadline - 15:
        U = _remove_element(U, deadline)
    
    # Another local search pass with different seed
    if time.time() < deadline - 15:
        U = _local_search(U, deadline, max_iter=15000, rng_seed=999)
    
    # Final add/remove
    if time.time() < deadline - 10:
        U = _add_element(U, deadline)
    if time.time() < deadline - 5:
        U = _remove_element(U, deadline)
    
    # Ensure 0 is in the set and all are nonnegative
    U = sorted(set([x for x in U if x >= 0]))
    if 0 not in U:
        U = [0] + U
    
    return U

# EVOLVE-BLOCK-END
