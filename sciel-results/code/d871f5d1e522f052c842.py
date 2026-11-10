"""Optimized doubly-covered prefix: Simulated Annealing with NumPy acceleration."""
import time
import random
import numpy as np

def _primes_up_to(n):
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = [False] * len(sieve[i*i::i])
    return [i for i in range(n + 1) if sieve[i]]

_PRIMES = _primes_up_to(1000)
_NPRIMES = len(_PRIMES)
_PRIMES_ARR = np.array(_PRIMES)

def _build_count_np(residues, L_MAX):
    """Build count array for x in [1, L_MAX] using NumPy."""
    count = np.zeros(L_MAX + 1, dtype=np.int8)
    for i, p in enumerate(_PRIMES):
        r = residues[i]
        start = p if r == 0 else r
        if start <= L_MAX:
            count[start:L_MAX+1:p] += 1
    return count

def _find_prefix_L_np(count, L_MAX):
    """Find largest L such that count[x] >= 2 for all x in [1..L]."""
    # Find first index where count < 2
    # count[0] is ignored, we care about 1..L_MAX
    # If all are >= 2, return L_MAX
    # np.where returns indices where condition is True
    # We want the first False
    valid = count[1:L_MAX+1] >= 2
    if not np.any(~valid):
        return L_MAX
    first_invalid = np.where(~valid)[0][0]
    return first_invalid

def _score(residues):
    """Compute the actual score (largest L with full double coverage)."""
    L_MAX = 100000
    count = _build_count_np(residues, L_MAX)
    return _find_prefix_L_np(count, L_MAX)

def _greedy_construct(deadline):
    """
    Greedy construction: build residues to maximize double coverage.
    Strategy: for each prime, choose residue to help cover the current gap.
    """
    residues = [0] * _NPRIMES
    L_MAX = 5000
    count = [0] * (L_MAX + 1)
    
    # Process primes in order of increasing size (small primes give more coverage)
    prime_order = sorted(range(_NPRIMES), key=lambda i: _PRIMES[i])
    
    current_L = 0
    for idx in prime_order:
        if time.time() > deadline:
            break
        p = _PRIMES[idx]
        best_r = 0
        best_score = -1
        
        # The gap is at current_L + 1
        gap = current_L + 1
        candidates = set()
        candidates.add(gap % p)
        candidates.add(0)
        candidates.add(1)
        candidates.add(2)
        for dx in range(-5, 6):
            x = gap + dx
            if x >= 1:
                candidates.add(x % p)
        
        for r in candidates:
            if r >= p:
                continue
            start = p if r == 0 else r
            coverage_gain = 0
            for x in range(start, L_MAX + 1, p):
                if count[x] <= 1:
                    coverage_gain += 1
                elif count[x] == 2:
                    coverage_gain -= 0.5
            
            if gap % p == r:
                coverage_gain += 100
            
            if coverage_gain > best_score:
                best_score = coverage_gain
                best_r = r
        
        residues[idx] = best_r
        start = p if best_r == 0 else best_r
        for x in range(start, L_MAX + 1, p):
            count[x] += 1
        
        while current_L + 1 <= L_MAX and count[current_L + 1] >= 2:
            current_L += 1
    
    return residues, current_L

def _sa_optimize(residues, deadline):
    """
    Simulated Annealing to refine residues.
    Objective: Maximize L, with a secondary objective to maximize coverage density just past L.
    """
    residues = list(residues)
    L_MAX_SEARCH = 10000 # Search space for SA evaluation
    count = _build_count_np(residues, L_MAX_SEARCH)
    current_L = _find_prefix_L_np(count, L_MAX_SEARCH)
    
    # Calculate initial potential: sum of counts in [L+1, L+50]
    def get_potential(count_arr, L_val):
        end = min(L_val + 50, L_MAX_SEARCH)
        if end <= L_val:
            return 0
        return np.sum(count_arr[L_val+1:end+1])

    current_potential = get_potential(count, current_L)
    
    T = 100.0
    T_min = 1.0
    cooling_rate = 0.995
    
    best_residues = list(residues)
    best_L = current_L
    best_potential = current_potential
    
    start_time = time.time()
    
    while time.time() < deadline and T > T_min:
        # Pick a random prime to change
        idx = random.randint(0, _NPRIMES - 1)
        p = _PRIMES[idx]
        old_r = residues[idx]
        
        if p == 2:
            new_r = 1 - old_r
        else:
            new_r = random.randint(0, p - 1)
            if new_r == old_r:
                continue
                
        # Apply move
        old_start = p if old_r == 0 else old_r
        new_start = p if new_r == 0 else new_r
        
        count[old_start:L_MAX_SEARCH+1:p] -= 1
        count[new_start:L_MAX_SEARCH+1:p] += 1
        
        new_L = _find_prefix_L_np(count, L_MAX_SEARCH)
        new_potential = get_potential(count, new_L)
        
        # Composite score: L is dominant, potential is secondary
        score_old = current_L + 0.1 * current_potential
        score_new = new_L + 0.1 * new_potential
        
        delta_score = score_new - score_old
        
        if delta_score >= 0:
            # Accept
            residues[idx] = new_r
            current_L = new_L
            current_potential = new_potential
            
            if new_L > best_L or (new_L == best_L and new_potential > best_potential):
                best_L = new_L
                best_potential = new_potential
                best_residues = list(residues)
        else:
            # Metropolis acceptance
            if random.random() < np.exp(delta_score / T):
                residues[idx] = new_r
                current_L = new_L
                current_potential = new_potential
            else:
                # Revert
                count[new_start:L_MAX_SEARCH+1:p] -= 1
                count[old_start:L_MAX_SEARCH+1:p] += 1
        
        T *= cooling_rate
        
    return best_residues, best_L

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill
    
    best_residues = None
    best_score = 0
    
    # 1. Greedy Initialization
    residues, current_L = _greedy_construct(deadline - 20)
    
    # 2. Simulated Annealing
    residues, sa_L = _sa_optimize(residues, deadline - 10)
    
    # 3. Final Score Calculation
    if time.time() < deadline - 5:
        actual_score = _score(residues)
        if actual_score > best_score:
            best_score = actual_score
            best_residues = list(residues)
            
    # 4. If time remains, try another restart with random init + SA
    if time.time() < deadline - 30:
        random.seed(123)
        residues_rand = [random.randint(0, _PRIMES[i] - 1) for i in range(_NPRIMES)]
        residues_rand, sa_L_rand = _sa_optimize(residues_rand, deadline - 10)
        if time.time() < deadline - 5:
            actual_score_rand = _score(residues_rand)
            if actual_score_rand > best_score:
                best_score = actual_score_rand
                best_residues = list(residues_rand)

    if best_residues is None:
        # Fallback: simple seed
        best_residues = [(i + 1) % p for i, p in enumerate(_PRIMES)]
    
    return best_residues
