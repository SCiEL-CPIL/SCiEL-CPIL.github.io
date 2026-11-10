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
_L_LIMIT = 10000

def _build_count(residues):
    count = np.zeros(_L_LIMIT + 1, dtype=np.int32)
    for i, p in enumerate(_PRIMES):
        r = residues[i]
        start = p if r == 0 else r
        count[start::p] += 1
    return count

def _find_L(count):
    below = np.where(count[1:_L_LIMIT+1] < 2)[0]
    if len(below) == 0:
        return _L_LIMIT
    return int(below[0])

def _greedy_first_bad_point():
    """Greedy: for each prime (small first), assign residue to cover current first bad point."""
    count = np.zeros(_L_LIMIT + 1, dtype=np.int32)
    residues = [0] * _NPRIMES
    
    # Process primes from smallest to largest
    for i in range(_NPRIMES):
        p = _PRIMES[i]
        # Find current first bad point
        below = np.where(count[1:_L_LIMIT+1] < 2)[0]
        if len(below) == 0:
            # All covered, just assign 0
            residues[i] = 0
            start = p
            count[start::p] += 1
            continue
        
        target = below[0] + 1  # the actual value (0-indexed array)
        # Assign residue so that target is covered
        r = target % p
        residues[i] = r
        start = p if r == 0 else r
        count[start::p] += 1
    
    return residues

def _greedy_repair(residues, count, max_rounds=50):
    """Iteratively find first bad point and reassign a prime to cover it."""
    current_L = _find_L(count)
    
    for _ in range(max_rounds):
        if current_L >= _L_LIMIT:
            break
        
        gap = current_L + 1
        if count[gap] >= 2:
            # Shouldn't happen if _find_L is correct, but advance
            while gap <= _L_LIMIT and count[gap] >= 2:
                current_L = gap
                gap += 1
            continue
        
        # Find best prime to reassign to cover gap
        best_i = -1
        best_cost = float('inf')
        
        for i in range(_NPRIMES):
            p = _PRIMES[i]
            new_r = gap % p
            old_r = residues[i]
            if new_r == old_r:
                continue
            
            start_old = p if old_r == 0 else old_r
            # Cost: how many points in [1, current_L] go from count 2 to count 1
            if start_old <= current_L:
                pos = np.arange(start_old, current_L + 1, p)
                cost = np.sum(count[pos] == 2)
            else:
                cost = 0
            
            if cost < best_cost:
                best_cost = cost
                best_i = i
                if cost == 0:
                    break
        
        if best_i == -1:
            break
        
        i = best_i
        p = _PRIMES[i]
        old_r = residues[i]
        new_r = gap % p
        start_old = p if old_r == 0 else old_r
        start_new = p if new_r == 0 else new_r
        
        count[start_old::p] -= 1
        count[start_new::p] += 1
        residues[i] = new_r
        
        new_L = _find_L(count)
        if new_L > current_L:
            current_L = new_L
        else:
            # Revert
            count[start_new::p] -= 1
            count[start_old::p] += 1
            residues[i] = old_r
            break
    
    return current_L, residues, count

def _simulated_annealing(residues, count, deadline):
    """SA with vectorized count updates."""
    current_L = _find_L(count)
    best_L = current_L
    best_residues = list(residues)
    
    T = 50.0
    T_min = 0.1
    cooling = 0.998
    
    while time.time() < deadline - 3 and T > T_min:
        i = random.randint(0, _NPRIMES - 1)
        p = _PRIMES[i]
        old_r = residues[i]
        new_r = random.randint(0, p - 1)
        
        if new_r == old_r:
            continue
        
        start_old = p if old_r == 0 else old_r
        start_new = p if new_r == 0 else new_r
        
        count[start_old::p] -= 1
        count[start_new::p] += 1
        
        new_L = _find_L(count)
        
        if new_L >= current_L:
            residues[i] = new_r
            current_L = new_L
            if current_L > best_L:
                best_L = current_L
                best_residues = list(residues)
        else:
            delta = new_L - current_L
            if random.random() < np.exp(delta / T):
                residues[i] = new_r
                current_L = new_L
            else:
                count[start_new::p] -= 1
                count[start_old::p] += 1
        
        T *= cooling
    
    return best_L, best_residues, count

def _targeted_local_search(residues, count, deadline):
    """Targeted search to fix gaps, focusing on bottleneck."""
    current_L = _find_L(count)
    best_L = current_L
    best_residues = list(residues)
    
    max_iters = 200
    for _ in range(max_iters):
        if time.time() > deadline - 3:
            break
        
        if current_L >= _L_LIMIT:
            break
        
        gap = current_L + 1
        if count[gap] >= 2:
            while gap <= _L_LIMIT and count[gap] >= 2:
                current_L = gap
                gap += 1
            if current_L > best_L:
                best_L = current_L
                best_residues = list(residues)
            continue
        
        # Find best move to cover gap
        best_move = None
        min_cost = float('inf')
        
        for i in range(_NPRIMES):
            p = _PRIMES[i]
            new_r = gap % p
            old_r = residues[i]
            if new_r == old_r:
                continue
            
            start_old = p if old_r == 0 else old_r
            if start_old > current_L:
                cost = 0
            else:
                pos = np.arange(start_old, current_L + 1, p)
                cost = np.sum(count[pos] == 2)
            
            if cost < min_cost:
                min_cost = cost
                best_move = (i, new_r)
                if cost == 0:
                    break
        
        if best_move is None:
            break
        
        i, new_r = best_move
        p = _PRIMES[i]
        old_r = residues[i]
        start_old = p if old_r == 0 else old_r
        start_new = p if new_r == 0 else new_r
        
        count[start_old::p] -= 1
        count[start_new::p] += 1
        residues[i] = new_r
        
        new_L = _find_L(count)
        
        if new_L > current_L:
            current_L = new_L
            if current_L > best_L:
                best_L = current_L
                best_residues = list(residues)
        else:
            count[start_new::p] -= 1
            count[start_old::p] += 1
            residues[i] = old_r
    
    return best_L, best_residues, count

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    random.seed(42)
    
    best_L_overall = 0
    best_residues_overall = None
    
    # Phase 1: Greedy first-bad-point initialization
    residues = _greedy_first_bad_point()
    count = _build_count(residues)
    L = _find_L(count)
    
    if L > best_L_overall:
        best_L_overall = L
        best_residues_overall = list(residues)
    
    # Phase 2: Greedy repair
    if time.time() < deadline - 30:
        L, residues, count = _greedy_repair(residues, count, max_rounds=100)
        if L > best_L_overall:
            best_L_overall = L
            best_residues_overall = list(residues)
    
    # Phase 3: SA
    if time.time() < deadline - 20:
        sa_deadline = min(time.time() + 40, deadline - 15)
        L_sa, residues_sa, count_sa = _simulated_annealing(residues, count, sa_deadline)
        if L_sa > best_L_overall:
            best_L_overall = L_sa
            best_residues_overall = list(residues_sa)
    
    # Phase 4: Targeted local search
    if time.time() < deadline - 10:
        L_ls, residues_ls, count_ls = _targeted_local_search(residues, count, deadline)
        if L_ls > best_L_overall:
            best_L_overall = L_ls
            best_residues_overall = list(residues_ls)
    
    # Phase 5: Another round of repair + SA with different seed
    if time.time() < deadline - 10:
        random.seed(123)
        residues2 = _greedy_first_bad_point()
        count2 = _build_count(residues2)
        L2 = _find_L(count2)
        
        if time.time() < deadline - 15:
            L2, residues2, count2 = _greedy_repair(residues2, count2, max_rounds=50)
        
        if time.time() < deadline - 10:
            sa_deadline2 = min(time.time() + 20, deadline - 5)
            L2, residues2, count2 = _simulated_annealing(residues2, count2, sa_deadline2)
        
        if L2 > best_L_overall:
            best_L_overall = L2
            best_residues_overall = list(residues2)
    
    if best_residues_overall is None:
        best_residues_overall = [0] * _NPRIMES
    
    return [int(r) for r in best_residues_overall]
