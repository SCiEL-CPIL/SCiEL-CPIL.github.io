"""Optimized doubly-covered prefix: greedy construction + fast local search."""
import time
import random

def _primes_up_to(n):
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = [False] * len(sieve[i*i::i])
    return [i for i in range(n + 1) if sieve[i]]

_PRIMES = _primes_up_to(1000)
_NPRIMES = len(_PRIMES)

def _build_count(residues, L_MAX):
    """Build count array for x in [1, L_MAX]."""
    count = [0] * (L_MAX + 1)
    for i, p in enumerate(_PRIMES):
        r = residues[i]
        start = p if r == 0 else r
        for x in range(start, L_MAX + 1, p):
            count[x] += 1
    return count

def _find_prefix_L(count, L_MAX):
    """Find largest L such that count[x] >= 2 for all x in [1..L]."""
    L = 0
    for x in range(1, L_MAX + 1):
        if count[x] >= 2:
            L = x
        else:
            break
    return L

def _score(residues):
    """Compute the actual score (largest L with full double coverage)."""
    L_MAX = 100000
    count = _build_count(residues, L_MAX)
    return _find_prefix_L(count, L_MAX)

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

def _local_search(residues, deadline, L_MAX=3000):
    """
    Local search: try single-residue flips to extend the doubly-covered prefix.
    """
    residues = list(residues)
    count = _build_count(residues, L_MAX)
    current_L = _find_prefix_L(count, L_MAX)
    
    improved = True
    stall_count = 0
    max_stall = 50
    
    while improved and time.time() < deadline:
        improved = False
        stall_count += 1
        if stall_count > max_stall:
            break
        
        gap = current_L + 1
        if gap > L_MAX:
            break
        
        if count[gap] >= 2:
            while gap <= L_MAX and count[gap] >= 2:
                current_L = gap
                gap += 1
            improved = True
            stall_count = 0
            continue
        
        best_move = None
        best_gain = 0
        
        for i in range(_NPRIMES):
            p = _PRIMES[i]
            if p > 200:
                continue
            
            current_r = residues[i]
            desired_r = gap % p
            
            if current_r == desired_r:
                continue
            
            old_start = p if current_r == 0 else current_r
            safe = True
            for x in range(old_start, current_L + 1, p):
                if count[x] < 3:
                    safe = False
                    break
            
            if not safe:
                continue
            
            new_start = p if desired_r == 0 else desired_r
            
            for x in range(old_start, L_MAX + 1, p):
                count[x] -= 1
            for x in range(new_start, L_MAX + 1, p):
                count[x] += 1
            
            new_L = current_L
            while new_L + 1 <= L_MAX and count[new_L + 1] >= 2:
                new_L += 1
            
            for x in range(new_start, L_MAX + 1, p):
                count[x] -= 1
            for x in range(old_start, L_MAX + 1, p):
                count[x] += 1
            
            gain = new_L - current_L
            if gain > best_gain:
                best_gain = gain
                best_move = (i, desired_r, new_L)
        
        if best_move is None:
            for i in range(_NPRIMES):
                p = _PRIMES[i]
                if p <= 200:
                    continue
                
                current_r = residues[i]
                desired_r = gap % p
                
                if current_r == desired_r:
                    continue
                
                old_start = p if current_r == 0 else current_r
                safe = True
                for x in range(old_start, current_L + 1, p):
                    if count[x] < 3:
                        safe = False
                        break
                
                if not safe:
                    continue
                
                new_start = p if desired_r == 0 else desired_r
                
                for x in range(old_start, L_MAX + 1, p):
                    count[x] -= 1
                for x in range(new_start, L_MAX + 1, p):
                    count[x] += 1
                
                new_L = current_L
                while new_L + 1 <= L_MAX and count[new_L + 1] >= 2:
                    new_L += 1
                
                for x in range(new_start, L_MAX + 1, p):
                    count[x] -= 1
                for x in range(old_start, L_MAX + 1, p):
                    count[x] += 1
                
                gain = new_L - current_L
                if gain > best_gain:
                    best_gain = gain
                    best_move = (i, desired_r, new_L)
        
        if best_move is not None:
            i, new_r, new_L = best_move
            p = _PRIMES[i]
            old_r = residues[i]
            
            old_start = p if old_r == 0 else old_r
            new_start = p if new_r == 0 else new_r
            
            for x in range(old_start, L_MAX + 1, p):
                count[x] -= 1
            for x in range(new_start, L_MAX + 1, p):
                count[x] += 1
            
            residues[i] = new_r
            current_L = new_L
            improved = True
            stall_count = 0
        else:
            # Try random perturbation as escape
            i = random.randint(0, _NPRIMES - 1)
            p = _PRIMES[i]
            old_r = residues[i]
            new_r = random.randint(0, p - 1)
            if new_r != old_r:
                old_start = p if old_r == 0 else old_r
                new_start = p if new_r == 0 else new_r
                
                for x in range(old_start, L_MAX + 1, p):
                    count[x] -= 1
                for x in range(new_start, L_MAX + 1, p):
                    count[x] += 1
                
                new_L = _find_prefix_L(count, L_MAX)
                
                if new_L > current_L:
                    residues[i] = new_r
                    current_L = new_L
                    improved = True
                    stall_count = 0
                else:
                    for x in range(new_start, L_MAX + 1, p):
                        count[x] -= 1
                    for x in range(old_start, L_MAX + 1, p):
                        count[x] += 1
    
    return residues, current_L

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill
    
    best_residues = None
    best_score = 0
    
    # Multi-restart strategy: 3 independent greedy initializations
    for restart in range(3):
        if time.time() > deadline - 20:
            break
        
        random.seed(42 + restart * 1000)
        
        # Different greedy strategies per restart
        if restart == 0:
            # Standard greedy
            residues, current_L = _greedy_construct(deadline - 30)
        elif restart == 1:
            # Greedy with different candidate set
            residues = [0] * _NPRIMES
            L_MAX = 5000
            count = [0] * (L_MAX + 1)
            prime_order = sorted(range(_NPRIMES), key=lambda i: -_PRIMES[i])  # Large primes first
            current_L = 0
            for idx in prime_order:
                if time.time() > deadline - 30:
                    break
                p = _PRIMES[idx]
                best_r = 0
                best_score = -1
                gap = current_L + 1
                candidates = set()
                candidates.add(gap % p)
                for dx in range(-10, 11):
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
                    if gap % p == r:
                        coverage_gain += 50
                    if coverage_gain > best_score:
                        best_score = coverage_gain
                        best_r = r
                residues[idx] = best_r
                start = p if best_r == 0 else best_r
                for x in range(start, L_MAX + 1, p):
                    count[x] += 1
                while current_L + 1 <= L_MAX and count[current_L + 1] >= 2:
                    current_L += 1
        else:
            # Random initialization followed by local search
            residues = [random.randint(0, _PRIMES[i] - 1) for i in range(_NPRIMES)]
            count = _build_count(residues, 5000)
            current_L = _find_prefix_L(count, 5000)
        
        # Local search refinement
        residues, current_L = _local_search(residues, deadline - 10, L_MAX=3000)
        
        # Compute actual score
        if time.time() < deadline - 5:
            actual_score = _score(residues)
            if actual_score > best_score:
                best_score = actual_score
                best_residues = list(residues)
    
    if best_residues is None:
        # Fallback: simple seed
        best_residues = [(i + 1) % p for i, p in enumerate(_PRIMES)]
    
    return best_residues
