"""Optimized doubly-covered prefix: greedy construction + fast local search with restarts."""
import time
import random

# EVOLVE-BLOCK-START

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
    L_MAX = 5000  # Working range for greedy
    count = [0] * (L_MAX + 1)
    
    # Track which primes have been "assigned"
    assigned = [False] * _NPRIMES
    
    # Process primes in order of increasing size (small primes give more coverage)
    prime_order = sorted(range(_NPRIMES), key=lambda i: _PRIMES[i])
    
    current_L = 0
    for idx in prime_order:
        if time.time() > deadline:
            break
        p = _PRIMES[idx]
        best_r = 0
        best_score = -1
        
        # Try candidate residues: focus on covering the gap
        # The gap is at current_L + 1
        gap = current_L + 1
        candidates = set()
        candidates.add(gap % p)
        candidates.add(0)
        candidates.add(1)
        candidates.add(2)
        # Also try residues that cover positions near the gap
        for dx in range(-10, 11):
            x = gap + dx
            if x >= 1:
                candidates.add(x % p)
        
        for r in candidates:
            if r >= p:
                continue
            # Simulate: what's the new prefix L if we set residue[idx] = r?
            # We need to check if this helps extend coverage
            # Quick heuristic: count how many positions in [1, L_MAX] this covers
            # and how many of those are currently under-covered
            start = p if r == 0 else r
            coverage_gain = 0
            for x in range(start, L_MAX + 1, p):
                if count[x] <= 1:
                    coverage_gain += 1
                elif count[x] == 2:
                    coverage_gain -= 0.5  # Slight penalty for over-covering already-covered
            
            # Prefer residues that cover the gap
            if gap % p == r:
                coverage_gain += 100
            
            if coverage_gain > best_score:
                best_score = coverage_gain
                best_r = r
        
        residues[idx] = best_r
        assigned[idx] = True
        
        # Update count
        start = p if best_r == 0 else best_r
        for x in range(start, L_MAX + 1, p):
            count[x] += 1
        
        # Update current_L
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
            # Extend
            while gap <= L_MAX and count[gap] >= 2:
                current_L = gap
                gap += 1
            improved = True
            stall_count = 0
            continue
        
        # Need to increase count[gap]. Try flipping residues of primes.
        best_move = None
        best_gain = 0
        
        # Prioritize small primes (more frequent coverage)
        for i in range(_NPRIMES):
            p = _PRIMES[i]
            if p > 200:  # Skip large primes for speed
                continue
            
            current_r = residues[i]
            desired_r = gap % p
            
            if current_r == desired_r:
                continue
            
            # Check safety: flipping i from current_r to desired_r
            # Removes coverage at positions x where x%p == current_r
            # Adds coverage at positions x where x%p == desired_r
            # For safety, all positions in [1, current_L] that lose coverage must have count >= 3
            
            old_start = p if current_r == 0 else current_r
            safe = True
            for x in range(old_start, current_L + 1, p):
                if count[x] < 3:
                    safe = False
                    break
            
            if not safe:
                continue
            
            # Simulate the flip
            new_start = p if desired_r == 0 else desired_r
            
            # Apply
            for x in range(old_start, L_MAX + 1, p):
                count[x] -= 1
            for x in range(new_start, L_MAX + 1, p):
                count[x] += 1
            
            # Find new L
            new_L = current_L
            while new_L + 1 <= L_MAX and count[new_L + 1] >= 2:
                new_L += 1
            
            # Undo
            for x in range(new_start, L_MAX + 1, p):
                count[x] -= 1
            for x in range(old_start, L_MAX + 1, p):
                count[x] += 1
            
            gain = new_L - current_L
            if gain > best_gain:
                best_gain = gain
                best_move = (i, desired_r, new_L)
        
        if best_move is None:
            # Try larger primes as fallback
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
    
    return residues, current_L

def _local_search_2flip(residues, deadline, L_MAX=3000):
    """
    2-flip local search: try pairs of residue flips to extend the doubly-covered prefix.
    This is more powerful than 1-flip but slower.
    """
    residues = list(residues)
    count = _build_count(residues, L_MAX)
    current_L = _find_prefix_L(count, L_MAX)
    
    improved = True
    stall_count = 0
    max_stall = 20
    
    while improved and time.time() < deadline:
        improved = False
        stall_count += 1
        if stall_count > max_stall:
            break
        
        gap = current_L + 1
        if gap > L_MAX:
            break
        
        if count[gap] >= 2:
            # Extend
            while gap <= L_MAX and count[gap] >= 2:
                current_L = gap
                gap += 1
            improved = True
            stall_count = 0
            continue
        
        # Need to increase count[gap]. Try pairs of flips.
        best_move = None
        best_gain = 0
        
        # Only try small primes for 2-flips to keep it fast
        small_primes = [i for i in range(_NPRIMES) if _PRIMES[i] <= 100]
        
        for i1 in range(len(small_primes)):
            if time.time() > deadline:
                break
            i = small_primes[i1]
            p = _PRIMES[i]
            current_r1 = residues[i]
            desired_r1 = gap % p
            
            if current_r1 == desired_r1:
                continue
            
            old_start1 = p if current_r1 == 0 else current_r1
            new_start1 = p if desired_r1 == 0 else desired_r1
            
            # Apply first flip
            for x in range(old_start1, L_MAX + 1, p):
                count[x] -= 1
            for x in range(new_start1, L_MAX + 1, p):
                count[x] += 1
            
            for i2 in range(i1 + 1, len(small_primes)):
                j = small_primes[i2]
                q = _PRIMES[j]
                current_r2 = residues[j]
                desired_r2 = gap % q
                
                if current_r2 == desired_r2:
                    continue
                
                old_start2 = q if current_r2 == 0 else current_r2
                new_start2 = q if desired_r2 == 0 else desired_r2
                
                # Check safety for both flips
                safe = True
                for x in range(old_start1, current_L + 1, p):
                    if count[x] < 3:
                        safe = False
                        break
                if safe:
                    for x in range(old_start2, current_L + 1, q):
                        if count[x] < 3:
                            safe = False
                            break
                
                if not safe:
                    continue
                
                # Apply second flip
                for x in range(old_start2, L_MAX + 1, q):
                    count[x] -= 1
                for x in range(new_start2, L_MAX + 1, q):
                    count[x] += 1
                
                # Find new L
                new_L = current_L
                while new_L + 1 <= L_MAX and count[new_L + 1] >= 2:
                    new_L += 1
                
                # Undo second flip
                for x in range(new_start2, L_MAX + 1, q):
                    count[x] -= 1
                for x in range(old_start2, L_MAX + 1, q):
                    count[x] += 1
                
                gain = new_L - current_L
                if gain > best_gain:
                    best_gain = gain
                    best_move = (i, desired_r1, j, desired_r2, new_L)
            
            # Undo first flip
            for x in range(new_start1, L_MAX + 1, p):
                count[x] -= 1
            for x in range(old_start1, L_MAX + 1, p):
                count[x] += 1
        
        if best_move is not None:
            i, r1, j, r2, new_L = best_move
            p = _PRIMES[i]
            q = _PRIMES[j]
            old_r1 = residues[i]
            old_r2 = residues[j]
            old_start1 = p if old_r1 == 0 else old_r1
            new_start1 = p if r1 == 0 else r1
            old_start2 = q if old_r2 == 0 else old_r2
            new_start2 = q if r2 == 0 else r2
            
            for x in range(old_start1, L_MAX + 1, p):
                count[x] -= 1
            for x in range(new_start1, L_MAX + 1, p):
                count[x] += 1
            for x in range(old_start2, L_MAX + 1, q):
                count[x] -= 1
            for x in range(new_start2, L_MAX + 1, q):
                count[x] += 1
            
            residues[i] = r1
            residues[j] = r2
            current_L = new_L
            improved = True
            stall_count = 0
    
    return residues, current_L

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Phase 1: Greedy construction
    residues, L = _greedy_construct(deadline - 60)
    best_residues = list(residues)
    best_L = L
    
    # Phase 2: Local search to extend
    if time.time() < deadline - 30:
        residues, L = _local_search(residues, deadline - 30, L_MAX=3000)
        if L > best_L:
            best_L = L
            best_residues = list(residues)
    
    # Phase 3: 2-flip local search if time permits
    if time.time() < deadline - 15:
        residues, L = _local_search_2flip(best_residues, deadline - 15, L_MAX=3000)
        if L > best_L:
            best_L = L
            best_residues = list(residues)
            
    # Phase 4: Random Restarts
    # Try to escape local optima by perturbing the best solution and re-optimizing
    rng = random.Random(42)
    while time.time() < deadline - 10:
        # Create a perturbed version of the best solution
        perturbed = list(best_residues)
        # Flip 5-10 random residues
        num_flips = rng.randint(5, 10)
        for _ in range(num_flips):
            idx = rng.randint(0, _NPRIMES - 1)
            p = _PRIMES[idx]
            perturbed[idx] = rng.randint(0, p - 1)
            
        # Run local search on the perturbed solution
        if time.time() > deadline - 10:
            break
            
        residues, L = _local_search(perturbed, deadline - 10, L_MAX=3000)
        
        # If we found a better solution, update best
        if L > best_L:
            best_L = L
            best_residues = list(residues)
            
    # Final verification with full range
    final_L = _score(best_residues)
    if final_L > best_L:
        best_L = final_L
    
    return best_residues

# EVOLVE-BLOCK-END
