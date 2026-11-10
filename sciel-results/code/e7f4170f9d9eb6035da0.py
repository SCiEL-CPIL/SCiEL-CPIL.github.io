"""Optimized doubly-covered prefix: targeted greedy + bottleneck-focused local search."""
import time

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
    Greedy construction: fix small primes for dense coverage, then repair gaps.
    """
    residues = [0] * _NPRIMES
    L_MAX = 3000  # Working range for greedy
    count = [0] * (L_MAX + 1)
    
    # Phase A: Assign small primes (p <= 100) to maximize coverage in [1, 2000]
    small_prime_indices = [i for i in range(_NPRIMES) if _PRIMES[i] <= 100]
    
    for idx in small_prime_indices:
        if time.time() > deadline:
            break
        p = _PRIMES[idx]
        best_r = 0
        best_score = -1
        
        # For small primes, try all residues and pick the one that maximizes
        # coverage of positions with count <= 1 in [1, 2000]
        for r in range(p):
            start = p if r == 0 else r
            gain = 0
            for x in range(start, 2001, p):
                if count[x] <= 1:
                    gain += 1
                elif count[x] == 2:
                    gain -= 0.1  # Small penalty for over-covering
            if gain > best_score:
                best_score = gain
                best_r = r
        
        residues[idx] = best_r
        start = p if best_r == 0 else best_r
        for x in range(start, L_MAX + 1, p):
            count[x] += 1
    
    # Phase B: For larger primes, use coverage deficit heuristic
    # Find the first gap and assign residues to repair it
    large_prime_indices = [i for i in range(_NPRIMES) if _PRIMES[i] > 100]
    
    current_L = 0
    while current_L + 1 <= L_MAX and count[current_L + 1] >= 2:
        current_L += 1
    
    for idx in large_prime_indices:
        if time.time() > deadline:
            break
        p = _PRIMES[idx]
        gap = current_L + 1
        
        # Desired residue to cover the gap
        desired_r = gap % p
        
        # Check if this residue is safe (doesn't break existing coverage)
        safe = True
        old_r = residues[idx]
        old_start = p if old_r == 0 else old_r
        for x in range(old_start, current_L + 1, p):
            if count[x] < 3:
                safe = False
                break
        
        if safe:
            residues[idx] = desired_r
        else:
            # Try to find a safe residue that still helps
            best_r = old_r
            best_gain = 0
            for r in range(p):
                if r == old_r:
                    continue
                start = p if r == 0 else r
                gain = 0
                for x in range(start, L_MAX + 1, p):
                    if count[x] <= 1:
                        gain += 1
                if gain > best_gain:
                    # Check safety
                    s = True
                    for x in range(old_start, current_L + 1, p):
                        if count[x] < 3:
                            s = False
                            break
                    if s and gain > best_gain:
                        best_gain = gain
                        best_r = r
            residues[idx] = best_r
        
        # Update count
        old_r = residues[idx]
        start = p if old_r == 0 else old_r
        for x in range(start, L_MAX + 1, p):
            count[x] += 1
        
        # Extend current_L
        while current_L + 1 <= L_MAX and count[current_L + 1] >= 2:
            current_L += 1
    
    return residues, current_L

def _identify_bottleneck_primes(residues, count, current_L, L_MAX):
    """
    Identify primes that are critical for the bottleneck position (current_L + 1).
    A prime is critical if flipping it could potentially help cover the gap.
    """
    gap = current_L + 1
    if gap > L_MAX:
        return []
    
    critical = []
    for i in range(_NPRIMES):
        p = _PRIMES[i]
        r = residues[i]
        # This prime covers the gap if gap % p == r
        if gap % p == r:
            # Check if this prime is currently providing necessary coverage
            # i.e., if we remove it, count[gap] drops below 2
            if count[gap] <= 2:
                critical.append(i)
    
    # Also include primes that could be flipped to cover the gap
    for i in range(_NPRIMES):
        p = _PRIMES[i]
        r = residues[i]
        if gap % p != r:
            # This prime could be flipped to cover the gap
            # Check if it's safe to flip
            old_start = p if r == 0 else r
            safe = True
            for x in range(old_start, current_L + 1, p):
                if count[x] < 3:
                    safe = False
                    break
            if safe:
                critical.append(i)
    
    return critical

def _local_search_targeted(residues, deadline, L_MAX=5000):
    """
    Targeted local search: only consider flips that affect the bottleneck.
    """
    residues = list(residues)
    count = _build_count(residues, L_MAX)
    current_L = _find_prefix_L(count, L_MAX)
    
    improved = True
    stall_count = 0
    max_stall = 30
    
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
        
        # Identify critical primes for the bottleneck
        critical = _identify_bottleneck_primes(residues, count, current_L, L_MAX)
        
        if not critical:
            # No critical primes found, try all small primes
            critical = [i for i in range(_NPRIMES) if _PRIMES[i] <= 200]
        
        best_move = None
        best_gain = 0
        
        for i in critical:
            p = _PRIMES[i]
            current_r = residues[i]
            desired_r = gap % p
            
            if current_r == desired_r:
                continue
            
            # Check safety
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
            # Try 2-flips among critical primes
            if len(critical) >= 2:
                for i1 in range(len(critical)):
                    if time.time() > deadline:
                        break
                    i = critical[i1]
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
                    
                    for i2 in range(i1 + 1, len(critical)):
                        j = critical[i2]
                        q = _PRIMES[j]
                        current_r2 = residues[j]
                        desired_r2 = gap % q
                        
                        if current_r2 == desired_r2:
                            continue
                        
                        old_start2 = q if current_r2 == 0 else current_r2
                        new_start2 = q if desired_r2 == 0 else desired_r2
                        
                        # Check safety for both
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
            if len(best_move) == 3:
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
            else:
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

def _polish_pass(residues, deadline, L_MAX=10000):
    """
    Final polish pass with larger L_MAX to extend the prefix further.
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
            while gap <= L_MAX and count[gap] >= 2:
                current_L = gap
                gap += 1
            improved = True
            stall_count = 0
            continue
        
        # Try all primes, prioritizing small ones
        best_move = None
        best_gain = 0
        
        for i in range(_NPRIMES):
            p = _PRIMES[i]
            if p > 500:  # Skip very large primes for speed
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

def construct():
    start_time = time.time()
    deadline = start_time + 135  # Safety margin
    
    # Phase 1: Greedy construction
    residues, L = _greedy_construct(deadline - 30)
    best_residues = list(residues)
    best_L = L
    
    # Phase 2: Targeted local search
    if time.time() < deadline - 15:
        residues, L = _local_search_targeted(residues, deadline - 15, L_MAX=5000)
        if L > best_L:
            best_L = L
            best_residues = list(residues)
    
    # Phase 3: Polish pass with larger range
    if time.time() < deadline - 5:
        residues, L = _polish_pass(best_residues, deadline - 5, L_MAX=10000)
        if L > best_L:
            best_L = L
            best_residues = list(residues)
    
    # Final verification
    final_L = _score(best_residues)
    if final_L > best_L:
        best_L = final_L
    
    return best_residues

# EVOLVE-BLOCK-END
