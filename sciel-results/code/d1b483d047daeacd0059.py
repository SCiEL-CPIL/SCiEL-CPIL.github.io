"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def _sieve_primes(limit):
    """Return list of primes up to limit using Sieve of Eratosthenes."""
    if limit < 2:
        return []
    is_prime = np.ones(limit + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(limit ** 0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    return [i for i in range(2, limit + 1) if is_prime[i]]

def _get_L(counts, max_check=100000):
    """Find the largest L such that counts[1..L] >= 2."""
    below_two = np.where(counts[1:max_check+1] < 2)[0]
    if len(below_two) == 0:
        return max_check
    return int(below_two[0])

def _greedy_construct(primes, order, max_check=100000):
    """
    Greedy set-cover heuristic.
    For each prime in 'order', pick residue to cover the first gap.
    """
    counts = np.zeros(max_check + 1, dtype=np.int32)
    residues = np.zeros(len(primes), dtype=np.int32)
    current_L = 0
    
    prime_to_idx = {p: i for i, p in enumerate(primes)}
    
    for p in order:
        if current_L < max_check:
            start_search = current_L + 1
            if start_search > max_check:
                g1 = max_check + 1
            else:
                segment = counts[start_search:]
                below_two = np.where(segment < 2)[0]
                if len(below_two) == 0:
                    g1 = max_check + 1
                else:
                    g1 = start_search + int(below_two[0])
        else:
            g1 = max_check + 1
            
        if g1 > max_check:
            r = 0
        else:
            r = g1 % p
        
        idx = prime_to_idx[p]
        residues[idx] = r
        start = r if r > 0 else p
        if start <= max_check:
            counts[start::p] += 1
            
        new_L = _get_L(counts, max_check)
        if new_L > current_L:
            current_L = new_L
        else:
            current_L = new_L
            
    return residues, current_L

def _fast_hill_climb(residues, primes, counts, current_L, max_check=100000, deadline=None, allow_neutral=False):
    """
    Fast local search using incremental gap logic.
    allow_neutral: if True, accept moves that don't decrease L (helps escape local minima).
    """
    n_primes = len(primes)
    improved = True
    passes = 0
    
    while improved and passes < 50:
        passes += 1
        improved = False
        
        for i in range(n_primes):
            if deadline and time.time() > deadline:
                return residues, counts, current_L
                
            p = primes[i]
            current_r = residues[i]
            
            if p <= 40:
                candidates = list(range(p))
            else:
                candidates = [current_r]
                
                if current_L < max_check:
                    start_search = current_L + 1
                    segment = counts[start_search:]
                    below_two = np.where(segment < 2)[0]
                    if len(below_two) > 0:
                        g1 = start_search + int(below_two[0])
                        target_r = g1 % p
                        if target_r != current_r:
                            candidates.append(target_r)
                        
                        # Cover positions around the first gap and beyond
                        for offset in [1, 2, 3, 4, 5, 7, 10, 15, 20, 30]:
                            pos = g1 + offset
                            if pos <= max_check:
                                r_off = pos % p
                                if r_off not in candidates:
                                    candidates.append(r_off)
                            
                            pos2 = current_L + offset
                            if pos2 <= max_check:
                                r_off2 = pos2 % p
                                if r_off2 not in candidates:
                                    candidates.append(r_off2)
                
                for _ in range(12):
                    r_rand = np.random.randint(0, p)
                    if r_rand not in candidates:
                        candidates.append(r_rand)
                
                if len(candidates) > 30:
                    candidates = candidates[:30]
                    
            best_r = current_r
            best_L = current_L
            
            for r in candidates:
                if r == current_r:
                    continue
                
                old_start = current_r if current_r > 0 else p
                if old_start <= max_check:
                    counts[old_start::p] -= 1
                
                new_start = r if r > 0 else p
                if new_start <= max_check:
                    counts[new_start::p] += 1
                
                new_L = _get_L(counts, max_check)
                
                if new_L > best_L or (allow_neutral and new_L == best_L and r != current_r):
                    best_L = new_L
                    best_r = r
                else:
                    counts[new_start::p] -= 1
                    if old_start <= max_check:
                        counts[old_start::p] += 1
                    
            if best_r != current_r:
                residues[i] = best_r
                current_L = best_L
                improved = True
                
    return residues, counts, current_L

def _targeted_gap_fix(residues, primes, counts, current_L, max_check=100000, deadline=None):
    """
    Targeted pass: find the first gap and try to fix it by adjusting one prime's residue.
    Picks the BEST single move that improves L, rather than the first one.
    """
    if current_L >= max_check:
        return residues, counts, current_L, False
    
    start_search = current_L + 1
    segment = counts[start_search:]
    below_two = np.where(segment < 2)[0]
    if len(below_two) == 0:
        return residues, counts, current_L, False
    
    g1 = start_search + int(below_two[0])
    
    n_primes = len(primes)
    best_L = current_L
    best_i = -1
    best_r = -1
    
    for i in range(n_primes):
        if deadline and time.time() > deadline:
            break
            
        p = primes[i]
        current_r = residues[i]
        
        target_r = g1 % p
        
        if target_r == current_r:
            continue
        
        old_start = current_r if current_r > 0 else p
        if old_start <= max_check:
            counts[old_start::p] -= 1
        
        new_start = target_r if target_r > 0 else p
        if new_start <= max_check:
            counts[new_start::p] += 1
        
        new_L = _get_L(counts, max_check)
        
        if new_L > best_L:
            best_L = new_L
            best_i = i
            best_r = target_r
            
        counts[new_start::p] -= 1
        if old_start <= max_check:
            counts[old_start::p] += 1

    if best_i != -1:
        p = primes[best_i]
        current_r = residues[best_i]
        
        old_start = current_r if current_r > 0 else p
        if old_start <= max_check:
            counts[old_start::p] -= 1
        
        new_start = best_r if best_r > 0 else p
        if new_start <= max_check:
            counts[new_start::p] += 1
            
        residues[best_i] = best_r
        current_L = best_L
        
    return residues, counts, current_L, best_i != -1

def _polish_pass(residues, primes, counts, current_L, max_check=100000, deadline=None):
    """
    Final high-precision polish: try all residues for small primes and targeted moves for large primes.
    """
    n_primes = len(primes)
    
    # Phase 1: For primes <= 50, try all residues
    for i in range(n_primes):
        if deadline and time.time() > deadline:
            break
        p = primes[i]
        if p > 50:
            continue
        current_r = residues[i]
        best_r = current_r
        best_L = current_L
        
        for r in range(p):
            if r == current_r:
                continue
            
            old_start = current_r if current_r > 0 else p
            if old_start <= max_check:
                counts[old_start::p] -= 1
            
            new_start = r if r > 0 else p
            if new_start <= max_check:
                counts[new_start::p] += 1
            
            new_L = _get_L(counts, max_check)
            
            if new_L > best_L:
                best_L = new_L
                best_r = r
            else:
                counts[new_start::p] -= 1
                if old_start <= max_check:
                    counts[old_start::p] += 1
        
        if best_r != current_r:
            residues[i] = best_r
            current_L = best_L
    
    # Phase 2: Targeted moves for larger primes
    for i in range(n_primes):
        if deadline and time.time() > deadline:
            break
        p = primes[i]
        if p <= 50:
            continue
        current_r = residues[i]
        best_r = current_r
        best_L = current_L
        
        # Generate candidates around first gap
        if current_L < max_check:
            start_search = current_L + 1
            segment = counts[start_search:]
            below_two = np.where(segment < 2)[0]
            if len(below_two) > 0:
                g1 = start_search + int(below_two[0])
                for offset in range(0, 50):
                    r_cand = (g1 + offset) % p
                    if r_cand == current_r:
                        continue
                    
                    old_start = current_r if current_r > 0 else p
                    if old_start <= max_check:
                        counts[old_start::p] -= 1
                    
                    new_start = r_cand if r_cand > 0 else p
                    if new_start <= max_check:
                        counts[new_start::p] += 1
                    
                    new_L = _get_L(counts, max_check)
                    
                    if new_L > best_L:
                        best_L = new_L
                        best_r = r_cand
                    else:
                        counts[new_start::p] -= 1
                        if old_start <= max_check:
                            counts[old_start::p] += 1
        
        if best_r != current_r:
            residues[i] = best_r
            current_L = best_L
    
    return residues, counts, current_L

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    primes = _sieve_primes(1000)
    n_primes = len(primes)
    max_check = 100000
    
    best_residues = None
    best_score = 0
    
    orders = [
        primes,
        primes[::-1],
        list(np.random.permutation(primes)),
        list(np.random.permutation(primes))
    ]
    
    for order in orders:
        if time.time() > deadline:
            break
            
        residues, L = _greedy_construct(primes, order, max_check)
        
        counts = np.zeros(max_check + 1, dtype=np.int32)
        for r, p in zip(residues, primes):
            start = r if r > 0 else p
            if start <= max_check:
                counts[start::p] += 1
        
        # Iterative refinement with neutral moves to escape local minima
        for _ in range(8):
            if time.time() > deadline - 10:
                break
                
            residues, counts, L, improved = _targeted_gap_fix(residues, primes, counts, L, max_check, deadline)
            
            # Allow neutral moves in later passes to escape plateaus
            allow_neutral = _ >= 2
            residues, counts, L = _fast_hill_climb(residues, primes, counts, L, max_check, deadline, allow_neutral=allow_neutral)
            
            if not improved and time.time() > deadline - 40:
                break

        # Final polish pass
        if time.time() < deadline - 15:
            residues, counts, L = _polish_pass(residues, primes, counts, L, max_check, deadline)
            
        if L > best_score:
            best_score = L
            best_residues = residues.copy()
            
    if best_residues is None:
        best_residues = np.array([((i + 2) // 2) % p for i, p in enumerate(primes)], dtype=np.int32)
        
    return best_residues.tolist()
# EVOLVE-BLOCK-END
