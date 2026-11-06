import numpy as np
import time
import random

def _sieve_primes(limit):
    """Return list of primes <= limit using sieve of Eratosthenes."""
    if limit < 2:
        return []
    is_prime = bytearray(b'\x01') * (limit + 1)
    is_prime[0] = 0
    is_prime[1] = 0
    for i in range(2, int(limit ** 0.5) + 1):
        if is_prime[i]:
            step = i
            start = i * i
            is_prime[start:limit+1:step] = bytearray(len(range(start, limit+1, step)))
    return [i for i in range(2, limit + 1) if is_prime[i]]

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    primes = _sieve_primes(1000)
    n_primes = len(primes)
    
    N_max = 100000
    
    best_residues = None
    best_score = 0
    
    rng = np.random.default_rng(42)
    random.seed(42)
    
    def build_counts(residues, N):
        counts = np.zeros(N + 1, dtype=np.int32)
        for i, p in enumerate(primes):
            r = int(residues[i])
            if r == 0:
                if p <= N:
                    counts[p::p] += 1
            else:
                if r <= N:
                    counts[r::p] += 1
        return counts

    def get_score(counts, N):
        failures_mask = counts[1:N+1] < 2
        if not np.any(failures_mask):
            return N
        return int(np.argmax(failures_mask)) + 1

    def apply_move(counts, residues, i, r_old, r_new, N):
        p = primes[i]
        if r_old == 0:
            if p <= N:
                counts[p::p] -= 1
        else:
            if r_old <= N:
                counts[r_old::p] -= 1
                
        if r_new == 0:
            if p <= N:
                counts[p::p] += 1
        else:
            if r_new <= N:
                counts[r_new::p] += 1

    def revert_move(counts, residues, i, r_old, r_new, N):
        p = primes[i]
        if r_new == 0:
            if p <= N:
                counts[p::p] -= 1
        else:
            if r_new <= N:
                counts[r_new::p] -= 1
                
        if r_old == 0:
            if p <= N:
                counts[p::p] += 1
        else:
            if r_old <= N:
                counts[r_old::p] += 1

    # Initial Strategies
    def init_random():
        return np.array([rng.integers(0, p) for p in primes], dtype=np.int64)
    
    def init_zero():
        return np.zeros(n_primes, dtype=np.int64)
        
    def init_one():
        return np.ones(n_primes, dtype=np.int64)
        
    def init_half():
        return np.array([p // 2 for p in primes], dtype=np.int64)
        
    def init_greedy():
        # Greedy construction: for each prime, pick residue that covers the most uncovered numbers
        residues = np.zeros(n_primes, dtype=np.int64)
        N_init = 5000
        counts = np.zeros(N_init + 1, dtype=np.int32)
        
        # Sort primes by size (smaller primes cover more numbers)
        prime_order = list(range(n_primes))
        
        for i in prime_order:
            p = primes[i]
            # Try each possible residue and pick the one that maximizes coverage
            best_r = 0
            best_gain = -1
            
            # Sample a few residues for speed
            sample_residues = set()
            for _ in range(min(20, p)):
                sample_residues.add(int(rng.integers(0, p)))
            # Also include 0, 1, and some structured choices
            sample_residues.add(0)
            sample_residues.add(1)
            if p > 2:
                sample_residues.add(p // 2)
            
            for r in sample_residues:
                # Calculate how many new numbers this residue would cover
                if r == 0:
                    if p <= N_init:
                        new_covered = np.sum(counts[p::p] == 1)
                    else:
                        new_covered = 0
                else:
                    if r <= N_init:
                        new_covered = np.sum(counts[r::p] == 1)
                    else:
                        new_covered = 0
                
                if new_covered > best_gain:
                    best_gain = new_covered
                    best_r = r
            
            residues[i] = best_r
            p = primes[i]
            if best_r == 0:
                if p <= N_init:
                    counts[p::p] += 1
            else:
                if best_r <= N_init:
                    counts[best_r::p] += 1
        
        return residues
    
    def init_structured():
        # Use a structured pattern based on prime properties
        residues = np.zeros(n_primes, dtype=np.int64)
        for i, p in enumerate(primes):
            # Use a hash-like function based on prime index and value
            residues[i] = (i * 7919 + p * 104729) % p
        return residues
    
    initial_strategies = [
        init_random,
        init_zero,
        init_one,
        init_half,
        init_greedy,
        init_structured,
        lambda: np.array([(i * 13 + 5) % p for i, p in enumerate(primes)], dtype=np.int64),
        lambda: np.array([(i * 7 + 3) % p for i, p in enumerate(primes)], dtype=np.int64),
        lambda: np.array([(p * 3 + 1) % p for p in primes], dtype=np.int64),
    ]
    
    for strat in initial_strategies:
        if time.time() > deadline - 10.0:
            break
            
        residues = strat()
        N_cur = 5000
        counts = build_counts(residues, N_cur)
        current_score = get_score(counts, N_cur)
        
        if current_score > best_score:
            best_score = current_score
            best_residues = residues.copy()
            
        # Local Search with Dynamic Expansion
        stall_count = 0
        max_stall = 300
        
        while time.time() < deadline - 5.0:
            # If we covered the current window, try to expand
            if current_score == N_cur:
                if N_cur * 2 > N_max:
                    break
                    
                N_cur = min(N_cur * 2, N_max)
                counts = build_counts(residues, N_cur)
                current_score = get_score(counts, N_cur)
                
                if current_score > best_score:
                    best_score = current_score
                    best_residues = residues.copy()
                continue
            
            # Find the first failure
            x_fail = current_score
            
            # Strategy: Pick a prime that doesn't cover x_fail, prefer small primes
            # Small primes affect more numbers, so changing them has bigger impact
            
            # Find primes that don't cover x_fail
            non_covering = []
            for i in range(n_primes):
                p = primes[i]
                if x_fail % p != residues[i]:
                    non_covering.append(i)
            
            if len(non_covering) == 0:
                # All primes cover x_fail, need to do something else
                # Try a random perturbation
                i = int(rng.integers(n_primes))
                p = primes[i]
                r_old = int(residues[i])
                r_new = int(rng.integers(0, p))
                if r_new == r_old:
                    continue
                    
                apply_move(counts, residues, i, r_old, r_new, N_cur)
                new_score = get_score(counts, N_cur)
                
                if new_score > current_score:
                    accepted = True
                elif new_score == current_score:
                    accepted = rng.random() < 0.1
                else:
                    accepted = False
                    
                if accepted:
                    residues[i] = r_new
                    current_score = new_score
                    stall_count = 0
                    if current_score > best_score:
                        best_score = current_score
                        best_residues = residues.copy()
                else:
                    revert_move(counts, residues, i, r_old, r_new, N_cur)
                    stall_count += 1
                    if stall_count > max_stall:
                        break
                continue
            
            # Pick from non-covering primes, bias towards smaller primes
            # Use a weighted random selection
            weights = np.array([1.0 / (primes[i] + 1) for i in non_covering])
            weights = weights / weights.sum()
            idx = int(rng.choice(len(non_covering), p=weights))
            i = non_covering[idx]
            p = primes[i]
            r_old = int(residues[i])
            r_new = int(x_fail % p)
            
            apply_move(counts, residues, i, r_old, r_new, N_cur)
            new_score = get_score(counts, N_cur)
            
            if new_score > current_score:
                accepted = True
            elif new_score == current_score:
                accepted = rng.random() < 0.3
            else:
                accepted = False
            
            if accepted:
                residues[i] = r_new
                current_score = new_score
                stall_count = 0
                if current_score > best_score:
                    best_score = current_score
                    best_residues = residues.copy()
            else:
                revert_move(counts, residues, i, r_old, r_new, N_cur)
                stall_count += 1
                if stall_count > max_stall:
                    break

    # Final Verification
    if best_residues is not None and time.time() < deadline - 1.0:
        N_verify = 100000
        counts_v = build_counts(best_residues, N_verify)
        final_score = get_score(counts_v, N_verify)
        # Don't update best_residues, just verify

    return best_residues.tolist()
