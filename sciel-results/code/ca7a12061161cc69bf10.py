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
    
    # Dynamic window strategy
    # Start small to explore quickly, expand if we cover the window.
    N_current = 5000
    N_max = 100000
    N_expansion_factor = 2
    
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
        # Find first index >= 1 where counts < 2
        # If no failure, return N
        failures_mask = counts[1:N+1] < 2
        if not np.any(failures_mask):
            return N
        return int(np.argmax(failures_mask)) + 1

    # Precompute coverage sets for each prime residue? No, too big.
    # Precompute which primes cover which x? 
    # For a given x, the primes that cover it are those where x % p == r_p.
    
    # Initial Strategies
    def init_random():
        return np.array([rng.integers(0, p) for p in primes], dtype=np.int64)
    
    def init_zero():
        return np.zeros(n_primes, dtype=np.int64)
        
    def init_one():
        return np.ones(n_primes, dtype=np.int64)
        
    def init_half():
        return np.array([p // 2 for p in primes], dtype=np.int64)
        
    initial_strategies = [
        init_random,
        init_zero,
        init_one,
        init_half,
        lambda: np.array([(i * 13 + 5) % p for i, p in enumerate(primes)], dtype=np.int64),
        lambda: np.array([(i * 7 + 3) % p for i, p in enumerate(primes)], dtype=np.int64),
    ]
    
    # Helper to apply a move and update counts
    def apply_move(counts, residues, i, r_new, N):
        p = primes[i]
        r_old = int(residues[i])
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
        residues[i] = r_new

    # Helper to revert a move
    def revert_move(counts, residues, i, r_old, N):
        p = primes[i]
        r_new = int(residues[i])
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
        residues[i] = r_old

    for strat in initial_strategies:
        if time.time() > deadline - 10.0:
            break
            
        residues = strat()
        N_cur = N_current
        counts = build_counts(residues, N_cur)
        current_score = get_score(counts, N_cur)
        
        if current_score > best_score:
            best_score = current_score
            best_residues = residues.copy()
            
        # Local Search with Dynamic Expansion
        stall_count = 0
        max_stall = 500 # Increased stall limit to allow more exploration
        
        while time.time() < deadline - 5.0:
            # If we covered the current window, try to expand
            if current_score == N_cur:
                # Expand window
                if N_cur * N_expansion_factor > N_max:
                    break # Reached max size, can't expand further
                    
                N_cur = min(N_cur * N_expansion_factor, N_max)
                counts = build_counts(residues, N_cur)
                current_score = get_score(counts, N_cur)
                
                if current_score > best_score:
                    best_score = current_score
                    best_residues = residues.copy()
                
                # If we still cover the new window, loop will expand again
                continue
            
            # Find the first failure
            x_fail = current_score
            
            # Identify primes that do NOT cover x_fail
            # We want to pick one of these to add coverage to x_fail.
            # Let's find indices of such primes.
            non_covering_indices = []
            for i in range(n_primes):
                p = primes[i]
                r = int(residues[i])
                if x_fail % p != r:
                    non_covering_indices.append(i)
            
            if not non_covering_indices:
                # All primes cover x_fail. This means count[x_fail] >= 168.
                # This is impossible for x_fail to be a failure (needs count < 2).
                # So this branch should never be reached if current_score is correct.
                # But just in case, pick a random move.
                i = int(rng.integers(n_primes))
                r_new = int(rng.integers(0, primes[i]))
                r_old = int(residues[i])
                if r_new == r_old:
                    continue
                apply_move(counts, residues, i, r_new, N_cur)
                new_score = get_score(counts, N_cur)
                if new_score > current_score:
                    current_score = new_score
                    stall_count = 0
                    if current_score > best_score:
                        best_score = current_score
                        best_residues = residues.copy()
                else:
                    revert_move(counts, residues, i, r_old, N_cur)
                    stall_count += 1
                    if stall_count > max_stall:
                        break
                continue
            
            # Pick a random non-covering prime
            i = non_covering_indices[int(rng.integers(len(non_covering_indices)))]
            p = primes[i]
            r_new = int(x_fail % p)
            r_old = int(residues[i])
            
            # Apply move
            apply_move(counts, residues, i, r_new, N_cur)
            new_score = get_score(counts, N_cur)
            
            # Acceptance
            if new_score > current_score:
                accepted = True
            elif new_score == current_score:
                if rng.random() < 0.3:
                    accepted = True
                else:
                    accepted = False
            else:
                accepted = False
            
            if accepted:
                current_score = new_score
                stall_count = 0
                if current_score > best_score:
                    best_score = current_score
                    best_residues = residues.copy()
            else:
                revert_move(counts, residues, i, r_old, N_cur)
                stall_count += 1
                if stall_count > max_stall:
                    break

    # Final Verification
    if best_residues is not None and time.time() < deadline - 1.0:
        N_verify = 100000
        counts_v = build_counts(best_residues, N_verify)
        final_score = get_score(counts_v, N_verify)
        
    return best_residues.tolist()
