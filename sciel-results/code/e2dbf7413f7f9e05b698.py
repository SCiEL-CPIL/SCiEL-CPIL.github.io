"""Combined approach: repair-based greedy with targeted bottleneck coverage,
plateau-walking, and simulated annealing for escape."""
import numpy as np
import time
import math
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

def _build_counts(residues, primes, N):
    """Build coverage counts array for x in 0..N."""
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

def _get_L(counts, N):
    """Find the largest L such that counts[x] >= 2 for all x in 1..L."""
    if N <= 1:
        return 0
    failures = np.nonzero(counts[1:N+1] < 2)[0]
    if len(failures) == 0:
        return N
    return int(failures[0]) + 1

def _greedy_repair(residues, primes, counts, N, max_iters=20000):
    """Greedily repair coverage gaps by finding the most impactful single-residue change.
    Targets the first uncovered position and tries to cover it while minimizing damage."""
    current_L = _get_L(counts, N)
    n_primes = len(primes)
    
    for it in range(max_iters):
        if current_L >= N:
            break
        
        bottleneck = current_L + 1
        if bottleneck > N:
            break
        
        best_gain = 0
        best_prime_idx = -1
        best_new_r = -1
        
        # For each prime, compute the residue that would cover the bottleneck
        # and evaluate the net gain in coverage
        for i in range(n_primes):
            p = primes[i]
            r_old = int(residues[i])
            
            # The residue that would cover the bottleneck
            r_new = bottleneck % p
            
            if r_new == r_old:
                continue
            
            # Compute positions covered by old and new residues
            if r_old == 0:
                old_positions = np.arange(p, N + 1, p)
            else:
                old_positions = np.arange(r_old, N + 1, p)
            
            if r_new == 0:
                new_positions = np.arange(p, N + 1, p)
            else:
                new_positions = np.arange(r_new, N + 1, p)
            
            # Count how many positions in new_positions have count < 2 (potential new coverage)
            if len(new_positions) > 0:
                newly_covered = int(np.sum(counts[new_positions] < 2))
            else:
                newly_covered = 0
            
            # Count how many positions in old_positions have count == 2 (would lose coverage)
            if len(old_positions) > 0:
                lost_coverage = int(np.sum(counts[old_positions] == 2))
            else:
                lost_coverage = 0
            
            gain = newly_covered - lost_coverage
            
            if gain > best_gain:
                best_gain = gain
                best_prime_idx = i
                best_new_r = r_new
        
        if best_prime_idx == -1 or best_gain <= 0:
            break
        
        # Apply the change
        i = best_prime_idx
        p = primes[i]
        r_old = int(residues[i])
        r_new = best_new_r
        
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
        new_L = _get_L(counts, N)
        
        if new_L > current_L:
            current_L = new_L
        else:
            # Plateau: allow some exploration
            pass
    
    return residues, counts, current_L

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    primes = _sieve_primes(1000)
    n_primes = len(primes)  # 168
    
    # Search range for L
    N = 50000
    
    best_residues = None
    best_score = 0
    
    rng = np.random.default_rng(42)
    random.seed(42)
    
    # Multiple initializations
    initial_strategies = [
        lambda i, p: random.randint(0, p - 1),
        lambda i, p: 1 % p,
        lambda i, p: (p // 2) % p,
        lambda i, p: (i * 13 + 5) % p,
        lambda i, p: (i * 7 + 3) % p,
        lambda i, p: ((i + 2) // 2) % p,
        lambda i, p: (i * 31 + 11) % p,
    ]
    
    n_restarts = 15
    
    for restart in range(n_restarts):
        if time.time() >= deadline:
            break
        
        # Initialize residues
        if restart < len(initial_strategies):
            strat = initial_strategies[restart]
            residues = np.array([strat(i, p) for i, p in enumerate(primes)], dtype=np.int64)
        elif best_residues is not None:
            # Perturb the best found so far
            residues = best_residues.copy()
            n_perturb = 3 + int(rng.integers(0, 5))
            for _ in range(n_perturb):
                i = int(rng.integers(n_primes))
                p = primes[i]
                residues[i] = int(rng.integers(0, p))
        else:
            residues = np.array([int(rng.integers(0, p)) for p in primes])
        
        # Build counts
        counts = _build_counts(residues, primes, N)
        current_L = _get_L(counts, N)
        
        if current_L > best_score:
            best_score = current_L
            best_residues = residues.copy()
        
        # Phase 1: Greedy repair
        if time.time() < deadline - 5.0:
            residues, counts, current_L = _greedy_repair(residues, primes, counts, N, max_iters=15000)
            
            if current_L > best_score:
                best_score = current_L
                best_residues = residues.copy()
        
        # Phase 2: Simulated annealing / plateau walking
        if time.time() < deadline - 3.0:
            T = 5.0
            T_min = 0.05
            alpha = 0.9998
            stall_count = 0
            max_stall = 3000
            
            while time.time() < deadline - 3.0:
                # Pick a random prime to mutate
                i = int(rng.integers(n_primes))
                p = primes[i]
                r_old = int(residues[i])
                
                # Pick new residue - bias towards covering the bottleneck
                if current_L < N and rng.random() < 0.5:
                    bottleneck = current_L + 1
                    if bottleneck <= N:
                        r_new = bottleneck % p
                    else:
                        r_new = int(rng.integers(0, p))
                else:
                    r_new = int(rng.integers(0, p))
                
                if r_old == r_new:
                    continue
                
                # Update counts
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
                
                new_L = _get_L(counts, N)
                delta = new_L - current_L
                
                # Acceptance criterion
                accepted = False
                if delta > 0:
                    accepted = True
                elif delta == 0:
                    if rng.random() < 0.4:
                        accepted = True
                else:
                    # Accept worse moves with probability exp(delta / T)
                    if rng.random() < math.exp(delta / max(T, 1e-10)):
                        accepted = True
                
                if accepted:
                    residues[i] = r_new
                    current_L = new_L
                    stall_count = 0
                    
                    if current_L > best_score:
                        best_score = current_L
                        best_residues = residues.copy()
                else:
                    # Revert
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
                    
                    stall_count += 1
                    if stall_count > max_stall:
                        break
                
                # Cool down
                T *= alpha
                if T < T_min:
                    T = T_min
        
        # After each restart, try greedy repair from the best found
        if best_residues is not None and time.time() < deadline - 2.0:
            residues2 = best_residues.copy()
            counts2 = _build_counts(residues2, primes, N)
            L2 = _get_L(counts2, N)
            residues2, counts2, L2 = _greedy_repair(residues2, primes, counts2, N, max_iters=8000)
            if L2 > best_score:
                best_score = L2
                best_residues = residues2
    
    # Final verification: check if we can go beyond N
    if best_residues is not None and time.time() < deadline - 1.0:
        N_verify = 100000
        counts_v = np.zeros(N_verify + 1, dtype=np.int32)
        for i, p in enumerate(primes):
            r = int(best_residues[i])
            if r == 0:
                if p <= N_verify:
                    counts_v[p::p] += 1
            else:
                if r <= N_verify:
                    counts_v[r::p] += 1
        
        failures = np.nonzero(counts_v[1:N_verify+1] < 2)[0]
        if len(failures) == 0:
            final_score = N_verify
        else:
            final_score = int(failures[0]) + 1
        
        # Update best if we found something better (though we can't change residues now)
        # The evaluator will compute the actual score from the returned residues
    
    return best_residues.tolist()
