"""Combined approach: numpy-vectorized simulated annealing with targeted bottleneck repair."""
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

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    primes = _sieve_primes(1000)
    n_primes = len(primes)  # 168
    primes_np = np.array(primes)
    
    # Search range for L - use a generous cap
    N = 20000
    
    best_residues = None
    best_score = 0
    
    # Simulated Annealing Parameters
    T = 50.0
    T_min = 0.5
    alpha = 0.9998
    
    rng = np.random.default_rng(42)
    
    while time.time() < deadline:
        # Initialize residues - use a heuristic initial assignment
        # Try to cover small integers: assign residues so that each prime covers at least one small number
        residues = np.array([int(rng.integers(0, p)) for p in primes])
        
        # Calculate initial counts using numpy slicing
        counts = np.zeros(N, dtype=np.int32)
        for i, p in enumerate(primes):
            r = residues[i]
            if r == 0:
                # x ≡ 0 (mod p) means x = p, 2p, 3p, ...
                # In 0-indexed array, indices p, 2p, ... correspond to x = p, 2p, ...
                # But we want x >= 1, and x=0 is index 0
                # x % p == 0 means x is a multiple of p
                # In the array, index 0 is x=0, index p is x=p, etc.
                # We want to mark x = p, 2p, 3p, ... (not x=0)
                # So start from index p
                if p < N:
                    counts[p::p] += 1
            else:
                # x ≡ r (mod p) means x = r, r+p, r+2p, ...
                # Index r corresponds to x=r (if r >= 1)
                # But if r == 0, we handled above
                if r < N:
                    counts[r::p] += 1
        
        # Calculate initial score
        def get_score(counts):
            # We care about x=1..N
            # Index 0 in counts corresponds to x=0, which is not in 1..L
            # So we check indices 1..N-1 (since counts is 0..N-1)
            # Actually, counts has N elements: indices 0 to N-1
            # x=1 corresponds to index 1, x=N-1 corresponds to index N-1
            # We want the largest L such that counts[x] >= 2 for all x in 1..L
            # But we only have counts up to index N-1, which is x=N-1
            # So max possible L from this array is N-1
            # Let's find the first x where counts[x] < 2
            # We need to check x=1, 2, 3, ...
            # If all counts[1:N] >= 2, then L >= N-1 (or more, but we can't tell from this array)
            # For simplicity, let's say if all covered, score = N
            # Otherwise, score = first failure index
            failures = np.nonzero(counts[1:N] < 2)[0]
            if len(failures) == 0:
                return N
            return failures[0] + 1
        
        current_score = get_score(counts)
        
        if current_score > best_score:
            best_score = current_score
            best_residues = residues.copy()
        
        # Annealing loop
        while time.time() < deadline:
            # Pick a random prime
            i = int(rng.integers(n_primes))
            p = primes[i]
            r_old = residues[i]
            
            # Pick a new residue
            # With some probability, target the bottleneck
            if best_score < N and rng.random() < 0.3:
                # Target the bottleneck: try to cover the first uncovered point
                bottleneck = best_score + 1
                if bottleneck < N:
                    r_new = bottleneck % p
                else:
                    r_new = int(rng.integers(0, p))
            else:
                r_new = int(rng.integers(0, p))
            
            if r_old == r_new:
                continue
                
            # Update counts
            if r_old == 0:
                if p < N:
                    counts[p::p] -= 1
            else:
                if r_old < N:
                    counts[r_old::p] -= 1
            
            if r_new == 0:
                if p < N:
                    counts[p::p] += 1
            else:
                if r_new < N:
                    counts[r_new::p] += 1
            
            new_score = get_score(counts)
            delta = new_score - current_score
            
            # Acceptance probability
            if delta > 0 or rng.random() < math.exp(delta / T):
                current_score = new_score
                residues[i] = r_new
                if current_score > best_score:
                    best_score = current_score
                    best_residues = residues.copy()
            else:
                # Revert
                if r_new == 0:
                    if p < N:
                        counts[p::p] -= 1
                else:
                    if r_new < N:
                        counts[r_new::p] -= 1
                
                if r_old == 0:
                    if p < N:
                        counts[p::p] += 1
                else:
                    if r_old < N:
                        counts[r_old::p] += 1
                
            # Cool down
            T *= alpha
            if T < T_min:
                T = T_min
        
        # Reset temperature for next restart
        T = 50.0 * (0.9 ** (best_score / 1000))  # Lower initial T as we make progress
        if T < 1.0:
            T = 1.0
    
    return best_residues.tolist()
