import numpy as np
import time

def get_primes(n):
    """Sieve of Eratosthenes to get primes up to n."""
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n**0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = [False] * len(sieve[i*i::i])
    return [i for i, is_prime in enumerate(sieve) if is_prime]

def construct():
    primes = get_primes(1000)
    n_primes = len(primes)
    
    # Set a reasonably high L_max to allow growth
    L_max = 5000
    
    # Initialize residues
    # Use the seed's logic as a starting point, or random
    # Seed: r_i = ceil(i/2) % p
    best_residues = [(i + 2) // 2 % p for i, p in enumerate(primes)]
    
    def evaluate(residues):
        counts = np.zeros(L_max + 1, dtype=np.int16)
        for r, p in zip(residues, primes):
            if r == 0:
                if p <= L_max:
                    counts[p::p] += 1
            else:
                if r <= L_max:
                    counts[r:L_max+1:p] += 1
        
        # Find L: largest L such that counts[1..L] >= 2
        bad = counts[1:] < 2
        if not np.any(bad):
            L = L_max
        else:
            L = np.argmax(bad) # index in counts[1:], so L is the count of good numbers
        
        if L > 0:
            # Calculate min count in 1..L to use as tie-breaker/robustness metric
            min_count = np.min(counts[1:L+1])
        else:
            min_count = 0
            
        # Composite score: Prioritize L, then robustness
        # If L increases by 1, score jumps by 1000
        # If L same, robustness (min_count - 2) adds small value
        robustness = max(0, min_count - 2)
        return L * 1000 + robustness

    start_time = time.time()
    time_limit = 140.0
    
    current_residues = best_residues.copy()
    current_score = evaluate(current_residues)
    
    best_score = current_score
    best_residues = current_residues.copy()
    
    T = 10.0
    cooling = 0.99995
    
    while time.time() - start_time < time_limit:
        i = np.random.randint(n_primes)
        p = primes[i]
        
        old_r = current_residues[i]
        new_r = np.random.randint(p)
        
        if new_r == old_r:
            continue
            
        current_residues[i] = new_r
        new_score = evaluate(current_residues)
        
        if new_score > current_score:
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_residues = current_residues.copy()
        else:
            delta = new_score - current_score
            if delta < 0:
                # Accept with probability exp(delta / T)
                # delta is negative
                if np.random.rand() < np.exp(delta / T):
                    current_score = new_score
                else:
                    current_residues[i] = old_r
            else:
                # delta == 0, accept to explore
                current_score = new_score
        
        T *= cooling
        if T < 0.5:
            T = 10.0
            # Reset to best to escape deep local minima
            # This is a "restart" strategy
            current_residues = best_residues.copy()
            current_score = best_score

    return best_residues
