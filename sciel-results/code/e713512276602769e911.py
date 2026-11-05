"""
Simulated Annealing solver for the covering problem.
Uses vectorized batch evaluation to efficiently explore the solution space.
"""
import random
import time
import numpy as np

def _get_primes(n):
    """Sieve of Eratosthenes to get primes up to n."""
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = [False] * len(sieve[i*i::i])
    return [i for i, is_prime in enumerate(sieve) if is_prime]

def _count_L(residues, primes, max_L):
    """
    Count the largest L such that every integer 1..L is covered at least twice.
    Vectorized count accumulation.
    """
    counts = np.zeros(max_L + 1, dtype=np.int32)
    for r, p in zip(residues, primes):
        if r == 0:
            start = p
        else:
            start = r
        if start > max_L:
            continue
        # Vectorized increment
        # Positions: start, start+p, ... <= max_L
        # np.arange(start, max_L + 1, p)
        # This creates a temporary array. For small p, this is large.
        # But max_L=100000, 168 primes.
        # Average p is ~500. 100000/500 = 200 elements per prime on average.
        # Total operations ~ 168 * 200 = 33600 additions. Very fast.
        pos = np.arange(start, max_L + 1, p)
        counts[pos] += 1
    
    # Find first index where count < 2
    # We care about 1..max_L
    # If all >= 2, return max_L
    # Else return first index - 1 (if index is 1, return 0? No, L is length. If 1 fails, L=0. But problem says 1..L. So if 1 is not covered twice, L=0.)
    # Actually, "largest L such that every integer 1..L satisfies".
    # If x=1 fails, L=0.
    # So we find the first x in 1..max_L where counts[x] < 2.
    # If found at x, L = x - 1.
    # If not found, L = max_L.
    
    # Efficient check:
    # below_two = np.where(counts[1:] < 2)[0]
    # if len(below_two) == 0: return max_L
    # else: return below_two[0]
    
    # Even faster:
    # min_count = np.min(counts[1:])
    # if min_count >= 2: return max_L
    # else:
    #    # find first
    #    idx = np.argmax(counts[1:] < 2)
    #    return idx
    
    # Let's use the where approach as it's robust
    below_two = np.where(counts[1:] < 2)[0]
    if len(below_two) == 0:
        return max_L
    else:
        return below_two[0]

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin for 150s limit
    
    primes = _get_primes(1000)
    n_primes = len(primes)
    max_L = 100000  # Cap for evaluation
    
    # Initialize
    # Start with a random or heuristic solution
    # Heuristic: r_p = p // 2 ? Or random?
    # Let's start with a decent heuristic: distribute residues somewhat evenly?
    # Or just random. SA is good with random starts.
    best_residues = [random.randint(0, p - 1) for p in primes]
    current_residues = best_residues[:]
    
    # Evaluate initial
    current_L = _count_L(current_residues, primes, max_L)
    best_L = current_L
    
    # Simulated Annealing parameters
    # Temperature schedule: T = T0 * (cooling_rate)^iteration
    T0 = 10.0
    cooling_rate = 0.999
    T = T0
    
    # Precompute positions for each prime to speed up evaluation?
    # We need to evaluate L for a given set of residues.
    # The bottleneck is _count_L.
    # Can we make it faster?
    # Using numpy is already fast.
    
    # To speed up SA, we can try to evaluate multiple candidates at once?
    # Or just accept the cost of _count_L per step.
    # How fast is _count_L?
    # 168 primes. Inner loop: arange + add.
    # Let's estimate: ~168 * (100000/500) = 33600 ops.
    # In Python/Numpy, this is likely < 1ms.
    # So we can do ~1000 steps/second.
    # 145s * 1000 = 145,000 steps. That's a lot of SA steps.
    
    # Let's run SA
    random.seed(42) # For reproducibility in debugging, but maybe not for final?
    # Actually, let's use system random for variety.
    
    while T > 0.1 and (time.time() - start_time) < deadline:
        # Pick a random prime index
        i = random.randint(0, n_primes - 1)
        p = primes[i]
        
        # Pick a new random residue
        new_r = random.randint(0, p - 1)
        if new_r == current_residues[i]:
            continue
            
        # Evaluate new state
        # To avoid copying the list every time, we can modify in place and revert
        old_r = current_residues[i]
        current_residues[i] = new_r
        
        new_L = _count_L(current_residues, primes, max_L)
        
        # Decide acceptance
        delta = new_L - current_L
        if delta > 0 or random.random() < np.exp(delta / T):
            # Accept
            current_L = new_L
            if new_L > best_L:
                best_L = new_L
                best_residues = current_residues[:]
        else:
            # Reject
            current_residues[i] = old_r
            
        T *= cooling_rate
        
        # Check time
        if time.time() - start_time > deadline - 5:
            break
            
    # Final polish: Hill climbing on the best solution found
    # Try to improve best_residues with single-step moves (greedy)
    improved = True
    while improved and (time.time() - start_time) < deadline - 2:
        improved = False
        for i in range(n_primes):
            if time.time() - start_time > deadline - 2:
                break
            p = primes[i]
            current_r = best_residues[i]
            
            # Try all residues? Or sample?
            # For small p, try all. For large p, sample.
            if p <= 50:
                candidates = [r for r in range(p) if r != current_r]
            else:
                # Sample 20 candidates
                candidates = [random.randint(0, p - 1) for _ in range(20)]
                candidates = list(set(candidates) - {current_r})
            
            best_new_r = current_r
            best_new_L = _count_L(best_residues, primes, max_L) # Cache current best L
            
            for new_r in candidates:
                if time.time() - start_time > deadline - 2:
                    break
                test_residues = best_residues[:]
                test_residues[i] = new_r
                L = _count_L(test_residues, primes, max_L)
                if L > best_new_L:
                    best_new_L = L
                    best_new_r = new_r
            
            if best_new_r != current_r:
                best_residues[i] = best_new_r
                improved = True
                
    return best_residues
