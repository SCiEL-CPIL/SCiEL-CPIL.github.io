import time
import random
import math

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    time_limit = 140.0  # Safety margin
    
    # Generate primes up to 1000
    primes = []
    for p in range(2, 1001):
        is_prime = True
        for q in range(2, int(math.isqrt(p)) + 1):
            if p % q == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(p)
    
    n_primes = len(primes)
    W = 2000  # Window size for coverage calculation
    
    def get_L(residues, counts):
        """
        Finds the largest L such that counts[x] >= 2 for all x in 1..L.
        Assumes counts is up to date.
        """
        for x in range(1, W + 1):
            if counts[x] < 2:
                return x - 1
        return W

    def update_counts(residues, counts, idx, old_r, new_r):
        """
        Updates counts array when residue at idx changes from old_r to new_r.
        """
        if old_r == new_r:
            return
        p = primes[idx]
        # Subtract old
        for x in range(old_r, W + 1, p):
            counts[x] -= 1
        # Add new
        for x in range(new_r, W + 1, p):
            counts[x] += 1

    # 1. Greedy Initialization
    residues = [0] * n_primes
    counts = [0] * (W + 1)
    
    # Greedy: For each prime, try to fix the first gap
    for i, p in enumerate(primes):
        # Find first x where counts[x] < 2
        x_fail = -1
        for x in range(1, W + 1):
            if counts[x] < 2:
                x_fail = x
                break
        
        if x_fail != -1:
            r = x_fail % p
        else:
            # If no gap, pick randomly or 0
            r = random.randint(0, p - 1)
        
        residues[i] = r
        # Update counts
        for x in range(r, W + 1, p):
            counts[x] += 1
            
    current_L = get_L(residues, counts)
    best_L = current_L
    best_residues = residues[:]
    
    # 2. Simulated Annealing / Local Search
    T = 10.0  # Initial temperature
    cooling_rate = 0.999
    
    while time.time() - start_time < time_limit:
        # Pick a random index
        i = random.randint(0, n_primes - 1)
        old_r = residues[i]
        p = primes[i]
        
        # Pick a random new residue
        new_r = random.randint(0, p - 1)
        if new_r == old_r:
            continue
            
        # Apply change
        update_counts(residues, counts, i, old_r, new_r)
        new_L = get_L(residues, counts)
        
        # Acceptance criterion
        if new_L > current_L:
            # Always accept improvement
            current_L = new_L
            residues[i] = new_r
            if new_L > best_L:
                best_L = new_L
                best_residues = residues[:]
        else:
            # Accept with probability exp((new_L - current_L) / T)
            # Since new_L <= current_L, delta is negative or zero
            delta = new_L - current_L
            if delta == 0:
                # Tie-breaking: accept to explore
                if random.random() < 0.5:
                    current_L = new_L
                    residues[i] = new_r
            else:
                prob = math.exp(delta / T)
                if random.random() < prob:
                    current_L = new_L
                    residues[i] = new_r
                else:
                    # Reject: revert change
                    update_counts(residues, counts, i, new_r, old_r)
        
        T *= cooling_rate
        if T < 0.1:
            T = 10.0  # Restart temperature occasionally? Or just keep it low.
            # Let's keep it low to converge, but restart if stuck?
            # For now, simple cooling.
            
    return best_residues
# EVOLVE-BLOCK-END
