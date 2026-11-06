import numpy as np
import time
import random

def construct():
    # Generate primes <= 1000
    primes = [p for p in range(2, 1001) if all(p % q for q in range(2, int(p ** 0.5) + 1))]
    n_primes = len(primes)
    
    # We track coverage up to a limit. 
    # The max possible L is likely not huge, but we search up to 20000 to be safe.
    # The score is capped at 10^5, but 20000 is a reasonable search window for optimization.
    L_MAX = 20000
    counts = np.zeros(L_MAX + 1, dtype=np.int16)
    
    # --- Phase 1: Smart Initialization ---
    # Goal: Create a high initial L by greedily assigning residues to cover small integers.
    # We prioritize covering integers in [1, 1000] densely.
    
    residues = np.zeros(n_primes, dtype=np.int64)
    
    # We will process primes in increasing order.
    # For each prime, we pick the residue that covers the most "valuable" positions.
    # Value of a position x is higher if x is small.
    # Specifically, we want to fix the "gaps" in coverage.
    
    # To make it fast, we'll just look at the first 500 positions for the greedy choice.
    # This ensures the initial segment [1, 500] is as well-covered as possible.
    INIT_WINDOW = 500
    
    for i, p in enumerate(primes):
        best_r = 0
        best_score = -1
        
        # We can't iterate all p residues for large p if p is large? 
        # Max p is 997. 168 * 1000 = 168,000 iterations. 
        # Inside each, we check ~500/p positions. 
        # Total ops ~ sum(500/p) for p in primes ~ 500 * log(log(1000)) ~ 500 * 2 = 1000 ops per prime? 
        # No, sum(1/p) for primes up to N is log(log N). 
        # So total work is small.
        
        for r in range(p):
            score = 0
            # Check coverage in INIT_WINDOW
            # Positions covered: r, r+p, r+2p... <= INIT_WINDOW
            # Note: if r==0, positions are p, 2p...
            # If r>0, positions are r, r+p...
            
            start = r if r > 0 else p
            if start > INIT_WINDOW:
                # This residue covers nothing in the initial window
                # But it might cover later? We prioritize initial window.
                # Give it a very low score, maybe based on random tie-breaking later.
                score = 0 
            else:
                # Count how many positions in [1, INIT_WINDOW] are covered
                # and add weight based on current coverage (prefer filling gaps)
                for x in range(start, INIT_WINDOW + 1, p):
                    # If current coverage is 0, this is very good (starts coverage)
                    # If current coverage is 1, this is good (completes pair)
                    # If current coverage >= 2, this is less useful for extending L
                    if counts[x] == 0:
                        score += 10
                    elif counts[x] == 1:
                        score += 5
                    else:
                        score += 1
            
            # Add a small random factor to break ties and avoid deterministic local minima
            score += random.random() * 0.1
            
            if score > best_score:
                best_score = score
                best_r = r
        
        residues[i] = best_r
        # Update counts
        p = primes[i]
        r = best_r
        start = r if r > 0 else p
        if start <= L_MAX:
            counts[start:L_MAX+1:p] += 1

    def get_L():
        # Find the first index where count < 2
        # counts[0] is irrelevant, we care about 1..L_MAX
        fail = counts[1:] < 2
        if not np.any(fail):
            return L_MAX
        idx = np.argmax(fail)
        return idx

    current_L = get_L()
    best_L = current_L
    best_residues = residues.copy()
    
    start_time = time.time()
    time_limit = 145.0 # Safety margin
    
    # --- Phase 2: Simulated Annealing ---
    
    # Temperature schedule
    T = 20.0
    T_min = 0.1
    cooling = 0.9998
    
    while time.time() - start_time < time_limit:
        # Move Selection
        # 70% chance: Target the first gap (current_L + 1)
        # 30% chance: Random move
        
        if random.random() < 0.7:
            # Target the bottleneck
            target_x = current_L + 1
            if target_x > L_MAX:
                # We have covered everything up to L_MAX. 
                # Just do random moves to potentially find a better configuration 
                # that might extend further if we increased L_MAX, or just maintain.
                # For now, random move.
                i = random.randint(0, n_primes - 1)
                p = primes[i]
                r_new = random.randint(0, p - 1)
            else:
                # Pick a random prime
                i = random.randint(0, n_primes - 1)
                p = primes[i]
                # Set residue such that target_x is covered
                r_new = target_x % p
        else:
            i = random.randint(0, n_primes - 1)
            p = primes[i]
            r_new = random.randint(0, p - 1)
            
        r_old = residues[i]
        
        if r_new == r_old:
            continue
            
        # Update counts
        old_start = r_old if r_old > 0 else p
        new_start = r_new if r_new > 0 else p
        
        if old_start <= L_MAX:
            counts[old_start:L_MAX+1:p] -= 1
        if new_start <= L_MAX:
            counts[new_start:L_MAX+1:p] += 1
        
        new_L = get_L()
        
        if new_L > current_L:
            current_L = new_L
            residues[i] = r_new
            if current_L > best_L:
                best_L = current_L
                best_residues = residues.copy()
        else:
            delta = new_L - current_L
            # Accept with probability exp(delta / T)
            # If delta is very negative, prob is ~0
            if delta > -100 * T: # Optimization: avoid underflow/exp of large negative
                prob = np.exp(delta / T)
            else:
                prob = 0.0
            
            if random.random() < prob:
                current_L = new_L
                residues[i] = r_new
            else:
                # Revert
                if new_start <= L_MAX:
                    counts[new_start:L_MAX+1:p] -= 1
                if old_start <= L_MAX:
                    counts[old_start:L_MAX+1:p] += 1
                residues[i] = r_old
        
        T *= cooling
        if T < T_min:
            T = T_min

    return best_residues.tolist()
