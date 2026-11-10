import random
import math
import time

def construct():
    # Generate primes up to 1000
    primes = [p for p in range(2, 1001) if all(p % q for q in range(2, int(p ** 0.5) + 1))]
    n_primes = len(primes)
    
    # We will search for a solution that covers up to M
    M = 2500
    
    # Initialize residues randomly
    residues = [random.randint(0, p - 1) for p in primes]
    
    def compute_counts(res):
        counts = [0] * (M + 1)
        for i in range(n_primes):
            p = primes[i]
            r = res[i]
            # Determine start index
            if r == 0:
                start = p
            else:
                start = r
            # Add counts
            for x in range(start, M + 1, p):
                counts[x] += 1
        return counts

    def get_score(counts):
        for x in range(1, M + 1):
            if counts[x] < 2:
                return x - 1
        return M

    # Initial counts and score
    counts = compute_counts(residues)
    current_score = get_score(counts)
    best_score = current_score
    best_residues = residues[:]
    
    # Simulated Annealing parameters
    T = 1.0
    cooling = 0.9995
    min_T = 0.001
    start_time = time.time()
    time_limit = 100.0 # Leave some buffer for 120s limit
    
    while T > min_T and (time.time() - start_time) < time_limit:
        # Determine move strategy
        # 80% Greedy (target current failure), 20% Random
        if random.random() < 0.8:
            # Find current failure
            x_fail = current_score + 1
            if x_fail > M:
                # We have covered M, try to extend? 
                # For now, just do random moves to explore
                i = random.randint(0, n_primes - 1)
                p = primes[i]
                old_r = residues[i]
                new_r = random.randint(0, p - 1)
            else:
                # Find primes that do NOT cover x_fail
                candidates = []
                for i in range(n_primes):
                    p = primes[i]
                    if x_fail % p != residues[i]:
                        candidates.append(i)
                
                if not candidates:
                    # All primes cover x_fail, but count is still < 2?
                    # This means count is 1 (since <2).
                    # This implies only 1 prime covers it.
                    # But we checked all primes. If all cover it, count >= n_primes.
                    # So this case is impossible if n_primes > 1.
                    # Wait, if x_fail % p == residues[i], it covers it.
                    # If count < 2, at most 1 prime covers it.
                    # So there are n_primes - 1 candidates.
                    i = random.randint(0, n_primes - 1)
                    p = primes[i]
                    old_r = residues[i]
                    new_r = random.randint(0, p - 1)
                else:
                    i = random.choice(candidates)
                    p = primes[i]
                    old_r = residues[i]
                    new_r = x_fail % p
        else:
            i = random.randint(0, n_primes - 1)
            p = primes[i]
            old_r = residues[i]
            new_r = random.randint(0, p - 1)
            
        if new_r == old_r:
            continue
            
        # Apply move
        residues[i] = new_r
        
        # Update counts
        # Decrement old
        start_old = old_r if old_r > 0 else p
        for x in range(start_old, M + 1, p):
            counts[x] -= 1
        # Increment new
        start_new = new_r if new_r > 0 else p
        for x in range(start_new, M + 1, p):
            counts[x] += 1
            
        new_score = get_score(counts)
        
        # Acceptance
        if new_score > current_score:
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_residues = residues[:]
        else:
            delta = new_score - current_score
            # Acceptance probability
            if random.random() < math.exp(delta / T):
                current_score = new_score
            else:
                # Revert
                residues[i] = old_r
                # Revert counts
                for x in range(start_new, M + 1, p):
                    counts[x] -= 1
                for x in range(start_old, M + 1, p):
                    counts[x] += 1
        
        T *= cooling

    return best_residues
