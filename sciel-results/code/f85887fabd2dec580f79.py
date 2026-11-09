import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s kill
    
    # Generate primes <= 1000
    def get_primes(limit):
        sieve = [True] * (limit + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(limit ** 0.5) + 1):
            if sieve[i]:
                for j in range(i * i, limit + 1, i):
                    sieve[j] = False
        return [i for i, is_p in enumerate(sieve) if is_p]
    
    primes = get_primes(1000)
    n_primes = len(primes)
    primes_arr = np.array(primes, dtype=np.int64)
    
    # We evaluate up to a large limit to ensure we can find improvements
    # The score is the largest L such that 1..L is doubly covered.
    MAX_EVAL = 20000
    
    xs = np.arange(1, MAX_EVAL + 1, dtype=np.int64)
    
    def evaluate(residues):
        """
        Computes the largest L such that every integer 1..L is covered by at least 2 congruences.
        Returns L (capped at MAX_EVAL).
        """
        diff = xs[None, :] - residues[:, None]
        mods = np.remainder(diff, primes_arr[:, None])
        hits = (mods == 0)
        coverage = np.sum(hits, axis=0)
        
        mask = coverage < 2
        if not np.any(mask):
            return MAX_EVAL
        first_fail_idx = np.argmax(mask)
        return int(xs[first_fail_idx] - 1)

    def get_coverage(residues):
        diff = xs[None, :] - residues[:, None]
        mods = np.remainder(diff, primes_arr[:, None])
        hits = (mods == 0)
        return np.sum(hits, axis=0)

    def calculate_energy(residues, target_score):
        """
        Energy function for MCMC.
        We want to maximize the prefix length L.
        Let L be the current score.
        Energy = (MAX_EVAL - L) + alpha * deficit_in_prefix
        
        The deficit in the prefix helps guide the search towards configurations
        that are "close" to being doubly covered in the entire prefix, 
        not just at the boundary.
        """
        L = evaluate(residues)
        if L >= MAX_EVAL:
            return 0.0
        
        coverage = get_coverage(residues)
        # Deficit in the prefix 1..L+1 (the failing part)
        # We care about the first failure and how "bad" it is.
        # A simple proxy: sum of max(0, 2 - coverage) for x in 1..L+5
        # This encourages fixing the immediate neighborhood of the failure.
        window = min(L + 10, MAX_EVAL)
        prefix_coverage = coverage[:window]
        deficit = np.sum(np.maximum(0, 2 - prefix_coverage))
        
        # Weight the prefix length heavily
        # If L is small, energy is high.
        # If L is large, energy is low.
        # Deficit adds a secondary gradient.
        return (MAX_EVAL - L) * 10.0 + deficit

    # Initial solution: Random residues
    residues = np.array([random.randint(0, p - 1) for p in primes], dtype=np.int64)
    
    current_score = evaluate(residues)
    best_residues = residues.copy()
    best_score = current_score
    
    # MCMC Parameters
    T = 50.0  # Initial temperature
    T_min = 1.0
    cooling = 0.9995
    restart_threshold = 500 # Steps without improvement before restart
    
    current_energy = calculate_energy(residues, best_score)
    best_energy = current_energy
    
    step_count = 0
    steps_since_improvement = 0
    
    while time.time() < deadline:
        # Pick a random prime to change
        idx = random.randint(0, n_primes - 1)
        old_res = residues[idx]
        
        # Choose a new residue. 
        # Strategy: Sometimes random, sometimes biased towards covering the current failure.
        if current_score < MAX_EVAL:
            target_x = current_score + 1
            # 50% chance to try to cover the target
            if random.random() < 0.5:
                new_res = target_x % primes[idx]
            else:
                new_res = random.randint(0, primes[idx] - 1)
        else:
            new_res = random.randint(0, primes[idx] - 1)
            
        if new_res == old_res:
            continue
            
        residues[idx] = new_res
        new_energy = calculate_energy(residues, best_score)
        
        # Metropolis-Hastings Acceptance
        delta_E = new_energy - current_energy
        if delta_E < 0 or random.random() < np.exp(-delta_E / T):
            # Accept
            current_energy = new_energy
            new_score = evaluate(residues)
            
            if new_score > best_score:
                best_score = new_score
                best_residues = residues.copy()
                best_energy = new_energy
                steps_since_improvement = 0
            else:
                steps_since_improvement += 1
        else:
            # Reject
            residues[idx] = old_res
            steps_since_improvement += 1
            
        step_count += 1
        
        # Cooling
        T *= cooling
        if T < T_min:
            T = T_min
            
        # Restart Strategy
        # If we are stuck (no improvement for a while) and T is low, 
        # we might be in a local minimum. 
        # We can "kick" the system by accepting a large energy increase 
        # or by resetting to a slightly perturbed version of the best solution.
        if steps_since_improvement > restart_threshold:
            # Check time
            if time.time() > deadline - 5:
                break
                
            # Strategy: Perturb the best solution found so far and restart the chain
            # This is a form of "iterated" MCMC.
            residues = best_residues.copy()
            
            # Apply a small random perturbation to escape the local minimum
            num_perturb = random.randint(3, 10)
            for _ in range(num_perturb):
                idx_p = random.randint(0, n_primes - 1)
                residues[idx_p] = random.randint(0, primes[idx_p] - 1)
                
            current_score = evaluate(residues)
            current_energy = calculate_energy(residues, best_score)
            steps_since_improvement = 0
            
            # Reset temperature to a moderate value to allow exploration
            T = max(T_min, T * 2.0)
            
            # If the perturbed solution is worse than best, that's expected.
            # We continue from here.
            
    return best_residues.tolist()
