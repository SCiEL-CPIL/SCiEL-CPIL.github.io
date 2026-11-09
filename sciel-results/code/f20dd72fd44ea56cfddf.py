import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 118.0

    def get_primes(limit):
        sieve = [True] * (limit + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(limit**0.5) + 1):
            if sieve[i]:
                for j in range(i*i, limit+1, i):
                    sieve[j] = False
        return [i for i, is_prime in enumerate(sieve) if is_prime]

    primes = get_primes(1000)
    num_primes = len(primes)
    primes_arr = np.array(primes, dtype=np.int64)

    # We evaluate up to a large limit to ensure we can find improvements
    MAX_EVAL = 50000
    xs = np.arange(1, MAX_EVAL + 1, dtype=np.int64)

    # Precompute x % p for each prime to speed up coverage calculations
    # x_mod_p[i] is an array of length MAX_EVAL where x_mod_p[i][j] = xs[j] % primes[i]
    x_mod_p = np.zeros((num_primes, MAX_EVAL), dtype=np.int64)
    for i in range(num_primes):
        x_mod_p[i] = xs % primes[i]

    def evaluate(residues):
        """
        Computes the largest L such that every integer 1..L is covered by at least 2 congruences.
        Returns L (capped at MAX_EVAL).
        """
        # Vectorized calculation of coverage
        # hits[i, j] is True if xs[j] == residues[i] (mod primes[i])
        # residues[:, None] has shape (num_primes, 1)
        # x_mod_p has shape (num_primes, MAX_EVAL)
        hits = (x_mod_p == residues[:, None])
        coverage = np.sum(hits, axis=0)

        # Find the first index where coverage < 2
        mask = coverage < 2
        if not np.any(mask):
            return MAX_EVAL
        first_fail_idx = np.argmax(mask)
        return int(xs[first_fail_idx] - 1)

    def get_coverage(residues):
        """
        Returns the coverage array for all x in 1..MAX_EVAL.
        """
        hits = (x_mod_p == residues[:, None])
        return np.sum(hits, axis=0)

    def check_prefix(residues, limit):
        """
        Checks if all x in 1..limit have coverage >= 2.
        """
        if limit <= 0:
            return True
        hits = (x_mod_p[:, :limit] == residues[:, None])
        coverage = np.sum(hits, axis=0)
        return np.all(coverage >= 2)

    # Initial solution: Random residues
    residues = np.array([random.randint(0, p - 1) for p in primes], dtype=np.int64)
    
    current_L = evaluate(residues)
    best_residues = residues.copy()
    best_L = current_L

    # Conflict-Driven Local Repair
    # Strategy:
    # 1. Find current L.
    # 2. Let x_fail = L + 1.
    # 3. Identify primes that do NOT cover x_fail.
    # 4. Try to change one such prime's residue to x_fail % p.
    # 5. Check if the change breaks coverage for any x in 1..L.
    # 6. If not, accept.
    # 7. If stuck, try a "2-step repair": change two primes simultaneously.
    # 8. If still stuck, perturb.

    stuck_count = 0
    MAX_STUCK = 20

    while time.time() < deadline:
        current_L = evaluate(residues)
        if current_L > best_L:
            best_L = current_L
            best_residues = residues.copy()
            stuck_count = 0

        if current_L >= MAX_EVAL:
            break

        target_x = current_L + 1

        # Identify candidates: primes that do NOT cover target_x
        # (target_x - residues[i]) % primes[i] != 0
        # This is equivalent to x_mod_p[i, target_x-1] != residues[i]
        # Note: xs[target_x-1] is target_x.
        # So we check x_mod_p[i, target_x-1] != residues[i]
        
        # We want to change residues[i] to target_x % primes[i]
        # which is x_mod_p[i, target_x-1]
        
        target_mods = x_mod_p[:, target_x - 1]
        
        # Candidates are indices where residues[i] != target_mods[i]
        candidates = np.where(residues != target_mods)[0]
        
        if len(candidates) == 0:
            # All primes cover target_x? Then coverage is >= 168 >= 2.
            # This shouldn't happen if current_L is the first failure.
            # But if it does, break.
            break

        # Shuffle candidates
        random.shuffle(candidates)
        
        found_repair = False
        
        # Try single-step repair
        for idx in candidates:
            old_res = residues[idx]
            new_res = target_mods[idx]
            
            if old_res == new_res:
                continue

            residues[idx] = new_res
            
            # Check if prefix 1..current_L is still valid
            if check_prefix(residues, current_L):
                # Accept change
                # Re-evaluate to find new L
                new_L = evaluate(residues)
                found_repair = True
                stuck_count = 0
                break
            else:
                # Revert
                residues[idx] = old_res

        if not found_repair:
            # Try 2-step repair
            # This is more expensive, so we only do it if stuck_count is high
            if stuck_count > 5:
                # Pick two random candidates
                if len(candidates) >= 2:
                    c1, c2 = random.sample(candidates, 2)
                    
                    old_res1 = residues[c1]
                    old_res2 = residues[c2]
                    
                    new_res1 = target_mods[c1]
                    new_res2 = target_mods[c2]
                    
                    if old_res1 != new_res1 and old_res2 != new_res2:
                        residues[c1] = new_res1
                        residues[c2] = new_res2
                        
                        if check_prefix(residues, current_L):
                            new_L = evaluate(residues)
                            found_repair = True
                            stuck_count = 0
                        else:
                            residues[c1] = old_res1
                            residues[c2] = old_res2

        if not found_repair:
            stuck_count += 1
            if stuck_count > MAX_STUCK:
                # Perturb: change a few random residues
                num_perturb = random.randint(3, 10)
                for _ in range(num_perturb):
                    i = random.randint(0, num_primes - 1)
                    p = primes[i]
                    residues[i] = random.randint(0, p - 1)
                stuck_count = 0
                current_L = evaluate(residues)
                if current_L < best_L:
                    residues = best_residues.copy()
                    current_L = best_L

    return [int(r) for r in best_residues]
