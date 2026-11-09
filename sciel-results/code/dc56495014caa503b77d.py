import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0

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

    MAX_EVAL = 20000
    xs = np.arange(1, MAX_EVAL + 1, dtype=np.int64)

    def compute_coverage(residues, limit):
        """Compute coverage array for x in 1..limit."""
        xs_sub = xs[:limit]
        diff = xs_sub[None, :] - residues[:, None]
        mods = np.remainder(diff, primes_arr[:, None])
        hits = (mods == 0)
        return np.sum(hits, axis=0)

    def evaluate_from_coverage(coverage, limit):
        """Compute largest L from coverage array."""
        mask = coverage[:limit] < 2
        if not np.any(mask):
            return limit
        first_fail_idx = np.argmax(mask)
        return int(xs[first_fail_idx] - 1)

    def update_coverage(coverage, idx, old_res, new_res, limit):
        """Update coverage when residue of prime idx changes."""
        p = primes[idx]
        
        if old_res == 0:
            start_old = p
        else:
            start_old = old_res
        
        if start_old <= limit:
            idxs_old = np.arange(start_old - 1, limit, p)
            coverage[idxs_old] -= 1
        
        if new_res == 0:
            start_new = p
        else:
            start_new = new_res
        
        if start_new <= limit:
            idxs_new = np.arange(start_new - 1, limit, p)
            coverage[idxs_new] += 1

    random.seed(42)
    residues = np.array([random.randint(0, p - 1) for p in primes], dtype=np.int64)
    
    coverage = compute_coverage(residues, MAX_EVAL)
    current_L = evaluate_from_coverage(coverage, MAX_EVAL)
    best_residues = residues.copy()
    best_coverage = coverage.copy()
    best_L = current_L

    stuck_count = 0
    MAX_STUCK = 30

    while time.time() < deadline:
        current_L = evaluate_from_coverage(coverage, MAX_EVAL)
        if current_L > best_L:
            best_L = current_L
            best_residues = residues.copy()
            best_coverage = coverage.copy()
            stuck_count = 0

        if current_L >= MAX_EVAL:
            break

        target_x = current_L + 1
        cov_target = coverage[target_x - 1]
        
        if cov_target >= 2:
            continue

        covers_target = np.array([(target_x - residues[i]) % primes[i] == 0 for i in range(num_primes)])

        candidates = [i for i in range(num_primes) if not covers_target[i]]

        if not candidates:
            break

        candidates.sort(key=lambda i: primes[i], reverse=True)

        found_repair = False

        for idx in candidates:
            if time.time() > deadline - 2.0:
                break
            old_res = residues[idx]
            new_res = target_x % primes[idx]
            
            if old_res == new_res:
                continue
            
            residues[idx] = new_res
            update_coverage(coverage, idx, old_res, new_res, MAX_EVAL)
            
            new_L = evaluate_from_coverage(coverage, MAX_EVAL)
            if new_L > current_L:
                found_repair = True
                stuck_count = 0
                break
            else:
                update_coverage(coverage, idx, new_res, old_res, MAX_EVAL)
                residues[idx] = old_res

        if not found_repair and time.time() < deadline - 5.0:
            tried_pairs = set()
            max_pairs = min(len(candidates), 10)
            for i_idx in range(max_pairs):
                for j_idx in range(i_idx + 1, max_pairs):
                    if time.time() > deadline - 3.0:
                        break
                    i, j = candidates[i_idx], candidates[j_idx]
                    if (i, j) in tried_pairs:
                        continue
                    tried_pairs.add((i, j))
                    
                    old_res_i = residues[i]
                    old_res_j = residues[j]
                    new_res_i = target_x % primes[i]
                    new_res_j = target_x % primes[j]
                    
                    if old_res_i == new_res_i and old_res_j == new_res_j:
                        continue
                    
                    residues[i] = new_res_i
                    residues[j] = new_res_j
                    update_coverage(coverage, i, old_res_i, new_res_i, MAX_EVAL)
                    update_coverage(coverage, j, old_res_j, new_res_j, MAX_EVAL)
                    
                    new_L = evaluate_from_coverage(coverage, MAX_EVAL)
                    if new_L > current_L:
                        found_repair = True
                        stuck_count = 0
                        break
                    else:
                        update_coverage(coverage, i, new_res_i, old_res_i, MAX_EVAL)
                        update_coverage(coverage, j, new_res_j, old_res_j, MAX_EVAL)
                        residues[i] = old_res_i
                        residues[j] = old_res_j
                
                if found_repair:
                    break

        if not found_repair:
            stuck_count += 1
            if stuck_count > MAX_STUCK:
                num_perturb = random.randint(3, 8)
                for _ in range(num_perturb):
                    i = random.randint(0, num_primes - 1)
                    p = primes[i]
                    old_res = residues[i]
                    residues[i] = random.randint(0, p - 1)
                    update_coverage(coverage, i, old_res, residues[i], MAX_EVAL)
                stuck_count = 0
                current_L = evaluate_from_coverage(coverage, MAX_EVAL)
                if current_L < best_L:
                    residues = best_residues.copy()
                    coverage = best_coverage.copy()
                    current_L = best_L

    return [int(r) for r in best_residues]
