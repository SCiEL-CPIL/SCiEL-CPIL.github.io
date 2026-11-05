import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 138.0
    
    # Generate primes up to 1000
    primes = [p for p in range(2, 1001) if all(p % q for q in range(2, int(p ** 0.5) + 1))]
    n_primes = len(primes)
    primes_arr = np.array(primes)
    
    MAX_EVAL = 12000
    
    def get_positions(p, r):
        if r == 0:
            first = p
        else:
            first = r
            if first < 1:
                first += p
        if first > MAX_EVAL:
            return np.array([], dtype=np.int32)
        return np.arange(first, MAX_EVAL + 1, p, dtype=np.int32)
    
    def compute_L(counts, max_check=None):
        if max_check is None:
            max_check = MAX_EVAL
        if counts[1] < 2:
            return 0
        # Check up to max_check
        check_range = min(max_check, MAX_EVAL)
        bad = np.where(counts[1:check_range+1] < 2)[0]
        if len(bad) == 0:
            return min(check_range, 100000)
        return bad[0]
    
    # Initial: seed-based
    residues = np.zeros(n_primes, dtype=np.int64)
    for i, p in enumerate(primes):
        residues[i] = ((i + 2) // 2) % p
    
    # Build counts
    counts = np.zeros(MAX_EVAL + 1, dtype=np.int32)
    for i, p in enumerate(primes):
        r = residues[i]
        pos = get_positions(p, r)
        if len(pos) > 0:
            counts[pos] += 1
    
    best_L = compute_L(counts)
    best_residues = residues.copy()
    
    random.seed(42)
    
    def local_search(residues, counts, L, max_rounds=50):
        """Local search to improve L."""
        for rnd in range(max_rounds):
            if time.time() > deadline - 10:
                break
            improved = False
            order = list(range(n_primes))
            random.shuffle(order)
            for i in order:
                if time.time() > deadline - 10:
                    break
                p = primes[i]
                old_r = residues[i]
                best_r = old_r
                best_L_new = L
                
                # Candidates
                if p <= 100:
                    candidates = list(range(p))
                else:
                    max_hit = L + 2 * p
                    upper = min(p - 1, max_hit)
                    if upper > 500:  # cap for very large p
                        candidates = list(range(0, upper + 1, max(1, (upper+1)//200)))
                        candidates.append(upper)
                    else:
                        candidates = list(range(0, upper + 1))
                
                for r in candidates:
                    if r == old_r:
                        continue
                    # Apply change
                    old_pos = get_positions(p, old_r)
                    new_pos = get_positions(p, r)
                    if len(old_pos) > 0:
                        counts[old_pos] -= 1
                    if len(new_pos) > 0:
                        counts[new_pos] += 1
                    new_L = compute_L(counts, max_check=L + p + 100)
                    if new_L > best_L_new:
                        best_L_new = new_L
                        best_r = r
                    # Undo
                    if len(new_pos) > 0:
                        counts[new_pos] -= 1
                    if len(old_pos) > 0:
                        counts[old_pos] += 1
                
                if best_r != old_r:
                    old_pos = get_positions(p, old_r)
                    new_pos = get_positions(p, best_r)
                    if len(old_pos) > 0:
                        counts[old_pos] -= 1
                    if len(new_pos) > 0:
                        counts[new_pos] += 1
                    residues[i] = best_r
                    L = best_L_new
                    improved = True
            
            if not improved:
                break
        
        return residues, counts, L
    
    # Main local search
    residues, counts, best_L = local_search(residues, counts, best_L, max_rounds=80)
    best_residues = residues.copy()
    
    # Perturbation restarts
    for restart in range(15):
        if time.time() > deadline - 20:
            break
        perturbed = best_residues.copy()
        n_changes = random.randint(10, 40)
        for _ in range(n_changes):
            i = random.randint(0, n_primes - 1)
            p = primes[i]
            perturbed[i] = random.randint(0, p - 1)
        
        # Rebuild counts
        counts_p = np.zeros(MAX_EVAL + 1, dtype=np.int32)
        for i, p in enumerate(primes):
            r = perturbed[i]
            pos = get_positions(p, r)
            if len(pos) > 0:
                counts_p[pos] += 1
        
        L_p = compute_L(counts_p)
        
        # Local search
        residues_p, counts_p, L_p = local_search(perturbed, counts_p, L_p, max_rounds=30)
        
        if L_p > best_L:
            best_L = L_p
            best_residues = residues_p.copy()
    
    return [int(r) for r in best_residues]
