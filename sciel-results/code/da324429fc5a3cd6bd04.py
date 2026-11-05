import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115.0

    def get_primes(limit):
        sieve = [True] * (limit + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(limit**0.5) + 1):
            if sieve[i]:
                step = i
                start = i * i
                sieve[start:limit+1:step] = [False] * len(sieve[start:limit+1:step])
        return [i for i, is_prime in enumerate(sieve) if is_prime]

    primes = get_primes(1000)
    n_primes = len(primes)
    MAX_L = 100000

    def build_counts(residues, primes):
        counts = np.zeros(MAX_L + 1, dtype=np.int32)
        for i, p in enumerate(primes):
            r = residues[i]
            start = r if r > 0 else p
            if start <= MAX_L:
                counts[start::p] += 1
        return counts

    def find_L(counts, current_L):
        # Find the largest L such that counts[1..L] >= 2
        if current_L >= MAX_L:
            if np.min(counts[1:MAX_L+1]) >= 2:
                return MAX_L
        if current_L > 0:
            prefix = counts[1:current_L+1]
            if np.min(prefix) < 2:
                bad = np.where(prefix < 2)[0]
                return bad[0]
            if current_L < MAX_L:
                rest = counts[current_L+1:]
                bad_idx = np.where(rest < 2)[0]
                if len(bad_idx) == 0:
                    return MAX_L
                else:
                    return current_L + bad_idx[0]
        else:
            bad_idx = np.where(counts[1:] < 2)[0]
            if len(bad_idx) == 0:
                return MAX_L
            return bad_idx[0]

    def greedy_maximize(primes, n_primes, max_time):
        """
        Greedy submodular maximization:
        For each prime, choose the residue that maximizes the resulting L.
        Use lazy evaluation: only recompute L when a change is made.
        """
        residues = [0] * n_primes
        counts = np.zeros(MAX_L + 1, dtype=np.int32)
        current_L = 0
        
        # Phase 1: Greedy forward pass - for each prime in order, pick best residue
        for i in range(n_primes):
            p = primes[i]
            best_r = 0
            best_L = -1
            
            # Try a strategic subset of residues
            # For small primes, try all; for large primes, try a subset
            if p <= 50:
                candidates = list(range(p))
            else:
                # Try residues that cover positions near current_L
                # and some random ones
                candidates = set()
                # Residues that cover current_L or nearby
                for offset in range(-10, 11):
                    target = current_L + offset
                    if target > 0:
                        r = target % p
                        candidates.add(r)
                # Add 0 and a few random
                candidates.add(0)
                for _ in range(20):
                    candidates.add(np.random.randint(p))
                candidates = list(candidates)
            
            for r in candidates:
                start = r if r > 0 else p
                if start <= MAX_L:
                    counts[start::p] += 1
                    new_L = find_L(counts, current_L)
                    if new_L > best_L:
                        best_L = new_L
                        best_r = r
                    counts[start::p] -= 1
            
            # Apply best
            start = best_r if best_r > 0 else p
            if start <= MAX_L:
                counts[start::p] += 1
            residues[i] = best_r
            current_L = max(current_L, best_L)
        
        # Phase 2: Local search - try to improve by changing one residue at a time
        improved = True
        while improved and time.time() < max_time:
            improved = False
            for i in range(n_primes):
                p = primes[i]
                old_r = residues[i]
                best_r = old_r
                best_L = current_L
                
                # Remove old
                start_old = old_r if old_r > 0 else p
                if start_old <= MAX_L:
                    counts[start_old::p] -= 1
                
                # Try candidates
                if p <= 30:
                    candidates = list(range(p))
                else:
                    candidates = set()
                    # Near current_L
                    for offset in range(-20, 21):
                        target = current_L + offset
                        if target > 0:
                            candidates.add(target % p)
                    # Also try to cover positions just beyond current_L
                    for offset in range(1, 50):
                        target = current_L + offset
                        if target > 0:
                            candidates.add(target % p)
                    candidates.add(0)
                    for _ in range(10):
                        candidates.add(np.random.randint(p))
                    candidates = list(candidates)
                
                for r in candidates:
                    if r == old_r:
                        continue
                    start_new = r if r > 0 else p
                    if start_new <= MAX_L:
                        counts[start_new::p] += 1
                    new_L = find_L(counts, current_L)
                    if new_L > best_L:
                        best_L = new_L
                        best_r = r
                    if start_new <= MAX_L:
                        counts[start_new::p] -= 1
                
                # Apply best
                if best_r != old_r:
                    start_new = best_r if best_r > 0 else p
                    if start_new <= MAX_L:
                        counts[start_new::p] += 1
                    residues[i] = best_r
                    current_L = best_L
                    improved = True
                else:
                    start_old = old_r if old_r > 0 else p
                    if start_old <= MAX_L:
                        counts[start_old::p] += 1
        
        return residues, counts, current_L

    # Run greedy maximization
    residues, counts, current_L = greedy_maximize(primes, n_primes, deadline - 10)
    best_L = current_L
    best_residues = residues.copy()

    # Phase 3: Simulated annealing for further improvement
    if time.time() < deadline - 5:
        T = 1.0
        min_T = 0.001
        cooling = 0.9999
        
        while time.time() < deadline - 2:
            i = np.random.randint(n_primes)
            p = primes[i]
            old_r = residues[i]
            new_r = np.random.randint(p)
            
            if new_r == old_r:
                continue
            
            # Apply change
            start_old = old_r if old_r > 0 else p
            if start_old <= MAX_L:
                counts[start_old::p] -= 1
            
            start_new = new_r if new_r > 0 else p
            if start_new <= MAX_L:
                counts[start_new::p] += 1
            
            # Calculate new L
            new_L = find_L(counts, current_L)
            
            # Acceptance
            if new_L >= current_L:
                residues[i] = new_r
                current_L = new_L
                if new_L > best_L:
                    best_L = new_L
                    best_residues = residues.copy()
            else:
                delta = new_L - current_L
                if delta > -1000:
                    prob = np.exp(delta / T)
                    if np.random.random() < prob:
                        residues[i] = new_r
                        current_L = new_L
                    else:
                        residues[i] = old_r
                        if start_new <= MAX_L:
                            counts[start_new::p] -= 1
                        if start_old <= MAX_L:
                            counts[start_old::p] += 1
                else:
                    residues[i] = old_r
                    if start_new <= MAX_L:
                        counts[start_new::p] -= 1
                    if start_old <= MAX_L:
                        counts[start_old::p] += 1
            
            T *= cooling
            if T < min_T:
                T = min_T
                if time.time() < deadline - 10:
                    T = 0.5

    return best_residues
