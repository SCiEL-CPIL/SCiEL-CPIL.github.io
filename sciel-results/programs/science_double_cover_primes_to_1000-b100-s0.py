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
            # Check if MAX_L is still valid
            if np.min(counts[1:MAX_L+1]) >= 2:
                return MAX_L
        # Check current prefix
        if current_L > 0:
            prefix = counts[1:current_L+1]
            if np.min(prefix) < 2:
                bad = np.where(prefix < 2)[0]
                return bad[0]
            # Current prefix is valid, try to extend
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

    # --- Greedy Initialization ---
    def greedy_init(primes, n_primes):
        residues = [0] * n_primes
        # Start with all zeros, then greedily pick residues to maximize L
        # But all zeros means x=0 mod p, so x ≡ 0 mod p.
        # For x in 1..L, x satisfies x≡0 mod p iff p divides x.
        # Let's just do a simple greedy: for each prime, pick r to maximize
        # the minimum count in the current covered range, or extend L.
        
        # Actually, let's start with a random-ish but deterministic init
        # and then do a few greedy passes.
        
        # Simple init: r_p = p // 2 or something
        # Let's try r_p = 1 for all (covers all odd numbers with 2, but 2 is prime)
        # Better: let's just use the seed init as a starting point for greedy
        residues = [((i + 2) // 2) % p for i, p in enumerate(primes)]
        
        counts = build_counts(residues, primes)
        L = find_L(counts, 0)
        
        # Greedy pass: try to improve L by changing one residue at a time
        # For each prime, try all residues (or a subset) and pick the best
        for _pass in range(5):
            improved = False
            for i in range(n_primes):
                p = primes[i]
                old_r = residues[i]
                best_r = old_r
                best_L = L
                
                # Remove old contribution
                start_old = old_r if old_r > 0 else p
                if start_old <= MAX_L:
                    counts[start_old::p] -= 1
                    
                # Try a few candidate residues
                # Try 0, and a few random ones, and old_r ± small amounts
                candidates = set([0, old_r])
                if old_r > 0:
                    candidates.add(old_r - 1)
                if old_r < p - 1:
                    candidates.add(old_r + 1)
                # Add a few random ones
                for _ in range(3):
                    candidates.add(np.random.randint(p))
                    
                for r in candidates:
                    if r == old_r:
                        continue
                    start_new = r if r > 0 else p
                    if start_new <= MAX_L:
                        counts[start_new::p] += 1
                    new_L = find_L(counts, L)
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
                    L = best_L
                    improved = True
                else:
                    # Restore old
                    start_old = old_r if old_r > 0 else p
                    if start_old <= MAX_L:
                        counts[start_old::p] += 1
            
            if not improved:
                break
        
        return residues, counts, L

    residues, counts, current_L = greedy_init(primes, n_primes)
    best_L = current_L
    best_residues = residues.copy()

    # --- Simulated Annealing ---
    T = 2.0
    min_T = 0.001
    cooling = 0.9998
    
    # Track time
    while time.time() < deadline:
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
            if delta > -500:
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
            # Restart with higher T to escape local minima
            if time.time() < deadline - 20:
                T = 1.0

    return best_residues
