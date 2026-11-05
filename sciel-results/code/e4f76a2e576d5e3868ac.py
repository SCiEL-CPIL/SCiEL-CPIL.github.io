import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 112.0  # Safety margin for 120s limit

    def get_primes(limit):
        sieve = [True] * (limit + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(limit ** 0.5) + 1):
            if sieve[i]:
                sieve[i * i:limit + 1:i] = [False] * len(sieve[i * i:limit + 1:i])
        return [i for i, is_prime in enumerate(sieve) if is_prime]

    primes = get_primes(1000)
    n_primes = len(primes)
    MAX_L = 100000

    # Precompute positions for each prime: positions[start::p] where start = r if r>0 else p
    # For each prime p, we precompute the index arrays for all possible residues
    # But that's too much memory. Instead, we'll compute on the fly but efficiently.

    def build_counts(residues):
        """Build the count array from scratch."""
        counts = np.zeros(MAX_L + 1, dtype=np.int32)
        for i, p in enumerate(primes):
            r = residues[i]
            start = r if r > 0 else p
            if start <= MAX_L:
                counts[start::p] += 1
        return counts

    def find_L(counts, check_limit=None):
        """Find the largest L such that all 1..L have count >= 2."""
        if check_limit is None:
            check_limit = MAX_L
        check_limit = min(check_limit, MAX_L)
        if check_limit <= 0:
            return 0
        bad_idx = np.where(counts[1:check_limit + 1] < 2)[0]
        if len(bad_idx) == 0:
            return check_limit
        return bad_idx[0]

    def evaluate_L(residues, check_limit=None):
        """Build counts and find L."""
        counts = build_counts(residues)
        L = find_L(counts, check_limit)
        return L, counts

    # --- Phase 1: Smart Greedy Initialization ---
    # Process primes in order of increasing p. For each prime, choose the residue
    # that maximizes coverage of positions currently having count == 1 (to push them to 2),
    # while minimizing coverage of positions already >= 2 (waste).
    
    best_init_L = 0
    best_init_res = [0] * n_primes
    best_init_counts = np.zeros(MAX_L + 1, dtype=np.int32)

    # Try a few different orderings/strategies
    for attempt in range(5):
        residues = [0] * n_primes
        counts = np.zeros(MAX_L + 1, dtype=np.int32)

        # Process in order of prime size (small first for greedy)
        order = list(range(n_primes))
        if attempt == 1:
            # Try processing large primes first
            order = list(range(n_primes - 1, -1, -1))
        elif attempt == 2:
            # Random shuffle
            random.shuffle(order)
        elif attempt == 3:
            # Process by descending prime
            order = sorted(range(n_primes), key=lambda i: primes[i], reverse=True)
        elif attempt == 4:
            # Process by ascending prime but with random tie-breaking
            order = sorted(range(n_primes), key=lambda i: (primes[i], random.random()))

        for idx in order:
            p = primes[idx]
            best_r = 0
            best_score = -1e18

            # Determine candidate residues
            if p <= 30:
                candidates = list(range(p))
            else:
                # Focus on residues that hit positions near where counts are 1
                # Get positions with count == 1
                one_positions = np.where(counts[1:MAX_L + 1] == 1)[0] + 1
                if len(one_positions) > 0:
                    # For each position with count==1, the residue that covers it is pos % p
                    # Sample a subset
                    sample_size = min(200, len(one_positions))
                    sampled = np.random.choice(one_positions, size=sample_size, replace=False)
                    candidates = list(set([int(pos % p) for pos in sampled]))
                else:
                    candidates = []
                
                # Add some random candidates
                for _ in range(min(50, p)):
                    candidates.append(np.random.randint(p))
                
                # Add residue 0 and small residues
                for r in range(min(20, p)):
                    candidates.append(r)
                
                candidates = list(set(candidates))

            for r in candidates:
                start = r if r > 0 else p
                if start > MAX_L:
                    continue
                pos = np.arange(start, MAX_L + 1, p)
                current_counts = counts[pos]
                # Score: maximize positions going from 1->2, penalize 2->3
                gained = np.sum(current_counts == 1)
                wasted = np.sum(current_counts >= 2)
                score = gained - 0.3 * wasted
                if score > best_score:
                    best_score = score
                    best_r = r

            residues[idx] = best_r
            r = best_r
            start = r if r > 0 else p
            if start <= MAX_L:
                counts[start::p] += 1

        L = find_L(counts)
        if L > best_init_L:
            best_init_L = L
            best_init_res = residues.copy()
            best_init_counts = counts.copy()

    # Start from best greedy
    residues = best_init_res.copy()
    counts = best_init_counts.copy()
    current_L = best_init_L
    best_L = current_L
    best_residues = residues.copy()

    # --- Phase 2: Local Search with Targeted Moves ---
    # Instead of random SA, do targeted local search:
    # 1. Find the frontier (first position where count < 2)
    # 2. Identify which primes could help cover that position
    # 3. Try changing residues of those primes to better cover the frontier

    T = 5.0
    cooling_rate = 0.9995
    min_T = 0.05

    while time.time() < deadline:
        # Find current frontier
        if current_L < MAX_L:
            # Find first bad position
            bad_idx = np.where(counts[1:min(current_L + 1000, MAX_L + 1)] < 2)[0]
            if len(bad_idx) > 0:
                frontier = bad_idx[0] + 1
            else:
                frontier = min(current_L + 1000, MAX_L + 1)
        else:
            frontier = MAX_L + 1

        # Try to improve: pick a prime and change its residue
        # Strategy: focus on primes that can cover the frontier
        if frontier <= MAX_L:
            # Primes that can cover the frontier are those where frontier % p is a valid residue
            # All primes can cover any position, but some are more "useful"
            # Try changing a random prime, but bias towards smaller primes (more frequent)
            if random.random() < 0.5:
                # Pick a small prime
                idx = random.randint(0, min(20, n_primes - 1))
            else:
                idx = random.randint(0, n_primes - 1)
        else:
            idx = random.randint(0, n_primes - 1)

        p = primes[idx]
        old_r = residues[idx]
        
        # Generate new residue: try to cover the frontier
        if frontier <= MAX_L and random.random() < 0.7:
            # Set residue to cover the frontier
            new_r = frontier % p
        else:
            new_r = np.random.randint(p)

        if new_r == old_r:
            continue

        # Apply change
        start_old = old_r if old_r > 0 else p
        start_new = new_r if new_r > 0 else p

        if start_old <= MAX_L:
            counts[start_old::p] -= 1
        if start_new <= MAX_L:
            counts[start_new::p] += 1
        residues[idx] = new_r

        # Evaluate
        check_limit = min(current_L + 2000, MAX_L)
        new_L = find_L(counts, check_limit)

        # Acceptance
        if new_L >= current_L:
            current_L = new_L
            if new_L > best_L:
                best_L = new_L
                best_residues = residues.copy()
        else:
            delta = new_L - current_L
            prob = np.exp(delta / T)
            if random.random() < prob:
                current_L = new_L
            else:
                # Revert
                if start_new <= MAX_L:
                    counts[start_new::p] -= 1
                if start_old <= MAX_L:
                    counts[start_old::p] += 1
                residues[idx] = old_r

        T *= cooling_rate
        if T < min_T:
            T = min_T

        # Periodic restart: if stuck for a while, try a fresh greedy
        if random.random() < 0.001:
            # Quick greedy re-init
            temp_res = [0] * n_primes
            temp_counts = np.zeros(MAX_L + 1, dtype=np.int32)
            for i in range(n_primes):
                pp = primes[i]
                best_r = 0
                best_score = -1e18
                if pp <= 30:
                    candidates = list(range(pp))
                else:
                    candidates = [np.random.randint(pp) for _ in range(30)]
                    for r in range(min(10, pp)):
                        candidates.append(r)
                    candidates = list(set(candidates))
                for r in candidates:
                    start = r if r > 0 else pp
                    if start > MAX_L:
                        continue
                    pos = np.arange(start, MAX_L + 1, pp)
                    cc = temp_counts[pos]
                    score = np.sum(cc == 1) - 0.3 * np.sum(cc >= 2)
                    if score > best_score:
                        best_score = score
                        best_r = r
                temp_res[i] = best_r
                start = best_r if best_r > 0 else pp
                if start <= MAX_L:
                    temp_counts[start::pp] += 1
            temp_L = find_L(temp_counts)
            if temp_L > current_L:
                residues = temp_res
                counts = temp_counts
                current_L = temp_L
                if current_L > best_L:
                    best_L = current_L
                    best_residues = residues.copy()

    return best_residues
