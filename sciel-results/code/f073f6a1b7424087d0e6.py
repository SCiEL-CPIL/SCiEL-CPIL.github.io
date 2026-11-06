"""Rewritten with vectorized evaluation and simulated annealing."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    # Generate primes up to 1000
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n ** 0.5) + 1):
            if is_prime[i]:
                for j in range(i * i, n + 1, i):
                    is_prime[j] = False
        return [i for i, prime in enumerate(is_prime) if prime]

    primes = sieve(1000)
    n_primes = len(primes)
    primes_arr = np.array(primes, dtype=np.int64)

    # Precompute: for each prime p, the list of integers in [1, MAX_L] that are ≡ r (mod p)
    # We'll use a different approach: for each integer x in [1, L], count how many primes p
    # satisfy x ≡ r_p (mod p).
    # Vectorized: for a given residues array, compute coverage for all x in [1, L] at once.

    MAX_CHECK = 2000  # We'll check up to this for evaluation

    def compute_L(residues, check_limit=MAX_CHECK):
        """Compute largest L such that every integer 1..L has coverage >= 2.
        Vectorized using NumPy."""
        residues_arr = np.array(residues, dtype=np.int64)
        # For each x in [1, check_limit], count how many primes p satisfy x % p == r_p
        # We can do this efficiently by iterating over primes and using vectorized mod
        x_range = np.arange(1, check_limit + 1, dtype=np.int64)
        counts = np.zeros(check_limit + 1, dtype=np.int32)

        for i in range(n_primes):
            p = primes_arr[i]
            r = residues_arr[i]
            # x % p == r  =>  x in {r, r+p, r+2p, ...} within [1, check_limit]
            # Find all x in [1, check_limit] where x % p == r
            # Use: x = r + k*p, k >= 0, 1 <= x <= check_limit
            if r == 0:
                # x must be multiple of p: p, 2p, 3p, ...
                first = p
            else:
                first = r
            if first < 1:
                first += p
            # Generate all such x
            k_max = (check_limit - first) // p
            if k_max >= 0:
                xs = first + np.arange(k_max + 1, dtype=np.int64) * p
                xs = xs[xs <= check_limit]
                if len(xs) > 0:
                    counts[xs] += 1

        # Find largest L where counts[1..L] all >= 2
        # Use cumsum of (counts < 2) to find first gap
        gap_mask = (counts[1:] < 2).astype(np.int32)
        if gap_mask.size == 0:
            return check_limit
        first_gap = np.argmax(gap_mask)
        if gap_mask[first_gap] == 0:
            return check_limit
        return first_gap  # 0-indexed in gap_mask, which corresponds to x = first_gap + 1 - 1 = first_gap
        # Wait: gap_mask[0] corresponds to x=1. If gap_mask[j]==1, x=j+1 is the first gap.
        # So L = j (the number of consecutive good integers before the gap)
        # Actually if gap_mask[j] is first 1, then x=1..j are all good, x=j+1 is bad.
        # So L = j.

    def compute_L_fast(residues, check_limit=MAX_CHECK):
        """Faster version using direct modular comparison."""
        residues_arr = np.array(residues, dtype=np.int64)
        x_range = np.arange(1, check_limit + 1, dtype=np.int64)
        counts = np.zeros(check_limit + 1, dtype=np.int32)

        for i in range(n_primes):
            p = primes_arr[i]
            r = residues_arr[i]
            # Vectorized: which x in [1, check_limit] satisfy x % p == r?
            # Compute x_range % p and compare to r
            mods = x_range % p
            mask = (mods == r)
            if mask.any():
                idx = np.where(mask)[0] + 1  # 1-indexed positions
                counts[idx] += 1

        # Find L
        gap_mask = (counts[1:] < 2).astype(np.int32)
        if gap_mask.size == 0:
            return check_limit
        first_gap_idx = np.argmax(gap_mask)
        if gap_mask[first_gap_idx] == 0:
            return check_limit
        return first_gap_idx  # L = number of consecutive integers from 1 that are covered >= 2

    # Initial solutions
    random.seed(42)
    np.random.seed(42)

    best_residues = None
    best_L = 0

    # Strategy 1: All zeros
    r1 = [0] * n_primes
    L1 = compute_L_fast(r1)
    if L1 > best_L:
        best_L = L1
        best_residues = r1[:]

    # Strategy 2: r_p = 1
    r2 = [1 % p for p in primes]
    L2 = compute_L_fast(r2)
    if L2 > best_L:
        best_L = L2
        best_residues = r2[:]

    # Strategy 3: r_p = p//2
    r3 = [(p // 2) % p for p in primes]
    L3 = compute_L_fast(r3)
    if L3 > best_L:
        best_L = L3
        best_residues = r3[:]

    # Strategy 4: Random
    for _ in range(10):
        rr = [random.randint(0, p - 1) for p in primes]
        Lr = compute_L_fast(rr)
        if Lr > best_L:
            best_L = Lr
            best_residues = rr[:]

    # Simulated annealing
    if best_residues is not None:
        current = best_residues[:]
        current_L = best_L
        T = 10.0  # Initial temperature
        cooling = 0.9995
        min_T = 0.01

        # Focus on small primes first as they matter most
        # Weighted selection: smaller primes get more attention
        prime_weights = [1.0 / primes[i] for i in range(n_primes)]
        total_w = sum(prime_weights)
        prime_weights = [w / total_w for w in prime_weights]

        no_improve = 0
        max_no_improve = 5000

        while time.time() < deadline - 5.0 and no_improve < max_no_improve:
            # Pick a coordinate with weighted probability
            i = np.random.choice(n_primes, p=prime_weights)
            p = primes[i]
            old_r = current[i]

            # Try a random new value
            new_r = random.randint(0, p - 1)
            if new_r == old_r:
                continue

            current[i] = new_r
            new_L = compute_L_fast(current)

            # Accept if better, or with probability exp((new_L - current_L) / T)
            if new_L > current_L:
                current_L = new_L
                no_improve = 0
                if current_L > best_L:
                    best_L = current_L
                    best_residues = current[:]
            else:
                # Reject (we only accept improvements or equal with small prob)
                if new_L == current_L and random.random() < 0.1:
                    no_improve += 1
                else:
                    current[i] = old_r
                    no_improve += 1

            T *= cooling
            if T < min_T:
                T = min_T

        # Second phase: local search with pairwise swaps
        if time.time() < deadline - 10.0:
            current = best_residues[:]
            current_L = best_L
            no_improve = 0

            # Order: try small primes first
            order = list(range(n_primes))  # already in increasing prime order

            for _ in range(5):  # Multiple passes
                if time.time() > deadline - 10.0:
                    break
                improved = False
                for i in order:
                    if time.time() > deadline - 10.0:
                        break
                    p = primes[i]
                    old_r = current[i]
                    best_r = old_r
                    best_local_L = current_L

                    # Try all residues for small primes, sample for large
                    if p <= 30:
                        candidates = list(range(p))
                    else:
                        candidates = [old_r] + [random.randint(0, p - 1) for _ in range(15)]

                    for new_r in candidates:
                        if new_r == old_r:
                            continue
                        current[i] = new_r
                        new_L = compute_L_fast(current)
                        if new_L > best_local_L:
                            best_local_L = new_L
                            best_r = new_r

                    current[i] = best_r
                    if best_r != old_r:
                        current_L = best_local_L
                        improved = True
                        if current_L > best_L:
                            best_L = current_L
                            best_residues = current[:]

                if not improved:
                    break

    return best_residues if best_residues is not None else [0] * n_primes
# EVOLVE-BLOCK-END
