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
                sieve[i*i:limit+1:i] = [False] * len(sieve[i*i:limit+1:i])
        return [i for i, is_prime in enumerate(sieve) if is_prime]

    primes = get_primes(1000)
    n_primes = len(primes)
    MAX_L = 100000
    GREEDY_WINDOW = 5000

    def find_L(counts, current_L, max_L):
        """
        Find the largest L such that all positions 1..L have count >= 2.
        Uses current_L to optimize the search.
        """
        if current_L > 0:
            # Check if current prefix is still valid
            # If min count in 1..current_L is < 2, we have a hole
            if np.min(counts[1:current_L + 1]) < 2:
                bad = np.where(counts[1:current_L + 1] < 2)[0]
                return bad[0]
            # Try to extend
            L = current_L
            if L < max_L:
                rest = counts[L + 1:]
                bad_idx = np.where(rest < 2)[0]
                if len(bad_idx) == 0:
                    return max_L
                else:
                    return L + bad_idx[0]
        else:
            bad_idx = np.where(counts[1:] < 2)[0]
            if len(bad_idx) == 0:
                return max_L
            else:
                return bad_idx[0]

    # --- Phase 1: Greedy Construction ---
    # Build residues forward, picking for each prime the residue that maximizes
    # coverage of the currently deficient positions in an early window.
    residues = [0] * n_primes
    counts_greedy = np.zeros(GREEDY_WINDOW + 1, dtype=np.int32)

    for idx in range(n_primes):
        p = primes[idx]
        best_r = 0
        best_score = -1

        # For small primes, try all residues; for large primes, sample
        if p <= 200:
            candidates = range(p)
        else:
            # Sample a subset of residues
            candidates = np.random.choice(p, size=min(p, 50), replace=False)

        for r in candidates:
            # Compute the gain: how many deficient positions does this residue cover?
            start = r if r > 0 else p
            if start > GREEDY_WINDOW:
                # This residue doesn't affect the window at all
                continue
            # Count positions in [start, GREEDY_WINDOW] with step p that have count < 2
            positions = np.arange(start, GREEDY_WINDOW + 1, p)
            deficient = np.sum(counts_greedy[positions] < 2)
            # Also consider how many already-covered positions we'd hit (less good)
            already_covered = np.sum(counts_greedy[positions] >= 2)
            score = deficient - 0.5 * already_covered
            if score > best_score:
                best_score = score
                best_r = r

        residues[idx] = best_r
        r = best_r
        start = r if r > 0 else p
        if start <= GREEDY_WINDOW:
            counts_greedy[start::p] += 1

    # --- Phase 2: Simulated Annealing on full range ---
    # Use the greedy result as starting point
    counts = np.zeros(MAX_L + 1, dtype=np.int32)
    for i, p in enumerate(primes):
        r = residues[i]
        start = r if r > 0 else p
        if start <= MAX_L:
            counts[start::p] += 1

    current_L = find_L(counts, 0, MAX_L)
    best_L = current_L
    best_residues = residues.copy()

    # Simulated Annealing Parameters
    T = 5.0
    cooling = 0.99995
    min_T = 0.05

    # Precompute prime indices for faster access
    prime_indices = np.arange(n_primes)

    while time.time() < deadline:
        # Pick random prime
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
        new_L = find_L(counts, current_L, MAX_L)

        # Acceptance
        if new_L >= current_L:
            # Accept
            residues[i] = new_r
            current_L = new_L
            if new_L > best_L:
                best_L = new_L
                best_residues = residues.copy()
        else:
            # Reject or Accept with probability
            delta = new_L - current_L
            if delta > -100:  # Only compute exp for reasonable deltas
                prob = np.exp(delta / T)
                if np.random.random() < prob:
                    residues[i] = new_r
                    current_L = new_L
                else:
                    # Revert
                    residues[i] = old_r
                    if start_new <= MAX_L:
                        counts[start_new::p] -= 1
                    if start_old <= MAX_L:
                        counts[start_old::p] += 1
            else:
                # Revert
                residues[i] = old_r
                if start_new <= MAX_L:
                    counts[start_new::p] -= 1
                if start_old <= MAX_L:
                    counts[start_old::p] += 1

        T *= cooling
        if T < min_T:
            T = min_T

    return best_residues
