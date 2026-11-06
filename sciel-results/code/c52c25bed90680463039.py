import numpy as np
import time
import random

def _sieve_primes(limit):
    if limit < 2:
        return []
    is_prime = bytearray(b'\x01') * (limit + 1)
    is_prime[0] = 0
    is_prime[1] = 0
    for i in range(2, int(limit ** 0.5) + 1):
        if is_prime[i]:
            step = i
            start = i * i
            is_prime[start:limit+1:step] = bytearray(len(range(start, limit+1, step)))
    return [i for i in range(2, limit + 1) if is_prime[i]]

def construct():
    start_time = time.time()
    deadline = start_time + 142.0  # Safety margin before 150s limit

    primes = _sieve_primes(1000)
    n_primes = len(primes)
    primes_arr = np.array(primes, dtype=np.int64)
    
    N_max = 100000
    # Precompute index arrays for each prime for O(1) update
    # prime_indices[i] contains the indices in 1..N_max covered by residue 0 (i.e., multiples of p)
    # But residue r covers r, r+p, r+2p, ...
    # So for prime i and residue r, the indices are:
    #   if r == 0: p, 2p, ... (indices p, 2p, ...)
    #   if r > 0: r, r+p, ... (indices r, r+p, ...)
    # We can precompute the "step" pattern.

    # For efficiency in hill climbing, we need fast count updates.
    # counts is an array of size N_max+1.
    # Updating one prime's residue involves subtracting old coverage and adding new.
    # The number of elements affected is ~N_max / p.
    # Sum over all primes of N_max/p is N_max * H_{n_primes} ~ 100000 * 5 = 500,000.
    # This is feasible for a few thousand iterations if optimized.

    # However, the "score" is the first gap.
    # A smarter approach:
    # 1. Initialize residues.
    # 2. Compute counts.
    # 3. Find first gap x_fail.
    # 4. For each prime p that does NOT cover x_fail, changing r_p to x_fail % p will fix x_fail.
    #    But it might break earlier points.
    #    Specifically, it removes coverage of old_r, old_r+p, ...
    #    It adds coverage of new_r, new_r+p, ...
    #    We need to check if any y < x_fail becomes a gap.
    #    This is O(N_max/p) per candidate prime.
    #    168 primes * 500,000 ops = 84M ops. Too slow per iteration.

    # Alternative:
    # Use a smaller window for the "search" phase, then expand.
    # Or, use a heuristic that only checks local effects.
    # Or, accept that we might break earlier points, and use backtracking? No.
    # 
    # Let's try the "repair" heuristic but with a twist:
    # When we fix x_fail, we check if we created a new gap y < x_fail.
    # If so, we revert.
    # If not, we accept.
    # This is a "greedy repair" with local consistency check.

    # To make this fast, we can precompute the "coverage" of each residue for each prime.
    # But memory is 168 * 1000 * 100000 = 16.8 billion. Too much.

    # Let's stick to the incremental update but optimize it.
    # We will use a window N=10000 for the main search to keep updates fast.
    # Then verify at the end.

    N_search = 10000
    
    def build_counts(residues, N):
        counts = np.zeros(N + 1, dtype=np.int32)
        for i, p in enumerate(primes):
            r = int(residues[i])
            if r == 0:
                if p <= N:
                    counts[p::p] += 1
            else:
                if r <= N:
                    counts[r::p] += 1
        return counts

    def get_score(counts, N):
        failures = np.nonzero(counts[1:N+1] < 2)[0]
        if len(failures) == 0:
            return N
        return int(failures[0]) + 1

    best_residues = None
    best_score = 0

    rng = np.random.default_rng(42)
    random.seed(42)

    # Strategy: 
    # 1. Start with random residues.
    # 2. Greedy repair loop:
    #    - Find first gap x.
    #    - Pick a random prime p that does NOT cover x.
    #    - Set r_p = x % p.
    #    - Update counts.
    #    - If new first gap >= old first gap, accept. Else revert.
    # 3. If stuck (no improvement for K steps), try random perturbation.

    def init_random():
        return np.array([rng.integers(0, p) for p in primes], dtype=np.int64)

    def init_zero():
        return np.zeros(n_primes, dtype=np.int64)

    def init_one():
        return np.ones(n_primes, dtype=np.int64)

    # Let's try a "coverage maximization" initialization:
    # For each prime, pick the residue that covers the most currently uncovered points?
    # This is complex. Let's stick to simple inits + repair.

    initial_strategies = [
        init_random,
        init_zero,
        init_one,
        lambda: np.array([p // 2 for p in primes], dtype=np.int64),
        lambda: np.array([(i * 13 + 5) % p for i, p in enumerate(primes)], dtype=np.int64),
    ]

    for strat in initial_strategies:
        if time.time() > deadline - 10.0:
            break

        residues = strat()
        counts = build_counts(residues, N_search)
        current_score = get_score(counts, N_search)

        if current_score > best_score:
            best_score = current_score
            best_residues = residues.copy()

        # Local Search: Greedy Repair
        stall_count = 0
        max_stall = 200
        no_improve_count = 0
        max_no_improve = 50

        while time.time() < deadline - 5.0:
            if current_score == N_search:
                break

            x_fail = current_score

            # Find a prime that does NOT cover x_fail
            # To speed up, we can just pick a random prime and check.
            # If it covers, try again (up to a few times).
            # Or, precompute which primes cover x_fail?
            # x_fail % p == r_p.
            
            # Let's pick a random prime index
            i = int(rng.integers(n_primes))
            p = primes[i]
            r_old = int(residues[i])
            r_new = int(x_fail % p)

            if r_old == r_new:
                # This prime already covers x_fail.
                # Changing it might remove coverage from x_fail.
                # We should avoid this.
                # Try a few more times?
                # Or just do a random move?
                # Let's try another prime.
                # For efficiency, just skip this iteration.
                continue

            # Apply move
            if r_old == 0:
                if p <= N_search:
                    counts[p::p] -= 1
            else:
                if r_old <= N_search:
                    counts[r_old::p] -= 1

            if r_new == 0:
                if p <= N_search:
                    counts[p::p] += 1
            else:
                if r_new <= N_search:
                    counts[r_new::p] += 1

            new_score = get_score(counts, N_search)

            # Acceptance:
            # We accept if new_score > current_score.
            # Or if new_score == current_score (exploration).
            # If new_score < current_score, we created an earlier gap. Revert.
            
            if new_score > current_score:
                accepted = True
                no_improve_count = 0
            elif new_score == current_score:
                # Accept with some probability to explore
                if rng.random() < 0.2:
                    accepted = True
                else:
                    accepted = False
            else:
                accepted = False

            if accepted:
                residues[i] = r_new
                current_score = new_score
                stall_count = 0
                if current_score > best_score:
                    best_score = current_score
                    best_residues = residues.copy()
            else:
                # Revert
                if r_new == 0:
                    if p <= N_search:
                        counts[p::p] -= 1
                else:
                    if r_new <= N_search:
                        counts[r_new::p] -= 1

                if r_old == 0:
                    if p <= N_search:
                        counts[p::p] += 1
                else:
                    if r_old <= N_search:
                        counts[r_old::p] += 1

                stall_count += 1
                if stall_count > max_stall:
                    # Restart local search from current state?
                    # Or break to try next initial strategy?
                    # Let's try a random perturbation.
                    # Pick 5 random primes and set random residues.
                    for _ in range(5):
                        j = int(rng.integers(n_primes))
                        p_j = primes[j]
                        r_old_j = int(residues[j])
                        r_new_j = int(rng.integers(0, p_j))
                        if r_old_j == r_new_j:
                            continue
                        if r_old_j == 0:
                            if p_j <= N_search:
                                counts[p_j::p_j] -= 1
                        else:
                            if r_old_j <= N_search:
                                counts[r_old_j::p_j] -= 1

                        if r_new_j == 0:
                            if p_j <= N_search:
                                counts[p_j::p_j] += 1
                        else:
                            if r_new_j <= N_search:
                                counts[r_new_j::p_j] += 1

                        residues[j] = r_new_j

                    current_score = get_score(counts, N_search)
                    stall_count = 0
                    max_stall = 200  # Reset stall limit

    # Final Verification with larger N
    if best_residues is not None and time.time() < deadline - 1.0:
        N_verify = 100000
        counts_v = np.zeros(N_verify + 1, dtype=np.int32)
        for i, p in enumerate(primes):
            r = int(best_residues[i])
            if r == 0:
                if p <= N_verify:
                    counts_v[p::p] += 1
            else:
                if r <= N_verify:
                    counts_v[r::p] += 1

        failures = np.nonzero(counts_v[1:N_verify+1] < 2)[0]
        if len(failures) == 0:
            final_score = N_verify
        else:
            final_score = int(failures[0]) + 1

    return best_residues.tolist()

