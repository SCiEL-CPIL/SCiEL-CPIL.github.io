import time
import random
import numpy as np


def construct():
    start_time = time.time()
    deadline = start_time + 112.0  # safety margin before 120s kill

    def get_primes(limit):
        sieve = [True] * (limit + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(limit ** 0.5) + 1):
            if sieve[i]:
                for j in range(i * i, limit + 1, i):
                    sieve[j] = False
        return [i for i in range(2, limit + 1) if sieve[i]]

    primes = get_primes(1000)
    n_primes = len(primes)
    M = 100000  # search window for the score

    def calc_L(counts):
        """Largest L s.t. every integer 1..L has count >= 2 (capped at M)."""
        mask = counts[1:M + 1] < 2
        if not np.any(mask):
            return M
        return int(np.argmax(mask))  # 0-based index in counts[1:], equals L

    # ---- Initialization: deterministic seed, then random restarts, keep best ----
    def make_counts(residues):
        counts = np.zeros(M + 1, dtype=np.int32)
        for i, p in enumerate(primes):
            r = residues[i]
            counts[r::p] += 1
        return counts

    def greedy_repair(residues, counts, max_iter=200):
        """Target the bottleneck integer and fix it via a 1D repair over primes.

        Structural core: convert the high-dim global problem into a series of
        local 1D fixes that each directly raise the prefix length.
        """
        for _ in range(max_iter):
            L = calc_L(counts)
            if L >= M:
                break
            # The first uncovered integer is x = L + 1
            x = L + 1
            # Find primes whose current residue does NOT cover x
            candidates = []
            for i, p in enumerate(primes):
                r = residues[i]
                # x is covered by prime i iff x % p == r % p
                if (x % p) != (r % p):
                    candidates.append(i)
            if not candidates:
                break
            # For each candidate, find the residue r that would cover x,
            # i.e. r ≡ x (mod p). That is r = x % p.
            best_i = None
            best_gain = -1
            for i in candidates:
                p = primes[i]
                r_new = x % p
                r_old = residues[i]
                if r_new == r_old:
                    continue
                # Apply
                counts[r_old::p] -= 1
                counts[r_new::p] += 1
                new_L = calc_L(counts)
                gain = new_L - L
                if gain > best_gain:
                    best_gain = gain
                    best_i = i
                    best_r = r_new
                    best_r_old = r_old
                    best_p = p
                # Revert
                counts[r_new::p] -= 1
                counts[r_old::p] += 1
            if best_i is None or best_gain <= 0:
                # No single repair helps; try a small random perturbation
                k = random.randrange(n_primes)
                p = primes[k]
                r_old = residues[k]
                r_new = random.randrange(p)
                if r_new == r_old:
                    continue
                counts[r_old::p] -= 1
                counts[r_new::p] += 1
                new_L = calc_L(counts)
                if new_L <= L:
                    # revert
                    counts[r_new::p] -= 1
                    counts[r_old::p] += 1
                    break
            else:
                # Apply the best repair
                counts[best_r_old::best_p] -= 1
                counts[best_r::best_p] += 1
                residues[best_i] = best_r
        return residues, counts

    def local_search(residues, counts, budget):
        """Hill climbing: repeatedly target the bottleneck and repair it.
        Uses the measured greedy-repair strategy as the primary move, with
        random restarts of the perturbation when stuck."""
        best_L = calc_L(counts)
        best_residues = residues.copy()
        stuck = 0
        while time.time() < budget:
            L = calc_L(counts)
            if L >= M:
                break
            x = L + 1
            # collect candidates that don't cover x
            candidates = []
            for i, p in enumerate(primes):
                r = residues[i]
                if (x % p) != (r % p):
                    candidates.append(i)
            if not candidates:
                break
            best_i = None
            best_gain = -1
            best_r = None
            best_r_old = None
            best_p = None
            for i in candidates:
                p = primes[i]
                r_new = x % p
                r_old = residues[i]
                if r_new == r_old:
                    continue
                counts[r_old::p] -= 1
                counts[r_new::p] += 1
                new_L = calc_L(counts)
                gain = new_L - L
                if gain > best_gain:
                    best_gain = gain
                    best_i = i
                    best_r = r_new
                    best_r_old = r_old
                    best_p = p
                counts[r_new::p] -= 1
                counts[r_old::p] += 1
            if best_i is not None and best_gain > 0:
                counts[best_r_old::best_p] -= 1
                counts[best_r::best_p] += 1
                residues[best_i] = best_r
                new_L = calc_L(counts)
                if new_L > best_L:
                    best_L = new_L
                    best_residues = residues.copy()
                stuck = 0
            else:
                # stuck: random perturbation of a few residues
                for _ in range(random.randint(2, 5)):
                    k = random.randrange(n_primes)
                    p = primes[k]
                    r_old = residues[k]
                    r_new = random.randrange(p)
                    if r_new != r_old:
                        counts[r_old::p] -= 1
                        counts[r_new::p] += 1
                        residues[k] = r_new
                new_L = calc_L(counts)
                if new_L > best_L:
                    best_L = new_L
                    best_residues = residues.copy()
                else:
                    # revert to best
                    residues = best_residues.copy()
                    counts = make_counts(residues)
                stuck += 1
                if stuck > 30:
                    break
        return best_residues, best_L

    # ---- Main: random restarts, keep the globally best ----
    random.seed(12345)
    best_residues = None
    best_L = -1

    # Seed start
    residues = np.array([((i + 2) // 2) % primes[i] for i in range(n_primes)], dtype=np.int64)
    counts = make_counts(residues)
    r2, L2 = local_search(residues.copy(), counts.copy(), deadline - 8.0)
    if L2 > best_L:
        best_L = L2
        best_residues = r2

    # Random restarts with greedy-repair initialization
    restart_budget_each = 6.0
    while time.time() < deadline - 8.0:
        residues = np.array([random.randrange(p) for p in primes], dtype=np.int64)
        counts = make_counts(residues)
        # quick greedy repair to get out of a bad random state
        residues, counts = greedy_repair(residues, counts, max_iter=40)
        r2, L2 = local_search(residues.copy(), counts.copy(),
                              min(time.time() + restart_budget_each, deadline - 8.0))
        if L2 > best_L:
            best_L = L2
            best_residues = r2

    return [int(r) for r in best_residues]
