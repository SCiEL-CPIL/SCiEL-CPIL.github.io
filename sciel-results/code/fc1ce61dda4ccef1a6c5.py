import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin under 90s wall-clock

    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    num_primes = len(primes)

    # Precompute prime-factorization vectors for 2..40.
    # Subset products are distinct  <=>  all subset sum-vectors are distinct
    # (unique prime factorization), so we can work entirely in vector space.
    vectors = {}
    for n in range(2, 41):
        v = [0] * num_primes
        temp = n
        for i, p in enumerate(primes):
            while temp % p == 0:
                v[i] += 1
                temp //= p
        vectors[n] = v

    all_candidates = list(range(2, 41))

    def sum_set_of(S):
        """All 2^k subset sum-vectors of S (as a set of tuples)."""
        sums = {tuple([0] * num_primes)}
        for n in S:
            v = vectors[n]
            new = {tuple(s[i] + v[i] for i in range(num_primes)) for s in sums}
            sums |= new
        return sums

    def can_add(S, sums, cand):
        """True iff adding cand keeps all subset sum-vectors distinct."""
        v = vectors[cand]
        new = {tuple(s[i] + v[i] for i in range(num_primes)) for s in sums}
        # A collision happens iff some new sum already exists in sums.
        for s in new:
            if s in sums:
                return False
        return True

    def greedy_build(order, max_size=18):
        """Greedily add candidates in the given order (skipping ones already present)."""
        S = []
        sums = {tuple([0] * num_primes)}
        present = set()
        for n in order:
            if len(S) >= max_size:
                break
            if time.time() > deadline:
                break
            if n in present:
                continue
            if can_add(S, sums, n):
                S.append(n)
                present.add(n)
                v = vectors[n]
                sums |= {tuple(s[i] + v[i] for i in range(num_primes)) for s in sums}
        return S

    def local_search(S, max_iter=40):
        """1-for-1 swap hill climbing: remove one, add another, keep if it grows size."""
        current = list(S)
        best = list(current)
        present = set(current)
        for _ in range(max_iter):
            if time.time() > deadline:
                break
            improved = False
            for i in range(len(current)):
                if time.time() > deadline:
                    break
                trial = current[:i] + current[i + 1:]
                trial_present = set(trial)
                trial_sums = sum_set_of(trial)
                for cand in all_candidates:
                    if cand in trial_present:
                        continue
                    if can_add(trial, trial_sums, cand):
                        new_trial = trial + [cand]
                        if len(new_trial) > len(best):
                            best = new_trial
                            current = new_trial
                            present = set(current)
                            improved = True
                        break
                if improved:
                    break
            if not improved:
                break
        return best

    def valid(S):
        """Exact verification: all 2^k subset products distinct."""
        prods = [1]
        for a in S:
            prods = prods + [x * a for x in prods]
        return len(prods) == len(set(prods))

    best_set = []

    # ---- Strategy 1: deterministic orderings (primes first, then composites) ----
    prime_set = set(primes)
    composites = [c for c in all_candidates if c not in prime_set]

    orderings = [
        primes + composites,
        primes + list(reversed(composites)),
        sorted(all_candidates, key=lambda x: len([p for p in primes if vectors[x][primes.index(p)] > 0])),
        list(reversed(all_candidates)),
        sorted(all_candidates, key=lambda x: -max(vectors[x])),
        sorted(all_candidates, key=lambda x: max(vectors[x])),
    ]

    for order in orderings:
        if time.time() > deadline - 2:
            break
        S = greedy_build(order)
        if len(S) > len(best_set):
            best_set = S

    # ---- Strategy 2: drop one prime, rebuild (sometimes frees vector space) ----
    for i in range(len(primes)):
        if time.time() > deadline - 2:
            break
        psubset = primes[:i] + primes[i + 1:]
        for order in (psubset + composites, psubset + list(reversed(composites))):
            S = greedy_build(order)
            if len(S) > len(best_set):
                best_set = S

    # ---- Strategy 3: random restarts ----
    random.seed(42)
    restarts = 0
    while time.time() < deadline - 12:
        cands = list(all_candidates)
        random.shuffle(cands)
        S = greedy_build(cands)
        if len(S) > len(best_set):
            best_set = S
        restarts += 1

    # ---- Strategy 4: local search polish on the best found ----
    if best_set and time.time() < deadline - 5:
        improved = local_search(best_set)
        if len(improved) > len(best_set):
            best_set = improved

    # ---- Final exact verification + fallback ----
    if best_set and not valid(best_set):
        # vector check should be exact, but guard against any bug
        fb = greedy_build(primes + composites)
        if valid(fb):
            best_set = fb
        else:
            best_set = primes

    # Trim and coerce to plain Python ints
    best_set = [int(x) for x in best_set][:18]

    return best_set
