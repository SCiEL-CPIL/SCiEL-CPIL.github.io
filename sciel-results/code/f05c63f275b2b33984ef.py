import time

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before the 150s kill

    def is_prime(n):
        if n < 2:
            return False
        if n % 2 == 0:
            return n == 2
        q = 3
        while q * q <= n:
            if n % q == 0:
                return False
            q += 2
        return True

    primes = [p for p in range(2, 1001) if is_prime(p)]
    prime_set = set(primes)

    # Candidate primes: those whose reciprocal we can afford.
    # We will choose a subset via greedy coverage, tracking the reciprocal budget.
    max_L = 10000

    # ---------- Greedy coverage heuristic to build a strong initial solution ----------
    # uncovered: True if integer not yet covered
    uncovered = [False] * (max_L + 1)
    for n in range(1, max_L + 1):
        uncovered[n] = True
    covered_count = max_L

    used = set()
    classes = []  # list of [p, r]
    # budget as a float is fine for selection; we only need to keep sum(1/p) <= 1.
    budget = 0.0

    # Precompute, for each prime, the count of uncovered integers per residue,
    # but that's O(L) per prime per step. Instead, do a smarter greedy:
    # For the smallest uncovered integer n, the residue that covers it is r = n % p.
    # We pick the prime that, with r = n % p, covers the most uncovered integers.

    # To make this fast, maintain for each prime the count of uncovered in each residue.
    # Build once, then update as we cover.
    # counts[p] is a list of length p with counts of uncovered numbers.
    counts = {}
    for p in primes:
        c = [0] * p
        for n in range(1, max_L + 1):
            if uncovered[n]:
                c[n % p] += 1
        counts[p] = c

    # We'll greedily add classes until budget is near 1 or no progress.
    # Order: repeatedly find the smallest uncovered integer, then pick the prime
    # (unused, affordable) that maximizes the number of uncovered integers in its
    # covering residue (r = n % p). Ties broken by smaller reciprocal (bigger p)
    # to save budget.

    # For efficiency, iterate a limited number of greedy steps.
    steps = 0
    max_steps = 200
    while covered_count > 0 and budget < 1.0 and steps < max_steps and time.time() < deadline - 20:
        steps += 1
        # Find smallest uncovered integer
        n0 = None
        for n in range(1, max_L + 1):
            if uncovered[n]:
                n0 = n
                break
        if n0 is None:
            break

        best_p = None
        best_r = None
        best_gain = -1
        best_cost = None
        for p in primes:
            if p in used:
                continue
            cost = 1.0 / p
            if budget + cost > 1.0:
                continue
            r = n0 % p
            gain = counts[p][r]
            if gain <= 0:
                continue
            # Prefer larger gain; tie-break by smaller cost (bigger p)
            if gain > best_gain or (gain == best_gain and best_cost is not None and cost < best_cost):
                best_gain = gain
                best_p = p
                best_r = r
                best_cost = cost

        if best_p is None:
            break

        # Apply
        used.add(best_p)
        budget += best_cost
        classes.append([best_p, best_r])

        # Update uncovered and counts for the covered residue
        p = best_p
        r = best_r
        start = r if r >= 1 else p
        for n in range(start, max_L + 1, p):
            if uncovered[n]:
                uncovered[n] = False
                covered_count -= 1
                # decrement counts for all primes at this index
                for q in primes:
                    counts[q][n % q] -= 1
        # Also decrement the residue count for best_p by the number covered (already done above per-index)
        # (counts[best_p][r] is now updated via the per-index loop)

    # Fallback: if greedy covered nothing, use the seed.
    if not classes:
        initial_primes = [p for p in primes if p >= 13]
        classes = [[p, (i + 1) % p] for i, p in enumerate(initial_primes)]

    # ---------- Build coverage count array for local search ----------
    count = [0] * (max_L + 1)
    for p, r in classes:
        start = r if r >= 1 else p
        for n in range(start, max_L + 1, p):
            count[n] += 1

    def find_prefix_from(count, start):
        for n in range(start, max_L + 1):
            if count[n] == 0:
                return n - 1
        return max_L

    best_L = find_prefix_from(count, 1)

    # ---------- Local search: optimize residues (1-moves) ----------
    improved = True
    while improved and time.time() < deadline - 10:
        improved = False
        for i in range(len(classes)):
            if time.time() > deadline - 10:
                break
            p, r = classes[i]
            best_r = r
            best_L_local = best_L
            for new_r in range(p):
                if new_r == r:
                    continue
                # Apply change
                start_old = r if r >= 1 else p
                start_new = new_r if new_r >= 1 else p
                for n in range(start_old, max_L + 1, p):
                    count[n] -= 1
                for n in range(start_new, max_L + 1, p):
                    count[n] += 1
                # Evaluate new prefix
                L_local = best_L
                if best_L < max_L and count[best_L + 1] > 0:
                    L_local = find_prefix_from(count, best_L + 1)
                if L_local > best_L_local:
                    best_L_local = L_local
                    best_r = new_r
                # Revert
                for n in range(start_new, max_L + 1, p):
                    count[n] -= 1
                for n in range(start_old, max_L + 1, p):
                    count[n] += 1
            if best_r != r:
                start_old = r if r >= 1 else p
                start_new = best_r if best_r >= 1 else p
                for n in range(start_old, max_L + 1, p):
                    count[n] -= 1
                for n in range(start_new, max_L + 1, p):
                    count[n] += 1
                classes[i] = [p, best_r]
                best_L = best_L_local
                improved = True

    # ---------- Optional: try adding a few more primes if budget remains ----------
    # (skip to keep within time; local search already optimizes existing set)

    return [[p, r] for p, r in classes]
