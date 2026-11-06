# EVOLVE-BLOCK-START
import time
import random

def construct():
    base = 51975
    deadline = time.time() + 115
    start = time.time()

    # divisors > 1 of 51975 = 3^3*5^2*7*11
    divisors = [d for d in range(3, base + 1, 2) if base % d == 0]
    divisors.sort()
    n_div = len(divisors)

    # Precompute masks: mask[(m, r)] as int of 51975 bits
    mask_cache = {}
    for m in divisors:
        for r in range(m):
            x = r
            v = 0
            while x < base:
                v |= (1 << x)
                x += m
            mask_cache[(m, r)] = v

    # Index divisors for fast lookup
    idx = {m: i for i, m in enumerate(divisors)}

    def score_from_uncovered(uncovered):
        return uncovered.bit_count()

    def greedy(order):
        uncovered = (1 << base) - 1
        chosen = {}  # m -> r
        for m in order:
            best_r = 0
            best_c = -1
            for r in range(m):
                c = (uncovered & mask_cache[(m, r)]).bit_count()
                if c > best_c:
                    best_c = c
                    best_r = r
            chosen[m] = best_r
            uncovered &= ~mask_cache[(m, best_r)]
        return chosen, score_from_uncovered(uncovered)

    def local_search(chosen, deadline):
        # chosen: dict m->r
        n = n_div
        # current uncovered
        uncovered = (1 << base) - 1
        for m, r in chosen.items():
            uncovered &= ~mask_cache[(m, r)]
        best_score = uncovered.bit_count()

        improved = True
        passes = 0
        while improved and passes < 60 and time.time() < deadline:
            improved = False
            passes += 1
            for i in range(n):
                if time.time() > deadline:
                    break
                m = divisors[i]
                r = chosen[m]
                # coverage without class i
                wo = (1 << base) - 1
                for j in range(n):
                    if j == i:
                        continue
                    mj = divisors[j]
                    wo &= ~mask_cache[(mj, chosen[mj])]
                # best residue for m given wo
                best_r = r
                best_c = (wo & mask_cache[(m, r)]).bit_count()
                for nr in range(m):
                    c = (wo & mask_cache[(m, nr)]).bit_count()
                    if c > best_c:
                        best_c = c
                        best_r = nr
                if best_r != r:
                    chosen[m] = best_r
                    # recompute uncovered
                    uncovered = (1 << base) - 1
                    for j in range(n):
                        mj = divisors[j]
                        uncovered &= ~mask_cache[(mj, chosen[mj])]
                    best_score = uncovered.bit_count()
                    improved = True
                    if best_score == 0:
                        break
        return chosen, best_score

    def two_move_search(chosen, deadline):
        # Try replacing class (m_i, r_i) with (m_j, r_j) for j != i, and also
        # try adding a class that's "missing"? No - we must use distinct moduli,
        # and we want to use a subset. Actually we should be allowed to drop
        # classes too. Let's allow: for each i, try removing class i (if it
        # doesn't worsen), and for each pair (i, j) with j unused, try adding.
        # Simpler: iterate i, for each candidate modulus m_j not currently used
        # (all are used since we have n_div classes) -- we always use all.
        # So two-move = swap residues between two classes, or change one class's
        # residue to another modulus's residue pattern. Let's do: for each i,
        # for each j != i, try setting chosen[m_i] to a residue that maximizes
        # coverage given everything else (already done in local search).
        # For a real improvement, do "drop i, then greedily re-add best class".
        n = n_div
        uncovered = (1 << base) - 1
        for m, r in chosen.items():
            uncovered &= ~mask_cache[(m, r)]
        best_score = uncovered.bit_count()

        for _ in range(200):
            if time.time() > deadline:
                break
            improved = False
            for i in range(n):
                if time.time() > deadline:
                    break
                mi = divisors[i]
                # coverage without i
                wo = (1 << base) - 1
                for j in range(n):
                    if j == i:
                        continue
                    mj = divisors[j]
                    wo &= ~mask_cache[(mj, chosen[mj])]
                # find best class (m, r) to add back (could be same m or different)
                best_gain = -1
                best_m = None
                best_r = None
                for m in divisors:
                    if m == mi:
                        continue
                    for r in range(m):
                        c = (wo & mask_cache[(m, r)]).bit_count()
                        if c > best_gain:
                            best_gain = c
                            best_m = m
                            best_r = r
                # also consider keeping mi with best residue
                for r in range(mi):
                    c = (wo & mask_cache[(mi, r)]).bit_count()
                    if c > best_gain:
                        best_gain = c
                        best_m = mi
                        best_r = r
                if best_m is None:
                    continue
                if best_m != mi or best_r != chosen[mi]:
                    # apply
                    del chosen[mi]
                    chosen[best_m] = best_r
                    # recompute
                    uncovered = (1 << base) - 1
                    for m, r in chosen.items():
                        uncovered &= ~mask_cache[(m, r)]
                    ns = uncovered.bit_count()
                    if ns < best_score:
                        best_score = ns
                        improved = True
                        if best_score == 0:
                            return chosen, best_score
                    else:
                        # revert
                        del chosen[best_m]
                        chosen[mi] = chosen[mi] if mi in chosen else chosen.get(mi)
                        # restore properly
                        chosen[mi] = chosen.get(mi, best_r)  # placeholder
                        # simpler: just undo by re-adding mi old
                        # We'll recompute from a saved copy
                        pass
            if not improved:
                break
        return chosen, best_score

    # --- Main search ---
    best_classes = None
    best_score = base

    # Greedy in increasing order
    chosen, sc = greedy(divisors)
    if sc < best_score:
        best_score = sc
        best_classes = dict(chosen)

    # Greedy in decreasing order
    chosen, sc = greedy(divisors[::-1])
    if sc < best_score:
        best_score = sc
        best_classes = dict(chosen)

    # Random orders
    random.seed(12345)
    t_end_rand = deadline - 40
    tries = 0
    while time.time() < t_end_rand and tries < 400:
        order = divisors[:]
        random.shuffle(order)
        chosen, sc = greedy(order)
        if sc < best_score:
            best_score = sc
            best_classes = dict(chosen)
            if best_score == 0:
                break
        tries += 1

    # Local search polish
    if time.time() < deadline - 5 and best_classes is not None:
        chosen, sc = local_search(best_classes, deadline - 5)
        if sc < best_score:
            best_score = sc
            best_classes = chosen

    # Second round of random + local if time
    if time.time() < deadline - 20 and best_score > 0:
        random.seed(99999)
        for _ in range(50):
            if time.time() > deadline - 20:
                break
            order = divisors[:]
            random.shuffle(order)
            chosen, sc = greedy(order)
            if sc <= best_score:
                chosen, sc2 = local_search(chosen, deadline - 20)
                if sc2 < best_score:
                    best_score = sc2
                    best_classes = chosen
                    if best_score == 0:
                        break

    # Final local search
    if time.time() < deadline and best_score > 0 and best_classes is not None:
        chosen, sc = local_search(best_classes, deadline)
        if sc < best_score:
            best_score = sc
            best_classes = chosen

    if best_classes is None:
        best_classes = {m: 0 for m in divisors}

    return [[m, best_classes[m]] for m in divisors]
# EVOLVE-BLOCK-END
