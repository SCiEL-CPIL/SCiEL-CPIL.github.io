"""Combined: structural free-prime handling + fast prime-factor local search."""
# EVOLVE-BLOCK-START
import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 82.0  # safety margin before 90s kill

    n = 256

    # --- smallest prime factor sieve ---
    spf = list(range(n + 1))
    for i in range(2, int(math.isqrt(n)) + 1):
        if spf[i] == i:
            for j in range(i * i, n + 1, i):
                if spf[j] == j:
                    spf[j] = i

    def prime_factors(x):
        pf = set()
        while x > 1:
            p = spf[x]
            pf.add(p)
            while x % p == 0:
                x //= p
        return pf

    # --- factor each number once ---
    pf = [None] * (n + 1)
    pf[1] = set()
    for v in range(2, n + 1):
        pf[v] = prime_factors(v)

    # --- free primes: p > 128 has no multiple in 2..256 other than itself,
    #     so it conflicts with nothing else and is always included ---
    free_primes = [p for p in range(129, n + 1) if spf[p] == p]
    core = [v for v in range(1, n + 1) if v != 1 and v not in free_primes]

    def conflicts(a, b):
        return not (pf[a].isdisjoint(pf[b]))

    # --- build an initial independent set (greedy by value desc) ---
    def build_initial(order):
        sel = []
        selset = set()
        for v in order:
            ok = True
            for u in sel:
                if conflicts(v, u):
                    ok = False
                    break
            if ok:
                sel.append(v)
                selset.add(v)
        return sel, selset

    base_order = sorted(core, key=lambda v: -v)
    sel, selset = build_initial(base_order)
    cur_sum = sum(sel)

    def feasible(v, selset, exclude):
        for u in selset:
            if u != exclude and conflicts(v, u):
                return False
        return True

    def local_search(sel, selset, cur_sum):
        nonlocal deadline
        improved = True
        while improved:
            if time.time() > deadline:
                break
            improved = False

            # candidates: unselected core numbers that fit the current set
            candidates = [v for v in core if v not in selset and feasible(v, selset, None)]
            candidates.sort(key=lambda v: -v)
            top = candidates[:40]

            # 1-for-1 swap
            best = None  # (gain, i, j)
            for j in sel:
                for i in top:
                    if i > j and feasible(i, selset, j):
                        g = i - j
                        if best is None or g > best[0]:
                            best = (g, i, j)
            if best is not None and best[0] > 0:
                g, i, j = best
                sel.remove(j)
                selset.discard(j)
                sel.append(i)
                selset.add(i)
                cur_sum += g
                improved = True
                continue

            # 1-for-2 swap: drop j, add i1,i2 (coprime to each other & rest)
            best12 = None  # (gain, i1, i2, j)
            for j in sel:
                comp = [i for i in top if i > j and feasible(i, selset, j)]
                L = len(comp)
                for a in range(L):
                    i1 = comp[a]
                    for b in range(a + 1, L):
                        i2 = comp[b]
                        if not pf[i1].isdisjoint(pf[i2]):
                            continue
                        g = i1 + i2 - j
                        if best12 is None or g > best12[0]:
                            best12 = (g, i1, i2, j)
            if best12 is not None and best12[0] > 0:
                g, i1, i2, j = best12
                sel.remove(j)
                selset.discard(j)
                sel.append(i1)
                sel.append(i2)
                selset.add(i1)
                selset.add(i2)
                cur_sum += g
                improved = True
                continue

            # 2-for-1 swap: drop j1,j2, add i
            best21 = None  # (gain, i, j1, j2)
            Ls = len(sel)
            for a in range(Ls):
                j1 = sel[a]
                for b in range(a + 1, Ls):
                    j2 = sel[b]
                    for i in top:
                        if i > j2 and feasible(i, selset, j1) and feasible(i, selset, j2):
                            g = i - j1 - j2
                            if best21 is None or g > best21[0]:
                                best21 = (g, i, j1, j2)
            if best21 is not None and best21[0] > 0:
                g, i, j1, j2 = best21
                sel.remove(j1)
                sel.remove(j2)
                selset.discard(j1)
                selset.discard(j2)
                sel.append(i)
                selset.add(i)
                cur_sum += g
                improved = True
                continue

        return sel, selset, cur_sum

    sel, selset, cur_sum = local_search(sel, selset, cur_sum)

    # --- diversifying restarts (only if we still have time) ---
    import random
    rng = random.Random(7)
    if time.time() < deadline - 6.0:
        for _ in range(3):
            if time.time() > deadline - 6.0:
                break
            order = core[:]
            rng.shuffle(order)
            # bias: keep high-value items early-ish
            order.sort(key=lambda v: -(v / 2.0) + rng.random() * 8.0)
            s2, set2 = build_initial(order)
            s2, set2, sum2 = local_search(s2, set2, sum2)
            if sum2 > cur_sum:
                sel, selset, cur_sum = s2, set2, sum2

    result = [1] + free_primes + sorted(sel)
    return result
# EVOLVE-BLOCK-END
