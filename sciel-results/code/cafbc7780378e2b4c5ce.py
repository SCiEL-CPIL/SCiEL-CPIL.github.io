"""Rewritten program: exact reciprocal-subsum-free maximization via
integer-scaled subset sums with greedy, beam search, and local improvement."""

def construct():
    # scale = lcm(1..24) so that 1/k becomes the integer scale//k
    scale = 5354228880

    def greedy():
        sums = {0}
        chosen = []
        for i in range(24, 0, -1):
            w = scale // i
            if w not in sums:
                chosen.append(i)
                sums = sums | {s + w for s in sums}
        return sorted(chosen)

    def is_valid(chosen):
        # exact check: for each x, 1/x must not equal a subset-sum of
        # reciprocals of the other selected elements.
        # Because all terms are positive, only elements y > x can participate.
        ch = set(chosen)
        for x in ch:
            target = scale // x
            usable = [scale // y for y in ch if y > x]
            # subset sum check: can we form target from usable?
            n = len(usable)
            if n == 0:
                continue
            # use meet-in-the-middle for efficiency when n is moderate
            if n <= 20:
                half = n // 2
                A = usable[:half]
                B = usable[half:]
                
                def subset_sums(lst):
                    res = {0}
                    for v in lst:
                        res = res | {s + v for s in res}
                    return res
                
                SA = subset_sums(A)
                SB = subset_sums(B)
                for sa in SA:
                    if (target - sa) in SB:
                        return False
            else:
                # simple subset sum with early termination
                possible = {0}
                for v in usable:
                    new_sums = set()
                    for s in possible:
                        if s + v == target:
                            return False
                        if s + v < target:
                            new_sums.add(s + v)
                    possible = possible | new_sums
        return True

    def beam_search(beam_size=500, cap=10000):
        beam = [({0}, 0, ())]
        for i in range(24, 0, -1):
            w = scale // i
            cands = []
            for sums, cnt, ch in beam:
                cands.append((sums, cnt, ch))  # exclude
                if w not in sums:
                    ns = sums | {s + w for s in sums}
                    if len(ns) > cap:
                        # keep a deterministic subset to bound memory/time
                        ns = set(sorted(ns)[:cap])
                    cands.append((ns, cnt + 1, ch + (i,)))  # include
            seen = {}
            for sums, cnt, ch in cands:
                key = (frozenset(sums), cnt, ch)
                if key not in seen:
                    seen[key] = (sums, cnt, ch)
            items = list(seen.values())
            items.sort(key=lambda t: (-t[1], len(t[0])))
            beam = items[:beam_size]
        best = max(beam, key=lambda t: t[1])
        return sorted(best[2])

    g = greedy()
    b = beam_search()

    candidates = [g, b]
    # also try a couple of randomized greedy orders for diversity
    import random
    rnd = random.Random(12345)
    for _ in range(5):
        order = list(range(24, 0, -1))
        rnd.shuffle(order)
        sums = {0}
        chosen = []
        for i in order:
            w = scale // i
            if w not in sums:
                chosen.append(i)
                sums = sums | {s + w for s in sums}
        candidates.append(sorted(chosen))

    # pick the largest valid candidate
    best = None
    for c in candidates:
        if not c:
            continue
        if is_valid(c):
            if best is None or len(c) > len(best):
                best = c
    if best is None:
        best = g  # greedy is always valid by construction

    # local improvement: try to add any missing element that keeps validity
    improved = True
    cur = list(best)
    while improved:
        improved = False
        curset = set(cur)
        for i in range(24, 0, -1):
            if i not in curset:
                trial = cur + [i]
                if is_valid(trial):
                    cur = trial
                    curset.add(i)
                    improved = True
                    break
    return sorted(cur)


# EVOLVE-BLOCK-END
