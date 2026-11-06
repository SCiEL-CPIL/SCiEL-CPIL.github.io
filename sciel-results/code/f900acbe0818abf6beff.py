import time
import random
import sys

def construct():
    t0 = time.time()
    DEADLINE = t0 + 80.0

    N = 256
    FULL = (1 << N) - 1

    # conflict_mask[i]: bitmask of j where (i+1)*(j+1) % ((i+1)+(j+1)) == 0
    cm = [0] * N
    for i in range(N):
        a = i + 1
        m = 0
        for j in range(N):
            if i == j:
                continue
            b = j + 1
            if (a * b) % (a + b) == 0:
                m |= (1 << j)
        cm[i] = m

    odds = [i for i in range(N) if (i + 1) % 2 == 1]
    evens = [i for i in range(N) if (i + 1) % 2 == 0]

    # 2-adic valuation classes: v2(x) = number of trailing zeros in binary
    # Numbers with the same v2 tend to have fewer conflicts with each other
    # because (a*b)%(a+b)==0 requires specific divisibility structure
    def v2(n):
        c = 0
        while n % 2 == 0 and n > 0:
            c += 1
            n //= 2
        return c

    # Group indices by 2-adic valuation of (i+1)
    v2_groups = {}
    for i in range(N):
        val = v2(i + 1)
        if val not in v2_groups:
            v2_groups[val] = []
        v2_groups[val].append(i)

    # Precompute for each index, which other indices it conflicts with, grouped by v2 class
    # This helps prioritize swaps that maintain structural coherence

    def popcount(x):
        return x.bit_count()

    # ---- candidate builders (each returns a valid mask) ----

    def greedy_add(start, order):
        m = start
        for e in order:
            if (m >> e) & 1:
                continue
            if cm[e] & m:
                continue
            m |= (1 << e)
        return m

    def local_search(m, iters):
        best = m
        bestsz = popcount(m)
        # Track recent swaps to avoid oscillation
        recent_swaps = set()
        for it in range(iters):
            if time.time() > DEADLINE:
                break
            if random.random() < 0.65:
                e = random.randrange(N)
                if (m >> e) & 1:
                    continue
                if not (cm[e] & m):
                    m |= (1 << e)
                    s = popcount(m)
                    if s > bestsz:
                        best, bestsz = m, s
                    recent_swaps.clear()
                else:
                    conf = cm[e] & m
                    # Try multiple conflict candidates, prefer those with fewer conflicts
                    conf_list = []
                    tmp = conf
                    while tmp:
                        c = (tmp & -tmp).bit_length() - 1
                        tmp &= tmp - 1
                        # Count how many elements in m conflict with c (excluding e if present)
                        # c is in m, so count conflicts of c within m
                        c_conf_count = popcount(cm[c] & (m & ~(1 << c)))
                        conf_list.append((c_conf_count, c))
                    # Sort by conflict count (fewer conflicts = better removal candidate)
                    conf_list.sort()
                    swapped = False
                    for c_conf_count, c in conf_list[:5]:  # Try top 5 candidates
                        if (e, c) in recent_swaps or (c, e) in recent_swaps:
                            continue
                        newm = (m & ~(1 << c)) | (1 << e)
                        if cm[e] & (newm & ~(1 << e)):
                            continue
                        # Verify c's removal doesn't break others: check that no element in newm conflicts with c
                        # Actually we need: for all x in newm, cm[x] & newm == 0
                        # Quick check: elements that conflict with c are in cm[c]
                        # After removing c, those elements are fine unless they also conflict with e
                        # But e is new, so check: any element in (cm[c] & m & ~(1<<c)) that also conflicts with e?
                        # If x is in m, x != c, x != e, and cm[x] has c set, x is fine after c removal
                        # The only new potential issue: x in m, x != c, and cm[x] has e set -> but e wasn't in m before
                        # So we just need: for all x in newm, cm[x] & newm == 0
                        # Since newm = m - {c} + {e}, and m was valid:
                        #   - For x in m\{c}: cm[x] & (m\{c}) == 0 (was valid, removing c only helps)
                        #     but need cm[x] & {e} == 0, i.e., not (cm[x] & (1<<e))
                        #   - For x = e: cm[e] & (m\{c}) == 0, which we checked above
                        # So we need: no x in (m & ~(1<<c)) has cm[x] with bit e set
                        # Equivalent: (cm[e] & (m & ~(1<<c))) == 0, which is exactly our check above
                        s = popcount(newm)
                        if s >= bestsz:
                            m = newm
                            recent_swaps.add((e, c))
                            if len(recent_swaps) > 200:
                                recent_swaps.clear()
                            if s > bestsz:
                                best, bestsz = m, s
                            swapped = True
                            break
            else:
                if m == 0:
                    continue
                # Remove a random element and try to add back more
                c = random.choice([i for i in range(N) if (m >> i) & 1])
                trial = m & ~(1 << c)
                added = 0
                # Try to add elements in order of v2 class diversity
                for v2_val in sorted(v2_groups.keys(), key=lambda v: random.random()):
                    for e in v2_groups[v2_val]:
                        if added >= 3:
                            break
                        if (trial >> e) & 1:
                            continue
                        if not (cm[e] & trial):
                            trial |= (1 << e)
                            added += 1
                    if added >= 3:
                        break
                # Also try random additions
                for _ in range(30):
                    if added >= 5:
                        break
                    e = random.randrange(N)
                    if (trial >> e) & 1:
                        continue
                    if not (cm[e] & trial):
                        trial |= (1 << e)
                        added += 1
                s = popcount(trial)
                if s > popcount(m):
                    m = trial
                    if s > bestsz:
                        best, bestsz = m, s
        return best

    def polish(m, iters):
        """High-precision polish: try all single-element swaps systematically"""
        best = m
        bestsz = popcount(m)
        for _ in range(iters):
            if time.time() > DEADLINE:
                break
            improved = False
            # Try adding each missing element
            for e in range(N):
                if (m >> e) & 1:
                    continue
                if not (cm[e] & m):
                    m |= (1 << e)
                    s = popcount(m)
                    if s > bestsz:
                        best, bestsz = m, s
                    improved = True
            if not improved:
                break
        return best

    def is_valid(m):
        for i in range(N):
            if (m >> i) & 1:
                if cm[i] & m:
                    return False
        return True

    best = 0
    bestsz = 0

    def consider(m):
        nonlocal best, bestsz
        if not is_valid(m):
            return
        s = popcount(m)
        if s > bestsz:
            best, bestsz = m, s

    # 1) all odds baseline
    base = 0
    for i in odds:
        base |= (1 << i)
    consider(base)

    # 2) greedy evens on odds, several orders
    consider(greedy_add(base, evens))
    consider(greedy_add(base, list(reversed(evens))))
    for _ in range(30):
        if time.time() > DEADLINE:
            break
        ro = evens[:]
        random.shuffle(ro)
        consider(greedy_add(base, ro))

    # 3) random small cores + greedy
    for _ in range(40):
        if time.time() > DEADLINE:
            break
        core = 0
        for i in random.sample(range(N), random.randint(1, 6)):
            if not (cm[i] & core):
                core |= (1 << i)
        ro = evens[:]
        random.shuffle(ro)
        consider(greedy_add(core, ro))

    # 4) local search from best
    if time.time() < DEADLINE - 12:
        consider(local_search(best, 200000))

    # 5) if time remains, refine with more local search passes
    if time.time() < DEADLINE - 8:
        for _ in range(3):
            if time.time() > DEADLINE:
                break
            consider(local_search(best, 50000))

    # 6) Final polish pass: systematic single-additions
    if time.time() < DEADLINE - 3:
        consider(polish(best, 200))

    if best == 0:
        best = 1  # number 1 alone is valid
    # Final validation
    if not is_valid(best):
        # Fallback to odds
        base = 0
        for i in odds:
            base |= (1 << i)
        best = base
    result = sorted([i + 1 for i in range(N) if (best >> i) & 1])
    return result
