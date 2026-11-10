import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s wall-clock limit

    N = 192
    MAX_SIZE = 48

    def time_left():
        return deadline - time.time()

    # ---- Precompute 4-ary conflict data using Python ints as bitsets ----
    # need[a][b][c] = bitset of d such that a | (b + c + d), for a,b,c,d distinct in 1..N
    # (index by value-1 so bit v corresponds to integer v+1)
    # For a given a, the set of d depends only on (b+c) mod a, so build per-a buckets.
    need = [[None] * (N + 1) for _ in range(N + 1)]
    for a in range(1, N + 1):
        buckets = {}
        for b in range(1, N + 1):
            if b == a:
                continue
            for c in range(b + 1, N + 1):
                if c == a:
                    continue
                r = (b + c) % a
                buckets.setdefault(r, 0)
                buckets[r] |= (1 << (c - 1))
        for b in range(1, N + 1):
            if b == a:
                continue
            for c in range(b + 1, N + 1):
                if c == a:
                    continue
                r = (b + c) % a
                mask = buckets.get(r, 0)
                # remove a, b, c from the candidate set
                mask &= ~(1 << (a - 1))
                mask &= ~(1 << (b - 1))
                mask &= ~(1 << (c - 1))
                need[a][b][c] = mask

    def triple_conflicts(x, bits):
        """Number of triples (b,c,d) among selected (bits) such that x | (b+c+d)."""
        if bits.bit_count() < 3:
            return 0
        lst = [i + 1 for i in range(N) if (bits >> i) & 1]
        n = len(lst)
        cnt = 0
        for i in range(n):
            b = lst[i]
            row = need[x][b]
            for j in range(i + 1, n):
                c = lst[j]
                m = row[c]
                m &= bits
                m &= ~(1 << (b - 1))
                m &= ~(1 << (c - 1))
                cnt += m.bit_count()
        return cnt

    def can_add(bits, x):
        """Can x be added to the valid set represented by bits?"""
        if (bits >> (x - 1)) & 1:
            return False
        # x must not divide any triple sum of the current set
        if triple_conflicts(x, bits) > 0:
            return False
        # For each selected a, a must not divide (x + b + c) for any b,c in S\{a}
        lst = [i + 1 for i in range(N) if (bits >> i) & 1]
        for a in lst:
            row = need[a][x]
            for b in lst:
                if b == a:
                    continue
                m = row[b]
                m &= bits
                m &= ~(1 << (a - 1))
                m &= ~(1 << (b - 1))
                if m:
                    return False
        return True

    def is_valid(S):
        bits = 0
        for x in S:
            bits |= (1 << (x - 1))
        for x in S:
            if triple_conflicts(x, bits & ~(1 << (x - 1))) > 0:
                return False
        return True

    def greedy_build(order):
        bits = 0
        selected = []
        for x in order:
            if time_left() < 2.0:
                break
            if len(selected) >= MAX_SIZE:
                break
            if can_add(bits, x):
                bits |= (1 << (x - 1))
                selected.append(x)
        return selected

    def local_improve(bits):
        """Repeatedly try to add any feasible element (ascending)."""
        improved = True
        while improved and time_left() > 2.0:
            improved = False
            for y in range(1, N + 1):
                if time_left() < 2.0:
                    break
                if (bits >> (y - 1)) & 1:
                    continue
                if can_add(bits, y):
                    bits |= (1 << (y - 1))
                    improved = True
                    break
        return bits

    def walksat(bits, max_iters):
        """Stochastic local search: occasionally drop an element to free conflicts."""
        best = bits
        best_size = bits.bit_count()
        full = (1 << N) - 1
        for _ in range(max_iters):
            if time_left() < 2.0:
                break
            if bits.bit_count() >= MAX_SIZE:
                break
            comp = full & ~bits
            if comp == 0:
                break
            if random.random() < 0.35 and bits.bit_count() > 8:
                # drop a random element, then try to add several
                sel = [i + 1 for i in range(N) if (bits >> i) & 1]
                removed = random.choice(sel)
                bits &= ~(1 << (removed - 1))
                cand = [i + 1 for i in range(N) if (comp >> i) & 1] + [removed]
                random.shuffle(cand)
                for c in cand[:25]:
                    if (bits >> (c - 1)) & 1:
                        continue
                    if can_add(bits, c):
                        bits |= (1 << (c - 1))
                        break
            else:
                cand = [i + 1 for i in range(N) if (comp >> i) & 1]
                random.shuffle(cand)
                for c in cand[:12]:
                    if can_add(bits, c):
                        bits |= (1 << (c - 1))
                        break
            if bits.bit_count() > best_size:
                best_size = bits.bit_count()
                best = bits
            if best_size >= MAX_SIZE:
                break
        return best

    def branch_and_bound(available, current_bits, best_size):
        """Branch-and-bound over 'available' to beat best_size. Returns best_bits."""
        best = current_bits
        best_sz = best_size
        if not available:
            return best, best_sz

        # feasibility bound
        if current_bits.bit_count() + len(available) <= best_sz:
            return best, best_sz

        # order available: prefer elements that are "easy" to add (few conflicts)
        # but cheap heuristic: just try in given order; recursion prunes.
        for x in available:
            if time_left() < 1.5:
                break
            if (current_bits >> (x - 1)) & 1:
                continue
            if can_add(current_bits, x):
                nb = current_bits | (1 << (x - 1))
                # recurse on remaining available (excluding x)
                rest = [y for y in available if y != x]
                sub_best, sub_sz = branch_and_bound(rest, nb, best_sz)
                if sub_sz > best_sz:
                    best_sz = sub_sz
                    best = sub_best
                if best_sz >= MAX_SIZE:
                    return best, best_sz
        return best, best_sz

    # ---- Main search: multiple seeds + polish ----
    best_bits = 0
    best_size = 0

    def consider(bits):
        nonlocal best_bits, best_size
        if bits.bit_count() > best_size:
            best_size = bits.bit_count()
            best_bits = bits

    # Seed 1: descending greedy
    s = greedy_build(list(range(N, 0, -1)))
    bits = 0
    for x in s:
        bits |= (1 << (x - 1))
    bits = local_improve(bits)
    consider(bits)

    # Seed 2: WalkSAT from descending
    if time_left() > 5.0:
        consider(walksat(bits, 30000))

    # Seed 3: random greedy + WalkSAT
    random.seed(42)
    for trial in range(15):
        if time_left() < 5.0:
            break
        order = list(range(1, N + 1))
        random.shuffle(order)
        s = greedy_build(order)
        b = 0
        for x in s:
            b |= (1 << (x - 1))
        b = local_improve(b)
        consider(b)
        if time_left() > 5.0 and best_size < MAX_SIZE:
            consider(walksat(b, 20000))

    # Seed 4: upper-range-first
    if time_left() > 5.0:
        order = list(range(N, N // 2, -1)) + list(range(N // 2, 0, -1))
        s = greedy_build(order)
        b = 0
        for x in s:
            b |= (1 << (x - 1))
        b = local_improve(b)
        consider(b)
        if time_left() > 5.0 and best_size < MAX_SIZE:
            consider(walksat(b, 20000))

    # Seed 5: low-divisor-first (primes / few divisors)
    if time_left() > 5.0:
        def score_num(x):
            if x == 1:
                return -1000
            d = 0
            for i in range(2, int(x ** 0.5) + 1):
                if x % i == 0:
                    d += 2
            return -d
        order = sorted(range(1, N + 1), key=score_num, reverse=True)
        s = greedy_build(order)
        b = 0
        for x in s:
            b |= (1 << (x - 1))
        b = local_improve(b)
        consider(b)
        if time_left() > 5.0 and best_size < MAX_SIZE:
            consider(walksat(b, 20000))

    # Final WalkSAT from current best
    if time_left() > 3.0 and best_size < MAX_SIZE:
        consider(walksat(best_bits, 30000))

    # Branch-and-bound polish: try to push to MAX_SIZE using the fast bitset test
    if time_left() > 4.0 and best_size < MAX_SIZE:
        sel = [i + 1 for i in range(N) if (best_bits >> i) & 1]
        avail = [x for x in range(1, N + 1) if x not in set(sel)]
        # sort available by "easiness": fewer conflicts with current set first
        def ease(x):
            return triple_conflicts(x, best_bits)
        avail.sort(key=ease)
        pb, psz = branch_and_bound(avail, best_bits, best_size)
        if psz > best_size:
            best_size = psz
            best_bits = pb

    # Final verify
    sel = [i + 1 for i in range(N) if (best_bits >> i) & 1]
    if not is_valid(sel):
        # fallback to a safe valid set
        s = greedy_build(list(range(N, 0, -1)))
        sel = s

    return sel[:MAX_SIZE]
