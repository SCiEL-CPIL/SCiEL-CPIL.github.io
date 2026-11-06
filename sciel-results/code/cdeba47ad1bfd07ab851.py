import time
import random
from fractions import Fraction

def construct():
    start = time.time()
    DEADLINE = start + 145.0   # safety margin before 150s kill

    # ---- sieve primes <= 1000 ----
    def sieve(n):
        is_p = [True] * (n + 1)
        is_p[0] = is_p[1] = False
        for i in range(2, int(n ** 0.5) + 1):
            if is_p[i]:
                for j in range(i * i, n + 1, i):
                    is_p[j] = False
        return [i for i in range(2, n + 1) if is_p[i]]

    primes = sieve(1000)
    P = len(primes)

    # Precompute bit masks for each prime over a fixed horizon.
    # mask[p_index] has bit k set for k in [1..HORIZON] with k % p == 0 (i.e. r=0 base).
    # Shifting right by r gives the class [p, r] (covers k where k % p == r), k>=1.
    HORIZON = 6000
    masks = []
    for p in primes:
        m = 0
        # k from p to HORIZON step p  (these are the r=0 positions)
        for k in range(p, HORIZON + 1, p):
            m |= (1 << k)
        masks.append(m)

    # mask for class [p, r]: positions k in [1..HORIZON] with k % p == r
    # For r==0: that's exactly masks[i].
    # For r!=0: k = r, r+p, r+2p, ... but we only care k>=1.
    # masks[i] >> r  shifts the r=0 pattern down by r, so bit k of (masks[i]>>r)
    #   is 1 iff bit (k+r) of masks[i] is 1 iff (k+r) % p == 0 iff k % p == (-r)%p == (p-r)%p.
    # So (masks[i] >> r) covers class [p, (p - r) % p].
    # Therefore class [p, r] = masks[i] >> ((p - r) % p)  ... let's define shift = (p - r) % p.
    #   Actually we want k % p == r. (masks[i] >> s) has bit k set iff (k+s)%p==0 iff k%p == (-s)%p.
    #   So we need (-s)%p == r  =>  s == (p - r) % p.
    # Good.

    def class_mask(i, r):
        p = primes[i]
        s = (p - r) % p
        return masks[i] >> s

    def compute_L(covered, cap):
        # find largest L s.t. bits 1..L all set
        # L = number of consecutive set bits starting at bit 1
        # covered has bit k set if k covered.
        # We want max L with all of 1..L set.
        # (covered >> 1) & ... check consecutive.
        # Trick: x = covered >> 1  (bit 0 of x = bit1 of covered)
        # count trailing ones of x.
        x = covered >> 1
        if x == 0:
            return 0
        # trailing ones count
        # y = ~x ; trailing zeros of y = trailing ones of x
        y = ~x
        # but Python ints are infinite-precision with sign; use a mask.
        # count trailing zeros of (x+1) is not right either.
        # Use: t = x & -x gives lowest set bit of x (that's a 1 in x). Not trailing ones.
        # trailing ones of x = trailing zeros of (~x) but ~x is negative.
        # Instead: z = x + 1 ; the number of trailing ones of x equals trailing zeros of z.
        z = x + 1
        tz = (z & -z).bit_length() - 1   # trailing zeros of z
        L = tz
        if L > cap:
            L = cap
        return L

    def coverage_check(classes):
        covered = 0
        for p, r in classes:
            i = prime_index[p]
            covered |= class_mask(i, r)
        return covered

    prime_index = {p: i for i, p in enumerate(primes)}

    def build_greedy(target, seed, prefer_large):
        """Greedy: repeatedly add the class covering most new integers in [1..target]."""
        rng = random.Random(seed)
        covered = 0
        used = set()
        budget = Fraction(1, 1)
        classes = []
        # bit mask for [1..target]
        prefix_mask = 0
        for k in range(1, target + 1):
            prefix_mask |= (1 << k)

        while True:
            if time.time() > DEADLINE:
                break
            # find first uncovered in [1..target]
            unc = prefix_mask & ~covered
            if unc == 0:
                break
            # first uncovered = lowest set bit of unc
            first_bit = (unc & -unc).bit_length() - 1
            if first_bit > target:
                break
            U = first_bit  # 1-indexed position

            best = None  # (score, i, r, gain)
            for i, p in enumerate(primes):
                if p in used:
                    continue
                cost = Fraction(1, p)
                if budget < cost:
                    continue
                r = U % p
                cm = class_mask(i, r)
                newbits = cm & prefix_mask & ~covered
                gain = newbits.bit_count()
                if gain == 0:
                    continue
                # score: maximize gain; tie-break by prefer_large (bigger p) or cheaper
                if prefer_large:
                    tie = p
                else:
                    tie = -p
                # random jitter for diversity
                jitter = rng.random() * 1e-9
                score = (gain, tie, jitter)
                if best is None or score > best[0]:
                    best = (score, i, r, gain)

            if best is None:
                break

            _, i, r, gain = best
            p = primes[i]
            used.add(p)
            budget -= Fraction(1, p)
            classes.append([p, r])
            covered |= class_mask(i, r)

        return classes, covered, budget

    def local_search(classes, covered, budget, target, seed):
        """Try to extend coverage past current L using add + re-orient moves."""
        rng = random.Random(seed)
        classes = [list(c) for c in classes]
        used = {c[0] for c in classes}
        prefix_mask = 0
        for k in range(1, target + 1):
            prefix_mask |= (1 << k)

        improved = True
        while improved and time.time() < DEADLINE - 2:
            improved = False
            L = compute_L(covered, target)
            U = L + 1
            if U > target:
                break

            # Try move A: add a new unused prime with r = U % p
            candidates = []
            for i, p in enumerate(primes):
                if p in used:
                    continue
                cost = Fraction(1, p)
                if budget < cost:
                    continue
                r = U % p
                cm = class_mask(i, r)
                newbits = cm & prefix_mask & ~covered
                gain = newbits.bit_count()
                if gain <= 1:
                    continue
                candidates.append((gain, p, i, r, cm, cost))
            candidates.sort(key=lambda x: (-x[0], x[1]))

            best_move = None
            for gain, p, i, r, cm, cost in candidates:
                new_classes = classes + [[p, r]]
                new_covered = covered | cm
                new_L = compute_L(new_covered, target)
                if new_L > L:
                    best_move = ('add', p, i, r, cm, cost, new_classes, new_covered, new_L)
                    break

            # Try move B: re-orient an existing prime to cover U (free cost)
            for p, r_old in classes:
                i = prime_index[p]
                r_new = U % p
                if r_new == r_old:
                    continue
                cm_old = class_mask(i, r_old)
                cm_new = class_mask(i, r_new)
                # re-orient: remove old, add new
                new_covered = (covered & ~cm_old) | cm_new
                new_L = compute_L(new_covered, target)
                if new_L > L:
                    new_classes = [list(c) for c in classes]
                    for c in new_classes:
                        if c[0] == p and c[1] == r_old:
                            c[1] = r_new
                            break
                    best_move = ('reorient', p, i, r_new, cm_new, Fraction(0), new_classes, new_covered, new_L)
                    break

            if best_move is not None:
                improved = True
                kind = best_move[0]
                if kind == 'add':
                    _, p, i, r, cm, cost, new_classes, new_covered, new_L = best_move
                    classes = new_classes
                    covered = new_covered
                    budget -= cost
                    used.add(p)
                else:
                    _, p, i, r_new, cm_new, _, new_classes, new_covered, new_L = best_move
                    classes = new_classes
                    covered = new_covered
        return classes, covered, budget

    best_L = 0
    best_classes = []

    # ---- Phase 1: multiple greedy builds with different targets and seeds ----
    configs = []
    for target in [1200, 1800, 2500, 3500, 5000]:
        for seed in [0, 1, 2, 3, 7, 42]:
            configs.append((target, seed, False))
    for target in [1200, 1800, 2500]:
        for seed in [11, 23, 55]:
            configs.append((target, seed, True))

    # shuffle for diversity
    random.Random(12345).shuffle(configs)

    for target, seed, prefer_large in configs:
        if time.time() > DEADLINE - 15:
            break
        classes, covered, budget = build_greedy(target, seed, prefer_large)
        # local search
        classes, covered, budget = local_search(classes, covered, budget, target, seed + 1000)
        L = compute_L(covered, HORIZON)
        # verify budget
        b = sum(Fraction(1, p) for p, r in classes)
        if b > Fraction(1, 1):
            continue
        if L > best_L:
            best_L = L
            best_classes = [list(c) for c in classes]

    # ---- Phase 2: refine the best solution with more local search ----
    if best_classes and time.time() < DEADLINE - 5:
        covered = 0
        for p, r in best_classes:
            i = prime_index[p]
            covered |= class_mask(i, r)
        budget = Fraction(1, 1) - sum(Fraction(1, p) for p, r in best_classes)
        best_classes, covered, budget = local_search(best_classes, covered, budget, HORIZON, 999)
        L = compute_L(covered, HORIZON)
        if L > best_L:
            best_L = L

    # Final budget verification
    total = sum(Fraction(1, p) for p, r in best_classes)
    if total > Fraction(1, 1):
        # trim: remove the most expensive classes until within budget (shouldn't happen)
        best_classes.sort(key=lambda c: -1 / c[0])
        while best_classes and sum(Fraction(1, p) for p, r in best_classes) > Fraction(1, 1):
            best_classes.pop()

    return best_classes

# EVOLVE-BLOCK-END
