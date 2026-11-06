import time
import random
from fractions import Fraction

def construct():
    start = time.time()
    DEADLINE = start + 145.0

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
    prime_index = {p: i for i, p in enumerate(primes)}

    HORIZON = 10000

    # Precompute base masks (r=0) for each prime
    masks = []
    for p in primes:
        m = 0
        for k in range(p, HORIZON + 1, p):
            m |= (1 << k)
        masks.append(m)

    def class_mask(i, r):
        p = primes[i]
        s = (p - r) % p
        return masks[i] >> s

    def compute_L(covered, cap):
        x = covered >> 1
        if x == 0:
            return 0
        z = x + 1
        tz = (z & -z).bit_length() - 1
        L = tz
        if L > cap:
            L = cap
        return L

    def budget_sum(classes):
        return sum(Fraction(1, p) for p, r in classes)

    def greedy_build(target, seed, mode="efficiency"):
        """
        Greedy build with efficiency scoring.
        mode: 'efficiency' = maximize gain*p (new coverage per unit cost)
              'raw' = maximize gain
              'first' = cover first uncovered with cheapest prime
        """
        rng = random.Random(seed)
        covered = 0
        used = set()
        budget = Fraction(1, 1)
        classes = []
        prefix_mask = 0
        for k in range(1, target + 1):
            prefix_mask |= (1 << k)

        while time.time() < DEADLINE - 5:
            unc = prefix_mask & ~covered
            if unc == 0:
                break
            first_bit = (unc & -unc).bit_length() - 1
            if first_bit > target:
                break
            U = first_bit

            best = None
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

                if mode == "efficiency":
                    # gain * p = gain / (1/p) = new coverage per unit budget
                    score = gain * p
                elif mode == "raw":
                    score = gain
                else:
                    score = -p  # cheapest

                jitter = rng.random() * 0.01
                if best is None or score + jitter > best[0]:
                    best = (score + jitter, i, r, cm, gain, cost)

            if best is None:
                break

            _, i, r, cm, gain, cost = best
            p = primes[i]
            used.add(p)
            budget -= cost
            classes.append([p, r])
            covered |= cm

        return classes, covered, budget

    def local_improve(classes, covered, budget, target, seed, max_iters=5000):
        """
        Local search: try re-orienting existing primes and adding new ones.
        Focus on extending L.
        """
        rng = random.Random(seed)
        classes = [list(c) for c in classes]
        used = {c[0] for c in classes}
        prefix_mask = 0
        for k in range(1, target + 1):
            prefix_mask |= (1 << k)

        for _ in range(max_iters):
            if time.time() > DEADLINE - 3:
                break

            L = compute_L(covered, target)
            if L >= target:
                break
            U = L + 1

            # Try re-orienting an existing prime to cover U
            improved = False
            for idx, (p, r_old) in enumerate(classes):
                i = prime_index[p]
                r_new = U % p
                if r_new == r_old:
                    continue
                cm_old = class_mask(i, r_old)
                cm_new = class_mask(i, r_new)
                new_covered = (covered & ~cm_old) | cm_new
                new_L = compute_L(new_covered, target)
                if new_L > L:
                    classes[idx][1] = r_new
                    covered = new_covered
                    improved = True
                    break

            if improved:
                continue

            # Try adding a new prime
            best_add = None
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
                new_covered = covered | cm
                new_L = compute_L(new_covered, target)
                if new_L > L:
                    score = new_L * p + gain  # prefer extensions that are efficient
                    if best_add is None or score > best_add[0]:
                        best_add = (score, i, p, r, cm, cost, new_covered, new_L)

            if best_add is not None:
                _, i, p, r, cm, cost, new_covered, new_L = best_add
                classes.append([p, r])
                used.add(p)
                budget -= cost
                covered = new_covered
            else:
                # No improvement possible; try random perturbation
                if len(classes) > 2:
                    idx = rng.randint(0, len(classes) - 1)
                    p, r_old = classes[idx]
                    i = prime_index[p]
                    r_new = rng.randrange(p)
                    if r_new != r_old:
                        cm_old = class_mask(i, r_old)
                        cm_new = class_mask(i, r_new)
                        new_covered = (covered & ~cm_old) | cm_new
                        new_L = compute_L(new_covered, target)
                        old_L = compute_L(covered, target)
                        # Accept if not worse (annealing-like)
                        if new_L >= old_L or rng.random() < 0.1:
                            classes[idx][1] = r_new
                            covered = new_covered
                else:
                    break

        return classes, covered, budget

    best_L = 0
    best_classes = []

    # Phase 1: Multiple greedy builds with different modes and seeds
    configs = []
    for mode in ["efficiency", "raw", "first"]:
        for seed in range(20):
            configs.append((mode, seed))
    # Shuffle
    random.Random(42).shuffle(configs)

    for mode, seed in configs:
        if time.time() > DEADLINE - 20:
            break
        target = min(HORIZON, 2000 + seed * 100)
        classes, covered, budget = greedy_build(target, seed, mode)
        if budget_sum(classes) > Fraction(1, 1):
            continue
        L = compute_L(covered, HORIZON)
        if L > best_L:
            best_L = L
            best_classes = [list(c) for c in classes]

    # Phase 2: Local improvement on best and top candidates
    if best_classes and time.time() < DEADLINE - 10:
        covered = 0
        for p, r in best_classes:
            i = prime_index[p]
            covered |= class_mask(i, r)
        budget = Fraction(1, 1) - budget_sum(best_classes)
        best_classes, covered, budget = local_improve(best_classes, covered, budget, HORIZON, 777, max_iters=10000)
        L = compute_L(covered, HORIZON)
        if L > best_L:
            best_L = L

    # Phase 3: More diverse restarts if time remains
    if time.time() < DEADLINE - 15:
        for extra_seed in range(30, 80):
            if time.time() > DEADLINE - 10:
                break
            mode = ["efficiency", "raw", "first"][extra_seed % 3]
            target = 1500 + (extra_seed % 20) * 200
            classes, covered, budget = greedy_build(target, extra_seed, mode)
            if budget_sum(classes) > Fraction(1, 1):
                continue
            classes, covered, budget = local_improve(classes, covered, budget, HORIZON, extra_seed + 500, max_iters=3000)
            L = compute_L(covered, HORIZON)
            if L > best_L:
                best_L = L
                best_classes = [list(c) for c in classes]

    # Final budget check
    total = budget_sum(best_classes)
    if total > Fraction(1, 1):
        best_classes.sort(key=lambda c: -1 / c[0])
        while best_classes and budget_sum(best_classes) > Fraction(1, 1):
            best_classes.pop()

    return best_classes
