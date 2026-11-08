import time
from math import log, gcd, exp
from itertools import combinations
import random
import numpy as np

def _score_raw(U):
    """Score an already-normalized sorted set (contains 0, len>=2)."""
    n = len(U)
    if n < 2:
        return -1.0
    U_arr = np.array(U, dtype=np.int64)
    sums = U_arr[:, None] + U_arr[None, :]
    diffs = U_arr[:, None] - U_arr[None, :]
    sc = len(np.unique(sums))
    dc = len(np.unique(diffs))
    if sc == 0:
        return -1.0
    ratio = dc / sc
    if ratio <= 1:
        return -1.0
    D = 2 * U[-1] + 1
    if D <= 1:
        return -1.0
    return 1.0 + log(ratio) / log(D)

def normalize(U):
    """Return sorted distinct nonnegative ints containing 0, divided by gcd."""
    U = sorted(set(U))
    if 0 not in U:
        U.append(0)
        U.sort()
    g = 0
    for x in U:
        g = gcd(g, x)
    if g > 1:
        U = sorted(set(x // g for x in U))
    return U

def score_set(U):
    U = normalize(list(U))
    if len(U) < 2:
        return -1.0
    return _score_raw(U)

def _score_fast(U):
    """Pure-Python fast score for small sets (no numpy overhead)."""
    U = sorted(set(U))
    if 0 not in U or len(U) < 2:
        return -1.0
    n = len(U)
    sums = set()
    diffs = set()
    for i in range(n):
        ui = U[i]
        for j in range(i, n):
            sums.add(ui + U[j])
        for j in range(i + 1, n):
            d = U[j] - ui
            diffs.add(d)
            diffs.add(-d)
    diffs.add(0)
    sc = len(sums)
    dc = len(diffs)
    if sc == 0:
        return -1.0
    ratio = dc / sc
    if ratio <= 1.0:
        return -1.0
    D = 2 * U[-1] + 1
    if D <= 1:
        return -1.0
    return 1.0 + log(ratio) / log(D)

def digit_based_set(bases, digit_sets):
    if not bases or not digit_sets:
        return [0]
    if len(bases) != len(digit_sets):
        raise ValueError("bases and digit_sets must have same length")
    elements = [0]
    for pos, (base, digits) in enumerate(zip(bases, digit_sets)):
        if not digits:
            continue
        new_elements = []
        for elem in elements:
            for d in digits:
                new_elements.append(elem * base + d)
        elements = new_elements
    if 0 not in elements:
        elements.append(0)
    return normalize(elements)

def digit_local_search(bases, digit_sets, deadline, max_val=300, T0=0.02):
    cur_set = digit_based_set(bases, digit_sets)
    cur_score = _score_raw(cur_set)
    best_score = cur_score
    best_bases = list(bases)
    best_digit_sets = [set(ds) for ds in digit_sets]
    T = T0
    no_improve = 0
    max_no_improve = 250
    while time.time() < deadline - 1 and no_improve < max_no_improve:
        r = random.random()
        new_bases = list(bases)
        new_digit_sets = [set(ds) for ds in digit_sets]
        if r < 0.25:
            pos = random.randint(0, len(bases) - 1)
            delta = random.randint(-2, 2)
            new_bases[pos] = max(2, bases[pos] + delta)
        elif r < 0.45:
            pos = random.randint(0, len(digit_sets) - 1)
            base = bases[pos]
            new_digit = random.randint(0, base - 1)
            new_digit_sets[pos].add(new_digit)
        elif r < 0.60:
            pos = random.randint(0, len(digit_sets) - 1)
            if len(digit_sets[pos]) > 2:
                digits_list = list(digit_sets[pos])
                rem = random.choice(digits_list)
                new_digit_sets[pos].discard(rem)
        elif r < 0.75:
            if len(bases) < 5:
                new_base = random.randint(3, 12)
                new_digits = set(random.sample(range(new_base), min(3, new_base)))
                new_bases.append(new_base)
                new_digit_sets.append(new_digits)
        else:
            if len(bases) > 2:
                pos = random.randint(1, len(bases) - 1)
                new_bases.pop(pos)
                new_digit_sets.pop(pos)
        try:
            new_set = digit_based_set(new_bases, new_digit_sets)
            if len(new_set) < 2 or new_set[-1] > max_val:
                no_improve += 1
                continue
            new_score = _score_raw(new_set)
        except:
            no_improve += 1
            continue
        if new_score > cur_score:
            bases = new_bases
            digit_sets = [set(ds) for ds in new_digit_sets]
            cur_score = new_score
            no_improve = 0
            if cur_score > best_score:
                best_score = cur_score
                best_bases = list(bases)
                best_digit_sets = [set(ds) for ds in digit_sets]
        else:
            delta = cur_score - new_score
            if delta < 0 or random.random() < exp(-delta / max(T, 1e-9)):
                bases = new_bases
                digit_sets = [set(ds) for ds in new_digit_sets]
                cur_score = new_score
            no_improve += 1
        T = max(T * 0.998, 1e-5)
    return best_score, best_bases, best_digit_sets

def greedy_build(deadline, maxv=80, maxn=14):
    U = [0]
    while time.time() < deadline - 4:
        cur = score_set(U)
        best_v = None
        best_sc = cur
        for v in range(1, maxv + 1):
            if v in U:
                continue
            sc = _score_raw(normalize(U + [v]))
            if sc > best_sc:
                best_sc = sc
                best_v = v
        if best_v is None or len(U) >= maxn:
            break
        U = normalize(U + [best_v])
    return U

def local_search(U, deadline, max_val=300, max_n=30, T0=0.02):
    U = normalize(list(U))
    cur_score = _score_raw(U)
    best_score = cur_score
    best_set = list(U)
    T = T0
    no_improve = 0
    max_no_improve = 400
    U_set = set(U)
    while time.time() < deadline - 1 and no_improve < max_no_improve:
        r = random.random()
        n = len(U)
        new_U = None
        if r < 0.30 and n < max_n:
            v = random.randint(1, max_val)
            if v not in U_set:
                new_U = normalize(U + [v])
        elif r < 0.40 and n > 2:
            idx = random.randint(1, n - 1)
            new_U = normalize([x for j, x in enumerate(U) if j != idx])
        elif r < 0.58:
            idx = random.randint(1, n - 1)
            v = random.randint(1, max_val)
            if v not in U_set:
                cand = list(U)
                cand[idx] = v
                new_U = normalize(cand)
        elif r < 0.74 and n > 2:
            i = random.randint(1, n - 1)
            j = random.randint(1, n - 1)
            if i != j:
                cand = list(U)
                cand[i], cand[j] = cand[j], cand[i]
                new_U = normalize(cand)
        else:
            idx = random.randint(1, n - 1)
            v = U[idx] + random.randint(-20, 20)
            if v >= 1 and v not in U_set:
                cand = list(U)
                cand[idx] = v
                new_U = normalize(cand)
        if new_U is None or len(new_U) < 2:
            no_improve += 1
            continue
        new_score = _score_raw(new_U)
        if new_score > cur_score:
            U = new_U
            U_set = set(U)
            cur_score = new_score
            no_improve = 0
            if cur_score > best_score:
                best_score = cur_score
                best_set = list(U)
        else:
            delta = cur_score - new_score
            if delta < 0 or random.random() < exp(-delta / max(T, 1e-9)):
                U = new_U
                U_set = set(U)
                cur_score = new_score
            no_improve += 1
        T = max(T * 0.9985, 1e-5)
    return best_score, best_set

def exhaustive_small(n, maxv, deadline):
    """Exhaustive search over all n-element sets (containing 0) with max element <= maxv."""
    best_sc = -1.0
    best_U = None
    count = 0
    for comb in combinations(range(1, maxv + 1), n - 1):
        if time.time() > deadline:
            break
        U = normalize([0] + list(comb))
        if len(U) != n:
            continue
        sc = _score_fast(U)
        count += 1
        if sc > best_sc:
            best_sc = sc
            best_U = list(U)
    return best_sc, best_U

def construct():
    start = time.time()
    deadline = start + 145  # 5s safety margin before 150s kill

    best_set = [0, 1, 3, 4]
    best_score = score_set(best_set)

    # --- Structured seeds (expanded) ---
    structured = [
        [0,1,3,4],
        [0,1,3,4,9,10,12,13],
        [0,1,4,9,11],
        [0,1,4,10,20,35],
        [0,1,2,4,8,16],
        [0,1,3,7,15,31],
        [0,1,5,13,25,41],
        [0,1,3,9,27,81],
        [0,1,4,10,18,28],
        [0,1,2,5,13,34],
        [0,1,2,4,7,9],
        [0,1,3,6,7,11],
        [0,1,3,7],
        [0,1,3,8],
        [0,1,3,9],
        [0,1,4,16],
        [0,1,2,7],
        [0,1,3,7,15],
        [0,1,3,8,16],
        [0,1,4,9,16],
        [0,1,2,4,8],
        [0,1,2,4,9],
        [0,1,3,6,10],
        [0,1,2,5,10],
        [0,1,4,7,12],
        [0,1,3,5,8],
    ]
    for s in structured:
        sc = score_set(s)
        if sc > best_score:
            best_score = sc
            best_set = normalize(list(s))

    # --- Digit-based seeds ---
    digit_seeds = []
    for base in [3, 4, 5, 6, 7, 8, 10]:
        for ndig in [2, 3]:
            digits = set(random.sample(range(base), min(ndig, base)))
            if 0 not in digits:
                digits.add(0)
            s = digit_based_set([base]*2, [digits, set(range(base))])
            if len(s) >= 2 and s[-1] <= 300:
                digit_seeds.append(s)
    mixed_configs = [
        ([3, 4], [{0,1,2}, {0,1,2,3}]),
        ([4, 3], [{0,1,2,3}, {0,1,2}]),
        ([5, 4], [{0,1,2,3,4}, {0,1,2,3}]),
        ([3, 5], [{0,1,2}, {0,1,2,3,4}]),
        ([4, 5, 3], [{0,1,2,3}, {0,1,2,3,4}, {0,1,2}]),
        ([5, 3, 4], [{0,1,2,3,4}, {0,1,2}, {0,1,2,3}]),
        ([3, 4, 5], [{0,1,2}, {0,1,2,3}, {0,1,2,3,4}]),
        ([4, 3, 5], [{0,1,2,3}, {0,1,2}, {0,1,2,3,4}]),
    ]
    for bases, dsets in mixed_configs:
        s = digit_based_set(bases, [set(ds) for ds in dsets])
        if len(s) >= 2 and s[-1] <= 300:
            digit_seeds.append(s)
    for s in digit_seeds:
        sc = score_set(s)
        if sc > best_score:
            best_score = sc
            best_set = normalize(list(s))

    # --- Exhaustive small-n search (key structural improvement) ---
    # This is the main structural change: thorough exhaustive search for small sets
    # to find globally optimal small configurations, then use them as seeds.
    exhaustive_seeds = [list(best_set)]

    # n=3, maxv=120
    if time.time() < deadline - 90:
        sc, U = exhaustive_small(3, 120, deadline - 90)
        if sc > best_score and U is not None:
            best_score = sc
            best_set = U
        if U is not None:
            exhaustive_seeds.append(U)

    # n=4, maxv=60
    if time.time() < deadline - 70:
        sc, U = exhaustive_small(4, 60, deadline - 70)
        if sc > best_score and U is not None:
            best_score = sc
            best_set = U
        if U is not None:
            exhaustive_seeds.append(U)

    # n=5, maxv=40
    if time.time() < deadline - 55:
        sc, U = exhaustive_small(5, 40, deadline - 55)
        if sc > best_score and U is not None:
            best_score = sc
            best_set = U
        if U is not None:
            exhaustive_seeds.append(U)

    # n=6, maxv=28
    if time.time() < deadline - 45:
        sc, U = exhaustive_small(6, 28, deadline - 45)
        if sc > best_score and U is not None:
            best_score = sc
            best_set = U
        if U is not None:
            exhaustive_seeds.append(U)

    # n=7, maxv=20
    if time.time() < deadline - 38:
        sc, U = exhaustive_small(7, 20, deadline - 38)
        if sc > best_score and U is not None:
            best_score = sc
            best_set = U
        if U is not None:
            exhaustive_seeds.append(U)

    # --- Greedy seeds ---
    seeds = [list(best_set)]
    for _ in range(20):
        n = random.randint(3, 8)
        vals = random.sample(range(1, 80), n)
        s = normalize([0] + vals)
        seeds.append(s)
    for mv in (50, 80, 120):
        if time.time() < deadline - 100:
            g = greedy_build(deadline, maxv=mv, maxn=12)
            if len(g) >= 2:
                seeds.append(g)
    for s in seeds:
        sc = score_set(s)
        if sc > best_score:
            best_score = sc
            best_set = list(s)

    # --- Digit-based local search restarts ---
    if time.time() < deadline - 30:
        for bases, dsets in mixed_configs:
            if time.time() > deadline - 30:
                break
            sc, _, _ = digit_local_search(bases, [set(ds) for ds in dsets], deadline, max_val=300)
            if sc > best_score:
                best_score = sc
                best_set = digit_based_set(bases, [set(ds) for ds in dsets])

    # --- Local search restarts (multiple diverse seeds) ---
    restart_seeds = [list(best_set)] + exhaustive_seeds + seeds[:10]
    # Deduplicate
    seen = set()
    unique_seeds = []
    for s in restart_seeds:
        key = tuple(s)
        if key not in seen:
            seen.add(key)
            unique_seeds.append(s)
    restart_seeds = unique_seeds

    attempts = 0
    while time.time() < deadline - 20:
        base = restart_seeds[attempts % len(restart_seeds)]
        sc, s = local_search(base, deadline, max_val=300, max_n=30,
                             T0=random.uniform(0.01, 0.03))
        if sc > best_score:
            best_score = sc
            best_set = list(s)
            if len(restart_seeds) < 20:
                restart_seeds.append(list(s))
            else:
                restart_seeds[-1] = list(s)
        attempts += 1

    # --- Additional targeted search ---
    if time.time() < deadline - 5:
        pattern_seeds = [
            [0, 1, 3, 4, 9, 10, 12, 13, 27, 28, 30, 31],
            [0, 1, 4, 9, 10, 13, 18, 23, 28, 33],
            [0, 1, 5, 6, 11, 12, 16, 17, 22, 23],
            [0, 1, 2, 4, 8, 16, 32, 64],
            [0, 1, 3, 7, 15, 31, 63, 127],
            [0, 1, 4, 10, 18, 28, 40, 54],
            [0, 1, 3, 6, 10, 15, 21, 28],
        ]
        for ps in pattern_seeds:
            if time.time() > deadline - 5:
                break
            s = normalize(ps)
            sc = score_set(s)
            if sc > best_score:
                best_score = sc
                best_set = s
            sc, s2 = local_search(s, deadline, max_val=250, max_n=26, T0=0.015)
            if sc > best_score:
                best_score = sc
                best_set = list(s2)

    # --- Final high-precision polish ---
    if time.time() < deadline - 3:
        cur = list(best_set)
        cur_score = _score_raw(cur)
        improved = True
        it = 0
        cur_set = set(cur)
        while improved and time.time() < deadline - 2 and it < 200:
            improved = False
            it += 1
            n = len(cur)
            # replace with delta window
            for i in range(1, n):
                if time.time() > deadline - 2:
                    break
                old = cur[i]
                for delta in range(-50, 51):
                    if delta == 0:
                        continue
                    v = old + delta
                    if v < 1 or v in cur_set:
                        continue
                    cand = list(cur)
                    cand[i] = v
                    newU = normalize(cand)
                    if len(newU) != n:
                        continue
                    sc = _score_raw(newU)
                    if sc > cur_score:
                        cur, cur_score, improved = newU, sc, True
                        cur_set = set(cur)
                        break
                if improved:
                    break
            if improved:
                continue
            # add
            if n < 28:
                hi = min(200, cur[-1] + 50)
                for v in range(1, hi + 1):
                    if v in cur_set:
                        continue
                    newU = normalize(cur + [v])
                    sc = _score_raw(newU)
                    if sc > cur_score:
                        cur, cur_score, improved = newU, sc, True
                        cur_set = set(cur)
                        break
            if improved:
                continue
            # remove
            for i in range(1, n):
                newU = normalize([x for j, x in enumerate(cur) if j != i])
                if len(newU) < 2:
                    continue
                sc = _score_raw(newU)
                if sc > cur_score:
                    cur, cur_score, improved = newU, sc, True
                    cur_set = set(cur)
                    break
        if cur_score > best_score:
            best_score = cur_score
            best_set = cur

    return normalize(list(best_set))

if __name__ == "__main__":
    U = construct()
    print("Result:", U)
    print("Score:", score_set(U))
