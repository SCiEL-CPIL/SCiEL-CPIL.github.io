"""B2[2] set in 1..1000 via model-and-optimize.

Model: points as A + B, A on a Sidon base, B a 2-AP. The base set (small,
optimizable) controls how "room" is left in the sumset; we then repair/grow by
swapping in under-used candidates while tracking a vectorized count array so the
B2[2] constraint (every sum has <=2 representations a<=b) is enforced exactly.

Deadline guard: evaluator kills at 150s; we keep a margin and always return the
best valid set found.
"""
import time
import random
import numpy as np

DEADLINE_SECONDS = 120.0
START_TIME = time.time()


def _time_left():
    return DEADLINE_SECONDS - (time.time() - START_TIME)


def _valid(chosen, cap=1000):
    """Validate every sum has <=2 reps a<=b. Returns bool."""
    c = np.asarray(chosen, dtype=np.int64)
    n = len(c)
    if n == 0:
        return True
    ii, jj = np.triu_indices(n)
    sums = c[ii] + c[jj]
    counts = np.bincount(sums, minlength=2 * cap + 1)
    return bool(counts.max() <= 2)


def _add_check(x, chosen_np, counts, cap):
    """Return (feasible, new_sums) for adding x to chosen_np with current counts."""
    if chosen_np.size == 0:
        ns = np.array([2 * x], dtype=np.int64)
    else:
        ns = np.concatenate([chosen_np + x, np.array([2 * x], dtype=np.int64)])
    if (counts[ns] + 1).max() <= 2:
        return True, ns
    return False, ns


def _greedy_base(seed, cap=1000, tlimit=3.0):
    """Greedy Sidon base: pick integers in 1..cap so all pairwise sums (a<=b) are unique."""
    rng = random.Random(seed)
    chosen = []
    cnp = np.empty(0, dtype=np.int64)
    seen = set()
    full = list(range(1, cap + 1))
    rng.shuffle(full)
    t0 = time.time()
    for x in full:
        if time.time() - t0 > tlimit:
            break
        if cnp.size == 0:
            new_sums = [2 * x]
        else:
            new_sums = list(cnp + x) + [2 * x]
        if all(s not in seen for s in new_sums):
            seen.update(new_sums)
            chosen.append(x)
            cnp = np.concatenate([cnp, np.array([x], dtype=np.int64)])
    return chosen


def _build_model(base, B0, B1, cap=1000):
    """Build candidate set {a + b : a in base, b in B0..B1} within 1..cap, dedup."""
    out = set()
    for a in base:
        for b in range(B0, B1 + 1):
            v = a + b
            if 1 <= v <= cap:
                out.add(v)
    return sorted(out)


def _seed_set_from_model(base, B0, B1, cap=1000, order="asc"):
    """Greedy-select from the model's candidate pool to enforce B2[2] exactly."""
    pool = _build_model(base, B0, B1, cap=cap)
    if order == "desc":
        pool = pool[::-1]
    chosen = []
    cnp = np.empty(len(pool), dtype=np.int64)
    n = 0
    counts = np.zeros(2 * cap + 1, dtype=np.int32)
    for x in pool:
        if n == 0:
            ns = np.array([2 * x], dtype=np.int64)
        else:
            ns = np.concatenate([cnp[:n] + x, np.array([2 * x], dtype=np.int64)])
        if (counts[ns] + 1).max() <= 2:
            chosen.append(x)
            cnp[n] = x
            n += 1
            counts[ns] += 1
    return chosen


def _repair_grow(seed_set, cap=1000, time_fn=None, stop_fn=None, max_rounds=200, seed=0,
                 max_attempts=20000):
    """Phased local improvement: (1) pure adds, (2) remove-one-add-one swaps,
    (3) remove-one-add-two swaps.

    Uses vectorized feasibility checks. A max_attempts cap prevents infinite
    loops when no swaps are productive.
    """
    rng = random.Random(seed)
    best = list(seed_set)
    if not best:
        return best
    best_set = set(best)
    outside = [v for v in range(1, cap + 1) if v not in best_set]

    c = np.asarray(best, dtype=np.int64)
    n = len(c)
    ii, jj = np.triu_indices(n)
    counts = np.bincount(c[ii] + c[jj], minlength=2 * cap + 1).astype(np.int32)

    def rebuild(ch):
        cc = np.asarray(ch, dtype=np.int64)
        nn = len(cc)
        if nn == 0:
            return np.zeros(2 * cap + 1, dtype=np.int32), nn
        i2, j2 = np.triu_indices(nn)
        return np.bincount(cc[i2] + cc[j2], minlength=2 * cap + 1).astype(np.int32), nn

    def try_add(x, cur_counts, cur_arr):
        """Check if x can be added to cur_arr under cur_counts. Returns (ok, new_sums)."""
        if cur_arr.size == 0:
            ns = np.array([2 * x], dtype=np.int64)
        else:
            ns = np.concatenate([cur_arr + x, np.array([2 * x], dtype=np.int64)])
        if (cur_counts[ns] + 1).max() <= 2:
            return True, ns
        return False, ns

    attempts = 0
    for _ in range(max_rounds):
        if time_fn is not None and time_fn() <= 0:
            break
        if stop_fn is not None and stop_fn() is True:
            break
        if attempts >= max_attempts:
            break

        c = np.asarray(best, dtype=np.int64)
        n = len(c)

        # Phase 1: try pure adds
        if outside:
            oa = np.asarray(outside, dtype=np.int64)
            if n == 0:
                new_sums_all = 2 * oa[:, None]
            else:
                cross = oa[:, None] + c[None, :]
                new_sums_all = np.concatenate([cross, 2 * oa[:, None]], axis=1)
            gathered = counts[new_sums_all] + 1
            cand_max = gathered.max(axis=1)
            feasible = np.where(cand_max <= 2)[0]
            if feasible.size:
                tight = gathered[feasible].mean(axis=1)
                bi = int(feasible[int(np.argmin(tight))])
                x = int(oa[bi])
                ns = new_sums_all[bi]
                best.append(x)
                best_set.add(x)
                outside.remove(x)
                c = np.concatenate([c, np.array([x], dtype=np.int64)])
                n += 1
                counts[ns] += 1
                attempts += 1
                continue

        # Phase 2: targeted swaps — remove one element, try to add an outside candidate
        if outside and n > 0:
            batch_size = min(n, 30)
            found_swap = False
            for _ in range(batch_size):
                if time_fn is not None and time_fn() <= 0:
                    break
                if stop_fn is not None and stop_fn() is True:
                    break
                if attempts >= max_attempts:
                    break
                rem = rng.randrange(n)
                rem_val = int(c[rem])
                trial = [best[j] for j in range(n) if j != rem]
                cnt_t, _ = rebuild(trial)
                ta = np.asarray(trial, dtype=np.int64)
                oa = np.asarray(outside, dtype=np.int64)
                if ta.size == 0:
                    new_sums2 = 2 * oa[:, None]
                else:
                    cross2 = oa[:, None] + ta[None, :]
                    new_sums2 = np.concatenate([cross2, 2 * oa[:, None]], axis=1)
                gathered2 = cnt_t[new_sums2] + 1
                cand_max2 = gathered2.max(axis=1)
                feas2 = np.where(cand_max2 <= 2)[0]
                if feas2.size:
                    tight2 = gathered2[feas2].mean(axis=1)
                    bi2 = int(feas2[int(np.argmin(tight2))])
                    x = int(oa[bi2])
                    ns2 = new_sums2[bi2]
                    best = trial
                    best.append(x)
                    best_set = set(best)
                    best_set.add(x)
                    outside.remove(x)
                    cnt_t[ns2] += 1
                    counts = cnt_t
                    c = np.asarray(best, dtype=np.int64)
                    n = len(c)
                    attempts += 1
                    found_swap = True
                    break
            if found_swap:
                continue

        # Phase 3: remove one, add two (net +1)
        if outside and n > 0:
            batch_size = min(n, 15)
            found_double = False
            for _ in range(batch_size):
                if time_fn is not None and time_fn() <= 0:
                    break
                if stop_fn is not None and stop_fn() is True:
                    break
                if attempts >= max_attempts:
                    break
                rem = rng.randrange(n)
                trial = [best[j] for j in range(n) if j != rem]
                cnt_t, _ = rebuild(trial)
                ta = np.asarray(trial, dtype=np.int64)
                oa = np.asarray(outside, dtype=np.int64)
                # Find all outside elements that fit into trial
                if ta.size == 0:
                    ns_all = 2 * oa[:, None]
                else:
                    cross_a = oa[:, None] + ta[None, :]
                    ns_all = np.concatenate([cross_a, 2 * oa[:, None]], axis=1)
                gathered_a = cnt_t[ns_all] + 1
                cand_max_a = gathered_a.max(axis=1)
                feas_a = np.where(cand_max_a <= 2)[0]
                if feas_a.size < 2:
                    continue
                # Pick the two least "tight" feasible candidates
                tight_a = gathered_a[feas_a].mean(axis=1)
                order_idx = feas_a[np.argsort(tight_a)][:2]
                x1 = int(oa[order_idx[0]])
                x2 = int(oa[order_idx[1]])
                # Add x1
                if ta.size == 0:
                    ns1 = np.array([2 * x1], dtype=np.int64)
                else:
                    ns1 = np.concatenate([ta + x1, np.array([2 * x1], dtype=np.int64)])
                if (cnt_t[ns1] + 1).max() > 2:
                    continue
                cnt_t[ns1] += 1
                ta = np.concatenate([ta, np.array([x1], dtype=np.int64)])
                # Add x2
                ns2 = np.concatenate([ta + x2, np.array([2 * x2], dtype=np.int64)])
                if (cnt_t[ns2] + 1).max() > 2:
                    continue
                cnt_t[ns2] += 1
                # Commit
                best = trial
                best.append(x1)
                best.append(x2)
                best_set = set(best)
                best_set.add(x1)
                best_set.add(x2)
                outside.remove(x1)
                outside.remove(x2)
                counts = cnt_t
                c = np.asarray(best, dtype=np.int64)
                n = len(c)
                attempts += 1
                found_double = True
                break
            if found_double:
                continue

        break  # stuck
    return best


def construct():
    global START_TIME, DEADLINE_SECONDS
    START_TIME = time.time()
    DEADLINE_SECONDS = 120.0
    cap = 1000

    def time_left():
        return _time_left()

    def stopped():
        return _time_left() < 6.0

    best = []
    best_len = -1

    def consider(s):
        nonlocal best, best_len
        if s and _valid(s, cap=cap) and len(s) > best_len:
            best_len = len(s)
            best = list(s)

    rng = random.Random(7)

    # --- Model-and-optimize loop: parameterize base + B-interval, optimize via greedy ---
    # Parameters: base (Sidon, from greedy with a seed), B0, B1.
    # We try a grid of B0/B1 with several base seeds and orderings.
    base_seeds = [0, 1, 2, 3, 5, 8, 13, 21, 34, 55]
    B0s = [0, 1, 2, 3, 5]
    B1s = [2, 3, 4, 5, 6, 8]

    base_cache = {}
    for bs in base_seeds:
        base_cache[bs] = _greedy_base(bs, cap=cap, tlimit=1.0)

    for bs, base in base_cache.items():
        if stopped():
            break
        for B0 in B0s:
            if stopped():
                break
            for B1 in B1s:
                if B1 < B0:
                    continue
                if stopped():
                    break
                for order in ("asc", "desc"):
                    if stopped():
                        break
                    s = _seed_set_from_model(base, B0, B1, cap=cap, order=order)
                    consider(s)
                    if stopped():
                        break

    # --- Also run a couple of pure-greedy random orderings for diversity ---
    for it in range(4):
        if stopped():
            break
        full = list(range(1, cap + 1))
        rng.shuffle(full)
        # simple fast greedy with pre-allocated array
        chosen = []
        cnp = np.empty(cap, dtype=np.int64)
        n = 0
        counts = np.zeros(2 * cap + 1, dtype=np.int32)
        for x in full:
            if n == 0:
                ns = np.array([2 * x], dtype=np.int64)
            else:
                ns = np.concatenate([cnp[:n] + x, np.array([2 * x], dtype=np.int64)])
            if (counts[ns] + 1).max() <= 2:
                chosen.append(x)
                cnp[n] = x
                n += 1
                counts[ns] += 1
        consider(chosen)

    # --- Repair/grow on the best to push size up (multiple restarts) ---
    if best and _time_left() > 8.0:
        for rep_seed in range(11, 21):
            if _time_left() < 6.0:
                break
            improved = _repair_grow(best, cap=cap, time_fn=time_left, stop_fn=stopped,
                                    max_rounds=300, seed=rep_seed, max_attempts=20000)
            consider(improved)

    if not best:
        # absolute fallback
        full = list(range(1, cap + 1))
        chosen = []
        cnp = np.empty(0, dtype=np.int64)
        counts = np.zeros(2 * cap + 1, dtype=np.int32)
        for x in full:
            ok, ns = _add_check(x, cnp, counts, cap)
            if ok:
                chosen.append(x)
                cnp = np.concatenate([cnp, np.array([x], dtype=np.int64)])
                counts[ns] += 1
        best = chosen

    # Final validation guard
    if best and not _valid(best, cap=cap):
        full = list(range(1, cap + 1))
        chosen = []
        cnp = np.empty(0, dtype=np.int64)
        counts = np.zeros(2 * cap + 1, dtype=np.int32)
        for x in full:
            ok, ns = _add_check(x, cnp, counts, cap)
            if ok:
                chosen.append(x)
                cnp = np.concatenate([cnp, np.array([x], dtype=np.int64)])
                counts[ns] += 1
        best = chosen

    return best
