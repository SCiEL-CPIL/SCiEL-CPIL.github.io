"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

MASK = (1 << 201) - 1  # keep only bits 0..200
TARGET = 200


def _reach_bitset(selected):
    """Bitset of all subset sums of `selected` that are <= 200.
    Bit i set  <=>  sum i is reachable.  Bit 0 always set."""
    bits = 1
    for x in selected:
        bits |= (bits << x)
        bits &= MASK
    return bits


def _has_200(selected):
    """True if some subset of `selected` sums to 200 (each element once)."""
    bits = 1
    for x in selected:
        bits |= (bits << x)
        bits &= MASK
        if (bits >> TARGET) & 1:
            return True
    return False


def _greedy(order, deadline):
    """Greedy: add x (in given order) iff it keeps 200 unreachable.
    Returns (selected, reachable_bits)."""
    sel = []
    reach = 1
    for x in order:
        if time.time() > deadline:
            break
        new_bits = reach | (reach << x)
        new_bits &= MASK
        if not ((new_bits >> TARGET) & 1):
            sel.append(x)
            reach = new_bits
    return sel, reach


def construct():
    start = time.time()
    deadline = start + 70.0  # safety margin well under the 90 s kill

    best = None
    best_size = -1

    def try_set(s):
        nonlocal best, best_size
        if not s:
            return
        if _has_200(s):
            return
        if len(s) > best_size:
            best = list(s)
            best_size = len(s)

    # ---- Phase 1: multi-order greedy seeds (fast) ----
    orders = [
        list(range(100, 0, -1)),   # descending
        list(range(1, 101)),       # ascending
    ]
    rng = random.Random(20240517)
    for _ in range(40):
        if time.time() > deadline:
            break
        o = list(range(1, 101))
        rng.shuffle(o)
        orders.append(o)

    for o in orders:
        if time.time() > deadline:
            break
        sel, _ = _greedy(o, deadline)
        try_set(sel)

    if best is None:
        return [100]  # guaranteed valid fallback

    # ---- Phase 2: local search (add + swap) on the best seed ----
    cur = list(best)
    cur_set = set(cur)
    cur_bits = _reach_bitset(cur)
    cur_size = len(cur)
    allnums = set(range(1, 101))

    def add_ok(x):
        # can we add x without creating sum 200?
        return not ((cur_bits >> (TARGET - x)) & 1)

    def try_add(x):
        nonlocal cur, cur_set, cur_bits, cur_size
        if x in cur_set:
            return False
        if not add_ok(x):
            return False
        cur_bits |= (cur_bits << x)
        cur_bits &= MASK
        cur.append(x)
        cur_set.add(x)
        cur_size += 1
        return True

    # 2a: add-only pass (repeated until no progress)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for x in sorted(allnums - cur_set, reverse=True):
            if time.time() > deadline:
                break
            if try_add(x):
                improved = True

    # 2b: swap pass (drop one, add another) -- escapes add-only local optima
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # candidate additions that are currently blocked
        for add in sorted(allnums - cur_set, reverse=True):
            if time.time() > deadline:
                break
            if add_ok(add):
                # not blocked: just take it (cheap win)
                if try_add(add):
                    improved = True
                continue
            # `add` is blocked: find a `drop` whose removal unblocks it
            # add is blocked iff there exists s in cur_bits with s == 200 - add.
            # Removing `drop` kills sum s only if s is reachable solely via drop.
            # Cheap sufficient test: after removing drop, recompute and check.
            # To keep it fast, only try dropping elements > add (they block more).
            for drop in cur:
                if time.time() > deadline:
                    break
                if drop <= add:
                    continue
                # simulate removal of `drop`
                bits_without = 1
                for y in cur:
                    if y == drop:
                        continue
                    bits_without |= (bits_without << y)
                    bits_without &= MASK
                    if (bits_without >> TARGET) & 1:
                        break
                if (bits_without >> TARGET) & 1:
                    continue  # set still has 200 after drop (shouldn't, but safe)
                # now check add
                nb = bits_without | (bits_without << add)
                nb &= MASK
                if (nb >> TARGET) & 1:
                    continue
                # commit the swap
                cur.remove(drop)
                cur_set.discard(drop)
                cur_bits = nb
                cur.append(add)
                cur_set.add(add)
                cur_size = len(cur)
                improved = True
                break  # restart add scan with new state

    # 2c: final add-only sweep
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for x in sorted(allnums - cur_set, reverse=True):
            if time.time() > deadline:
                break
            if try_add(x):
                improved = True

    # 2d: randomized 2-opt local search (remove 2, add 1) with SA-style acceptance
    # This explores moves that the single-swap pass cannot reach.
    if time.time() < deadline:
        rng2 = random.Random(99731)
        cur_list = list(cur)
        cur_set2 = set(cur_list)
        cur_size2 = len(cur_list)
        T_init = 2.0
        T_min = 0.01
        cooling = 0.97
        T = T_init

        def _recompute_bits(lst):
            b = 1
            for y in lst:
                b |= (b << y)
                b &= MASK
            return b

        cur_bits2 = _recompute_bits(cur_list)

        while time.time() < deadline and T > T_min:
            # Try to add a missing element
            missing = [x for x in range(1, 101) if x not in cur_set2]
            if missing:
                add_cand = rng2.choice(missing)
                if not ((cur_bits2 >> (TARGET - add_cand)) & 1):
                    # Can add directly
                    cur_bits2 |= (cur_bits2 << add_cand)
                    cur_bits2 &= MASK
                    cur_list.append(add_cand)
                    cur_set2.add(add_cand)
                    cur_size2 += 1
                    continue

            # Blocked: try 2-opt (remove 2 elements, add 1)
            if len(cur_list) >= 2:
                d1, d2 = rng2.sample(cur_list, 2)
                # Pick an add candidate from missing
                missing_now = [x for x in range(1, 101) if x not in cur_set2]
                if not missing_now:
                    continue
                add_cand = rng2.choice(missing_now)

                # Simulate: remove d1, d2, add add_cand
                bits_tmp = 1
                for y in cur_list:
                    if y == d1 or y == d2:
                        continue
                    bits_tmp |= (bits_tmp << y)
                    bits_tmp &= MASK
                    if (bits_tmp >> TARGET) & 1:
                        break
                if (bits_tmp >> TARGET) & 1:
                    T *= cooling
                    continue

                nb = bits_tmp | (bits_tmp << add_cand)
                nb &= MASK
                if (nb >> TARGET) & 1:
                    T *= cooling
                    continue

                # Valid move: new size = cur_size2 - 2 + 1 = cur_size2 - 1
                # This is a worsening move (lose 1 element), so use SA acceptance
                delta = -1  # size change
                if delta >= 0 or rng2.random() < (1.0 / (1.0 + (1.0 - T) * 10.0)):
                    # Accept
                    cur_list.remove(d1)
                    cur_list.remove(d2)
                    cur_set2.discard(d1)
                    cur_set2.discard(d2)
                    cur_list.append(add_cand)
                    cur_set2.add(add_cand)
                    cur_size2 = len(cur_list)
                    cur_bits2 = nb
                else:
                    T *= cooling
            else:
                T *= cooling

            # Cool
            T *= cooling

        # After SA, do a final add-only sweep on the SA result
        if time.time() < deadline:
            improved = True
            while improved and time.time() < deadline:
                improved = False
                for x in sorted(set(range(1, 101)) - cur_set2, reverse=True):
                    if time.time() > deadline:
                        break
                    if not ((cur_bits2 >> (TARGET - x)) & 1):
                        cur_bits2 |= (cur_bits2 << x)
                        cur_bits2 &= MASK
                        cur_list.append(x)
                        cur_set2.add(x)
                        improved = True

        # Update best if SA result is better
        if cur_list and not _has_200(cur_list):
            if len(cur_list) > best_size:
                best = list(cur_list)
                best_size = len(cur_list)

    # keep the best of everything
    if not _has_200(cur):
        if len(cur) > best_size:
            best = cur
    best = sorted(set(best))
    return best
# EVOLVE-BLOCK-END
