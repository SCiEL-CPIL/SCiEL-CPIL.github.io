"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

MASK = (1 << 201) - 1  # keep only bits 0..200
TARGET = 200
ALLNUMS = list(range(1, 101))


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


def _local_search(cur, deadline, rng):
    """Aggressive add + swap local search on a valid set.
    Returns the improved set (list)."""
    cur = list(cur)
    cur_set = set(cur)
    cur_bits = _reach_bitset(cur)
    allnums = set(ALLNUMS)

    def add_ok(x):
        # adding x creates 200 iff (200 - x) is currently reachable
        return not ((cur_bits >> (TARGET - x)) & 1)

    def try_add(x):
        nonlocal cur, cur_set, cur_bits
        if x in cur_set:
            return False
        if not add_ok(x):
            return False
        cur_bits |= (cur_bits << x)
        cur_bits &= MASK
        cur.append(x)
        cur_set.add(x)
        return True

    # add-only passes until saturation
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for x in sorted(allnums - cur_set, reverse=True):
            if time.time() > deadline:
                break
            if try_add(x):
                improved = True

    # swap passes: drop one element, add another (possibly two to compensate)
    improved = True
    iters = 0
    while improved and time.time() < deadline:
        improved = False
        iters += 1
        # try to add any currently-blocked number by dropping some element
        for add in sorted(allnums - cur_set, reverse=True):
            if time.time() > deadline:
                break
            if add_ok(add):
                if try_add(add):
                    improved = True
                continue
            # `add` blocked: try dropping each element of cur
            for drop in cur:
                if time.time() > deadline:
                    break
                # simulate: remove drop, recompute reachability
                bits_without = 1
                for y in cur:
                    if y == drop:
                        continue
                    bits_without |= (bits_without << y)
                    bits_without &= MASK
                if (bits_without >> TARGET) & 1:
                    continue  # still invalid after drop (shouldn't happen)
                nb = bits_without | (bits_without << add)
                nb &= MASK
                if (nb >> TARGET) & 1:
                    continue  # adding still creates 200
                # commit swap
                cur.remove(drop)
                cur_set.discard(drop)
                cur_bits = nb
                cur.append(add)
                cur_set.add(add)
                improved = True
                break  # restart scan with new state
        # occasional: try a 1-drop that unblocks multiple adds
        if time.time() > deadline:
            break

    # final add-only sweep
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for x in sorted(allnums - cur_set, reverse=True):
            if time.time() > deadline:
                break
            if try_add(x):
                improved = True

    return cur


def construct():
    start = time.time()
    deadline = start + 78.0  # safety margin under 90 s kill

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

    rng = random.Random(20240517)

    # ---- Phase 1: many diverse greedy seeds ----
    orders = [
        list(range(100, 0, -1)),   # descending
        list(range(1, 101)),       # ascending
        list(range(50, 0, -1)) + list(range(100, 50, -1)),
        list(range(100, 49, -1)) + list(range(49, 0, -1)),
    ]
    for _ in range(120):
        if time.time() > deadline:
            break
        o = list(ALLNUMS)
        rng.shuffle(o)
        orders.append(o)

    for o in orders:
        if time.time() > deadline:
            break
        sel, _ = _greedy(o, deadline)
        try_set(sel)

    if best is None:
        return [100]  # guaranteed valid fallback

    # ---- Phase 2: local search from best seed and from a few other seeds ----
    seed_pool = [list(best)]
    # add a few more seeds for diversity
    extra = [
        list(range(100, 0, -1)),
        list(range(1, 101)),
    ]
    for o in extra:
        if time.time() > deadline:
            break
        sel, _ = _greedy(o, deadline)
        if sel and not _has_200(sel):
            seed_pool.append(sel)

    for seed in seed_pool:
        if time.time() > deadline:
            break
        improved = _local_search(seed, deadline, rng)
        try_set(improved)

    # ---- Phase 3: multi-restart local search with random perturbations ----
    while time.time() < deadline - 5.0:
        base = list(best)
        # perturb: remove a random small number of elements, then re-local-search
        k = rng.randint(1, 4)
        pert = base[:]
        if len(pert) > k:
            for _ in range(k):
                pert.pop(rng.randrange(len(pert)))
        if _has_200(pert):
            continue
        improved = _local_search(pert, deadline, rng)
        try_set(improved)

    best = sorted(set(best))
    # final safety: ensure validity
    if _has_200(best):
        # extremely unlikely; fall back to a known-valid set
        return [100]
    return best
# EVOLVE-BLOCK-END
