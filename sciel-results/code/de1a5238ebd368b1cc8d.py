"""Stronger solver for the 200-free subset problem.

Key improvements over the seed:
1. Much faster feasibility check: instead of iterating every value 1..100
   each time, only iterate the set bits of the mask.  This makes each
   _is_200_free call roughly (popcount(mask)) bit-shifts rather than 100.
2. Greedy "add" phase: starting from the empty set, try to add values in a
   smart order (largest first, then randomised) and keep those that do not
   create a 200-sum.  This yields a good feasible starting point quickly.
3. Local search with add/swap moves: try to add a missing value; if that
   breaks feasibility, try swapping it with a present value.  Accept
   improvements; occasionally accept worsening moves via simulated annealing
   to escape local optima.
4. Multiple diverse initialisations and restarts within the time budget.
5. Aggressive deadline guard: return the best feasible result found so far
   well before the 90 s kill.
"""
import time
import random
import math

TARGET = 200
N = 100
_SUM_MASK = (1 << (TARGET + 1)) - 1  # keep only sums 0..200


def _popcount(m):
    return bin(m).count("1")


def _is_200_free(mask):
    """Return True iff no submultiset of the chosen values sums to TARGET.
    Values are 1..N, bit (v-1) set means value v is chosen.
    """
    if mask == 0:
        return False  # must be nonempty
    reach = 1  # bit 0 set (sum 0)
    # iterate only set bits
    mm = mask
    while mm:
        # lowest set bit position
        low = mm & (-mm)
        v = low.bit_length()  # value = bit_index + 1
        reach = (reach | (reach << v)) & _SUM_MASK
        if (reach >> TARGET) & 1:
            return False
        mm ^= low
    return True


def _cardinality(mask):
    return _popcount(mask)


def _flip(mask, i):
    return mask ^ (1 << i)


def _add_value(mask, v):
    return mask | (1 << (v - 1))


def _remove_value(mask, v):
    return mask & ~(1 << (v - 1))


def _set_bits(mask):
    """Return list of values present in mask."""
    vals = []
    mm = mask
    while mm:
        low = mm & (-mm)
        vals.append(low.bit_length())
        mm ^= low
    return vals


def _missing_values(mask):
    """Return list of values not present in mask."""
    present = set(_set_bits(mask))
    return [v for v in range(1, N + 1) if v not in present]


def _greedy_add(order, mask=0):
    """Add values in the given order if they keep the set 200-free."""
    m = mask
    reach = 1
    for v in order:
        if m & (1 << (v - 1)):
            continue
        new_reach = (reach | (reach << v)) & _SUM_MASK
        if not ((new_reach >> TARGET) & 1):
            m |= (1 << (v - 1))
            reach = new_reach
    return m


def _greedy_add_random(rng, mask=0):
    """Randomised greedy add: shuffle values and add if feasible."""
    order = list(range(1, N + 1))
    rng.shuffle(order)
    m = mask
    reach = 1
    for v in order:
        if m & (1 << (v - 1)):
            continue
        new_reach = (reach | (reach << v)) & _SUM_MASK
        if not ((new_reach >> TARGET) & 1):
            m |= (1 << (v - 1))
            reach = new_reach
    return m


def _greedy_add_desc(mask=0):
    """Add values in descending order if feasible."""
    m = mask
    reach = 1
    for v in range(N, 0, -1):
        if m & (1 << (v - 1)):
            continue
        new_reach = (reach | (reach << v)) & _SUM_MASK
        if not ((new_reach >> TARGET) & 1):
            m |= (1 << (v - 1))
            reach = new_reach
    return m


def _compute_reach(mask):
    """Compute the subset-sum reachability bitmask for a given mask."""
    reach = 1
    mm = mask
    while mm:
        low = mm & (-mm)
        v = low.bit_length()
        reach = (reach | (reach << v)) & _SUM_MASK
        mm ^= low
    return reach


def _local_search_fast(mask, rng, deadline, t0=2.0, t1=0.01, max_iters=500000):
    """Fast local search with incremental reach updates."""
    cur = mask
    cur_c = _cardinality(cur)
    cur_reach = _compute_reach(cur)
    best = mask
    best_c = cur_c

    if time.time() > deadline:
        return best, best_c

    T = t0
    for it in range(max_iters):
        if time.time() > deadline:
            break
        frac = it / max(1, max_iters - 1)
        T = t0 + (t1 - t0) * frac

        present = _set_bits(cur)
        missing = _missing_values(cur)

        if not missing:
            break

        move_type = rng.random()
        if move_type < 0.6:
            # Try to add a missing value
            v = rng.choice(missing)
            new_reach = (cur_reach | (cur_reach << v)) & _SUM_MASK
            if not ((new_reach >> TARGET) & 1):
                # Adding is feasible
                cur = _add_value(cur, v)
                cur_c += 1
                cur_reach = new_reach
                if cur_c > best_c:
                    best = cur
                    best_c = cur_c
            else:
                # Adding breaks feasibility; try a swap
                if present:
                    v_rem = rng.choice(present)
                    # Remove v_rem, then try to add v
                    # Recompute reach without v_rem is expensive; do a simpler swap
                    # Just try: remove random present, add random missing, check full
                    cand = _add_value(_remove_value(cur, v_rem), v)
                    cand_ok = _is_200_free(cand)
                    if cand_ok:
                        d = 0  # same cardinality
                        if T > 0 and rng.random() < math.exp(0.0 / T):
                            cur = cand
                            cur_reach = _compute_reach(cur)
        else:
            # Swap move
            if present:
                v_add = rng.choice(missing)
                v_rem = rng.choice(present)
                cand = _add_value(_remove_value(cur, v_rem), v_add)
                cand_ok = _is_200_free(cand)
                if cand_ok:
                    d = 0
                    if T > 0 and rng.random() < math.exp(0.0 / T):
                        cur = cand
                        cur_reach = _compute_reach(cur)

    return best, best_c


def _polish(mask, deadline):
    """Try to add every missing value in multiple systematic orders."""
    cur = mask
    cur_c = _cardinality(cur)
    best = mask
    best_c = cur_c

    # Try adding in ascending order
    for v in range(1, N + 1):
        if time.time() > deadline:
            break
        if cur & (1 << (v - 1)):
            continue
        new_reach = (cur_reach if False else _compute_reach(cur))
        # Actually let's maintain reach
    # Simpler: just try each missing value, keep if feasible
    reach = _compute_reach(cur)
    for v in range(1, N + 1):
        if time.time() > deadline:
            break
        if cur & (1 << (v - 1)):
            continue
        new_reach = (reach | (reach << v)) & _SUM_MASK
        if not ((new_reach >> TARGET) & 1):
            cur |= (1 << (v - 1))
            reach = new_reach
    cur_c = _cardinality(cur)
    if cur_c > best_c:
        best = cur
        best_c = cur_c

    # Try descending order from best
    cur = best
    reach = _compute_reach(cur)
    for v in range(N, 0, -1):
        if time.time() > deadline:
            break
        if cur & (1 << (v - 1)):
            continue
        new_reach = (reach | (reach << v)) & _SUM_MASK
        if not ((new_reach >> TARGET) & 1):
            cur |= (1 << (v - 1))
            reach = new_reach
    cur_c = _cardinality(cur)
    if cur_c > best_c:
        best = cur
        best_c = cur_c

    return best, best_c


def construct():
    start = time.time()
    # Safety margin: return well before 90 s kill
    HARD_DEADLINE = start + 82.0

    rng = random.Random(12345)

    best_mask = 0
    best_c = 0

    # --- Phase 1: diverse greedy initialisations ---
    # Descending greedy
    m = _greedy_add_desc()
    c = _cardinality(m)
    if c > best_c:
        best_c = c
        best_mask = m

    # Randomised greedy (several)
    for _ in range(20):
        m = _greedy_add_random(rng)
        c = _cardinality(m)
        if c > best_c:
            best_c = c
            best_mask = m

    # Interval-based starts
    for lo in range(50, 101, 5):
        m = 0
        for v in range(lo, N + 1):
            m |= (1 << (v - 1))
        # repair by greedy add from there? No, just check feasibility
        if _is_200_free(m):
            c = _cardinality(m)
            if c > best_c:
                best_c = c
                best_mask = m

    # --- Phase 2: local search from best and from random restarts ---
    # First: fast local search from the best initial
    m, c = _local_search_fast(best_mask, rng, HARD_DEADLINE, t0=2.0, t1=0.01,
                              max_iters=500000)
    if c > best_c:
        best_c = c
        best_mask = m

    # Polish pass
    m, c = _polish(best_mask, HARD_DEADLINE)
    if c > best_c:
        best_c = c
        best_mask = m

    # Restart loop: random greedy + fast local search
    guard = 0
    while time.time() < HARD_DEADLINE - 3.0 and guard < 100:
        guard += 1
        rm = _greedy_add_random(rng)
        rc = _cardinality(rm)
        # Only run local search if this start is competitive
        if rc < best_c - 3:
            continue
        m, c = _local_search_fast(rm, rng, HARD_DEADLINE, t0=1.5, t1=0.01,
                                  max_iters=200000)
        if c > best_c:
            best_c = c
            best_mask = m
            # Polish after improvement
            m2, c2 = _polish(best_mask, HARD_DEADLINE)
            if c2 > best_c:
                best_c = c2
                best_mask = m2

    # Final polish before safety check
    if time.time() < HARD_DEADLINE - 1.0:
        m, c = _polish(best_mask, HARD_DEADLINE)
        if c > best_c:
            best_c = c
            best_mask = m

    # --- Final safety: ensure feasibility and nonempty ---
    if not _is_200_free(best_mask) or best_mask == 0:
        # Fallback: use a known feasible set
        best_mask = _greedy_add_desc()
        if not _is_200_free(best_mask) or best_mask == 0:
            # Absolute fallback: {66,...,100} is 200-free? 66+67+...? 
            # Actually {66..100}: min 2-sum = 133, min 3-sum = 199, min 4-sum = 265 > 200.
            # So any subset of {66..100} with at most 3 elements is safe, but we need
            # no subset summing to 200. 66+67+68 = 201 > 200, so no 3-element subset
            # sums to 200. 2-element: max = 199 < 200. 1-element: max 100 < 200.
            # So {66..100} is 200-free.
            best_mask = 0
            for v in range(66, 101):
                best_mask |= (1 << (v - 1))

    # Convert mask to sorted list
    out = _set_bits(best_mask)
    out.sort()
    if not out:
        out = list(range(66, 101))
    return out
