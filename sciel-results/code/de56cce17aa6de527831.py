"""Structural rewrite: greedy comparator removal + random restarts under a deadline guard.

Strategy:
  1. Start from the odd-even transposition seed network (153 comparators).
  2. Apply a fast, linear greedy pass: scan comparators in order, and for each one,
     check whether removing it still yields a valid sorting network (0-1 principle,
     all 2**18 Boolean inputs). If removable, drop it and continue (no restart).
  3. Repeat passes until a full pass removes nothing (local minimum).
  4. Random restarts: occasionally perturb the network (insert random comparators /
     reorder) to escape local minima, then re-run greedy reduction.
  5. Keep the best (fewest-comparator) valid network found.
  6. Hard deadline guard: return the best result with a safety margin before 90s.

The 0-1 principle: a compare-exchange network sorts all inputs iff it sorts all
2**N Boolean inputs. For N=18 that is 262,144 inputs, which is tractable.
"""
import time
import random

N = 18
SEED_COMPARATORS = 153  # odd-even transposition network size
DEADLINE_SECONDS = 90.0
SAFETY_MARGIN = 8.0     # stop searching this many seconds before the hard limit


def _odd_even_transposition():
    """Return the odd-even transposition sorting network for N wires."""
    return [(i, i + 1) for phase in range(N)
            for i in range(phase % 2, N - 1, 2)]


def _fast_verify(comparators):
    """Return True iff the comparator list sorts all 2**N Boolean inputs.

    Uses the 0-1 principle. For each of the 2**N masks, simulate the network
    (each comparator writes min to i, max to j) and check the final state is
    non-increasing (sorted).
    """
    n = N
    total = 1 << n
    for mask in range(total):
        # wire w holds bit w of mask
        state = [ (mask >> w) & 1 for w in range(n) ]
        for (i, j) in comparators:
            a = state[i]
            b = state[j]
            if a > b:
                state[i] = b
                state[j] = a
        # check sorted: state[0] >= state[1] >= ... >= state[n-1]
        # i.e. no inversion: for all k, state[k] >= state[k+1]
        ok = True
        for k in range(n - 1):
            if state[k] < state[k + 1]:
                ok = False
                break
        if not ok:
            return False
    return True


def _greedy_reduce(comparators, deadline):
    """Linear greedy removal pass. Returns (reduced_comparators, changed_flag).

    Scans comparators in order; for each, tentatively remove it and verify.
    If still valid, drop it and continue without restarting the scan.
    Stops early if the deadline is approached.
    """
    cur = list(comparators)
    changed = False
    i = 0
    while i < len(cur):
        if time.time() > deadline:
            break
        # try removing cur[i]
        candidate = cur[:i] + cur[i+1:]
        if _fast_verify(candidate):
            cur = candidate
            changed = True
            # do not increment i: re-check the same index (now the next element)
        else:
            i += 1
    return cur, changed


def _perturb(comparators, rng):
    """Create a perturbed copy: insert a few random comparators and shuffle a window.

    This gives the greedy pass a different starting point to escape local minima.
    """
    cur = list(comparators)
    # insert 1-3 random comparators
    for _ in range(rng.randint(1, 3)):
        i = rng.randrange(N)
        j = rng.randrange(N)
        if i == j:
            continue
        if i > j:
            i, j = j, i
        pos = rng.randrange(len(cur) + 1)
        cur.insert(pos, (i, j))
    # shuffle a small window
    if len(cur) >= 4:
        start = rng.randrange(len(cur) - 3)
        window = cur[start:start + rng.randint(2, 4)]
        rng.shuffle(window)
        cur[start:start + len(window)] = window
    return cur


def construct():
    """Construct a compare-exchange network that sorts all 18-input sequences.

    Returns an ordered list of integer pairs (i, j), 0 <= i < j < N.
    """
    t0 = time.time()
    deadline = t0 + DEADLINE_SECONDS - SAFETY_MARGIN
    rng = random.Random(12345)

    best = _odd_even_transposition()
    best_len = len(best)

    # First, reduce the seed network as much as possible (no perturbations).
    while True:
        if time.time() > deadline:
            break
        reduced, changed = _greedy_reduce(best, deadline)
        if not changed:
            break
        best = reduced
        best_len = len(best)

    # Random restarts: perturb, reduce, keep the best.
    while time.time() < deadline:
        perturbed = _perturb(best, rng)
        reduced, _ = _greedy_reduce(perturbed, deadline)
        if len(reduced) < best_len:
            best = reduced
            best_len = len(reduced)

    # Final safety: ensure the returned network is valid.
    # (It should be, since we only keep verified networks, but guard anyway.)
    if not _fast_verify(best):
        # Fallback to the seed network, which is known valid.
        best = _odd_even_transposition()

    return best
