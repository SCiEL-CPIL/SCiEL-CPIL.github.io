"""Size-efficient sorting network on 17 wires."""
import time
import random
import numpy as np

N = 17
ALL_MASKS = np.arange(1 << N, dtype=np.int64)


def _init_states():
    """Simulate a network of comparator pairs over all 2^N boolean inputs.
    Returns an array of shape (2^N, N) of uint8 where each row is the
    current state of the N wires for one input."""
    states = np.zeros((1 << N, N), dtype=np.uint8)
    for k in range(N):
        states[:, k] = ((ALL_MASKS >> k) & 1).astype(np.uint8)
    return states


def _time_left(deadline, margin=0.5):
    return (deadline - margin) - time.time()


def _apply_comparator(states, i, j):
    """Apply comparator (i, j): min to wire i, max to wire j."""
    a = states[:, i]
    b = states[:, j]
    states[:, i] = np.minimum(a, b)
    states[:, j] = np.maximum(a, b)


def _is_sorted(states):
    """Check whether states (2^N, N) is fully sorted on every row."""
    if N <= 1:
        return True
    return bool(np.all(states[:, :-1] <= states[:, 1:]))


def _apply_batch(states, comps):
    for (i, j) in comps:
        _apply_comparator(states, i, j)


def verify_network(comps):
    states = _init_states()
    _apply_batch(states, comps)
    return _is_sorted(states)


def _seed_network():
    # Odd-even transposition network: N phases, N(N-1)/2 comparators.
    return [(i, i + 1) for phase in range(N)
            for i in range(phase % 2, N - 1, 2)]


def _try_remove(comps, states, deadline, margin=0.5):
    """Try removing redundant comparators (greedy + random restarts)."""
    best_comps = list(comps)
    best_states = states.copy()

    # Greedy: iterate, try removing each, keep if still sorted.
    changed = True
    while changed:
        changed = False
        for idx in range(len(best_comps) - 1, -1, -1):
            if _time_left(deadline, margin) <= 0:
                return best_comps
            candidate = best_comps[:idx] + best_comps[idx + 1:]
            test_states = _init_states()
            _apply_batch(test_states, candidate)
            if _is_sorted(test_states):
                best_comps = candidate
                best_states = test_states
                changed = True

    # Randomized removal attempts with local search
    rng = random.Random(12345)
    for _ in range(50):
        if _time_left(deadline, margin) <= 0:
            break
        cur = list(best_comps)
        cur_states = best_states.copy()
        improved = True
        while improved:
            improved = False
            # Try removing a random comparator
            if cur:
                idx = rng.randrange(len(cur))
                candidate = cur[:idx] + cur[idx + 1:]
                test_states = _init_states()
                _apply_batch(test_states, candidate)
                if _is_sorted(test_states):
                    cur = candidate
                    cur_states = test_states
                    improved = True
            if len(cur) < len(best_comps):
                best_comps = cur
                best_states = cur_states
                improved = True
        if len(cur) < len(best_comps):
            best_comps = cur
            best_states = cur_states

    return best_comps


def _try_insert(comps, states, deadline, margin=0.5):
    """Try adding a comparator and then removing two to net -1, or just
    try swapping a comparator for a cheaper one. This is a placeholder for
    a 2-opt style move: remove one, add one, see if count drops.
    Since we only care about count, insertion only helps if it enables
    removal of more than one. We do a simple 1-in-2-out search."""
    best_comps = list(comps)
    best_states = states.copy()
    rng = random.Random(54321)

    for _ in range(30):
        if _time_left(deadline, margin) <= 0:
            break
        cur = list(best_comps)
        # Remove up to 2 random comparators
        remove_count = min(2, len(cur))
        if remove_count == 0:
            break
        idxs = rng.sample(range(len(cur)), remove_count)
        idxs.sort(reverse=True)
        for idx in idxs:
            cur.pop(idx)
        # Try adding up to 2 random comparators
        add_count = min(2, remove_count)
        for _ in range(add_count):
            i = rng.randrange(N)
            j = rng.randrange(N)
            if i == j:
                continue
            if i > j:
                i, j = j, i
            cur.append((i, j))
        test_states = _init_states()
        _apply_batch(test_states, cur)
        if _is_sorted(test_states) and len(cur) < len(best_comps):
            best_comps = cur
            best_states = test_states

    return best_comps


def construct():
    deadline = time.time() + 80.0  # 10s safety margin before 90s kill
    comps = _seed_network()
    states = _init_states()
    _apply_batch(states, comps)
    # Verify seed is correct
    assert _is_sorted(states), "Seed network should be correct"

    # Phase 1: greedy + randomized removal
    comps = _try_remove(comps, states, deadline, margin=5.0)

    # Phase 2: try 1-in-2-out improvements
    states = _init_states()
    _apply_batch(states, comps)
    comps = _try_insert(comps, states, deadline, margin=2.0)

    # Final verification
    final_states = _init_states()
    _apply_batch(final_states, comps)
    if not _is_sorted(final_states):
        # Fallback to seed if something went wrong
        return _seed_network()

    return comps
