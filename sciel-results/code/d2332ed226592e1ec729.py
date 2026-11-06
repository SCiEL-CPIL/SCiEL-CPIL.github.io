"""Combined program: one-pass incremental validation + multi-start local search."""
import time
import random
import numpy as np

N = 17
SEED_COMPARATORS = 136

_ALL_INPUTS = None

def _make_all_inputs():
    n_inputs = 1 << N
    indices = np.arange(n_inputs, dtype=np.uint32)[:, None]
    bit_positions = np.arange(N, dtype=np.uint32)[None, :]
    inputs = ((indices >> bit_positions) & 1).astype(np.uint8)
    return inputs

def _get_all_inputs():
    global _ALL_INPUTS
    if _ALL_INPUTS is None:
        _ALL_INPUTS = _make_all_inputs()
    return _ALL_INPUTS

def _simulate(comparators, inputs):
    state = inputs.copy()
    for (i, j) in comparators:
        mi = state[:, i] & state[:, j]
        mj = state[:, i] | state[:, j]
        state[:, i] = mi
        state[:, j] = mj
    return state

def is_sorted_network(comparators):
    """Full check on all 2^N Boolean inputs (one pass)."""
    inputs = _get_all_inputs()
    output = _simulate(comparators, inputs)
    for k in range(N - 1):
        if np.any(output[:, k] > output[:, k + 1]):
            return False
    return True

def _simulate_prefix(comparators, inputs, depth):
    """Simulate only the first `depth` comparators; return the intermediate state."""
    state = inputs.copy()
    for idx in range(min(depth, len(comparators))):
        i, j = comparators[idx]
        mi = state[:, i] & state[:, j]
        mj = state[:, i] | state[:, j]
        state[:, i] = mi
        state[:, j] = mj
    return state

def _fast_candidate_ok(best, i, inputs, prefix):
    """
    Fast necessary filter for the candidate that removes best[i].
    Simulates best up to depth i-1 (shared prefix), then branches:
      (a) apply best[i], continue to the end
      (b) skip best[i], continue to the end
    Returns True only if BOTH full networks sort all inputs (a genuine certificate
    for the removal), else False.
    """
    n = len(best)
    i, j = best[i]
    st = prefix.copy()
    mi = st[:, i] & st[:, j]
    mj = st[:, i] | st[:, j]
    st[:, i] = mi
    st[:, j] = mj
    for idx in range(i + 1, n):
        a, b = best[idx]
        x = st[:, a] & st[:, b]
        y = st[:, a] | st[:, b]
        st[:, a] = x
        st[:, b] = y
    for k in range(N - 1):
        if np.any(st[:, k] > st[:, k + 1]):
            return False
    st2 = prefix.copy()
    for idx in range(i + 1, n):
        a, b = best[idx]
        x = st2[:, a] & st2[:, b]
        y = st2[:, a] | st2[:, b]
        st2[:, a] = x
        st2[:, b] = y
    for k in range(N - 1):
        if np.any(st2[:, k] > st2[:, k + 1]):
            return False
    return True

def odd_even_merge_sort_network(n):
    if n == 1:
        return []

    def _build(n):
        if n == 1:
            return []
        if n == 2:
            return [(0, 1)]
        half = n // 2
        left = _build(half)
        right = [(i + half, j + half) for (i, j) in _build(n - half)]
        merged = _odd_even_merge(n)
        return left + right + merged

    def _odd_even_merge(n):
        if n <= 1:
            return []
        if n == 2:
            return [(0, 1)]
        comparators = []
        if n % 2 == 0:
            for i in range(0, n, 2):
                comparators.append((i, i + 1))
            even_merged = _odd_even_merge(n // 2)
            for (a, b) in even_merged:
                comparators.append((2 * a, 2 * b))
            odd_merged = _odd_even_merge(n // 2)
            for (a, b) in odd_merged:
                comparators.append((2 * a + 1, 2 * b + 1))
        else:
            for i in range(1, n - 1, 2):
                comparators.append((i, i + 1))
            if n - 2 > 0:
                mid_merged = _odd_even_merge(n - 2)
                for (a, b) in mid_merged:
                    comparators.append((a + 1, b + 1))
        return comparators

    return _build(n)

def odd_even_transposition_network(n):
    return [(i, i + 1) for phase in range(n)
            for i in range(phase % 2, n - 1, 2)]

def _remove_pass(best, inputs, deadline):
    """Sweep every position; drop any comparator whose removal keeps the network
    correct. Re-sweeps (without advancing) after a drop so newly-exposed
    redundancies are also caught. Returns a new list."""
    best = list(best)
    i = 0
    while i < len(best):
        if time.time() > deadline - 2:
            break
        prefix = _simulate_prefix(best, inputs, i)
        if _fast_candidate_ok(best, i, inputs, prefix):
            best = best[:i] + best[i + 1:]
            # stay at i to re-examine
        else:
            i += 1
    return best

def _insert_remove(best, inputs, deadline, max_attempts=40):
    """Insert a random comparator at a random position, then prune; keep if shorter."""
    best = list(best)
    best_len = len(best)
    for _ in range(max_attempts):
        if time.time() > deadline - 4:
            break
        pos = random.randint(0, len(best))
        i = random.randint(0, N - 2)
        j = random.randint(i + 1, N - 1)
        candidate = best[:pos] + [(i, j)] + best[pos:]
        candidate = _remove_pass(candidate, inputs, deadline)
        if len(candidate) < best_len:
            best = candidate
            best_len = len(best)
    return best

def _swap_remove(best, inputs, deadline, max_attempts=30):
    """Swap two comparators, then prune; keep if shorter (escape local minima)."""
    best = list(best)
    best_len = len(best)
    for _ in range(max_attempts):
        if time.time() > deadline - 4:
            break
        if len(best) < 2:
            break
        p1 = random.randint(0, len(best) - 1)
        p2 = random.randint(0, len(best) - 1)
        if p1 == p2:
            continue
        candidate = best[:]
        candidate[p1], candidate[p2] = candidate[p2], candidate[p1]
        candidate = _remove_pass(candidate, inputs, deadline)
        if len(candidate) < best_len:
            best = candidate
            best_len = len(best)
    return best

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 86  # safety margin before the 90s kill
    inputs = _get_all_inputs()

    base_merge = odd_even_merge_sort_network(N)
    if not is_sorted_network(base_merge):
        base_merge = odd_even_transposition_network(N)
    base_trans = odd_even_transposition_network(N)

    best = list(base_merge)
    best_len = len(best)

    # Multi-start: each iteration begins from a fresh base and prunes.
    # The one-pass incremental validation is cheap enough to afford several
    # independent starts, which escapes local minima that a single run cannot.
    starts = 0
    while time.time() < deadline - 12:
        starts += 1
        if starts == 1:
            cur = list(base_merge)
        elif starts % 2 == 0:
            cur = list(base_trans)
        else:
            cur = list(base_merge)
            # perturb the base a bit for variety
            for _ in range(3):
                pos = random.randint(0, len(cur))
                a = random.randint(0, N - 2)
                b = random.randint(a + 1, N - 1)
                cur = cur[:pos] + [(a, b)] + cur[pos:]

        cur = _remove_pass(cur, inputs, deadline)
        cur = _insert_remove(cur, inputs, deadline, max_attempts=30)
        cur = _swap_remove(cur, inputs, deadline, max_attempts=20)
        cur = _remove_pass(cur, inputs, deadline)

        if len(cur) < best_len:
            best = cur
            best_len = len(best)

    # Final cleanup pass on the overall best.
    if time.time() < deadline - 4:
        final = _remove_pass(best, inputs, deadline)
        if len(final) < best_len:
            best = final
            best_len = len(best)

    if not is_sorted_network(best):
        return base_trans
    return best
# EVOLVE-BLOCK-END
