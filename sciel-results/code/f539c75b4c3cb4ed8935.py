"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

N = 17
MAX_COMPARATORS = 4096
DEADLINE_SECONDS = 85  # safety margin under 90s kill


def _build_seed_network():
    """Odd-even transposition network for N wires."""
    return [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]


def _init_wires():
    """Initialize wire bitmasks for all Boolean inputs.

    For N wires, there are 2^N boolean inputs.
    wires[i] is a bitmask of length 2^N where bit k is 1 iff input k has a 1 on wire i.
    """
    total = 1 << N
    wires = []
    for i in range(N):
        val = 0
        block = 1 << i
        period = 2 * block
        for start in range(0, total, period):
            mask = ((1 << block) - 1) << (start + block)
            val |= mask
        wires.append(val)
    return wires


def _compute_expected():
    """Compute expected final wire values after sorting.

    After sorting, wire i (0-indexed, 0 is smallest) should contain 1s for all inputs
    where the (N-i)-th smallest value is 1.
    Since we're dealing with boolean inputs, the sorted output on wire i is 1 iff
    the input has at least (N-i) ones.
    """
    expected = []
    total = 1 << N
    for i in range(N):
        val = 0
        threshold = N - i
        for k in range(threshold, total):
            if k.bit_count() >= threshold:
                val |= (1 << k)
        expected.append(val)
    return expected


def _verify_network_bitmask(comparators, wires_init=None, expected=None):
    """Verify network sorts all Boolean inputs using bitmask simulation.

    This is the fastest verification method using Python big integers.
    Each wire is represented as a single big integer where bit k indicates
    the value on that wire for input k.
    """
    if wires_init is None:
        wires_init = _init_wires()
    if expected is None:
        expected = _compute_expected()

    wires = wires_init[:]
    for (a, b) in comparators:
        ma = wires[a] & wires[b]
        mb = wires[a] | wires[b]
        wires[a] = ma
        wires[b] = mb

    for i in range(N):
        if wires[i] != expected[i]:
            return False
    return True


def _depth_of_network(comps):
    """Compute parallel depth (longest path) of the network."""
    if not comps:
        return 0
    depth = [0] * N
    for (i, j) in comps:
        d = max(depth[i], depth[j]) + 1
        depth[i] = d
        depth[j] = d
    return max(depth)


def _greedy_remove(comps, wires_init, expected, deadline):
    """Repeatedly try to remove each comparator until a full pass removes none.
    Returns the reduced list."""
    comps = list(comps)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        n = len(comps)
        # Scan from the end so removals don't shift indices of not-yet-visited items
        for idx in range(n - 1, -1, -1):
            if time.time() > deadline:
                return comps
            new_comps = comps[:idx] + comps[idx + 1:]
            if _verify_network_bitmask(new_comps, wires_init, expected):
                comps = new_comps
                improved = True
                break  # restart scan from the end
    return comps


def _aggressive_local_search(comps, wires_init, expected, rng, deadline):
    """Perturb-Verify-Cleanup hill climbing.

    Each valid perturbation is immediately followed by a full greedy-removal
    cleanup so the search always descends to the nearest local minimum.
    Diverse perturbation operators (single swap, double swap, block rotation,
    3-cycle rotation, insert) with a small probability of accepting same-size
    reorders to escape plateaus.
    """
    comps = list(comps)
    best = comps[:]
    best_len = len(comps)

    while time.time() < deadline:
        n = len(comps)
        if n < 2:
            break

        moved = False
        # Try a batch of perturbations before giving up on this state
        for _ in range(8):
            if time.time() > deadline:
                return best

            r = rng.random()
            cand = None

            if r < 0.30:
                # Single swap of two comparators
                if n >= 2:
                    i = rng.randint(0, n - 1)
                    j = rng.randint(0, n - 1)
                    if i != j:
                        cand = comps[:]
                        cand[i], cand[j] = cand[j], cand[i]
            elif r < 0.45:
                # Double swap (two disjoint pairs)
                if n >= 4:
                    i = rng.randint(0, n - 1)
                    j = rng.randint(0, n - 1)
                    k = rng.randint(0, n - 1)
                    l = rng.randint(0, n - 1)
                    if len({i, j, k, l}) == 4:
                        cand = comps[:]
                        cand[i], cand[j] = cand[j], cand[i]
                        cand[k], cand[l] = cand[l], cand[k]
            elif r < 0.55:
                # Block rotation: move a contiguous block of 2-4 comparators
                block_len = rng.randint(2, min(4, n))
                start = rng.randint(0, n - block_len)
                end = start + block_len
                block = comps[start:end]
                # Rotate right by one
                block = [block[-1]] + block[:-1]
                cand = comps[:start] + block + comps[end:]
            elif r < 0.65:
                # 3-cycle rotation of three comparators
                if n >= 3:
                    i = rng.randint(0, n - 1)
                    j = rng.randint(0, n - 1)
                    k = rng.randint(0, n - 1)
                    if len({i, j, k}) == 3:
                        cand = comps[:]
                        a, b, c = cand[i], cand[j], cand[k]
                        cand[i], cand[j], cand[k] = c, a, b
            else:
                # Insert a random comparator, then clean up
                a = rng.randint(0, N - 1)
                b = rng.randint(a + 1, N)
                pos = rng.randint(0, n + 1)
                cand = comps[:pos] + [(a, b)] + comps[pos:]

            if cand is None:
                continue

            # Verify the perturbation
            if not _verify_network_bitmask(cand, wires_init, expected):
                continue

            # Cleanup: greedy removal to descend to nearest local minimum
            reduced = _greedy_remove(cand, wires_init, expected, deadline)

            if len(reduced) < len(comps):
                # Strict improvement
                comps = reduced
                if len(comps) < best_len:
                    best_len = len(comps)
                    best = comps[:]
                moved = True
                break  # restart perturbation batch from the new state
            elif len(reduced) == len(comps) and rng.random() < 0.25:
                # Same-size reorder: accept with some probability to escape plateaus
                comps = reduced
                if len(comps) < best_len:
                    best_len = len(comps)
                    best = comps[:]
                moved = True
                break

        if not moved:
            # Stuck at this local minimum; try a more aggressive escape
            if time.time() < deadline:
                escaped = False
                for _ in range(5):
                    if time.time() > deadline:
                        break
                    # Randomly permute a larger chunk to break out
                    n = len(comps)
                    if n < 4:
                        break
                    chunk = rng.randint(3, min(6, n))
                    start = rng.randint(0, n - chunk)
                    segment = comps[start:start + chunk]
                    rng.shuffle(segment)
                    cand = comps[:start] + segment + comps[start + chunk:]
                    if _verify_network_bitmask(cand, wires_init, expected):
                        reduced = _greedy_remove(cand, wires_init, expected, deadline)
                        if len(reduced) < len(comps):
                            comps = reduced
                            if len(comps) < best_len:
                                best_len = len(comps)
                                best = comps[:]
                            escaped = True
                            break
                        elif rng.random() < 0.15:
                            comps = reduced
                            if len(comps) < best_len:
                                best_len = len(comps)
                                best = comps[:]
                            escaped = True
                            break
                if not escaped:
                    # Give up on this starting point
                    break

    return best


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS

    # Build seed network
    comps = _build_seed_network()

    # Precompute verification data
    wires_init = _init_wires()
    expected = _compute_expected()

    # Verify seed is correct
    if not _verify_network_bitmask(comps, wires_init, expected):
        return comps

    best_comps = comps[:]
    best_count = len(comps)

    # Phase 1: Greedy removal
    phase1_deadline = deadline - 30
    comps = _greedy_remove(comps, wires_init, expected, phase1_deadline)
    if len(comps) < best_count:
        best_count = len(comps)
        best_comps = comps[:]

    # Phase 2: Aggressive local search
    if time.time() < deadline - 15:
        rng = random.Random(42)
        result = _aggressive_local_search(comps, wires_init, expected, rng, deadline - 5)
        if len(result) < best_count:
            best_count = len(result)
            best_comps = result[:]

    # Phase 3: Final greedy removal pass
    if time.time() < deadline - 2:
        comps = _greedy_remove(best_comps, wires_init, expected, deadline - 1)
        if len(comps) < best_count:
            best_count = len(comps)
            best_comps = comps[:]

    # Verify final result
    if not _verify_network_bitmask(best_comps, wires_init, expected):
        # Shouldn't happen, but return seed anyway
        return _build_seed_network()

    return best_comps
# EVOLVE-BLOCK-END
