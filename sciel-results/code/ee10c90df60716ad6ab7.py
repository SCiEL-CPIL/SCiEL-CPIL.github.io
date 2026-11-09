import time
import numpy as np
import random

N = 17
MAX_COMPARATORS = 4096
DEADLINE_SECONDS = 50

def _build_seed_network():
    """
    Constructs Batcher's Odd-Merge sort network for 17 inputs.
    Uses recursive odd-merge with proper index mapping.
    Expected ~105 comparators.
    """
    comps = []

    def _odd_merge(n, offset):
        """
        Batcher's Odd-Merge sort of n elements starting at wire offset.
        """
        if n <= 1:
            return

        n_odd = (n + 1) // 2  # positions 0, 2, 4, ...
        n_even = n // 2       # positions 1, 3, 5, ...

        # Wire indices for odd sub-sequence
        odd_wires = [offset + 2 * i for i in range(n_odd)]
        # Wire indices for even sub-sequence
        even_wires = [offset + 2 * i + 1 for i in range(n_even)]

        # Recursively sort each sub-sequence
        # We need to generate comparators on the actual wire numbers
        def _sort_wires(wires, length):
            """Sort elements at given wire positions using odd-merge."""
            if length <= 1:
                return
            if length == 2:
                a, b = wires[0], wires[1]
                if a < b:
                    comps.append((a, b))
                else:
                    comps.append((b, a))
                return

            half_odd = (length + 1) // 2
            half_even = length // 2

            # Split wires into odd and even positions within this sub-problem
            sub_odd = [wires[2 * i] for i in range(half_odd)]
            sub_even = [wires[2 * i + 1] for i in range(half_even)]

            _sort_wires(sub_odd, half_odd)
            _sort_wires(sub_even, half_even)

            # Merge step
            m = min(half_odd, half_even)
            for i in range(m):
                a = sub_odd[i]
                b = sub_even[i]
                if a < b:
                    comps.append((a, b))
                else:
                    comps.append((b, a))
            for i in range(m - 1):
                a = sub_odd[i + 1]
                b = sub_even[i]
                if a < b:
                    comps.append((a, b))
                else:
                    comps.append((b, a))

        _sort_wires(list(range(N)), N)

    _odd_merge(N, 0)

    if not comps:
        comps = [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]

    return comps


def _verify(comps):
    """
    Fast numpy-based verification using vectorized operations.
    Checks all 2^17 Boolean inputs (zero-one principle).
    """
    if not comps:
        return N == 1

    total = 1 << N
    masks = np.arange(total, dtype=np.uint32)

    for (i, j) in comps:
        bi = np.uint32(1 << i)
        bj = np.uint32(1 << j)
        cond = (masks & bi) & ~(masks & bj)
        if np.any(cond):
            swap_mask = np.uint32((1 << i) | (1 << j))
            masks = np.where(cond, masks ^ swap_mask, masks)

    # A sorted 17-bit number (zeros then ones) satisfies (x & (x+1)) == 0
    return bool(np.all((masks & (masks + 1)) == 0))


def _greedy_remove(comps, deadline):
    """Try to remove each comparator if the network still sorts."""
    comps = list(comps)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        n = len(comps)
        for idx in range(n - 1, -1, -1):
            if time.time() > deadline:
                return comps
            new_comps = comps[:idx] + comps[idx + 1:]
            if _verify(new_comps):
                comps = new_comps
                improved = True
                break
    return comps


def _local_search(comps, deadline):
    """
    Focused local search: try removals, swaps, and insertions iteratively.
    No GA overhead - just direct improvement moves.
    """
    comps = list(comps)
    best = list(comps)
    best_len = len(comps)

    while time.time() < deadline:
        moved = False
        n = len(comps)

        # Phase 1: Try removals
        for idx in range(n - 1, -1, -1):
            if time.time() > deadline:
                return best
            cand = comps[:idx] + comps[idx + 1:]
            if _verify(cand):
                comps = cand
                best = list(comps)
                best_len = len(comps)
                moved = True
                break

        if moved:
            continue

        # Phase 2: Try swaps of comparator positions
        if n >= 2:
            for i in range(n):
                if time.time() > deadline:
                    return best
                for j in range(i + 1, n):
                    if time.time() > deadline:
                        return best
                    cand = comps[:]
                    cand[i], cand[j] = cand[j], cand[i]
                    if _verify(cand):
                        reduced = _greedy_remove(cand, deadline)
                        if len(reduced) < best_len:
                            comps = reduced
                            best = list(comps)
                            best_len = len(comps)
                            moved = True
                            break
                    if time.time() > deadline:
                        return best
                if moved:
                    break

        if moved:
            continue

        # Phase 3: Try inserting a new comparator
        for a in range(N - 1):
            if time.time() > deadline:
                return best
            for b in range(a + 1, N):
                if time.time() > deadline:
                    return best
                for pos in range(len(comps) + 1):
                    if time.time() > deadline:
                        return best
                    cand = comps[:pos] + [(a, b)] + comps[pos:]
                    if _verify(cand):
                        reduced = _greedy_remove(cand, deadline)
                        if len(reduced) < best_len:
                            comps = reduced
                            best = list(comps)
                            best_len = len(comps)
                            moved = True
                            break
                    if time.time() > deadline:
                        return best
                if moved:
                    break
            if moved:
                break

        if not moved:
            break

    return best


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS

    # Build seed network (Odd-Merge, ~105 comparators)
    comps = _build_seed_network()

    # Verify seed
    if not _verify(comps):
        # Fallback to odd-even transposition if seed is broken
        comps = [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]

    # Initial greedy removal
    comps = _greedy_remove(comps, deadline)

    # Focused local search
    comps = _local_search(comps, deadline)

    return comps
