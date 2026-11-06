"""Size-efficient sorting network on 16 wires.

Combined strategy:
  * Foundation: Batcher's odd-even merge sort (63 comparators).
  * Refinement: Aggressive local search with "kicks" to escape local minima,
    targeting the known optimal 60-comparator network.
  * Speedups:
      - Precompute the 65536x16 bit matrix once and reuse it.
      - Use a faster, chunked simulation with early-exit.
      - Add a "swap" kick (replace one comparator with another) to the
        local-search toolbox.
      - Try a small set of deterministic permutations of the Batcher
        network before starting the stochastic search.
"""
import numpy as np
import time
import random

N = 16
_START = time.time()
_DEADLINE = _START + 80.0  # Safety margin below the 90 s wall clock


def _batcher_odd_even_merge(n):
    if n <= 1:
        return []
    if n == 2:
        return [(0, 1)]

    cmps = []
    for a, b in _batcher_odd_even_merge(n // 2):
        cmps.append((2 * a + 1, 2 * b + 1))
    for a, b in _batcher_odd_even_merge(n // 2):
        cmps.append((2 * a, 2 * b))
    for i in range(n // 2):
        cmps.append((2 * i, 2 * i + 1))
    for i in range(n // 2 - 1):
        cmps.append((2 * i + 1, 2 * i + 2))
    return cmps


def _build_matrix():
    M = 1 << N
    idx = np.arange(M, dtype=np.int64)
    bits = np.zeros((M, N), dtype=np.int8)
    for k in range(N):
        bits[:, k] = ((idx >> k) & 1).astype(np.int8)
    return bits


def _is_sorted(bits):
    return not np.any(bits[:, :-1] > bits[:, 1:])


def _apply(bits, cmps, start):
    for i, j in cmps[start:]:
        a = bits[:, i]
        b = bits[:, j]
        lo = np.minimum(a, b)
        hi = np.maximum(a, b)
        bits[:, i] = lo
        bits[:, j] = hi


def _verify_zero_one(cmps):
    bits = _build_matrix()
    _apply(bits, cmps, 0)
    return _is_sorted(bits)


def _parallel_depth(cmps):
    end_time = [0] * N
    depth = 0
    for i, j in cmps:
        t = max(end_time[i], end_time[j]) + 1
        depth = max(depth, t)
        end_time[i] = t
        end_time[j] = t
    return depth


def _reduce_network(initial_cmps, seed=42, base_bits=None):
    """Greedy reduction with kicks. Returns the best (shortest) network found."""
    current = list(initial_cmps)
    random.seed(seed)

    if base_bits is None:
        base_bits = _build_matrix()

    def _test_removal(cmps, idx_to_remove):
        cand = base_bits.copy()
        n = len(cmps)
        CHUNK = 16
        for start in range(0, n, CHUNK):
            end = min(start + CHUNK, n)
            for k in range(start, end):
                if k == idx_to_remove:
                    continue
                i, j = cmps[k]
                a = cand[:, i]
                b = cand[:, j]
                lo = np.minimum(a, b)
                hi = np.maximum(a, b)
                cand[:, i] = lo
                cand[:, j] = hi
            # Early exit check every 4 chunks (64 comparators)
            if (end // CHUNK) % 4 == 0 and end < n:
                if np.any(cand[:, :-1] > cand[:, 1:]):
                    return False
        return not np.any(cand[:, :-1] > cand[:, 1:])

    def _test_replacement(cmps, idx_to_replace, new_cmp):
        """Test replacing cmps[idx_to_replace] with new_cmp."""
        cand = base_bits.copy()
        n = len(cmps)
        for k in range(n):
            if k == idx_to_replace:
                i, j = new_cmp
            else:
                i, j = cmps[k]
            a = cand[:, i]
            b = cand[:, j]
            lo = np.minimum(a, b)
            hi = np.maximum(a, b)
            cand[:, i] = lo
            cand[:, j] = hi
        return not np.any(cand[:, :-1] > cand[:, 1:])

    def _kick(cmps):
        """Apply a random kick to escape local minimum."""
        n = len(cmps)
        if n < 2:
            return cmps
        kick_type = random.random()
        if kick_type < 0.3:
            # Swap two comparators
            i = random.randint(0, n - 2)
            j = random.randint(i + 1, n - 1)
            cmps[i], cmps[j] = cmps[j], cmps[i]
        elif kick_type < 0.5:
            # Move a comparator to a new position
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            cmps.insert(j, cmps.pop(i))
        elif kick_type < 0.8:
            # Insert a random comparator (increases size, but may enable later removals)
            i = random.randint(0, N - 2)
            j = random.randint(i + 1, N - 1)
            pos = random.randint(0, n)
            cmps.insert(pos, (i, j))
        else:
            # Replace one comparator with a random one
            idx = random.randint(0, n - 1)
            i = random.randint(0, N - 2)
            j = random.randint(i + 1, N - 1)
            cmps[idx] = (i, j)
        return cmps

    best = list(current)
    best_len = len(current)
    stall_count = 0

    while True:
        if time.time() > _DEADLINE:
            break
        n = len(current)
        if n == 60:
            break

        any_removed = False
        # Try to remove any comparator
        for idx in range(n):
            if time.time() > _DEADLINE:
                return best

            if _test_removal(current, idx):
                del current[idx]
                any_removed = True
                stall_count = 0
                if len(current) < best_len:
                    best_len = len(current)
                    best = list(current)
                break  # Restart scan from 0

        if not any_removed:
            stall_count += 1
            # If we've stalled a lot, try a more aggressive kick
            if n == 60:
                break
            if stall_count > 5:
                # Try multiple kicks in a row
                for _ in range(3):
                    current = _kick(current)
                stall_count = 0
            else:
                current = _kick(current)

    return best


def construct():
    net = _batcher_odd_even_merge(N)

    if not _verify_zero_one(net):
        return [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]

    best_net = net
    best_count = len(net)

    # Precompute the bit matrix once
    base_bits = _build_matrix()

    # Try multiple seeds with the reduced network
    seeds = [42, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500]
    for seed in seeds:
        if time.time() > _DEADLINE - 5:
            break
        if best_count == 60:
            break

        # For later seeds, perturb the initial network more aggressively
        if seed == 42:
            perturbed = list(net)
        else:
            random.seed(seed)
            perturbed = list(net)
            # Swap several random pairs to create diversity
            n_swaps = 5 if seed % 2 == 0 else 3
            for _ in range(n_swaps):
                if len(perturbed) > 1:
                    i = random.randint(0, len(perturbed) - 2)
                    j = random.randint(i + 1, len(perturbed) - 1)
                    perturbed[i], perturbed[j] = perturbed[j], perturbed[i]

        candidate = _reduce_network(perturbed, seed=seed, base_bits=base_bits)
        if len(candidate) < best_count:
            best_count = len(candidate)
            best_net = candidate

    if not _verify_zero_one(best_net):
        best_net = net

    return best_net


if __name__ == "__main__":
    net = construct()
    print("comparators:", len(net))
    print("depth:", _parallel_depth(net))
    print("zero-one verified:", _verify_zero_one(net))
