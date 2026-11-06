"""Size-efficient sorting network on 16 wires.

Strategy:
  1. Build Batcher's odd-even merge sort (63 comparators) as the starting point.
  2. Run a fast, vectorised zero-one verifier (all 2^16 inputs) using NumPy.
  3. Perform an aggressive greedy removal search: repeatedly try to drop a
     comparator, accepting the drop if the network still sorts all Boolean inputs.
     Use randomised restarts and multiple passes to escape local minima.
  4. Optionally, after removal, try a local "swap" move: replace one comparator
     with another to allow further removals (a light 1-opt on the comparator list).
  5. Return the best (fewest-comparator) verified network found within the deadline.

The target is 60 comparators (known optimal for 16 wires).
"""
import time
import random
import numpy as np

N = 16
_START = time.time()
# Safety margin: evaluator kills at 90 s; we stop at 80 s to be safe.
_DEADLINE = _START + 80.0


# ---------------------------------------------------------------------------
# Network builder: Batcher's odd-even merge sort (recursive).
# ---------------------------------------------------------------------------
def _batcher_odd_even_merge(n):
    """Return a fixed compare-exchange network sorting n wires (n a power of 2)."""
    if n <= 1:
        return []
    if n == 2:
        return [(0, 1)]

    cmps = []
    # 1) recursively sort odd-indexed wires: 1,3,5,...
    for a, b in _batcher_odd_even_merge(n // 2):
        cmps.append((2 * a + 1, 2 * b + 1))
    # 2) recursively sort even-indexed wires: 0,2,4,...
    for a, b in _batcher_odd_even_merge(n // 2):
        cmps.append((2 * a, 2 * b))
    # 3) odd-even cross comparisons (2i, 2i+1) then (2i+1, 2i+2)
    for i in range(n // 2):
        cmps.append((2 * i, 2 * i + 1))
    for i in range(n // 2 - 1):
        cmps.append((2 * i + 1, 2 * i + 2))
    return cmps


# ---------------------------------------------------------------------------
# Verification: zero-one principle (all 2**N Boolean inputs).
# Optimised with NumPy vectorisation.
# ---------------------------------------------------------------------------
def _verify_zero_one(cmps):
    """Return True iff `cmps` sorts every Boolean vector of length N."""
    M = 1 << N
    idx = np.arange(M, dtype=np.int64)
    bits = np.zeros((M, N), dtype=np.int8)
    for k in range(N):
        bits[:, k] = ((idx >> k) & 1).astype(np.int8)

    for i, j in cmps:
        a = bits[:, i]
        b = bits[:, j]
        lo = np.minimum(a, b)
        hi = np.maximum(a, b)
        bits[:, i] = lo
        bits[:, j] = hi

    # Sorted ascending: bits[:, k] <= bits[:, k+1] for all k
    for k in range(N - 1):
        if np.any(bits[:, k] > bits[:, k + 1]):
            return False
    return True


# ---------------------------------------------------------------------------
# Depth (parallelism) reporter.
# ---------------------------------------------------------------------------
def _parallel_depth(cmps):
    end_time = {i: 0 for i in range(N)}
    depth = 0
    for i, j in cmps:
        t = max(end_time[i], end_time[j]) + 1
        depth = max(depth, t)
        end_time[i] = t
        end_time[j] = t
    return depth


# ---------------------------------------------------------------------------
# Fast "would removal of comparator at index idx break sorting?" test.
# We re-use the bit matrix and only replay the network with one comparator removed.
# To keep it fast, we precompute the bit matrix once and replay the network.
# ---------------------------------------------------------------------------
def _build_bit_matrix():
    M = 1 << N
    idx = np.arange(M, dtype=np.int64)
    bits = np.zeros((M, N), dtype=np.int8)
    for k in range(N):
        bits[:, k] = ((idx >> k) & 1).astype(np.int8)
    return bits


def _is_sorted(bits):
    for k in range(N - 1):
        if np.any(bits[:, k] > bits[:, k + 1]):
            return False
    return True


def _apply_network(bits, cmps):
    """Apply the compare-exchange network in-place to the bit matrix."""
    for i, j in cmps:
        a = bits[:, i]
        b = bits[:, j]
        lo = np.minimum(a, b)
        hi = np.maximum(a, b)
        bits[:, i] = lo
        bits[:, j] = hi
    return bits


def _try_remove(cmps, idx, base_bits):
    """Return True if removing cmps[idx] still yields a sorting network."""
    bits = base_bits.copy()
    # Replay all comparators except idx
    for k, (i, j) in enumerate(cmps):
        if k == idx:
            continue
        a = bits[:, i]
        b = bits[:, j]
        lo = np.minimum(a, b)
        hi = np.maximum(a, b)
        bits[:, i] = lo
        bits[:, j] = hi
    return _is_sorted(bits)


# ---------------------------------------------------------------------------
# Reduction search: greedy removal with randomised restarts.
# ---------------------------------------------------------------------------
def _reduce_network(initial_cmps):
    """
    Greedy removal: repeatedly try to remove a comparator; keep it removed
    if the network remains valid. Use multiple randomised passes to escape
    local minima.
    """
    best_cmps = list(initial_cmps)
    base_bits = _build_bit_matrix()
    rng = random.Random(42)

    max_restarts = 30  # number of independent greedy passes

    for restart in range(max_restarts):
        if time.time() > _DEADLINE:
            break

        current_cmps = list(best_cmps)
        # Shuffle the order in which we consider removals
        indices = list(range(len(current_cmps)))
        rng.shuffle(indices)

        for idx in indices:
            if time.time() > _DEADLINE:
                break
            # Try removing current_cmps[idx]
            if _try_remove(current_cmps, idx, base_bits):
                current_cmps.pop(idx)
                # After a removal, restart the scan (indices shift)
                # But to keep it simple and fast, we just continue; the next
                # restart will re-shuffle. For efficiency, break and restart.
                break

        if len(current_cmps) < len(best_cmps):
            best_cmps = current_cmps
            # Continue to next restart to try further removals from the new state

    return best_cmps


# ---------------------------------------------------------------------------
# Additional local search: try replacing one comparator with a different one
# to enable further removals. This is a light 1-opt.
# ---------------------------------------------------------------------------
def _local_swap_search(cmps, base_bits):
    """
    Try to improve by swapping one comparator with another (or removing and
    adding). This is a heuristic to escape local minima in the removal search.
    """
    best_cmps = list(cmps)
    rng = random.Random(123)

    # Generate a pool of candidate comparators (all valid (i,j) pairs)
    all_pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]

    max_iter = 200
    for _ in range(max_iter):
        if time.time() > _DEADLINE:
            break

        # Try removing a random comparator and adding a random other one
        idx = rng.randrange(len(best_cmps))
        new_cmp = rng.choice(all_pairs)

        candidate = list(best_cmps)
        candidate.pop(idx)
        if new_cmp not in candidate:
            candidate.append(new_cmp)

        # Verify
        bits = base_bits.copy()
        _apply_network(bits, candidate)
        if _is_sorted(bits):
            best_cmps = candidate

    return best_cmps


# ---------------------------------------------------------------------------
# Entry point (name/signature kept exactly as in the seed program).
# ---------------------------------------------------------------------------
def construct():
    # 1. Generate Batcher's network
    net = _batcher_odd_even_merge(N)
    initial_count = len(net)

    # 2. Verify initial network (sanity check)
    if not _verify_zero_one(net):
        # Should not happen, but fallback to odd-even transposition
        return [(i, i + 1) for phase in range(N)
                for i in range(phase % 2, N - 1, 2)]

    # 3. Build the base bit matrix once for reuse
    base_bits = _build_bit_matrix()

    # 4. Greedy removal search
    reduced_net = _reduce_network(net)

    # 5. Local swap search to escape local minima
    swapped_net = _local_swap_search(reduced_net, base_bits)

    # 6. Choose the best (fewest comparators) verified network
    candidates = [net, reduced_net, swapped_net]
    best = None
    for cand in candidates:
        if _verify_zero_one(cand):
            if best is None or len(cand) < len(best):
                best = cand

    if best is None:
        best = net  # fallback

    # 7. Report
    depth = _parallel_depth(best)

    try:
        import sys
        sys.stderr.write(
            "[construct] Initial: %d, Reduced: %d, Swapped: %d, Best: %d, depth %d, verified: %s\n"
            % (initial_count, len(reduced_net), len(swapped_net), len(best), depth, _verify_zero_one(best))
        )
    except Exception:
        pass

    # Deadline guard
    if time.time() > _DEADLINE:
        return best

    return best


if __name__ == "__main__":
    net = construct()
    print("comparators:", len(net))
    print("depth:", _parallel_depth(net))
    print("zero-one verified:", _verify_zero_one(net))
