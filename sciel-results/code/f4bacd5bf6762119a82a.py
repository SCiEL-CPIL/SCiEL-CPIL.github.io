"""Sorting network search for N=18 wires.

Strategy:
  1. Start from a good seed (odd-even transposition or a known compact network).
  2. Use a vectorized inversion-count heuristic over a large sample of Boolean inputs.
  3. Perform local search: try adding/removing/moving comparators, accept if the
     inversion-count metric improves.
  4. Periodically verify correctness on ALL 2^18 Boolean inputs.
  5. Return the best verified (or best heuristic) network within the time budget.
"""
import time
import numpy as np
from itertools import combinations

N = 18
ALL_BOOLS = 1 << N  # 262144

# Deadline: leave safety margin before 90s wall-clock kill.
DEADLINE = time.time() + 85.0


def _apply_comparator(arr, i, j):
    """Vectorized compare-exchange: min to i, max to j."""
    # arr shape: (M, N)
    a = arr[:, i].copy()
    b = arr[:, j].copy()
    mn = np.minimum(a, b)
    mx = np.maximum(a, b)
    arr[:, i] = mn
    arr[:, j] = mx
    return arr


def _inversion_count(arr):
    """Count inversions in each row of arr (shape M x N). Returns (M,) int array."""
    # For each row, count pairs (i, j) with i < j and arr[i] > arr[j].
    # Vectorized: for each pair (i, j), count rows where arr[:, i] > arr[:, j].
    M, Nw = arr.shape
    inv = np.zeros(M, dtype=np.int32)
    for i in range(Nw):
        for j in range(i + 1, Nw):
            inv += (arr[:, i] > arr[:, j]).astype(np.int32)
    return inv


def _verify_all(network):
    """Check correctness on all 2^N Boolean inputs. Returns True if sorted."""
    # Generate all 2^N boolean inputs as a (2^N, N) uint8 array.
    # Use bit unpacking.
    vals = np.arange(ALL_BOOLS, dtype=np.uint32)
    # Unpack bits: for each wire k (0-indexed), bit k of val.
    # We want wire 0 to be the least significant bit? Convention: input bit k -> wire k.
    # Let's define: input integer x, wire k gets bit k of x.
    arr = np.zeros((ALL_BOOLS, N), dtype=np.uint8)
    for k in range(N):
        arr[:, k] = (vals >> k) & 1

    for (i, j) in network:
        a = arr[:, i].copy()
        b = arr[:, j].copy()
        arr[:, i] = np.minimum(a, b)
        arr[:, j] = np.maximum(a, b)

    # Check if each row is sorted: non-decreasing.
    # A row is sorted if for all k < N-1, arr[:, k] <= arr[:, k+1].
    diffs = np.diff(arr, axis=1)  # shape (ALL_BOOLS, N-1)
    # Sorted iff all diffs >= 0.
    return bool(np.all(diffs >= 0))


def _heuristic_score(network, sample):
    """Compute total inversions after applying network to sample. Lower is better."""
    arr = sample.copy()
    for (i, j) in network:
        a = arr[:, i].copy()
        b = arr[:, j].copy()
        arr[:, i] = np.minimum(a, b)
        arr[:, j] = np.maximum(a, b)
    inv = _inversion_count(arr)
    return inv.sum()


def _seed_network():
    """Odd-even transposition network."""
    return [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]


def _make_sample(n_samples=20000):
    """Random Boolean inputs."""
    rng = np.random.default_rng(42)
    return rng.integers(0, 2, size=(n_samples, N), dtype=np.uint8)


def construct():
    t0 = time.time()
    deadline = t0 + 85.0  # safety margin

    best_network = _seed_network()
    best_score = None
    best_verified = False

    sample = _make_sample(20000)
    best_score = _heuristic_score(best_network, sample)

    # Try to verify the seed
    if time.time() < deadline - 5.0:
        if _verify_all(best_network):
            best_verified = True

    # Local search: try removing comparators, then try adding useful ones.
    # Phase 1: Removal
    max_iterations = 500
    for it in range(max_iterations):
        if time.time() > deadline - 10.0:
            break

        # Try removing each comparator
        improved = False
        net = list(best_network)
        for idx in range(len(net)):
            if time.time() > deadline - 10.0:
                break
            candidate = net[:idx] + net[idx+1:]
            if len(candidate) == 0:
                break
            score = _heuristic_score(candidate, sample)
            if score < best_score:
                # Verify before accepting? Too slow. Accept on heuristic.
                best_network = candidate
                best_score = score
                best_verified = False
                improved = True
                net = list(best_network)
                break  # restart removal pass
        if not improved:
            break

    # Phase 2: Addition - try adding comparators that reduce inversions
    all_pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
    # Score each candidate pair by how much it would reduce inversions if added at the end
    # Actually, we should try inserting at various positions, but that's expensive.
    # Simpler: try appending each pair, see if it helps.
    for _ in range(200):  # max additions
        if time.time() > deadline - 10.0:
            break

        best_add = None
        best_add_score = best_score

        # Sample a subset of pairs to try
        rng = np.random.default_rng()
        trial_pairs = rng.choice(len(all_pairs), size=min(50, len(all_pairs)), replace=False)
        for ti in trial_pairs:
            i, j = all_pairs[ti]
            if (i, j) in best_network:
                continue
            candidate = best_network + [(i, j)]
            score = _heuristic_score(candidate, sample)
            if score < best_add_score:
                best_add_score = score
                best_add = (i, j)

        if best_add is not None:
            best_network = best_network + [best_add]
            best_score = best_add_score
            best_verified = False
        else:
            break

    # Final verification if time permits
    if time.time() < deadline - 5.0:
        if _verify_all(best_network):
            best_verified = True

    return list(best_network)
