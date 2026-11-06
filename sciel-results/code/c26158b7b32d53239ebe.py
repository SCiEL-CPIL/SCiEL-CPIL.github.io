"""
Hybrid approach: Bitonic sort seed + DAG-inspired redundancy peeling + local search.

Structural change from current program:
- Use bitonic sort network as initial seed (fewer comparators than odd-even transposition)
- Apply "topological peeling": iteratively remove comparators whose removal preserves
  sortedness, using a fast sample-based check first, then full verification
- Local search with move-to-position and random-swap moves
- Aggressive multi-pass deletion

The key insight from the inspiration: represent the network structurally and use
transitive-reduction-like ideas to identify redundant comparators without
simulating all 131,072 inputs for every single deletion attempt. We approximate
this by using a fast sample check (16K inputs) as a filter before the full 131K check.
"""
import time
import random
import numpy as np

N = 17
SEED_COMPARATORS = 136
_NUM_INPUTS = 1 << N  # 131072

# Precompute all Boolean inputs
def _make_all_inputs():
    indices = np.arange(_NUM_INPUTS, dtype=np.uint32)[:, None]
    bit_positions = np.arange(N, dtype=np.uint32)[None, :]
    inputs = ((indices >> bit_positions) & 1).astype(np.uint8)
    return inputs

_ALL_INPUTS = None
_SAMPLE_INPUTS = None
_SAMPLE_SIZE = 16384

def _get_all_inputs():
    global _ALL_INPUTS
    if _ALL_INPUTS is None:
        _ALL_INPUTS = _make_all_inputs()
    return _ALL_INPUTS

def _get_sample_inputs():
    global _SAMPLE_INPUTS
    if _SAMPLE_INPUTS is None:
        all_inputs = _get_all_inputs()
        rng = np.random.RandomState(42)
        idx = rng.choice(_NUM_INPUTS, size=_SAMPLE_SIZE, replace=False)
        _SAMPLE_INPUTS = all_inputs[idx]
    return _SAMPLE_INPUTS


def simulate_network(comparators, inputs):
    """Simulate compare-exchange network on given inputs."""
    state = inputs.copy()
    for i, j in comparators:
        mi = state[:, i] & state[:, j]
        mj = state[:, i] | state[:, j]
        state[:, i] = mi
        state[:, j] = mj
    return state


def _is_sorted_rows(output):
    """Check if each row is non-decreasing (sorted)."""
    for k in range(N - 1):
        if np.any(output[:, k] > output[:, k + 1]):
            return False
    return True


def is_sorted_network(comparators):
    """Full check over all 2^N Boolean inputs."""
    inputs = _get_all_inputs()
    output = simulate_network(comparators, inputs)
    return _is_sorted_rows(output)


def is_sorted_network_sample(comparators):
    """Fast check on a sample of inputs (necessary but not sufficient)."""
    inputs = _get_sample_inputs()
    output = simulate_network(comparators, inputs)
    return _is_sorted_rows(output)


def _odd_even_transposition(n):
    """Standard odd-even transposition sort network."""
    comparators = []
    for phase in range(n):
        for i in range(phase % 2, n - 1, 2):
            comparators.append((i, i + 1))
    return comparators


def _bitonic_sort_network(n):
    """
    Bitonic sort network for n wires.
    Pads to next power of 2, generates full network, then filters out
    comparators involving wires >= n.
    """
    p = 1
    while p < n:
        p *= 2

    def bitonic_merge(k, ascending=True):
        """Bitonic merge for k elements (k is power of 2)."""
        if k == 1:
            return []
        half = k // 2
        # Compare elements i and i+half
        comps = []
        for i in range(half):
            a = i
            b = i + half
            if ascending:
                comps.append((a, b))
            else:
                comps.append((b, a))
        # Recursively merge halves
        left = bitonic_merge(half, ascending)
        right = bitonic_merge(half, ascending)
        right = [(i + half, j + half) for i, j in right]
        return comps + left + right

    def bitonic_sort(k, ascending=True):
        """Bitonic sort for k elements."""
        if k == 1:
            return []
        half = k // 2
        # Sort halves in opposite directions to create bitonic sequence
        left = bitonic_sort(half, ascending)
        right = bitonic_sort(half, not ascending)
        right = [(i + half, j + half) for i, j in right]
        # Merge
        merge = bitonic_merge(k, ascending)
        return left + right + merge

    full = bitonic_sort(p, True)

    # Normalize and filter
    normalized = []
    for i, j in full:
        if i > j:
            i, j = j, i
        if i < n and j < n:
            normalized.append((i, j))
    return normalized


def _odd_even_merge_sort(n):
    """
    Odd-even merge sort network.
    """
    def merge(left_start, left_len, right_start, right_len):
        """Merge two sorted sequences of given lengths starting at given positions."""
        if left_len == 0:
            return []
        if right_len == 0:
            return []
        if left_len == 1 and right_len == 1:
            return [(left_start, right_start)]

        # Split each into odd and even indexed elements
        lo = (left_len + 1) // 2
        le = left_len // 2
        ro = (right_len + 1) // 2
        re = right_len // 2

        comps = []
        # Merge odd-indexed parts
        comps.extend(merge(left_start, lo, right_start, ro))
        # Merge even-indexed parts
        comps.extend(merge(left_start + 1, le, right_start + 1, re))

        # Interleave: compare (2i, 2i+1) for i in range
        # After merging, the odd part occupies positions left_start, left_start+2, ...
        # and the even part occupies left_start+1, left_start+3, ...
        # We need to compare adjacent pairs in the merged region
        total_merged = left_len + right_len
        # The merged result is in positions left_start .. left_start+total_merged-1
        # Odd positions (0-indexed within merged): left_start, left_start+2, ...
        # Even positions: left_start+1, left_start+3, ...
        # Compare (left_start + 2i, left_start + 2i + 1) for valid i
        for i in range(total_merged // 2):
            a = left_start + 2 * i
            b = left_start + 2 * i + 1
            if b < n:
                comps.append((a, b))

        return comps

    def sort_seq(start, length):
        if length <= 1:
            return []
        half = length // 2
        comps = []
        comps.extend(sort_seq(start, half))
        comps.extend(sort_seq(start + half, length - half))
        comps.extend(merge(start, half, start + half, length - half))
        return comps

    return sort_seq(0, n)


def _best_seed():
    """Pick the smallest valid seed network."""
    candidates = []

    oet = _odd_even_transposition(N)
    if is_sorted_network(oet):
        candidates.append(oet)

    bitonic = _bitonic_sort_network(N)
    if is_sorted_network(bitonic):
        candidates.append(bitonic)

    oems = _odd_even_merge_sort(N)
    if is_sorted_network(oems):
        candidates.append(oems)

    if not candidates:
        return oet
    return min(candidates, key=len)


def _peel_network(comparators, deadline):
    """
    Topological peeling: iteratively try to remove each comparator.
    Use sample check as a fast filter; only do full check if sample passes.
    Multiple passes until no more removals possible or deadline.
    """
    best = list(comparators)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        i = 0
        while i < len(best) and time.time() < deadline:
            candidate = best[:i] + best[i + 1:]
            # Fast sample check first
            if is_sorted_network_sample(candidate):
                # Full verification
                if is_sorted_network(candidate):
                    best = candidate
                    improved = True
                    # Don't increment i; re-check same position
                    continue
            i += 1
    return best


def _local_search(comparators, deadline, max_moves=200):
    """
    Local search: try moving a comparator to a different position,
    or swapping two comparators. Accept if still valid and same length
    (enables further peeling).
    """
    best = list(comparators)
    best_len = len(best)
    moves = 0

    while time.time() < deadline and moves < max_moves:
        moves += 1
        n = len(best)
        if n < 2:
            break

        # Try move-to-position
        idx = random.randint(0, n - 1)
        new_pos = random.randint(0, n - 1)
        if idx == new_pos:
            continue
        comp = best.pop(idx)
        if new_pos > idx:
            new_pos -= 1
        candidate = best[:new_pos] + [comp] + best[new_pos:]

        if is_sorted_network_sample(candidate) and is_sorted_network(candidate):
            best = candidate
            # Try peeling after a move
            best = _peel_network(best, deadline)
            if len(best) < best_len:
                best_len = len(best)

        # Try swap
        if time.time() < deadline and len(best) >= 2:
            a = random.randint(0, len(best) - 1)
            b = random.randint(0, len(best) - 1)
            if a == b:
                continue
            candidate = list(best)
            candidate[a], candidate[b] = candidate[b], candidate[a]
            if is_sorted_network_sample(candidate) and is_sorted_network(candidate):
                best = candidate
                best = _peel_network(best, deadline)
                if len(best) < best_len:
                    best_len = len(best)

    return best


def _random_restart(comparators, deadline):
    """
    Perturb the network: remove a random subset and re-add, then peel.
    """
    best = list(comparators)
    n = len(best)
    if n < 10:
        return best

    # Remove 10-20% of comparators
    remove_count = max(3, n // 10)
    remove_idx = set(random.sample(range(n), remove_count))
    candidate = [c for i, c in enumerate(best) if i not in remove_idx]

    # Try to fix by peeling (which may not help if broken)
    if is_sorted_network_sample(candidate):
        if is_sorted_network(candidate):
            candidate = _peel_network(candidate, deadline)
            if len(candidate) < len(best):
                best = candidate
    else:
        # Try re-adding some comparators at random positions
        for _ in range(remove_count):
            if time.time() > deadline:
                break
            i = random.randint(0, N - 2)
            j = random.randint(i + 1, N - 1)
            pos = random.randint(0, len(candidate))
            candidate = candidate[:pos] + [(i, j)] + candidate[pos:]

        if is_sorted_network_sample(candidate):
            if is_sorted_network(candidate):
                candidate = _peel_network(candidate, deadline)
                if len(candidate) < len(best):
                    best = candidate

    return best


def construct():
    """
    Construct a sorting network on 17 wires with as few comparators as possible.
    Returns an ordered list of integer pairs (i, j).
    """
    start_time = time.time()
    deadline = start_time + 82  # Safety margin before 90s kill

    random.seed(42)
    np.random.seed(42)

    # Get best seed
    best = _best_seed()

    # Verify
    if not is_sorted_network(best):
        raise ValueError("Seed network is not correct")

    # Phase 1: Aggressive peeling (multiple passes)
    if time.time() < deadline - 10:
        best = _peel_network(best, deadline - 5)

    # Phase 2: Local search with moves and swaps
    if time.time() < deadline - 10:
        best = _local_search(best, deadline - 5, max_moves=100)

    # Phase 3: More peeling after local search
    if time.time() < deadline - 5:
        best = _peel_network(best, deadline - 2)

    # Phase 4: Random restarts / perturbations
    restarts = 0
    while time.time() < deadline - 15 and restarts < 20:
        restarts += 1
        candidate = _random_restart(best, deadline - 10)
        if len(candidate) < len(best):
            best = candidate

    # Phase 5: Final peeling pass
    if time.time() < deadline - 2:
        best = _peel_network(best, deadline - 1)

    # Final local search if time remains
    if time.time() < deadline - 3:
        best = _local_search(best, deadline - 1, max_moves=50)

    # Final peel
    if time.time() < deadline - 1:
        best = _peel_network(best, deadline - 0.5)

    # Ensure validity
    if not is_sorted_network(best):
        # Fallback to seed
        best = _odd_even_transposition(N)

    return best


# EVOLVE-BLOCK-START
# construct() is defined above
# EVOLVE-BLOCK-END
