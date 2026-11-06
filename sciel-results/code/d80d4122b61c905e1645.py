"""Combined program: Odd-even merge base + fast sample validation + full verification."""
import time
import numpy as np

N = 17
DEADLINE = 85.0  # Safety margin before 90s kill
START_TIME = None


def _time_left():
    return DEADLINE - (time.time() - (START_TIME or 0))


def _make_odd_even_merge_network(n):
    """Generate an odd-even merge sort network for n wires.
    
    Returns list of (i, j) comparators.
    Odd-even merge sort:
    - Split into two halves
    - Recursively sort each half
    - Odd-even merge the two sorted halves
    """
    if n <= 1:
        return []
    if n == 2:
        return [(0, 1)]
    
    half = n // 2
    # Recursively sort each half
    left_net = _make_odd_even_merge_network(half)
    right_net = [(i + half, j + half) for i, j in _make_odd_even_merge_network(n - half)]
    
    # Odd-even merge
    merge_net = _odd_even_merge(n)
    
    return left_net + right_net + merge_net


def _odd_even_merge(n):
    """Odd-even merge for n wires (assumes two sorted halves of size n//2 and n-n//2)."""
    if n <= 1:
        return []
    if n == 2:
        return [(0, 1)]
    
    half = n // 2
    comps = []
    
    # Merge odd-indexed and even-indexed elements
    # First, merge the odd-indexed elements
    # Odd indices: 1, 3, 5, ...
    # Even indices: 0, 2, 4, ...
    
    # The odd-even merge algorithm:
    # 1. Compare (0, 1), (2, 3), (4, 5), ...
    # 2. Recursively merge the even-indexed subsequence
    # 3. Recursively merge the odd-indexed subsequence
    # 4. Compare (1, 2), (3, 4), (5, 6), ...
    
    # Step 1: Compare pairs (0,1), (2,3), ...
    for i in range(0, n - 1, 2):
        comps.append((i, i + 1))
    
    # Step 2: Recursively merge even-indexed elements (0, 2, 4, ...)
    # These are at positions 0, 2, 4, ... in the array
    # Map to a smaller problem
    even_count = (n + 1) // 2  # number of even-indexed positions
    odd_count = n // 2         # number of odd-indexed positions
    
    # For the even subsequence: positions 0, 2, 4, ...
    # We need to map indices: position 2*k in original -> position k in sub-problem
    even_sub = _odd_even_merge(even_count)
    for i, j in even_sub:
        comps.append((2 * i, 2 * j))
    
    # For the odd subsequence: positions 1, 3, 5, ...
    # Map: position 2*k+1 in original -> position k in sub-problem
    odd_sub = _odd_even_merge(odd_count)
    for i, j in odd_sub:
        comps.append((2 * i + 1, 2 * j + 1))
    
    # Step 4: Compare pairs (1,2), (3,4), ...
    for i in range(1, n - 1, 2):
        comps.append((i, i + 1))
    
    return comps


def _make_transposition_network(n):
    """Odd-even transposition sorting network for n wires."""
    comps = []
    for phase in range(n):
        start = phase % 2
        for i in range(start, n - 1, 2):
            comps.append((i, i + 1))
    return comps


def _verify_full(comps):
    """Full verification over all 2^N boolean inputs."""
    n_inputs = 1 << N
    indices = np.arange(n_inputs, dtype=np.uint32)[:, None]
    bit_positions = np.arange(N, dtype=np.uint32)[None, :]
    data = ((indices >> bit_positions) & 1).astype(np.uint8)
    
    for a, b in comps:
        lo = np.minimum(data[:, a], data[:, b])
        hi = np.maximum(data[:, a], data[:, b])
        data[:, a] = lo
        data[:, b] = hi
    
    return np.all(data[:, :-1] <= data[:, 1:])


def _verify_sample(comps, n_samples=4096, seed=42):
    """Fast sample-based verification."""
    rng = np.random.RandomState(seed)
    data = rng.randint(0, 2, size=(n_samples, N)).astype(np.uint8)
    
    for a, b in comps:
        lo = np.minimum(data[:, a], data[:, b])
        hi = np.maximum(data[:, a], data[:, b])
        data[:, a] = lo
        data[:, b] = hi
    
    return np.all(data[:, :-1] <= data[:, 1:])


def _try_remove(comps, idx):
    """Try removing comparator at index idx. Returns True if still valid."""
    new_comps = comps[:idx] + comps[idx + 1:]
    # Use sample verification for speed
    return _verify_sample(new_comps)


def construct():
    global START_TIME
    START_TIME = time.time()
    
    # Start with odd-even merge sort network
    comps = _make_odd_even_merge_network(N)
    
    # Verify it's actually correct
    if not _verify_full(comps):
        # Fallback to transposition
        comps = _make_transposition_network(N)
    
    # Try to remove redundant comparators
    # Work from the end backwards, trying to remove each one
    improved = True
    while improved and _time_left() > 5.0:
        improved = False
        for idx in range(len(comps) - 1, -1, -1):
            if _time_left() < 2.0:
                break
            if _try_remove(comps, idx):
                comps = comps[:idx] + comps[idx + 1:]
                improved = True
                break  # Restart scan from the end
    
    # Final full verification
    if not _verify_full(comps):
        # Shouldn't happen, but fallback
        comps = _make_transposition_network(N)
    
    return comps
