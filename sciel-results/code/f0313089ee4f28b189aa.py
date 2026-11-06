"""Sorting network for 17 wires: Batcher seed + aggressive comparator removal with fast validation."""
import time
import numpy as np

N = 17
DEADLINE = 85.0  # safety margin before 90s kill

def _odd_even_transposition(n):
    """Odd-even transposition sort network for n wires."""
    comparators = []
    for phase in range(n):
        for i in range(phase % 2, n - 1, 2):
            comparators.append((i, i + 1))
    return comparators

def _simulate_network(comparators, inputs):
    """Simulate compare-exchange network on boolean inputs.
    inputs: (num_inputs, N) array of 0/1.
    Returns sorted wires."""
    wires = inputs.copy()
    for (i, j) in comparators:
        ti = wires[:, i]
        tj = wires[:, j]
        wires[:, i] = np.minimum(ti, tj)
        wires[:, j] = np.maximum(ti, tj)
    return wires

def _is_sorted(wires):
    """Check if all rows are non-decreasing."""
    diffs = np.diff(wires, axis=1)
    return np.all(diffs >= 0)

def _validate_full(comparators):
    """Full validation over all 2^17 boolean inputs."""
    n_inputs = 1 << N
    idx = np.arange(n_inputs, dtype=np.uint32)
    bits = np.zeros((n_inputs, N), dtype=np.uint8)
    for k in range(N):
        bits[:, k] = (idx >> k) & 1
    wires = _simulate_network(comparators, bits)
    return _is_sorted(wires)

def _validate_sample(comparators, n_samples=4096, rng=None):
    """Sample-based validation for speed."""
    if rng is None:
        rng = np.random.RandomState(42)
    bits = (rng.randint(0, 2, size=(n_samples, N))).astype(np.uint8)
    wires = _simulate_network(comparators, bits)
    return _is_sorted(wires)

def construct():
    start_time = time.time()
    
    # Seed: odd-even transposition (136 comparators for 17 wires)
    comparators = _odd_even_transposition(N)
    
    # Verify seed
    if not _validate_full(comparators):
        return comparators
    
    best_comparators = comparators.copy()
    best_count = len(comparators)
    
    rng = np.random.RandomState(12345)
    
    # Iterative removal: try removing each comparator
    while (time.time() - start_time) < DEADLINE - 10.0:
        removed = False
        n_comp = len(comparators)
        for idx in range(n_comp):
            if (time.time() - start_time) > DEADLINE - 5.0:
                break
            candidate = comparators[:idx] + comparators[idx+1:]
            # Fast sample check
            if _validate_sample(candidate, n_samples=2048, rng=rng):
                # Full check
                if _validate_full(candidate):
                    comparators = candidate
                    best_count = len(comparators)
                    best_comparators = comparators.copy()
                    removed = True
                    break  # restart scan
    
        if not removed:
            break
    
    return best_comparators
