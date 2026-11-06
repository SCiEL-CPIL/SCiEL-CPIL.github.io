import numpy as np
import time

N = 16

def _verify_fast(network):
    """Verify sorting network using zero-one principle with numpy."""
    masks = np.arange(1 << N, dtype=np.uint16)
    inputs = np.zeros((1 << N, N), dtype=np.uint8)
    for i in range(N):
        inputs[:, i] = (masks >> i) & 1
    wires = inputs.copy()
    for (i, j) in network:
        ci = wires[:, i]
        cj = wires[:, j]
        wires[:, i] = np.minimum(ci, cj)
        wires[:, j] = np.maximum(ci, cj)
    for i in range(N - 1):
        if np.any(wires[:, i] > wires[:, i + 1]):
            return False
    return True

def construct():
    deadline = time.time() + 75
    # Start with odd-even transposition sorting network (120 comparators)
    network = [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]
    
    # Verify starting network is correct
    if not _verify_fast(network):
        return network
    
    # Greedy removal: try removing each comparator, keep if still correct
    removed = set()
    i = 0
    while i < len(network) and time.time() < deadline:
        if i in removed:
            i += 1
            continue
        test_network = [c for idx, c in enumerate(network) if idx != i and idx not in removed]
        if _verify_fast(test_network):
            removed.add(i)
        i += 1
    
    return [c for idx, c in enumerate(network) if idx not in removed]
