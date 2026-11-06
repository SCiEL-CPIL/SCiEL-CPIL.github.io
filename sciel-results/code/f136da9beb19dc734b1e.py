import time
import numpy as np
from itertools import combinations

N = 18
NUM_MASKS = 1 << N

def _apply_network(network, x):
    """Apply comparison network to input vector x."""
    x = list(x)
    for i, j in network:
        if x[i] > x[j]:
            x[i], x[j] = x[j], x[i]
    return x

def _is_sorted_network(network):
    """Check if network sorts all 0-1 inputs using 0-1 principle."""
    for mask in range(NUM_MASKS):
        x = [(mask >> i) & 1 for i in range(N)]
        result = _apply_network(network, x)
        if result != sorted(result):
            return False
    return True

def _is_sorted_network_fast(network):
    """Faster verification using bit manipulation."""
    # For each mask, simulate the network
    for mask in range(NUM_MASKS):
        x = [(mask >> i) & 1 for i in range(N)]
        for i, j in network:
            if x[i] > x[j]:
                x[i], x[j] = x[j], x[i]
        # Check if sorted: x[0] <= x[1] <= ... <= x[N-1]
        for k in range(N - 1):
            if x[k] > x[k + 1]:
                return False
    return True

def _depth(network):
    """Compute parallel depth of the network."""
    return len(network)

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s safety margin
    
    # Start with odd-even transposition
    network = [(i, i + 1) for phase in range(N)
               for i in range(phase % 2, N - 1, 2)]
    
    # Verify it works
    assert _is_sorted_network(network), "Initial network is invalid"
    
    # Local search: try removing comparators
    improved = True
    while improved and (time.time() - start_time) < deadline:
        improved = False
        for idx in range(len(network) - 1, -1, -1):
            if (time.time() - start_time) > deadline - 5:
                break
            # Try removing comparator at idx
            test_network = network[:idx] + network[idx+1:]
            if _is_sorted_network_fast(test_network):
                network = test_network
                improved = True
    
    # Try adding comparators to enable further removals
    if (time.time() - start_time) < deadline - 10:
        for _ in range(10):  # Limited number of add-remove cycles
            if (time.time() - start_time) > deadline - 10:
                break
            # Try all possible (i,j) pairs not already in network
            existing = set(network)
            for i in range(N):
                for j in range(i+1, N):
                    if (i,j) not in existing:
                        test_network = network + [(i,j)]
                        # Try removing one comparator from the extended network
                        for idx in range(len(test_network) - 1, -1, -1):
                            if (time.time() - start_time) > deadline - 10:
                                break
                            if idx == len(test_network) - 1:
                                continue  # Don't remove the one we just added
                            removed = test_network[:idx] + test_network[idx+1:]
                            if len(removed) < len(network) and _is_sorted_network_fast(removed):
                                network = removed
                                improved = True
                                break
                        if improved:
                            break
                if improved:
                    break
            if not improved:
                break
    
    return network
