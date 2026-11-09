"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 18

# EVOLVE-BLOCK-START
import time
import numpy as np

_START_TIME = time.time()
_DEADLINE = 85.0  # safety margin before 90s kill

def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)

def _apply_network(wires, network):
    """Apply network to a wire vector (numpy array)."""
    for i, j in network:
        a, b = wires[i], wires[j]
        if a > b:
            wires[i], wires[j] = b, a
    return wires

def _check_network(network):
    """Check if network sorts all 2^N Boolean inputs using 0-1 principle."""
    n = N
    for mask in range(1 << n):
        wires = np.array([(mask >> i) & 1 for i in range(n)], dtype=np.int8)
        _apply_network(wires, network)
        # Check if sorted
        for i in range(n - 1):
            if wires[i] > wires[i + 1]:
                return False
    return True

def _bitonic_merge(n, offset, step):
    """Generate comparators for bitonic merge of 'step' elements starting at offset."""
    if step == 1:
        return []
    half = step // 2
    comps = []
    for i in range(half):
        comps.append((offset + i, offset + i + half))
    comps.extend(_bitonic_merge(n, offset, half))
    comps.extend(_bitonic_merge(n, offset + half, half))
    return comps

def _bitonic_sort_network(n):
    """Generate a bitonic sorting network for n wires."""
    comps = []
    for k in range(2, n + 1, 2):
        for i in range(0, n, k):
            # Build bitonic sequence in block of size k
            for j in range(k // 2, 0, -1):
                comps.extend(_bitonic_merge(n, i, j))
    return comps

def _odd_even_transposition(n):
    """Generate odd-even transposition sorting network."""
    comps = []
    for phase in range(n):
        for i in range(phase % 2, n - 1, 2):
            comps.append((i, i + 1))
    return comps

def _try_prune(network, timeout=30.0):
    """Try to remove redundant comparators from the network."""
    network = list(network)
    changed = True
    while changed and _time_left() > 5.0:
        changed = False
        for idx in range(len(network) - 1, -1, -1):
            if _time_left() < 5.0:
                break
            candidate = network[:idx] + network[idx+1:]
            if _check_network(candidate):
                network = candidate
                changed = True
                break  # restart from end after removal
    return network

def construct():
    # Start with bitonic sort network
    network = _bitonic_sort_network(N)
    
    # Try to prune redundant comparators
    if _time_left() > 10.0:
        network = _try_prune(network)
    
    return network
# EVOLVE-BLOCK-END
