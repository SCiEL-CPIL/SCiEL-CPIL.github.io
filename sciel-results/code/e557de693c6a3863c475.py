"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 17

# EVOLVE-BLOCK-START
def _make_all_inputs():
    """Generate all 2^N boolean inputs as a (2^N, N) uint8 array."""
    n = 1 << N
    # Create bit patterns: for each bit position, 0 for first half, 1 for second half
    arr = np.zeros((n, N), dtype=np.uint8)
    for b in range(N):
        # Bit b is 1 for the second half of every 2^(b+1) block
        period = 1 << (b + 1)
        half = 1 << b
        # Create pattern: [0]*half + [1]*half repeated
        pattern = np.zeros(period, dtype=np.uint8)
        pattern[half:] = 1
        arr[:, b] = np.tile(pattern, n // period)
    return arr

def _simulate(network, inputs):
    """Simulate network on all inputs. Returns (num_inputs, N) array."""
    state = inputs.copy()
    for i, j in network:
        a = state[:, i]
        b = state[:, j]
        # min goes to i, max goes to j
        state[:, i] = np.minimum(a, b)
        state[:, j] = np.maximum(a, b)
    return state

def _is_sorting_network(network):
    """Check if network sorts all boolean inputs using zero-one principle."""
    inputs = _make_all_inputs()
    outputs = _simulate(network, inputs)
    # Check: for each input, outputs must be non-decreasing
    # i.e., outputs[:, k] <= outputs[:, k+1] for all k
    for k in range(N - 1):
        if np.any(outputs[:, k] > outputs[:, k + 1]):
            return False
    return True

def _try_remove(network, idx):
    """Try removing comparator at index idx. Return new network if valid, else None."""
    if idx < 0 or idx >= len(network):
        return None
    new_net = network[:idx] + network[idx+1:]
    if _is_sorting_network(new_net):
        return new_net
    return None

def _try_swap(network, i, j):
    """Try swapping comparators at positions i and j. Return new network if valid, else None."""
    if i == j:
        return None
    new_net = list(network)
    new_net[i], new_net[j] = new_net[j], new_net[i]
    if _is_sorting_network(new_net):
        return new_net
    return None

def _local_search_remove(network, deadline):
    """Iteratively try to remove comparators until no more can be removed or deadline hit."""
    net = list(network)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for idx in range(len(net)):
            if time.time() >= deadline:
                break
            new_net = _try_remove(net, idx)
            if new_net is not None:
                net = new_net
                improved = True
                break  # restart scan from beginning after a removal
    return net

def _local_search_swap_remove(network, deadline):
    """Try swapping adjacent comparators then removing, to find more reductions."""
    net = list(network)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(len(net) - 1):
            if time.time() >= deadline:
                break
            # Try swapping i and i+1, then see if we can remove something
            new_net = _try_swap(net, i, i + 1)
            if new_net is not None:
                # After swap, try to remove
                for idx in range(len(new_net)):
                    if time.time() >= deadline:
                        break
                    reduced = _try_remove(new_net, idx)
                    if reduced is not None:
                        net = reduced
                        improved = True
                        break
                if improved:
                    break
            # Also try direct removal at position i after swap attempt
            if not improved:
                reduced = _try_remove(net, i)
                if reduced is not None:
                    net = reduced
                    improved = True
                    break
    return net

def _odd_even_transposition():
    """Generate odd-even transposition sorting network for N wires."""
    return [(i, i + 1) for phase in range(N)
            for i in range(phase % 2, N - 1, 2)]

def construct():
    """Construct a sorting network with as few comparators as possible."""
    start_time = time.time()
    deadline = start_time + 75.0  # 75s deadline, 15s safety margin
    
    # Start with odd-even transposition network
    net = _odd_even_transposition()
    
    # First pass: pure removal local search
    net = _local_search_remove(net, deadline)
    
    # Second pass: swap-and-remove to find further reductions
    if time.time() < deadline:
        net = _local_search_swap_remove(net, deadline)
    
    # Final verification
    if not _is_sorting_network(net):
        # Fallback to original if something went wrong
        net = _odd_even_transposition()
    
    return net
# EVOLVE-BLOCK-END
