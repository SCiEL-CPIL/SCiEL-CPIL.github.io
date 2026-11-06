import time
import numpy as np
import random

N = 18
TOTAL_INPUTS = 1 << N

def _precompute_inputs():
    """
    Precompute all 2^N Boolean inputs as a (N, 2^N) uint8 array.
    """
    all_inputs = np.zeros((N, TOTAL_INPUTS), dtype=np.uint8)
    for i in range(N):
        period = 1 << (i + 1)
        half = 1 << i
        pattern = np.zeros(period, dtype=np.uint8)
        pattern[half:] = 1
        all_inputs[i] = np.tile(pattern, TOTAL_INPUTS // period)
    return all_inputs

_ALL_INPUTS = _precompute_inputs()

def _verify_network(network):
    """
    Vectorized verification of the sorting network.
    Returns True if the network sorts all 0-1 inputs.
    """
    x = _ALL_INPUTS.copy()
    for i, j in network:
        # Optimization: Only check if swap is needed
        # If x[i] > x[j], swap.
        # Since inputs are 0/1, swap happens if x[i]=1 and x[j]=0.
        mask = (x[i] == 1) & (x[j] == 0)
        if np.any(mask):
            x[i][mask] = 0
            x[j][mask] = 1
    
    # Check if sorted: x[k] <= x[k+1] for all k
    # i.e., not (x[k]=1 and x[k+1]=0)
    diffs = x[1:].astype(np.int8) - x[:-1].astype(np.int8)
    if np.any(diffs == -1):
        return False
    return True

def _odd_even_transposition_network(n):
    """Odd-even transposition network."""
    comps = []
    for phase in range(n):
        for i in range(phase % 2, n - 1, 2):
            comps.append((i, i + 1))
    return comps

def _random_greedy_removal(network, deadline, max_passes=100):
    """
    Repeatedly attempt to remove random comparators.
    If a removal maintains sorting, keep it.
    This is a structural change from deterministic sequential removal to randomized search.
    """
    current_network = network[:]
    pass_count = 0
    
    while time.time() < deadline and pass_count < max_passes:
        pass_count += 1
        # Shuffle indices to try different orders
        indices = list(range(len(current_network)))
        random.shuffle(indices)
        
        improved = False
        for idx in indices:
            if time.time() > deadline:
                return current_network
            
            # Try removing current_network[idx]
            test_network = current_network[:idx] + current_network[idx+1:]
            
            if _verify_network(test_network):
                current_network = test_network
                improved = True
        
        if not improved:
            # If no improvement in a full pass, we might be stuck.
            # Break if we've done several stagnant passes to save time for other strategies?
            # For now, just break if no improvement, as re-shuffling rarely helps in deep local minima without add/swap.
            # But we can try a few more shuffles.
            if pass_count > 5:
                break
    return current_network

def construct():
    start_time = time.time()
    # Safety margin: 85s out of 90s
    deadline = start_time + 85.0
    
    # Start with Odd-Even Transposition (153 comparators)
    # This is a valid sorting network.
    network = _odd_even_transposition_network(N)
    
    # Verify initial network (should be True)
    if not _verify_network(network):
        # Fallback: This shouldn't happen, but return empty or minimal if broken
        return []
    
    # Phase 1: Deterministic backward removal
    # This is fast and removes obvious redundancies.
    improved = True
    while improved and (time.time() < deadline - 10.0):
        improved = False
        for idx in range(len(network) - 1, -1, -1):
            if time.time() > deadline - 10.0:
                break
            test_network = network[:idx] + network[idx+1:]
            if _verify_network(test_network):
                network = test_network
                improved = True
    
    # Phase 2: Randomized Greedy Removal
    # This helps escape local minima created by the order of removal.
    network = _random_greedy_removal(network, deadline)
    
    return network

def _apply_network(network, x):
    x = list(x)
    for i, j in network:
        if x[i] > x[j]:
            x[i], x[j] = x[j], x[i]
    return x

def _is_sorted_network(network):
    for mask in range(1 << N):
        x = [(mask >> i) & 1 for i in range(N)]
        result = _apply_network(network, x)
        if result != sorted(result):
            return False
    return True

def _depth(network):
    return len(network)
