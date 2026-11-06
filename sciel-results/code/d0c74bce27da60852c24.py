import time
import numpy as np

N = 17

# EVOLVE-BLOCK-START
def _generate_seed_network():
    """Generates an Odd-Even Transposition sorting network."""
    comps = []
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            comps.append((i, i + 1))
    return comps

def _verify_network(network):
    """
    Verifies if the network sorts all 2^N boolean inputs.
    Uses NumPy for vectorized simulation.
    """
    if not hasattr(_verify_network, 'cached_inputs'):
        # Generate all 2^N boolean inputs
        # Shape: (2^N, N)
        bits = (np.arange(2**N, dtype=np.uint64)[:, None] >> np.arange(N, dtype=np.uint64)[None, :]) & 1
        _verify_network.cached_inputs = bits.astype(np.uint8)
        
    A = _verify_network.cached_inputs.copy()
    
    for i, j in network:
        min_ij = np.minimum(A[:, i], A[:, j])
        max_ij = np.maximum(A[:, i], A[:, j])
        A[:, i] = min_ij
        A[:, j] = max_ij
        
    diffs = np.diff(A, axis=1)
    return np.all(diffs >= 0)

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 5s safety margin
    
    # 1. Generate seed network
    network = _generate_seed_network()
    
    # 2. Local Search: Remove redundant comparators
    current_network = network[:]
    
    # Strategies for iteration order
    strategies = ["reverse", "forward", "random"]
    
    for strat in strategies:
        while time.time() < deadline:
            modified = False
            
            if strat == "reverse":
                indices = range(len(current_network) - 1, -1, -1)
            elif strat == "forward":
                indices = range(len(current_network))
            else:
                indices = list(range(len(current_network)))
                np.random.shuffle(indices)
                
            for idx in indices:
                if time.time() > deadline:
                    break
                
                # Try removing comparator at idx
                candidate = current_network[:idx] + current_network[idx+1:]
                
                if _verify_network(candidate):
                    current_network = candidate
                    modified = True
                    
            if not modified:
                break
                
    return current_network
# EVOLVE-BLOCK-END
