import numpy as np
import time
import random

N = 17
NUM_INPUTS = 1 << N

def _build_seed_network():
    """Build odd-even transposition network (seed)."""
    network = []
    for phase in range(N):
        if phase % 2 == 0:
            for i in range(0, N - 1, 2):
                network.append((i, i + 1))
        else:
            for i in range(1, N - 1, 2):
                network.append((i, i + 1))
    return network

def _build_batcher_network():
    """Build Batcher's bitonic merging network for N=17 (padded to 32)."""
    # For N=17, we can use a known construction or build from scratch
    # Using a recursive bitonic sort approach
    def _bitonic_sort(n):
        """Generate comparators for bitonic sort on n wires (n must be power of 2)."""
        if n == 1:
            return []
        # First, sort two halves
        left = _bitonic_sort(n // 2)
        right = _bitonic_sort(n // 2)
        # Shift right
        right_shifted = [(i + n // 2, j + n // 2) for (i, j) in right]
        # Merge
        merged = _bitonic_merge(n)
        return left + right_shifted + merged

    def _bitonic_merge(n):
        """Generate comparators for bitonic merge on n wires."""
        if n == 1:
            return []
        half = n // 2
        # Compare elements i and i+half for i in 0..half-1
        comparators = []
        for i in range(half):
            if i < half - 1:  # i and i+half: i < i+half always true
                comparators.append((i, i + half))
        # Recursively merge
        left = _bitonic_merge(half)
        right = [(i + half, j + half) for (i, j) in _bitonic_merge(half)]
        return comparators + left + right

    # We need to handle 17 wires. Pad to 32 and use only first 17.
    # Actually, let's use a simpler approach: use a known small sorting network
    # For 17 inputs, we can use the fact that 17 = 2^4 + 1
    # Use a hybrid approach: sort first 16 with bitonic, then insert 17th
    # For now, let's just use the seed network and improve it
    
    # Alternative: Use a constructive algorithm
    # For now, return a reasonable starting network
    # Let's use a simpler seed: the odd-even transposition is fine
    return _build_seed_network()

def _verify_network_np(network, _INITIAL_STATE):
    """
    Verify network using Numpy.
    Returns True if valid.
    """
    state = _INITIAL_STATE.copy()
    
    for i, j in network:
        si = state[i]
        sj = state[j]
        state[i] = si & sj
        state[j] = si | sj
        
    return np.all(state[:-1] <= state[1:])

def _precompute_initial_state():
    """Precompute initial state for all 2^N Boolean inputs."""
    indices = np.arange(NUM_INPUTS, dtype=np.uint32)
    state = np.zeros((N, NUM_INPUTS), dtype=np.uint8)
    for w in range(N):
        state[w] = ((indices >> w) & 1).astype(np.uint8)
    return state

_INITIAL_STATE = _precompute_initial_state()

def _greedy_remove(network, max_time=None):
    """
    Greedily remove comparators from the end.
    """
    start_time = time.time()
    changed = True
    while changed:
        changed = False
        if max_time and time.time() - start_time > max_time:
            break
        for idx in range(len(network) - 1, -1, -1):
            new_net = network[:idx] + network[idx+1:]
            if _verify_network_np(new_net, _INITIAL_STATE):
                network = new_net
                changed = True
                break
    return network

def _local_search_with_swaps(network, max_time):
    """
    Local search: try swapping adjacent comparators, or removing/inserting.
    """
    current_net = list(network)
    best_net = list(current_net)
    best_size = len(current_net)
    start_time = time.time()
    
    # Temperature-based acceptance for non-improving moves
    temperature = 2.0
    cooling = 0.999
    min_temp = 0.1
    
    while time.time() - start_time < max_time and temperature > min_temp:
        # Randomly pick a move type
        move_type = random.random()
        
        if move_type < 0.3 and len(current_net) > 0:
            # Delete a random comparator
            idx = random.randint(0, len(current_net) - 1)
            new_net = current_net[:idx] + current_net[idx+1:]
        elif move_type < 0.6 and len(current_net) > 1:
            # Swap two adjacent comparators
            i = random.randint(0, len(current_net) - 2)
            j = i + 1
            new_net = list(current_net)
            new_net[i], new_net[j] = new_net[j], new_net[i]
        elif move_type < 0.8:
            # Move a comparator to a random position
            i = random.randint(0, len(current_net) - 1)
            j = random.randint(0, len(current_net) - 1)
            if i != j:
                new_net = list(current_net)
                cmp = new_net.pop(i)
                new_net.insert(j, cmp)
            else:
                new_net = list(current_net)
        else:
            # Insert a random comparator at a random position
            a = random.randint(0, N - 2)
            b = random.randint(a + 1, N - 1)
            pos = random.randint(0, len(current_net))
            new_net = current_net[:pos] + [(a, b)] + current_net[pos:]
        
        new_size = len(new_net)
        
        # Verify
        is_valid = _verify_network_np(new_net, _INITIAL_STATE)
        
        if is_valid:
            delta = new_size - len(current_net)
            if delta < 0:
                # Improvement
                current_net = new_net
                if new_size < best_size:
                    best_net = new_net
                    best_size = new_size
            else:
                # Non-improving or worsening: accept with probability
                prob = np.exp(-delta / max(temperature, 1e-10))
                if random.random() < prob:
                    current_net = new_net
        # If invalid, reject
        
        temperature *= cooling
    
    return best_net

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin before 90s kill
    
    # 1. Build seed network
    network = _build_seed_network()
    
    # 2. Greedy removal to quickly reduce size
    network = _greedy_remove(network, max_time=10.0)
    
    # 3. Local search with swaps to escape local optima
    best_net = _local_search_with_swaps(network, max_time=deadline - time.time())
    
    return best_net
