import numpy as np
import time
import random

N = 18
DEADLINE = 85.0  # 5 seconds margin

def generate_seed_network():
    """Odd-even transposition network for N wires."""
    return [(i, i + 1) for phase in range(N)
            for i in range(phase % 2, N - 1, 2)]

def verify_network(network, states=None):
    """
    Verify if the network sorts all 2^N boolean inputs.
    Uses NumPy for parallel simulation.
    """
    if states is None:
        # Generate all 2^N boolean states
        # Shape: (2^N, N)
        # Using bit tricks to generate
        states = np.zeros((1 << N, N), dtype=np.int8)
        for i in range(N):
            # Fill column i with bits of 0..2^N-1
            # Bit i is set if (x >> i) & 1
            # We can generate this vectorized
            # x = np.arange(1 << N)
            # states[:, i] = (x >> i) & 1
            # This is fast enough for N=18
            pass
        
        # More efficient generation
        x = np.arange(1 << N, dtype=np.int64)
        for i in range(N):
            states[:, i] = (x >> i) & 1

    # Apply network
    for i, j in network:
        # min to i, max to j
        min_val = np.minimum(states[:, i], states[:, j])
        max_val = np.maximum(states[:, i], states[:, j])
        states[:, i] = min_val
        states[:, j] = max_val

    # Check if sorted
    # A sequence is sorted if a[i] <= a[i+1] for all i
    # diff >= 0
    diffs = np.diff(states, axis=1)
    return np.all(diffs >= 0)

def construct():
    start_time = time.time()
    
    # Start with seed
    current_network = generate_seed_network()
    best_network = current_network[:]
    best_score = len(current_network)
    
    # Pre-generate states for verification to avoid overhead in loop
    # However, states are modified in place by verify_network if not copied.
    # We must copy states or use a function that doesn't mutate the original.
    # Better: create a function that takes a fresh copy or uses a buffer.
    # Given 2^18 * 18 bytes ~ 4.7 MB, copying is cheap.
    
    def is_valid(net):
        # Create fresh states
        x = np.arange(1 << N, dtype=np.int64)
        states = np.zeros((1 << N, N), dtype=np.int8)
        for i in range(N):
            states[:, i] = (x >> i) & 1
            
        for i, j in net:
            min_val = np.minimum(states[:, i], states[:, j])
            max_val = np.maximum(states[:, i], states[:, j])
            states[:, i] = min_val
            states[:, j] = max_val
        diffs = np.diff(states, axis=1)
        return np.all(diffs >= 0)

    # Local Search
    # Operations:
    # 1. Remove a random comparator
    # 2. Move a random comparator to a random position
    # 3. Swap two comparators
    
    while time.time() < DEADLINE:
        new_network = current_network[:]
        op = random.randint(0, 2)
        
        if op == 0 and len(new_network) > 0:
            # Remove
            idx = random.randint(0, len(new_network) - 1)
            new_network.pop(idx)
        elif op == 1 and len(new_network) > 1:
            # Move
            idx = random.randint(0, len(new_network) - 1)
            comp = new_network.pop(idx)
            new_idx = random.randint(0, len(new_network))
            new_network.insert(new_idx, comp)
        elif op == 2 and len(new_network) > 1:
            # Swap
            i = random.randint(0, len(new_network) - 1)
            j = random.randint(0, len(new_network) - 1)
            if i != j:
                new_network[i], new_network[j] = new_network[j], new_network[i]
        
        if is_valid(new_network):
            current_network = new_network
            if len(current_network) < best_score:
                best_score = len(current_network)
                best_network = current_network[:]
        else:
            # Revert
            pass
            # Optional: try a smaller perturbation or accept with probability (Simulated Annealing)
            # But for now, simple hill climbing is safer.
            
    return best_network
