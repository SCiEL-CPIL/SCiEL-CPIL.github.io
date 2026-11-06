import numpy as np
import random
import time

N = 17
NUM_STATES = 1 << N
SORTED_MASKS = set()
for k in range(N + 1):
    # k zeros, N-k ones
    mask = ((1 << (N - k)) - 1) << k
    SORTED_MASKS.add(mask)

# Precompute sorted masks as a numpy array for fast isin
SORTED_MASKS_ARR = np.array(list(SORTED_MASKS), dtype=np.uint32)

def verify_network(network):
    """
    Verify if the network sorts all 2^N boolean inputs.
    Returns True if valid, False otherwise.
    """
    # Initialize states with all possible inputs
    # states[i] represents the state of the i-th input after processing
    states = np.arange(NUM_STATES, dtype=np.uint32)
    
    for i, j in network:
        # Compare-exchange (i, j): min to i, max to j
        # If bit i is 1 and bit j is 0, swap them.
        # mask = (1 << i) | (1 << j)
        # cond = (states & (1 << i)) & ~(states & (1 << j))
        # states[cond] ^= mask
        
        mask = (1 << i) | (1 << j)
        bit_i = states & (1 << i)
        bit_j = states & (1 << j)
        cond = bit_i & ~bit_j
        states[cond] ^= mask
        
    # Check if all states are sorted
    # A state is sorted if it is in SORTED_MASKS
    # np.isin is fast for small sets
    is_sorted = np.isin(states, SORTED_MASKS_ARR)
    return np.all(is_sorted)

def construct():
    # Initial network: Odd-even transposition
    network = []
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            network.append((i, i + 1))
    
    # Verify initial network
    if not verify_network(network):
        # Fallback: Simple bubble sort network
        network = []
        for i in range(N):
            for j in range(N - 1 - i):
                network.append((j, j + 1))
    
    best_network = list(network)
    best_len = len(network)
    
    # Search loop
    start_time = time.time()
    timeout = 55  # Leave 5s for safety
    
    # Greedy removal phase
    improved = True
    while improved and time.time() - start_time < timeout:
        improved = False
        for idx in range(len(network) - 1, -1, -1):
            candidate = network[:idx] + network[idx+1:]
            if verify_network(candidate):
                network = candidate
                improved = True
                if len(network) < best_len:
                    best_network = list(network)
                    best_len = len(network)
                break # Restart loop with new length
    
    # Local search phase: Try to move comparators to enable further removal
    # We try to move a random comparator to a random position
    # If the new network is valid, we keep it.
    # We do this for a fixed number of iterations or until time limit.
    
    # To make it more effective, we can try all moves for a few iterations
    # But "all moves" is O(C^2) verifications. C ~ 70, so ~5000 verifications.
    # Each verification is fast.
    
    while time.time() - start_time < timeout:
        moved = False
        # Try moving each comparator to each other position
        # This is expensive, so we do it randomly or in chunks
        for idx in range(len(network)):
            if time.time() - start_time > timeout:
                break
            comp = network[idx]
            # Try moving to a random position
            new_idx = random.randint(0, len(network) - 1)
            if new_idx == idx:
                continue
            
            # Create candidate
            candidate = list(network)
            candidate.pop(idx)
            candidate.insert(new_idx, comp)
            
            if verify_network(candidate):
                network = candidate
                moved = True
                if len(network) < best_len:
                    best_network = list(network)
                    best_len = len(network)
                break # Restart move loop
        
        if not moved:
            # No single move improved validity (or we tried and failed)
            # Try adding a random comparator and then removing?
            # Or just break and try a different strategy?
            # Let's try to add a random comparator and then run greedy removal again
            # This might help escape local minima
            if len(network) < 200: # Safety bound
                i = random.randint(0, N-2)
                j = random.randint(i+1, N-1)
                network.append((i, j))
                # Run greedy removal again
                improved = True
                while improved and time.time() - start_time < timeout:
                    improved = False
                    for ridx in range(len(network) - 1, -1, -1):
                        candidate = network[:ridx] + network[ridx+1:]
                        if verify_network(candidate):
                            network = candidate
                            improved = True
                            if len(network) < best_len:
                                best_network = list(network)
                                best_len = len(network)
                            break
            
    return best_network
