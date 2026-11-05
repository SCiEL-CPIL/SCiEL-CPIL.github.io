"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 18
import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin before 90s kill
    
    # Start with odd-even transposition network
    def odd_even_transposition():
        comparators = []
        for phase in range(N):
            for i in range(phase % 2, N - 1, 2):
                comparators.append((i, i + 1))
        return comparators
    
    def is_sorting_network(comparators):
        """Check if the network sorts all 2^N Boolean inputs."""
        # Test all 2^N Boolean inputs
        for mask in range(1 << N):
            wires = [(mask >> i) & 1 for i in range(N)]
            for (i, j) in comparators:
                if wires[i] > wires[j]:
                    wires[i], wires[j] = wires[j], wires[i]
            # Check sorted
            for i in range(N - 1):
                if wires[i] > wires[i + 1]:
                    return False
        return True
    
    def is_sorting_network_fast(comparators):
        """Faster check using numpy for all Boolean inputs at once."""
        # Represent all 2^N inputs as a (2^N, N) binary matrix
        total = 1 << N
        # Create input matrix: each row is a different boolean assignment
        bits = np.arange(total, dtype=np.int32)
        wires = np.zeros((total, N), dtype=np.int8)
        for i in range(N):
            wires[:, i] = (bits >> i) & 1
        
        # Apply comparators
        for (i, j) in comparators:
            wi = wires[:, i].copy()
            wj = wires[:, j].copy()
            # min goes to i, max goes to j
            wires[:, i] = np.minimum(wi, wj)
            wires[:, j] = np.maximum(wi, wj)
        
        # Check if all rows are sorted (non-decreasing)
        diffs = np.diff(wires, axis=1)
        return np.all(diffs >= 0)
    
    # Start with odd-even transposition
    best_network = odd_even_transposition()
    best_count = len(best_network)
    
    # Try to improve by removing redundant comparators
    # Strategy: try removing each comparator and check if still valid
    # If valid, remove it and continue
    
    current = list(best_network)
    
    # First, try to remove redundant comparators (greedy)
    changed = True
    while changed and time.time() < deadline - 5.0:
        changed = False
        for idx in range(len(current) - 1, -1, -1):
            if time.time() > deadline - 5.0:
                break
            # Try removing comparator at idx
            test = current[:idx] + current[idx+1:]
            if is_sorting_network_fast(test):
                current = test
                changed = True
                best_network = list(current)
                best_count = len(current)
                break  # Restart the loop since indices changed
    
    # Now try local search: swap or modify comparators
    # Try replacing a comparator with a different one
    for iteration in range(500):
        if time.time() > deadline - 5.0:
            break
        if not current:
            break
        
        # Randomly pick a comparator to modify
        idx = random.randint(0, len(current) - 1)
        i, j = current[idx]
        
        # Try changing to a different comparator
        # Pick a random different pair
        new_i = random.randint(0, N - 1)
        new_j = random.randint(0, N - 1)
        if new_i > new_j:
            new_i, new_j = new_j, new_i
        if new_i == new_j:
            continue
        if (new_i, new_j) == (i, j):
            continue
        
        test = list(current)
        test[idx] = (new_i, new_j)
        
        if is_sorting_network_fast(test):
            current = test
            best_network = list(current)
            best_count = len(current)
    
    # Try adding comparators that allow further removals
    # This is more complex, skip for now due to time
    
    # Final check
    if not is_sorting_network_fast(best_network):
        # Fallback to odd-even transposition
        best_network = odd_even_transposition()
    
    return best_network
# EVOLVE-BLOCK-END
