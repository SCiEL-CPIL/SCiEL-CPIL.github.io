"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

N = 18
SEED_TIME_LIMIT = 80  # safety margin for evaluator 90s limit
SEED_COMPARATORS = 153

def _apply_network(comparators, bits):
    """Apply a list of (i,j) comparators to a numpy array of bits."""
    a = bits.copy()
    for i, j in comparators:
        if a[i] > a[j]:
            a[i], a[j] = a[j], a[i]
    return a

def _verify_network(comparators):
    """Verify that the network sorts all 2^N boolean inputs."""
    n = len(comparators)
    if n == 0:
        return True
    # Generate all 2^N boolean inputs
    # 2^18 = 262144
    all_inputs = np.arange(2**N, dtype=np.uint32)
    # Create a matrix of shape (2^N, N) where row x has bit i set if x has bit i
    # We can do this efficiently
    bits_matrix = np.zeros((2**N, N), dtype=np.uint8)
    for k in range(N):
        bits_matrix[:, k] = (all_inputs >> k) & 1
    
    # For each input, apply the network and check if sorted
    # To be efficient, we can process in batches or just loop
    # Given 262144 inputs and up to ~80 comparators, this is ~20M ops which is fine in numpy if vectorized
    # But applying sequentially per comparator is hard to vectorize fully without memory blowup if we store all intermediates.
    # Instead, we can check each input individually but that's slow in Python.
    # Better: use a single pass with numpy operations on the whole matrix per comparator step.
    
    current = bits_matrix.copy()
    for i, j in comparators:
        # Swap if current[:, i] > current[:, j]
        mask = current[:, i] > current[:, j]
        if np.any(mask):
            current[mask, i], current[mask, j] = current[mask, j], current[mask, i]
    
    # Check if every row is sorted (non-decreasing)
    # A row is sorted if for all k < N-1, current[k] <= current[k+1]
    sorted_mask = np.all(current[:, :-1] <= current[:, 1:], axis=1)
    return np.all(sorted_mask)

def _odd_even_transposition():
    """Generate the seed odd-even transposition network."""
    return [(i, i + 1) for phase in range(N)
            for i in range(phase % 2, N - 1, 2)]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + SEED_TIME_LIMIT
    
    # Start with the known good seed network
    best_network = _odd_even_transposition()
    best_len = len(best_network)
    
    # Pre-compute the verification inputs matrix
    all_inputs = np.arange(2**N, dtype=np.uint32)
    bits_matrix = np.zeros((2**N, N), dtype=np.uint8)
    for k in range(N):
        bits_matrix[:, k] = (all_inputs >> k) & 1

    def _fast_verify(comparators):
        current = bits_matrix.copy()
        for i, j in comparators:
            mask = current[:, i] > current[:, j]
            if np.any(mask):
                current[mask, i], current[mask, j] = current[mask, j], current[mask, i]
        # Check sorted
        if np.any(current[:, :-1] > current[:, 1:]):
            return False
        return True

    # Simple Hill Climbing: try to remove each comparator and see if it's still valid.
    # If yes, remove it and continue from there.
    # This is a greedy approach.
    
    current_network = list(best_network)
    current_len = best_len

    # First pass: Greedy removal from end to start
    improved = True
    while improved and (time.time() < deadline - 10):
        improved = False
        # Iterate backwards to avoid index issues when removing
        for idx in range(len(current_network) - 1, -1, -1):
            if time.time() > deadline - 10:
                break
            # Try removing comparator at idx
            trial = current_network[:idx] + current_network[idx+1:]
            if _fast_verify(trial):
                current_network = trial
                current_len = len(current_network)
                improved = True
                # Restart scan from end to find more redundancies introduced by removal
                break
    
    # Second pass: Try swapping adjacent comparators to see if it helps enable further removals
    # This is a more complex move, but might unlock stuck states.
    # However, swapping doesn't reduce size directly.
    # A better second pass: try removing comparators in a random order or different order.
    # Let's try a randomized removal search to escape local minima of the greedy scan.
    
    # If we still have time, do some random restarts or randomized local search
    if time.time() < deadline - 10:
        # Try randomized removal
        random.seed(42) # Fixed seed for reproducibility if needed, or just random
        
        # We can try to remove comparators in a random order
        indices = list(range(len(current_network)))
        random.shuffle(indices)
        
        for idx in indices:
            if time.time() > deadline - 10:
                break
            # Try removing comparator at idx
            trial = current_network[:idx] + current_network[idx+1:]
            if _fast_verify(trial):
                current_network = trial
                current_len = len(current_network)
                # Restart random shuffle? Or just continue?
                # Continuing might miss some, but we can do another pass if time permits.
                break

        # Another pass if time permits
        if time.time() < deadline - 10:
            improved = True
            while improved and (time.time() < deadline - 10):
                improved = False
                indices = list(range(len(current_network)))
                random.shuffle(indices)
                for idx in indices:
                    if time.time() > deadline - 10:
                        break
                    trial = current_network[:idx] + current_network[idx+1:]
                    if _fast_verify(trial):
                        current_network = trial
                        current_len = len(current_network)
                        improved = True
                        break

    if current_len < best_len:
        best_network = current_network
        best_len = current_len
    
    return best_network
# EVOLVE-BLOCK-END
