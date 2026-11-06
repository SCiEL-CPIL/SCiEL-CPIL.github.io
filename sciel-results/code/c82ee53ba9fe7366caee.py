"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

N = 16

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a size-efficient sorting network on 16 wires.
    
    Uses a hybrid approach:
    1. Start with a known good construction (odd-even transposition: 120 comparators)
    2. Try to find smaller networks using local search / simulated annealing
    3. Return the best valid network found within the time budget
    
    The zero-one principle is used for verification: a sorting network sorts all inputs
    if and only if it sorts all 2^16 = 65536 binary inputs.
    """
    deadline = time.time() + 50  # Safety margin before 90s wall-clock limit
    
    def verify_network(comparators):
        """Verify that a sorting network sorts all 65536 binary inputs using zero-one principle."""
        # Use numpy for efficient batch verification
        # Each input is a 16-bit integer; we need to check that after applying all comparators,
        # the output is sorted (all 0s then all 1s)
        
        for mask in range(2 ** N):
            # Create the input vector
            vec = [(mask >> i) & 1 for i in range(N)]
            
            # Apply each comparator
            for i, j in comparators:
                if vec[i] > vec[j]:
                    vec[i], vec[j] = vec[j], vec[i]
            
            # Check if sorted: all 0s followed by all 1s
            seen_one = False
            for v in vec:
                if v == 1:
                    seen_one = True
                elif seen_one:
                    return False
        return True
    
    def verify_network_fast(comparators):
        """Faster verification using numpy vectorized operations over all 65536 inputs at once."""
        # Create all 65536 binary inputs as a matrix of shape (65536, 16)
        # Input k has bit i = (k >> i) & 1
        inputs = np.array([[(k >> i) & 1 for i in range(N)] for k in range(2 ** N)], dtype=np.uint8)
        
        # Apply each comparator
        for i, j in comparators:
            # Where input[i] > input[j], swap
            swap_mask = inputs[:, i] > inputs[:, j]
            # Perform swap using temporary
            tmp = inputs[:, i].copy()
            inputs[:, i] = np.where(swap_mask, inputs[:, j], inputs[:, i])
            inputs[:, j] = np.where(swap_mask, tmp, inputs[:, j])
        
        # Check if all rows are sorted (all 0s then all 1s)
        # A row is sorted if it's non-decreasing
        diffs = np.diff(inputs, axis=1)
        # All diffs should be >= 0
        return np.all(diffs >= 0)
    
    def count_comparators(comparators):
        return len(comparators)
    
    def compute_depth(comparators):
        """Compute parallel depth of the network."""
        # Build a graph where each comparator is a node, edges indicate dependencies
        # Two comparators conflict if they share a wire
        # Use topological sort / longest path in DAG
        
        # Assign levels: process comparators in order
        # For each wire, track the last level it was used
        wire_level = [0] * N
        max_depth = 0
        
        for i, j in comparators:
            # This comparator can start after both wires are free
            level = max(wire_level[i], wire_level[j]) + 1
            wire_level[i] = level
            wire_level[j] = level
            max_depth = max(max_depth, level)
        
        return max_depth
    
    # Start with odd-even transposition network (120 comparators)
    best_network = [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]
    best_count = count_comparators(best_network)
    
    # Verify the initial network
    if not verify_network_fast(best_network):
        # This shouldn't happen, but just in case
        return best_network
    
    # Local search: try to remove or swap comparators to reduce count
    # Strategy: 
    # 1. Try removing each comparator and check if still valid
    # 2. Try swapping pairs of comparators
    # 3. Use simulated annealing to escape local minima
    
    rng = np.random.RandomState(42)
    
    # First, try greedy removal: go through each comparator and try to remove it
    # If removing it still keeps the network valid, keep it removed
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        current = list(best_network)
        for idx in range(len(current) - 1, -1, -1):
            if time.time() > deadline - 5:
                break
            # Try removing comparator at idx
            candidate = current[:idx] + current[idx+1:]
            if len(candidate) == 0:
                break
            if verify_network_fast(candidate):
                best_network = candidate
                best_count = len(candidate)
                improved = True
                break
        # After one removal, restart the pass
    
    # Now try local moves: swap two comparators, or replace one with another
    # Simulated annealing approach
    T = 10.0  # Temperature
    T_min = 0.1
    cooling = 0.995
    
    current = list(best_network)
    current_count = len(current)
    
    while time.time() < deadline - 3 and T > T_min:
        # Generate a random move
        move_type = rng.random()
        
        if move_type < 0.3 and len(current) > 0:
            # Remove a random comparator
            idx = rng.randint(0, len(current) - 1)
            candidate = current[:idx] + current[idx+1:]
            delta = -1
        elif move_type < 0.6:
            # Add a random comparator
            i = rng.randint(0, N - 1)
            j = rng.randint(i + 1, N)
            candidate = current + [(i, j)]
            delta = 1
        elif move_type < 0.8:
            # Swap two comparators
            if len(current) >= 2:
                idx1 = rng.randint(0, len(current) - 1)
                idx2 = rng.randint(0, len(current) - 1)
                if idx1 != idx2:
                    candidate = list(current)
                    candidate[idx1], candidate[idx2] = candidate[idx2], candidate[idx1]
                    delta = 0
                else:
                    T *= cooling
                    continue
            else:
                T *= cooling
                continue
        else:
            # Replace a comparator with a random one
            if len(current) > 0:
                idx = rng.randint(0, len(current) - 1)
                i = rng.randint(0, N - 1)
                j = rng.randint(i + 1, N)
                candidate = list(current)
                candidate[idx] = (i, j)
                delta = 0
            else:
                T *= cooling
                continue
        
        # Evaluate candidate
        if verify_network_fast(candidate):
            # Valid move, always accept
            current = candidate
            current_count = len(candidate)
            if current_count < best_count:
                best_network = candidate
                best_count = current_count
        else:
            # Invalid: accept with probability exp(-delta/T) if delta > 0
            if delta > 0 and rng.random() < np.exp(-delta / T):
                current = candidate
                current_count = len(candidate)
        
        T *= cooling
    
    # Final greedy removal pass on the best network
    if time.time() < deadline - 2:
        final = list(best_network)
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            for idx in range(len(final) - 1, -1, -1):
                if time.time() > deadline - 2:
                    break
                candidate = final[:idx] + final[idx+1:]
                if len(candidate) == 0:
                    break
                if verify_network_fast(candidate):
                    final = candidate
                    improved = True
                    break
        if len(final) < best_count:
            best_network = final
            best_count = len(final)
    
    return best_network
# EVOLVE-BLOCK-END
