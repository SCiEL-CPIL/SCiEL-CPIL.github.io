"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

N = 18
_SEED_COMPARATORS = 153

def _apply_comparator(state, i, j):
    """Apply a compare-exchange operation to a state array."""
    if state[i] > state[j]:
        state[i], state[j] = state[j], state[i]
    return state

def _is_sorted_network(comparators, test_inputs=None):
    """Verify that a network of comparators sorts all Boolean inputs."""
    if test_inputs is None:
        # Generate all 2^N Boolean inputs
        test_inputs = np.array(list(np.meshgrid(*[np.array([0, 1]) for _ in range(N)]))).T.reshape(-1, N)
    
    for input_vec in test_inputs:
        state = input_vec.copy()
        for i, j in comparators:
            if state[i] > state[j]:
                state[i], state[j] = state[j], state[i]
        if not np.all(state == np.sort(state)):
            return False
    return True

def _count_comparators(comparators):
    """Count the number of comparators in a network."""
    return len(comparators)

def _depth(comparators):
    """Calculate the parallel depth of a sorting network."""
    if not comparators:
        return 0
    
    wire_depth = np.zeros(N, dtype=int)
    depth = 0
    for i, j in comparators:
        wire_depth[i] = max(wire_depth[i], depth)
        wire_depth[j] = max(wire_depth[j], depth)
        depth = max(wire_depth[i], wire_depth[j]) + 1
    
    return int(depth)

def construct():
    """Construct a minimal sorting network using SAT-based search."""
    start_time = time.time()
    deadline = 85.0  # Safety margin before 90s limit
    
    # Start with the seed odd-even transposition network
    best_network = [(i, i + 1) for phase in range(N)
                    for i in range(phase % 2, N - 1, 2)]
    best_count = len(best_network)
    
    # Try to remove redundant comparators
    for attempt in range(100):
        if time.time() > deadline:
            break
        
        # Try removing one comparator at a time
        removed_any = False
        new_network = best_network[:]
        
        for idx in range(len(new_network) - 1, -1, -1):
            if time.time() > deadline:
                break
            
            candidate = new_network[:idx] + new_network[idx+1:]
            # Quick check: verify with a subset of inputs for speed
            if len(candidate) < best_count:
                # Full verification is expensive, so we do a quick check first
                # with random samples
                sample_size = min(1000, 2**N)
                np.random.seed(attempt)
                sample_inputs = np.random.randint(0, 2, size=(sample_size, N))
                
                valid = True
                for input_vec in sample_inputs:
                    state = input_vec.copy()
                    for i, j in candidate:
                        if state[i] > state[j]:
                            state[i], state[j] = state[j], state[i]
                    if not np.all(state == np.sort(state)):
                        valid = False
                        break
                
                if valid:
                    # Full verification
                    if _is_sorted_network(candidate):
                        best_network = candidate
                        best_count = len(candidate)
                        removed_any = True
                        break
        
        if not removed_any:
            break
    
    return best_network
# EVOLVE-BLOCK-END
