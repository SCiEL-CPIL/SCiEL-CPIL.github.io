"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools

N = 18

# EVOLVE-BLOCK-START
def _apply_comparator(state, i, j):
    if state[i] > state[j]:
        state[i], state[j] = state[j], state[i]
    return state

def _is_sorted_network(network):
    # Test all 2^N Boolean inputs
    for mask in range(1 << N):
        state = [(mask >> i) & 1 for i in range(N)]
        for (i, j) in network:
            if state[i] > state[j]:
                state[i], state[j] = state[j], state[i]
        # Check sorted
        for i in range(N - 1):
            if state[i] > state[i + 1]:
                return False
    return True

def _remove_redundant_comparators(network):
    # Try to remove each comparator if the network still sorts
    current = list(network)
    improved = True
    while improved:
        improved = False
        for idx in range(len(current) - 1, -1, -1):
            candidate = current[:idx] + current[idx+1:]
            if _is_sorted_network(candidate):
                current = candidate
                improved = True
                break
    return current

def construct():
    # Odd-even transposition sorting network
    network = [(i, i + 1) for phase in range(N)
               for i in range(phase % 2, N - 1, 2)]
    
    # Deadline guard: 85s safety margin before 90s limit
    start_time = time.time()
    deadline = start_time + 85.0
    
    # Try to remove redundant comparators within time budget
    improved = True
    while improved and (time.time() < deadline):
        improved = False
        for idx in range(len(network) - 1, -1, -1):
            if time.time() >= deadline:
                break
            candidate = network[:idx] + network[idx+1:]
            if _is_sorted_network(candidate):
                network = candidate
                improved = True
                break
    
    return network
# EVOLVE-BLOCK-END
