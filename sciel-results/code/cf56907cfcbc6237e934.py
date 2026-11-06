"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

N = 18

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a sorting network on 18 wires with as few comparators as possible.
    
    Strategy:
    1. Start from the odd-even transposition network (153 comparators).
    2. Use a heuristic search to remove redundant comparators.
    3. Verify correctness on all 2^18 Boolean inputs.
    
    The search tries to remove comparators one at a time, checking if the
    network still sorts all Boolean inputs. This is a greedy approach that
    may not find the optimal solution but should find a good one.
    """
    start_time = time.time()
    deadline = start_time + 80  # Leave 10s safety margin
    
    # Start with odd-even transposition network
    network = [(i, i + 1) for phase in range(N)
               for i in range(phase % 2, N - 1, 2)]
    
    # Verify it's correct
    if not _verify_network(network):
        return network
    
    # Try to remove comparators
    improved = True
    while improved:
        if time.time() > deadline - 5:
            break
        improved = False
        # Try removing each comparator
        for idx in range(len(network) - 1, -1, -1):
            if time.time() > deadline - 5:
                break
            # Try removing comparator at idx
            candidate = network[:idx] + network[idx+1:]
            if _verify_network(candidate):
                network = candidate
                improved = True
                break
    
    return network


def _verify_network(network):
    """
    Verify that the network sorts all 2^N Boolean inputs.
    Uses the zero-one principle: a comparator network sorts all inputs
    if and only if it sorts all 0-1 inputs.
    """
    n = len(network)
    if n == 0:
        return True
    
    # Check all 2^18 = 262144 Boolean inputs
    for mask in range(1 << N):
        # Create the input vector
        vec = [(mask >> i) & 1 for i in range(N)]
        
        # Apply the network
        for i, j in network:
            if vec[i] > vec[j]:
                vec[i], vec[j] = vec[j], vec[i]
        
        # Check if sorted
        for k in range(N - 1):
            if vec[k] > vec[k + 1]:
                return False
    
    return True


def _verify_network_fast(network):
    """
    Faster verification using bit operations.
    """
    # For each input mask, simulate the network
    for mask in range(1 << N):
        vec = mask
        for i, j in network:
            vi = (vec >> i) & 1
            vj = (vec >> j) & 1
            if vi > vj:
                # Swap: set i to 0, j to 1
                vec &= ~(1 << i)
                vec |= (1 << j)
            # else: already sorted, do nothing
        
        # Check if sorted: should be of the form 0...01...1
        # i.e., once we see a 1, all subsequent bits must be 1
        seen_one = False
        for k in range(N):
            if (vec >> k) & 1:
                seen_one = True
            elif seen_one:
                return False
    
    return True
# EVOLVE-BLOCK-END
