"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

N = 18

# EVOLVE-BLOCK-START
def _verify_network(net):
    """Verify that the network sorts all 2^N Boolean inputs."""
    if not net:
        return False
    # Check each comparator
    for i, j in net:
        if not (0 <= i < j < N):
            return False
    # Simulate all Boolean inputs
    # Use vectorized simulation: for each input (2^N), track the state
    # State is a 2^N x N matrix of bits
    # For each comparator (i,j): min goes to i, max goes to j
    # min(a,b) = a & b, max(a,b) = a | b for bits
    states = np.arange(2**N, dtype=np.uint32)
    # Extract bits: state[:, k] = bit k of states
    # We'll use a different approach: represent as array of shape (2^N, N)
    bits = np.zeros((2**N, N), dtype=np.uint8)
    for k in range(N):
        bits[:, k] = (states >> k) & 1
    
    for i, j in net:
        # min goes to i, max goes to j
        # For bits: min = AND, max = OR
        mi = bits[:, i] & bits[:, j]
        mj = bits[:, i] | bits[:, j]
        bits[:, i] = mi
        bits[:, j] = mj
    
    # Check if sorted: for each row, bits should be non-decreasing (0s then 1s)
    # i.e., for each row, once we see a 1, all subsequent must be 1
    # Equivalent: the row is sorted iff for all k < N-1, bits[k] <= bits[k+1]
    for k in range(N - 1):
        if np.any(bits[:, k] > bits[:, k+1]):
            return False
    return True


def _count_comparators(net):
    return len(net)


def _depth(net):
    """Compute the parallel depth of the network."""
    if not net:
        return 0
    # Track the last time each wire was used
    last_used = [0] * N
    depth = 0
    for i, j in net:
        # The comparator (i,j) can only happen after both wires i and j are free
        # We need to find the maximum depth among all comparators
        # Actually, depth is the length of the longest path in the DAG of comparators
        # A simpler approach: process sequentially, tracking when each wire becomes free
        # But that's not quite right for parallel depth.
        # Let's use a proper topological approach:
        # Actually, for sorting networks, depth = number of layers
        # Two comparators are in the same layer if they don't share wires
        # Let's compute it properly
        pass
    # Proper depth calculation
    # Assign each comparator to a layer
    # A comparator (i,j) at position p can be in layer L if no other comparator in layer L shares wire i or j
    # And the comparator must come after all comparators that use wire i or j and are in earlier layers
    # This is equivalent to: layer(c) = 1 + max(layer(c') for c' before c that shares a wire with c)
    layer = [0] * len(net)
    wire_layer = [-1] * N  # last layer a wire was used in
    for idx, (i, j) in enumerate(net):
        # This comparator must come after the last use of wire i and wire j
        l = max(wire_layer[i], wire_layer[j]) + 1
        layer[idx] = l
        wire_layer[i] = l
        wire_layer[j] = l
    return max(layer) if layer else 0


def construct():
    """Construct a sorting network on 18 wires using iterative improvement."""
    start_time = time.time()
    deadline = start_time + 75.0  # Safety margin: 90s limit, leave 15s buffer
    
    # Start with odd-even transposition network
    def odd_even_transposition():
        return [(i, i + 1) for phase in range(N)
                for i in range(phase % 2, N - 1, 2)]
    
    best_net = odd_even_transposition()
    best_count = len(best_net)
    
    # Try to improve by removing or reordering comparators
    # Strategy: for each comparator, try removing it and check if still valid
    # Also try swapping adjacent comparators that don't share wires
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Try removing each comparator
        for idx in range(len(best_net)):
            if time.time() > deadline:
                break
            candidate = best_net[:idx] + best_net[idx+1:]
            if _verify_network(candidate):
                best_net = candidate
                best_count = len(best_net)
                improved = True
                break  # Restart the loop with the new shorter network
        
        # Try swapping adjacent comparators that don't share wires
        for idx in range(len(best_net) - 1):
            if time.time() > deadline:
                break
            i1, j1 = best_net[idx]
            i2, j2 = best_net[idx+1]
            # Check if they don't share wires
            if len({i1, j1, i2, j2}) == 4:
                candidate = best_net[:idx] + [best_net[idx+1], best_net[idx]] + best_net[idx+2:]
                # This swap might allow further removals, but let's just check validity
                # Actually, swapping doesn't change the network's correctness if both are valid
                # But it might enable different removals. Let's just try removals again.
                pass
    
    # Try a more aggressive approach: hill climbing with random restarts
    # Actually, let's try a different strategy: start from a known good network
    # and try to remove comparators greedily
    
    # Another approach: use the fact that we can verify quickly
    # Try removing multiple comparators at once using a greedy approach
    
    # Let's try a smarter removal: process from the end, since later comparators
    # are more likely to be redundant
    for _ in range(5):  # Multiple passes
        if time.time() > deadline:
            break
        improved = False
        for idx in range(len(best_net) - 1, -1, -1):
            if time.time() > deadline:
                break
            candidate = best_net[:idx] + best_net[idx+1:]
            if _verify_network(candidate):
                best_net = candidate
                best_count = len(best_net)
                improved = True
        if not improved:
            break
    
    return best_net
# EVOLVE-BLOCK-END
