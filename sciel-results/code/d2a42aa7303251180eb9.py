"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

N = 17
# EVOLVE-BLOCK-START

def _run_network(network, mask):
    """Run a sorting network on a 17-bit mask (0/1 inputs) using big-int bitwise ops.
    Returns the final 17-bit mask."""
    for (i, j) in network:
        # extract bits i and j
        bi = (mask >> i) & 1
        bj = (mask >> j) & 1
        if bi > bj:
            # swap: set bit i to 0, bit j to 1
            mask ^= (1 << i) | (1 << j)
    return mask

def _verify(network):
    """Verify network sorts all 2^17 boolean inputs. Returns True if correct."""
    for mask in range(1 << N):
        result = _run_network(network, mask)
        # sorted means all 1s come before all 0s (min on wire 0)
        # i.e., result should be of form 0...0 1...1
        # count number of 1s
        ones = bin(result).count('1')
        # sorted form: top 'ones' bits are 1, rest 0
        expected = ((1 << ones) - 1) << (N - ones)
        if result != expected:
            return False
    return True

def _build_seed():
    """Build the odd-even transposition seed network."""
    return [(i, i + 1) for phase in range(N)
            for i in range(phase % 2, N - 1, 2)]

def construct():
    deadline = time.time() + 85.0  # safety margin before 90s kill
    network = _build_seed()
    
    # Greedy removal: try removing each comparator, verify, keep if still valid
    # Use random order for diversity
    import random
    rng = random.Random(42)
    
    # First pass: try removing comparators in reverse order (later ones less critical)
    i = len(network) - 1
    while i >= 0 and time.time() < deadline:
        # Try removing comparator i
        candidate = network[:i] + network[i+1:]
        if _verify(candidate):
            network = candidate
        else:
            i -= 1
    
    # Second pass: try removing in forward order
    i = 0
    while i < len(network) and time.time() < deadline:
        candidate = network[:i] + network[i+1:]
        if _verify(candidate):
            network = candidate
        else:
            i += 1
    
    # Third pass: randomized removal attempts
    attempts = 0
    while time.time() < deadline - 5.0:
        attempts += 1
        idx = rng.randrange(len(network))
        candidate = network[:idx] + network[idx+1:]
        if _verify(candidate):
            network = candidate
    
    # Final verification
    assert _verify(network)
    return network

# EVOLVE-BLOCK-END
