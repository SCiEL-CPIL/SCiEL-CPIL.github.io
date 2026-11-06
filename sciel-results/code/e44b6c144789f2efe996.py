"""Size-efficient sorting networks on 17 wires.

Uses bit-parallel zero-one principle verification and local search to reduce
comparator count from the odd-even transposition seed.
"""

N = 17

# EVOLVE-BLOCK-START

def _verify_network_bitparallel(network, N):
    """
    Verify network sorts all 2^N Boolean inputs using bit-parallel simulation.
    Each wire's state is represented as an integer where bit k is the value
    of wire at input k (k from 0 to 2^N - 1).
    """
    num_inputs = 1 << N
    # Initialize: for input mask, wire i has bit (mask >> i) & 1
    # So wire i's bitset has bit k set iff (k >> i) & 1 == 1
    # That is, bits where the i-th bit of k is 1.
    wires = []
    for i in range(N):
        # Pattern: for bit position k in 0..2^N-1, bit is 1 iff k has bit i set
        # This is a repeating pattern: 2^i zeros, 2^i ones, repeating
        pattern = 0
        for k in range(num_inputs):
            if (k >> i) & 1:
                pattern |= (1 << k)
        wires.append(pattern)
    
    for (i, j) in network:
        # Compare wires i and j: min goes to i, max goes to j
        # min = wires[i] & wires[j], max = wires[i] | wires[j]
        new_i = wires[i] & wires[j]
        new_j = wires[i] | wires[j]
        wires[i] = new_i
        wires[j] = new_j
    
    # Check sorted: for each k, wires[0][k] <= wires[1][k] <= ... <= wires[N-1][k]
    # i.e., if wires[k][m] is 1 then wires[k][m+1] must be 1
    # Equivalently: wires[m] & ~wires[m+1] should be 0 for all m
    for m in range(N - 1):
        if (wires[m] & ~wires[m + 1]) != 0:
            return False
    return True

def _odd_even_transposition(N):
    """Generate odd-even transposition sorting network."""
    comps = []
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            comps.append((i, i + 1))
    return comps

def _try_remove_comparator(network, idx, N):
    """Try removing comparator at idx; return new network if still valid."""
    new_net = network[:idx] + network[idx+1:]
    if _verify_network_bitparallel(new_net, N):
        return new_net
    return None

def _local_search_reduce(network, N, max_iters=200):
    """Greedy local search: repeatedly try removing random comparators."""
    import random
    net = list(network)
    for _ in range(max_iters):
        if not net:
            break
        idx = random.randint(0, len(net) - 1)
        new_net = _try_remove_comparator(net, idx, N)
        if new_net is not None:
            net = new_net
    return net

def construct():
    """Return a sorting network for N=17 wires as a list of (i, j) pairs."""
    import random
    # Start with odd-even transposition (136 comparators)
    network = _odd_even_transposition(N)
    
    # Use local search to reduce comparator count
    random.seed(42)  # Deterministic for reproducibility
    reduced = _local_search_reduce(network, N, max_iters=200)
    
    # Final verification
    if _verify_network_bitparallel(reduced, N):
        return reduced
    
    # Fallback: return the original valid network
    return network

# EVOLVE-BLOCK-END
