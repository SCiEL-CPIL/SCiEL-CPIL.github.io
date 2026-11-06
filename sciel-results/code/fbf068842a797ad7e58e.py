"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from itertools import product

N = 16

# EVOLVE-BLOCK-START
def _verify_network(comparators):
    """Verify sorting network using zero-one principle.
    Returns True if network sorts all 2^16 binary inputs."""
    # Track for each wire: min value and max value seen across all inputs
    # Initially each wire can be 0 or 1
    # We process all 2^16 inputs simultaneously using bitmasks
    
    # For each wire, track:
    # - zeros_mask: bitmask of inputs where wire is 0
    # - ones_mask: bitmask of inputs where wire is 1
    # Initially wire i: zeros = inputs where bit i is 0, ones = inputs where bit i is 1
    
    # But 2^16 = 65536, we can use Python integers as bitmasks
    
    # Initialize: for input x (0..65535), wire i has value (x >> i) & 1
    # zeros_mask[i] = bitmask of x where bit i of x is 0
    # ones_mask[i] = bitmask of x where bit i of x is 1
    
    zeros = [0] * N
    ones = [0] * N
    
    for i in range(N):
        # For bit position i in the input index:
        # inputs where bit i is 0: pattern of 0s and 1s in the bitmask
        # We need to construct this efficiently
        # For bit i: the pattern repeats every 2^(i+1), with 2^i zeros then 2^i ones
        # We'll build the bitmask
        mask = 0
        period = 1 << (i + 1)
        half = 1 << i
        for base in range(0, 1 << N, period):
            # In this period, first half has bit i = 0, second half has bit i = 1
            for x in range(base, base + half):
                mask |= (1 << x)
        zeros[i] = mask
        ones[i] = mask ^ ((1 << (1 << N)) - 1)
    
    # Apply comparators
    for (a, b) in comparators:
        # min goes to a, max goes to b
        # new_zeros[a] = zeros[a] | zeros[b]  (a gets 0 if either was 0)
        # new_ones[a] = ones[a] & ones[b]    (a gets 1 if both were 1)
        # new_zeros[b] = zeros[a] & zeros[b]  (b gets 0 if both were 0)
        # new_ones[b] = ones[a] | ones[b]     (b gets 1 if either was 1)
        
        za, zb = zeros[a], zeros[b]
        oa, ob = ones[a], ones[b]
        
        zeros[a] = za | zb
        ones[a] = oa & ob
        zeros[b] = za & zb
        ones[b] = oa | ob
    
    # After sorting, wire 0 should be 0 for all inputs, wire N-1 should be 1 for all inputs
    # Actually, for a sorted network: wire i should be 0 for inputs with <= i ones, and 1 otherwise
    # But simpler check: wire 0 must be all 0s, wire N-1 must be all 1s
    # And the pattern must be monotonic
    
    # The correct check: after sorting, for each input x with k ones,
    # wires 0..k-1 should be 1 and wires k..N-1 should be 0? No wait.
    # min to wire i, max to wire j. So sorted output has smallest values first.
    # For binary: sorted means 0s then 1s. Wire 0..(k-1) = 0, wire k..N-1 = 1 where k = number of ones.
    
    # Check: zeros[0] should be all 1s (wire 0 is always 0)
    # ones[N-1] should be all 1s (wire N-1 is always 1)
    # And for consistency: zeros[i] & ones[i] should be 0 for all i (no input has wire both 0 and 1)
    # And the boundary should be correct.
    
    all_mask = (1 << (1 << N)) - 1
    
    # Wire 0 must be 0 for all inputs
    if zeros[0] != all_mask:
        return False
    # Wire N-1 must be 1 for all inputs  
    if ones[N-1] != all_mask:
        return False
    
    # Check that for each wire, zeros[i] and ones[i] partition the space
    for i in range(N):
        if (zeros[i] & ones[i]) != 0:
            return False
        if (zeros[i] | ones[i]) != all_mask:
            return False
    
    # Check monotonicity: if wire i is 1, wire i+1 should also be 1
    # i.e., ones[i] should be subset of ones[i+1]
    for i in range(N - 1):
        if (ones[i] & ~ones[i + 1]) != 0:
            return False
    
    return True


def _verify_network_fast(comparators):
    """Faster verification using the fact that we only need to check the final state.
    Uses a different approach: track the set of possible values for each wire."""
    # Alternative: use the property that a sorting network is correct iff
    # for every subset S of {0..N-1} of size k, the network maps the indicator
    # of S to the indicator of {0..k-1}.
    # But that's still 2^N checks.
    
    # Let's stick with the bitmask approach but optimize the initial setup
    # Precompute the masks
    total_inputs = 1 << N
    all_mask = (1 << total_inputs) - 1
    
    # Precompute bit masks for each input bit position
    # mask_for_bit[i] = bitmask of input indices where bit i is set
    mask_for_bit = []
    for i in range(N):
        m = 0
        period = 1 << (i + 1)
        half = 1 << i
        for base in range(half, total_inputs, period):
            # In this period, second half has bit i = 1
            # Set bits from base to base+half-1
            # This is a contiguous range of half bits starting at base
            m |= ((1 << half) - 1) << base
        mask_for_bit.append(m)
    
    # Initialize: wire i has value = bit i of input index
    # zeros[i] = inputs where bit i is 0 = all_mask ^ mask_for_bit[i]
    # ones[i] = inputs where bit i is 1 = mask_for_bit[i]
    zeros = [all_mask ^ mask_for_bit[i] for i in range(N)]
    ones = mask_for_bit[:]
    
    # Apply comparators
    for (a, b) in comparators:
        za, zb = zeros[a], zeros[b]
        oa, ob = ones[a], ones[b]
        
        zeros[a] = za | zb
        ones[a] = oa & ob
        zeros[b] = za & zb
        ones[b] = oa | ob
    
    # Verify
    if zeros[0] != all_mask:
        return False
    if ones[N - 1] != all_mask:
        return False
    
    for i in range(N):
        if (zeros[i] & ones[i]) != 0:
            return False
    
    for i in range(N - 1):
        if (ones[i] & ~ones[i + 1]) != 0:
            return False
    
    return True


def _count_comparators(comparators):
    return len(comparators)


def _random_swap(comparators, rng):
    """Return a new list with a random swap of two comparators."""
    lst = comparators[:]
    if len(lst) < 2:
        return lst
    i, j = rng.sample(range(len(lst)), 2)
    lst[i], lst[j] = lst[j], lst[i]
    return lst


def _remove_comparator(comparators, idx):
    """Return a new list with comparator at idx removed."""
    lst = comparators[:]
    lst.pop(idx)
    return lst


def _insert_comparator(comparators, comparator, position=None):
    """Return a new list with comparator inserted."""
    lst = comparators[:]
    if position is None:
        position = len(lst)
    lst.insert(position, comparator)
    return lst


def _get_all_comparators():
    """Get all possible comparators."""
    return [(i, j) for i in range(N) for j in range(i + 1, N)]


def construct():
    """Construct a sorting network with minimum comparators."""
    start_time = time.time()
    deadline = start_time + 85  # 85 second deadline (5 sec safety margin)
    
    rng = random.Random(42)
    
    # Start with odd-even transposition network
    best_network = [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]
    best_count = len(best_network)
    
    # Verify initial network
    assert _verify_network_fast(best_network), "Initial network should be valid"
    
    all_comps = _get_all_comparators()
    all_comp_set = set(all_comps)
    
    # Local search: try removing comparators
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        current = best_network[:]
        current_set = set(current)
        
        # Try removing each comparator
        for idx in range(len(current) - 1, -1, -1):
            if time.time() > deadline - 5:
                break
            
            comp = current[idx]
            # Try removing this comparator
            candidate = current[:idx] + current[idx + 1:]
            
            # Quick check: is this comparator type already present elsewhere?
            # Not necessarily redundant, but let's verify
            if _verify_network_fast(candidate):
                best_network = candidate
                best_count = len(best_network)
                improved = True
                break  # Restart the loop with the new network
        
        if not improved and time.time() < deadline - 10:
            # Try swapping comparators
            for _ in range(100):
                if time.time() > deadline - 10:
                    break
                candidate = _random_swap(best_network, rng)
                if _verify_network_fast(candidate):
                    # Swapping doesn't change count, but might enable future removals
                    best_network = candidate
                    break
    
    # Try to add and remove to find better configurations
    if time.time() < deadline - 15:
        for _ in range(50):
            if time.time() > deadline - 15:
                break
            
            # Add a random comparator
            comp = rng.choice(all_comps)
            if comp not in best_network:
                candidate = best_network + [comp]
                if _verify_network_fast(candidate):
                    # Now try to remove some comparators
                    for idx in range(len(candidate) - 1, -1, -1):
                        if time.time() > deadline - 15:
                            break
                        c2 = candidate[:idx] + candidate[idx + 1:]
                        if _verify_network_fast(c2):
                            if len(c2) < len(best_network):
                                best_network = c2
                                break
    
    return best_network
# EVOLVE-BLOCK-END

