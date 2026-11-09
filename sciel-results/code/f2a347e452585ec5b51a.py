"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

N = 18
SEED_COMPARATORS = 153  # odd-even transposition for 18 wires

# EVOLVE-BLOCK-START

def _odd_even_transposition():
    """Generate the seed odd-even transposition network for N wires."""
    comps = []
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            comps.append((i, i + 1))
    return comps


def _apply_network(comparators, values):
    """Apply a list of comparators to a list/array of values.
    Returns a new list/array with the result.
    """
    v = list(values)
    for i, j in comparators:
        if v[i] > v[j]:
            v[i], v[j] = v[j], v[i]
    return v


def _is_sorted_network(comparators):
    """Check if the network sorts all 2^N boolean inputs.
    Uses the zero-one principle: a comparator network sorts all inputs
    if and only if it sorts all 0-1 inputs.
    """
    n_comps = len(comparators)
    # Pre-compute for speed: use numpy
    for mask in range(1 << N):
        # Build input vector
        vec = np.array([(mask >> k) & 1 for k in range(N)], dtype=np.int8)
        # Apply network
        for i, j in comparators:
            if vec[i] > vec[j]:
                vec[i], vec[j] = vec[j], vec[i]
        # Check sorted (non-decreasing)
        if np.any(vec[1:] < vec[:-1]):
            return False
    return True


def _is_sorted_network_fast(comparators, precomputed_masks=None):
    """Faster check using bit-level operations.
    For boolean inputs, we can represent the state as a bitmask.
    """
    # We'll use the standard 0-1 principle check but with optimized inner loop
    for mask in range(1 << N):
        state = mask
        for i, j in comparators:
            bit_i = (state >> i) & 1
            bit_j = (state >> j) & 1
            if bit_i > bit_j:
                # Swap: set bit_j to 1, bit_i to 0
                state |= (1 << j)
                state &= ~(1 << i)
        # Check if sorted: all 1s should be at the high end
        # A sorted boolean vector has form: 0...0 1...1
        # So if state != 0, the lowest set bit should be at position (N - popcount)
        # Actually, sorted means: for all k, if bit k is 1 then all bits below k are 0? No.
        # Sorted non-decreasing: 0,0,...,0,1,1,...,1
        # So the set bits should form a suffix.
        # Check: state should be of the form ((1 << N) - 1) ^ ((1 << (N - popcount)) - 1)
        # i.e., state should equal (1 << N) - 1 - ((1 << (N - popcount(state))) - 1)
        # Let's just check: for each bit position from 0 to N-2, if bit k is 1 and bit k+1 is 0, it's not sorted.
        # But that's O(N) per mask. Let's use a trick:
        # A number is a "suffix of ones" iff state & (state + 1) == 0 is NOT the right check.
        # Actually: state is a suffix of ones iff state == ((1 << N) - 1) - ((1 << k) - 1) for some k.
        # Equivalent: state has the form 000...0111...111.
        # We can check: let p = number of trailing zeros in (~state & ((1<<N)-1))... 
        # Simpler: check that state equals ((1<<N)-1) ^ (((1<<N)-1) >> popcount) ... no.
        # Let's just do the O(N) check but it's fast enough.
        # Actually, for 2^18 * 153 * 18 operations, that's ~710M ops, too slow.
        # Let me use a different approach: precompute which masks are sorted.
        # A mask is sorted (non-decreasing) iff it is of the form: lower k bits 0, upper N-k bits 1.
        # There are only N+1 such masks: 0, 1, 3, 7, ..., (1<<N)-1, i.e., (1<<k)-1 reversed? 
        # No: non-decreasing means bit0 <= bit1 <= ... <= bit_{N-1}.
        # So the mask must be 0...0111...1 where the transition happens once.
        # The sorted masks are: for k in 0..N: mask = ((1 << N) - 1) ^ ((1 << k) - 1) = ((1<<N)-1) - ((1<<k)-1)
        # i.e., mask = (1 << N) - 1 - ((1 << k) - 1) for k=0..N
        # Let's precompute the set of sorted masks.
        pass  # We'll handle this differently
    
    # Better approach: precompute sorted_masks
    sorted_masks = set()
    for k in range(N + 1):
        # k ones at the top, N-k zeros at the bottom
        # mask = ((1 << N) - 1) ^ ((1 << k) - 1)  -- this puts k ones at the bottom, wrong.
        # We want: bits 0 to N-k-1 are 0, bits N-k to N-1 are 1.
        # mask = ((1 << N) - 1) - ((1 << (N - k)) - 1)
        mask = ((1 << N) - 1) - ((1 << (N - k)) - 1)
        sorted_masks.add(mask)
    
    for mask in range(1 << N):
        state = mask
        for i, j in comparators:
            bit_i = (state >> i) & 1
            bit_j = (state >> j) & 1
            if bit_i > bit_j:
                state |= (1 << j)
                state &= ~(1 << i)
        if state not in sorted_masks:
            return False
    return True


def _depth(comparators):
    """Compute the parallel depth of a comparator network."""
    # depth[i] = the earliest time step at which wire i is free
    depth = [0] * N
    for i, j in comparators:
        t = max(depth[i], depth[j]) + 1
        depth[i] = t
        depth[j] = t
    return max(depth)


def _random_comparator():
    """Generate a random comparator (i, j) with 0 <= i < j < N."""
    i = random.randint(0, N - 1)
    j = random.randint(0, N - 1)
    if i == j:
        j = (i + 1) % N
    if i > j:
        i, j = j, i
    return (i, j)


def construct():
    """Construct a sorting network on 18 wires.
    Returns a list of comparator pairs (i, j) with 0 <= i < j < N.
    """
    start_time = time.time()
    deadline = start_time + 80  # 10 second safety margin
    
    # Start with the seed network
    best_network = _odd_even_transposition()
    best_len = len(best_network)
    
    if best_len <= 0:
        return best_network
    
    # Local search: try to remove comparators
    # Strategy: iterate through comparators, try removing each one,
    # check if the network still sorts. If yes, keep it removed.
    # Repeat until no more improvements.
    
    # For speed, we'll use a randomized order and try to remove
    # We need a fast checker. Let's use the bitmask approach.
    
    # Precompute sorted masks for fast checking
    sorted_masks = set()
    for k in range(N + 1):
        mask = ((1 << N) - 1) - ((1 << (N - k)) - 1)
        sorted_masks.add(mask)
    
    def check_network(comps):
        """Check if a network sorts all boolean inputs. Returns True if valid."""
        for mask in range(1 << N):
            state = mask
            for i, j in comps:
                bit_i = (state >> i) & 1
                bit_j = (state >> j) & 1
                if bit_i > bit_j:
                    state |= (1 << j)
                    state &= ~(1 << i)
            if state not in sorted_masks:
                return False
        return True
    
    # Try to remove comparators one by one
    # We'll do multiple passes
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Shuffle the order of comparators to try
        indices = list(range(len(best_network)))
        random.shuffle(indices)
        for idx in indices:
            if time.time() >= deadline:
                break
            # Try removing comparator at idx
            candidate = best_network[:idx] + best_network[idx+1:]
            if check_network(candidate):
                best_network = candidate
                improved = True
                # Restart the pass since the list changed
                break
    
    # Also try swapping adjacent comparators to see if we can enable further removals
    # This is a simple 2-opt style local search
    if time.time() < deadline:
        for _ in range(100):  # Limited iterations
            if time.time() >= deadline:
                break
            if len(best_network) < 2:
                break
            i = random.randint(0, len(best_network) - 2)
            # Swap comparators i and i+1
            candidate = best_network[:]
            candidate[i], candidate[i+1] = candidate[i+1], candidate[i]
            if check_network(candidate):
                # Check if this swap enables a removal
                # Try removing a random comparator from the swapped network
                if len(candidate) > 1:
                    j = random.randint(0, len(candidate) - 1)
                    candidate2 = candidate[:j] + candidate[j+1:]
                    if check_network(candidate2):
                        best_network = candidate2
                        continue
                # Accept the swap even if no removal, as it might help later
                best_network = candidate
    
    return best_network

# EVOLVE-BLOCK-END
