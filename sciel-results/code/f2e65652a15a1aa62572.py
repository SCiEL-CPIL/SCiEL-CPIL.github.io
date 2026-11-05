"""Size-efficient sorting network on 16 wires via heuristic search + zero-one verification."""
import time
import random
import itertools

N = 16

# Known good reference: odd-even transposition (120 comparators)
def _baseline_network():
    return [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]

def _verify_network(net):
    """Verify network sorts all 2^16 zero-one inputs."""
    # Use the zero-one principle: check all 65536 bitmasks
    # Represent state as a bitmask; wire i holds bit i.
    # After a comparator (i,j): if bit i is 0 and bit j is 1, swap them.
    # Sorted means bits are in non-decreasing order: 0...0 1...1
    # i.e., the bitmask equals ((1<<k)-1) << (N-k) for some k, or simply
    # the set of 1-bits is a suffix.
    # Actually sorted non-decreasing: smallest at wire 0. So 0s then 1s.
    # Bitmask: bit i = value on wire i. Sorted => bits 0..k-1 are 0, bits k..N-1 are 1.
    # That means the bitmask has the form: 000...0111...1 (low bits 0, high bits 1).
    # Equivalently, the bitmask equals ((1<<N)-1) ^ ((1<<k)-1) for some k in [0,N].
    # Or: the bitmask is of the form ((1 << (N - k)) - 1) << k.
    
    def is_sorted(mask):
        # mask should be of form 0...01...1
        # Find the position of the lowest 1-bit; all bits below must be 0, all above must be 1.
        if mask == 0 or mask == (1 << N) - 1:
            return True
        # lowest set bit position
        low = (mask & -mask).bit_length() - 1
        # All bits from low to N-1 must be 1, bits 0 to low-1 must be 0
        expected = ((1 << N) - 1) ^ ((1 << low) - 1)
        return mask == expected

    for mask in range(1 << N):
        m = mask
        for (i, j) in net:
            bi = (m >> i) & 1
            bj = (m >> j) & 1
            if bi == 0 and bj == 1:
                m |= (1 << i)
                m &= ~(1 << j)
        if not is_sorted(m):
            return False
    return True

def _verify_network_fast(net):
    """Faster verification using bitmask operations."""
    # Precompute for each comparator (i,j) the mask manipulation
    # For each input mask, apply all comparators
    # We can vectorize somewhat but 65536 * len(net) ops might be slow in pure Python.
    # Let's try a different approach: track the set of possible masks? No, too complex.
    # Alternative: use the fact that sorting network correctness can be checked by
    # simulating all 2^N inputs. For 16 wires that's 65536 inputs.
    # Each input goes through len(net) comparators.
    # Total ops: 65536 * len(net). For 60 comparators, ~4M ops. Should be fine in <1s.
    
    n = len(net)
    for mask in range(1 << N):
        m = mask
        for (i, j) in net:
            # Extract bits
            bi = (m >> i) & 1
            bj = (m >> j) & 1
            if bi == 0 and bj == 1:
                m |= (1 << i)
                m &= ~(1 << j)
        # Check sorted: bits must be 0...01...1
        # The sorted form for a mask with k ones is: lower (N-k) bits are 0, upper k bits are 1
        # Count ones:
        k = bin(m).count('1')
        expected = ((1 << k) - 1) << (N - k)
        if m != expected:
            return False
    return True

def _construct_heuristic(deadline):
    """
    Heuristic construction of a sorting network.
    Strategy: Build a network using a divide-and-conquer / merge-sort style approach,
    then try to optimize by removing redundant comparators.
    
    Actually, let's use a known construction first, then optimize.
    
    Known: For N=16, the best known is 60 comparators.
    Let's try to construct one using a standard algorithm.
    
    Approach: Use a "bitonic" or "merge" based construction, then prune.
    """
    # Start with a known correct network (e.g., odd-even transposition)
    net = _baseline_network()
    
    # Try to remove redundant comparators
    # A comparator is redundant if removing it still yields a valid sorting network.
    # We can try removing each comparator one at a time.
    
    removed = True
    while removed:
        removed = False
        for idx in range(len(net) - 1, -1, -1):
            if time.time() > deadline:
                return net
            # Try removing net[idx]
            candidate = net[:idx] + net[idx+1:]
            if _verify_network_fast(candidate):
                net = candidate
                removed = True
                break  # Restart scanning from the end
    
    return net

def _construct_known_60():
    """
    Try to return a known 60-comparator sorting network for 16 wires.
    I'll attempt to construct one using a systematic approach.
    
    Actually, let me use a well-known construction. The optimal or near-optimal
    networks for 16 wires are known in the literature. Let me try to build one
    using a recursive merge-sort based approach and then optimize.
    """
    # Use a bitonic sorting network as a starting point, then optimize
    # Bitonic network for 16 wires: depth log2(16)*log2(16)/2 = 4*4/2... 
    # Actually bitonic sort network for N=16 has 60 comparators? Let me check.
    # Bitonic sort: for N=2^k, the number of comparators is N/2 * k * (k+1)/2
    # For N=16, k=4: 16/2 * 4 * 5/2 = 8 * 10 = 80 comparators.
    # That's more than 60.
    
    # Let's just use the heuristic approach starting from odd-even transposition.
    pass

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Leave 10s safety margin
    
    # First, try to quickly construct a good network
    # Start with the baseline (120 comparators)
    net = _baseline_network()
    
    # Now iteratively remove redundant comparators
    # To speed up verification, we can use a smarter approach
    # But first, let's just do the removal heuristic
    
    # Optimization: process comparators in order of "least useful" first
    # A comparator (i,j) is less useful if |i-j| is small and it's in a "late" phase
    # Actually, let's just try removing from the end first (later comparators are often redundant)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try removing each comparator, starting from the last
        for idx in range(len(net) - 1, -1, -1):
            if time.time() > deadline:
                break
            candidate = net[:idx] + net[idx+1:]
            if _verify_network_fast(candidate):
                net = candidate
                improved = True
                break
    
    return net
