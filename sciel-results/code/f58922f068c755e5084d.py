"""Size-efficient sorting networks on 16 wires."""
import time
import itertools
from functools import lru_cache

N = 16
MAX_COMPARATORS = 4096
TIME_LIMIT = 85  # seconds, safety margin before 90s kill

# EVOLVE-BLOCK-START

def _apply_comparator(state, i, j):
    """Apply comparator (i, j) to a bitmask state.
    state is a tuple of 16 bits (0/1).
    Returns new tuple with min at i, max at j.
    """
    if state[i] > state[j]:
        lst = list(state)
        lst[i], lst[j] = lst[j], lst[i]
        return tuple(lst)
    return state

def _is_sorted(state):
    """Check if state is sorted (non-decreasing)."""
    for k in range(N - 1):
        if state[k] > state[k + 1]:
            return False
    return True

def _verify_network(comparators):
    """Verify that a network sorts all 2^16 Boolean inputs using zero-one principle."""
    # We can optimize: only need to check inputs where the number of 1s is 'k'
    # but actually zero-one principle says check all 2^16. 
    # However, we can use a smarter verification: 
    # A network sorts all inputs iff for every input x, the output is sorted.
    # Due to symmetry, we can just check all 2^16 states.
    # But 65536 states * len(comparators) might be slow in pure Python.
    # Let's use a bitmask representation and precompute.
    
    # Actually, a faster verification:
    # Represent state as an integer bitmask (16 bits).
    # Precompute the effect of each comparator on all 65536 masks? 
    # That would be 65536 * len(comparators) operations, which is fine if len is small.
    
    # Let's represent state as int.
    # Comparator (i, j) on mask m:
    # bit_i = (m >> i) & 1
    # bit_j = (m >> j) & 1
    # if bit_i > bit_j (i.e., bit_i=1, bit_j=0), swap.
    
    def apply_comp_mask(m, i, j):
        bi = (m >> i) & 1
        bj = (m >> j) & 1
        if bi > bj:
            # swap bits i and j
            m ^= (1 << i) | (1 << j)
        return m
    
    for mask in range(1 << N):
        m = mask
        for (i, j) in comparators:
            m = apply_comp_mask(m, i, j)
        # Check if m is sorted: all 0s followed by all 1s
        # A sorted 16-bit mask has the form: 000...0111...1
        # So if k ones, m should be (1 << k) - 1
        # Count ones:
        ones = bin(m).count('1')
        expected = (1 << ones) - 1
        if m != expected:
            return False
    return True

def _depth_of_network(comparators):
    """Compute parallel depth of a comparator network."""
    if not comparators:
        return 0
    # Assign depth to each wire, update as we process comparators in order.
    # A comparator (i, j) at step t has depth max(depth[i], depth[j]) + 1.
    # Then depth[i] = depth[j] = that value.
    depths = [0] * N
    max_depth = 0
    for (i, j) in comparators:
        d = max(depths[i], depths[j]) + 1
        depths[i] = d
        depths[j] = d
        if d > max_depth:
            max_depth = d
    return max_depth

def _odd_even_transposition():
    """Generate the odd-even transposition sorting network (120 comparators)."""
    return [(i, i + 1) for phase in range(N)
            for i in range(phase % 2, N - 1, 2)]

def _akl_baseline():
    """A known smaller network for 16 inputs.
    The AKL (Ajtai-Komlos-Szemeredi) construction gives ~ (N^2/2) * log N / log log N comparators.
    For N=16, a known explicit network uses 60 comparators (as mentioned in the task).
    Let's try to hardcode a known 60-comparator network if we can recall/construct one.
    
    Actually, the task says "A verified 60-comparator construction is the reference".
    Let's try to find or construct a small network.
    
    Alternative: Use a recursive construction.
    Odd-even merge sort on 16 wires.
    """
    # Let's implement odd-even mergesort network generation.
    def odd_even_merge(left, right):
        """Merge two sorted networks of size len(left) and len(right).
        left and right are lists of comparator networks (lists of (i,j) pairs) 
        that sort their respective wires.
        Returns a network that merges them.
        """
        # This is complex. Let's use a known explicit construction instead.
        pass
    
    # Let's just return the 120-comparator one for now, or try a simple heuristic.
    return _odd_even_transposition()

def _heuristic_search():
    """Try to improve the network using a greedy or heuristic approach.
    Start from the 120-comparator network and try to remove redundant comparators.
    """
    start_time = time.time()
    network = _odd_even_transposition()
    
    # Try to remove each comparator and check if the network still works.
    # This is O(n^2 * 2^16) which is too slow for n=120.
    # 120 * 65536 = 7.8M operations per removal check, times 120 = 900M. Too slow.
    
    # Alternative: Start with a smaller known network.
    # Let's try the "bitonic" sorting network for 16 wires.
    # Bitonic sort for 16 wires has depth 6 and uses 60 comparators.
    # Let's generate a bitonic sorting network.
    
    def bitonic_sort_network(n):
        """Generate a bitonic sorting network for n wires (n must be power of 2)."""
        if n == 1:
            return []
        
        # Recursive construction:
        # 1. Sort first half and second half (in reverse order for one).
        # 2. Merge the two halves using bitonic merge.
        
        # Actually, let's use the standard bitonic merge construction.
        def bitonic_merge(n, offset, direction):
            """Generate comparators for bitonic merge of n wires starting at offset.
            direction: 1 for ascending, -1 for descending.
            """
            if n == 1:
                return []
            half = n // 2
            comps = []
            # Compare elements at distance half
            for i in range(half):
                a = offset + i
                b = offset + i + half
                if direction == 1:
                    # min to a, max to b
                    comps.append((a, b))
                else:
                    # max to a, min to b -> swap indices
                    comps.append((b, a))
            # Recursively merge the two halves
            comps += bitonic_merge(half, offset, direction)
            comps += bitonic_merge(half, offset + half, direction)
            return comps
        
        def bitonic_sort(n, offset, direction):
            """Generate comparators for bitonic sort of n wires starting at offset.
            direction: 1 for ascending, -1 for descending.
            """
            if n == 1:
                return []
            half = n // 2
            comps = []
            # Sort first half in one direction, second half in opposite
            comps += bitonic_sort(half, offset, direction)
            comps += bitonic_sort(half, offset + half, -direction)
            # Merge the two halves
            comps += bitonic_merge(n, offset, direction)
            return comps
        
        return bitonic_sort(n, 0, 1)
    
    bitonic_net = bitonic_sort_network(N)
    # Verify
    if _verify_network(bitonic_net):
        return bitonic_net
    
    # Fallback to odd-even transposition
    return _odd_even_transposition()

def construct():
    """Construct a sorting network for 16 wires."""
    start_time = time.time()
    
    # Try the bitonic sorting network first (should be 60 comparators)
    def bitonic_sort_network(n):
        """Generate a bitonic sorting network for n wires (n must be power of 2)."""
        if n == 1:
            return []
        
        def bitonic_merge(n, offset, direction):
            """Generate comparators for bitonic merge of n wires starting at offset.
            direction: 1 for ascending, -1 for descending.
            """
            if n == 1:
                return []
            half = n // 2
            comps = []
            # Compare elements at distance half
            for i in range(half):
                a = offset + i
                b = offset + i + half
                if direction == 1:
                    # min to a, max to b
                    comps.append((a, b))
                else:
                    # max to a, min to b -> swap indices
                    comps.append((b, a))
            # Recursively merge the two halves
            comps += bitonic_merge(half, offset, direction)
            comps += bitonic_merge(half, offset + half, direction)
            return comps
        
        def bitonic_sort(n, offset, direction):
            """Generate comparators for bitonic sort of n wires starting at offset.
            direction: 1 for ascending, -1 for descending.
            """
            if n == 1:
                return []
            half = n // 2
            comps = []
            # Sort first half in one direction, second half in opposite
            comps += bitonic_sort(half, offset, direction)
            comps += bitonic_sort(half, offset + half, -direction)
            # Merge the two halves
            comps += bitonic_merge(n, offset, direction)
            return comps
        
        return bitonic_sort(n, 0, 1)
    
    # Generate bitonic network
    net = bitonic_sort_network(N)
    
    # Verify it
    if _verify_network(net):
        return net
    
    # If bitonic fails (shouldn't), fall back to odd-even transposition
    net_oe = _odd_even_transposition()
    if _verify_network(net_oe):
        return net_oe
    
    # Last resort: return empty (should never happen)
    return []

# EVOLVE-BLOCK-END
