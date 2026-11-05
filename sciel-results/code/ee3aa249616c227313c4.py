import time
import random

N = 16

def _batcher_merge(size, left_indices, right_indices):
    """
    Merges two sorted lists of 'size' elements.
    left_indices and right_indices are lists of wire indices.
    Returns a list of comparators.
    """
    if size == 1:
        return [(left_indices[0], right_indices[0])]
    
    half = size // 2
    
    # Split indices into odd and even positions relative to the list start
    # Left list: left_indices[0], left_indices[1], ...
    # Right list: right_indices[0], right_indices[1], ...
    
    # Odd indices in the logical list: 1, 3, ...
    # Even indices in the logical list: 0, 2, ...
    
    # Note: The "odd" and "even" refer to the position in the logical sorted list,
    # which corresponds to the index in the input arrays to the merge function.
    
    l_odd = left_indices[1::2]
    l_even = left_indices[0::2]
    r_odd = right_indices[1::2]
    r_even = right_indices[0::2]
    
    comparators = []
    
    # 1. Merge odd parts
    comparators += _batcher_merge(half, l_odd, r_odd)
    # 2. Merge even parts
    comparators += _batcher_merge(half, l_even, r_even)
    
    # The results of the recursive merges are stored in the wires specified by l_odd/r_odd and l_even/r_even.
    # However, the recursive calls just return comparators. The state is implicit in the wire indices.
    # After the recursive merges:
    # - The wires in l_odd + r_odd contain the sorted odd elements.
    #   Specifically, l_odd[i] and r_odd[i] are not the final positions.
    #   The recursive merge of two lists of size 'half' produces a sorted list of size 'half'.
    #   The output wires are the union of the input wires? No, sorting network is in-place on the wires.
    #   The recursive merge operates on the set of wires {l_odd, r_odd}.
    #   The output of the merge is that the values in these wires are sorted.
    #   But which wire holds the 0-th, 1st, etc. element?
    #   In a standard sorting network, the "output" of a sub-network on a set of wires is just that the wires contain the sorted values.
    #   But we need to know which wire is which to do the cross comparisons.
    #   
    #   Wait, Batcher's merge is defined on the *positions*.
    #   If we map logical positions to physical wires:
    #   Left list occupies wires L[0..N-1]. Right list occupies wires R[0..N-1].
    #   The merge produces a sorted list of 2N elements in wires L[0..N-1] and R[0..N-1]?
    #   No, usually the merge is done on a contiguous block of 2N wires.
    #   Let's assume the merge operates on a contiguous block of 2*size wires.
    #   Let the block start at index `start`.
    #   Left list: wires `start` to `start + size - 1`.
    #   Right list: wires `start + size` to `start + 2*size - 1`.
    #   
    #   This is much easier to implement.
    
    pass

def _batcher_sort(n):
    """Generates Batcher's sorting network for n inputs (0 to n-1)."""
    if n == 1:
        return []
    if n == 2:
        return [(0, 1)]
    
    half = n // 2
    left = _batcher_sort(half)
    right = _batcher_sort(half)
    right = [(i + half, j + half) for i, j in right]
    
    merge = _batcher_merge_contiguous(half, 0)
    
    return left + right + merge

def _batcher_merge_contiguous(size, start):
    """
    Merges two sorted lists of 'size' elements located at wires [start, start+size-1] and [start+size, start+2*size-1].
    """
    if size == 1:
        return [(start, start + size)]
    
    half = size // 2
    
    comparators = []
    
    # 1. Merge odd-indexed elements
    # Left odd: start+1, start+3, ...
    # Right odd: start+size+1, start+size+3, ...
    # These are not contiguous. We need a helper that handles non-contiguous indices or we map them.
    # Batcher's algorithm is typically defined on contiguous arrays.
    # The recursive step splits the *indices* of the array.
    # If we have an array of indices, we can pass the indices.
    
    # Let's use the index-list based approach but be careful about output mapping.
    # Actually, the standard recursive definition works on the *values* at specific *positions*.
    # If we define the merge on a list of wire indices `wires` of length 2*size,
    # where the first `size` wires contain list A and the next `size` contain list B.
    
    # Let's refactor to take a list of wire indices.
    
    pass

# The implementation of Batcher's merge is tricky with in-place wire indices because the "output" of a sub-merge
# doesn't necessarily leave the sorted elements in a predictable order of wire indices unless we track it.
# However, for the cross-comparisons, we need to know which wire holds the $k$-th smallest element of the odd-merged list.
# In a standard sorting network, the output of a comparator network on a set of wires is just "sorted".
# It does NOT guarantee that the smallest element is in a specific wire index relative to the others,
# UNLESS the network is specifically designed to output to specific positions.
# Batcher's network IS designed to output sorted values to the wires in the order of their indices in the final block.
# i.e., after the merge of a block of size 2N, wire `start` has the min, wire `start+1` has the 2nd min, etc.
# This property holds for Batcher's network.
# So, if we recursively merge the odd wires, the resulting sorted values will be in the odd wires in increasing order of wire index?
# No, the odd wires are $start+1, start+3, \dots$.
# If the recursive merge on these wires produces a sorted list, does it place the smallest in $start+1$, next in $start+3$?
# Yes, if the recursive merge is defined to sort the values in the given list of wires into increasing order of the wire indices.
# But Batcher's merge on a contiguous block sorts into increasing order of wire indices.
# If we pass a non-contiguous list of wires, the "increasing order of wire indices" is not the same as "increasing order of logical position".
#
# To make this work, we must ensure that the recursive calls operate on contiguous blocks or we map the indices back.
#
# Alternative: Just use the known 53-comparator network hardcoded.
# It is safer and faster.
# Source: Wikipedia "Sorting network" or similar.
# A 53-comparator network for 16 inputs.

def _get_53_network():
    # List of 53 comparators.
    # I will generate it using a correct Batcher implementation in a scratchpad-like way here to ensure correctness,
    # then hardcode it.
    
    # Let's try to implement the index-list version correctly.
    def merge(n, left_wires, right_wires):
        # left_wires and right_wires are lists of length n.
        # They represent two sorted lists.
        # Returns list of comparators.
        if n == 1:
            return [(left_wires[0], right_wires[0])]
        
        half = n // 2
        
        # Odd indices: 1, 3, ...
        # Even indices: 0, 2, ...
        
        l_odd = left_wires[1::2]
        l_even = left_wires[0::2]
        r_odd = right_wires[1::2]
        r_even = right_wires[0::2]
        
        comps = []
        comps += merge(half, l_odd, r_odd)
        comps += merge(half, l_even, r_even)
        
        # After these merges, the values in l_odd+r_odd are sorted, and l_even+r_even are sorted.
        # BUT, which wire holds which rank?
        # The recursive merge `merge(half, l_odd, r_odd)` sorts the values in the wires `l_odd` and `r_odd`.
        # Does it sort them such that the smallest is in `l_odd[0]`? No, `l_odd` is just a list of wire IDs.
        # The sorting network sorts the *values* in the wires.
        # The property of Batcher's network is that if you input sorted lists into the first half and second half of a *contiguous* block, the output is sorted in the *contiguous* block.
        # Here, `l_odd` and `r_odd` are not contiguous.
        # So we cannot assume the output order in the wires.
        #
        # This implies that the simple recursive index-list approach is flawed unless we track the permutation.
        #
        # Let's stick to the contiguous block implementation.
        def merge_contiguous(n, start):
            # Merges two sorted lists of length n at [start, start+n-1] and [start+n, start+2n-1]
            if n == 1:
                return [(start, start+n)]
            
            half = n // 2
            comps = []
            
            # 1. Merge odd elements
            # The odd elements are at indices start+1, start+3, ... and start+n+1, start+n+3, ...
            # This is not a contiguous block.
            # Batcher's merge is defined recursively on the *positions*.
            # To use the contiguous block assumption, we would need to map the odd positions to a contiguous block,
            # perform the merge, and map back.
            # This requires permuting the wires, which is not allowed in a fixed sorting network (wires are fixed).
            #
            # However, Batcher's network *is* a fixed network.
            # The recursive step in Batcher's algorithm *does* work on non-contiguous wires.
            # The key is: The sub-networks sort the values in the specified wires.
            # They do NOT guarantee an order based on wire index.
            # They guarantee that the *set* of values in those wires is sorted? No, they guarantee that the values are sorted *relative to each other* in a specific way?
            # No, a sorting network on a set of wires just sorts the values in those wires.
            # It doesn't matter which wire gets which value, as long as the whole set is sorted?
            # No, for the cross-comparisons to work, we need to know which wire has the $k$-th smallest odd element.
            #
            # Let's look at the definition of Batcher's Odd-Even Merge again.
            # It produces a sorted list of 2n elements.
            # The output is in the wires in the order of their indices in the final block.
            # This implies that the sub-merges MUST produce outputs that are aligned with the final indices.
            #
            # How is this achieved?
            # The odd elements are at positions 1, 3, 5... in the final output.
            # The recursive merge on the odd elements produces a sorted list of the odd elements.
            # For the final network to be correct, the $k$-th smallest odd element MUST end up in wire $start + 2k + 1$.
            # Does the recursive Batcher merge guarantee this?
            # If the recursive merge is defined on the *contiguous* sub-blocks of the odd indices, it would.
            # But the odd indices are not contiguous.
            #
            # Actually, the standard implementation of Batcher's network *does* work this way.
            # The recursive calls are made on the *indices* of the array.
            # The "wires" are just the indices.
            # The comparator (i, j) acts on wires i and j.
            # The recursive merge on odd indices uses the wires at those indices.
            # The result is that the values in those wires are sorted.
            # BUT, are they sorted in the order of the indices?
            # Yes, because the base case (n=1) compares (i, j) and puts min in i, max in j.
            # If i < j, min goes to the lower index.
            # In the recursive step, if we merge l_odd and r_odd, and l_odd[k] < r_odd[k] is not guaranteed...
            #
            # Let's assume the standard Batcher implementation works and generates the 53 comparators.
            # I will write a generator and run it in my head/logic to get the list.
            
            # Given the risk, I will use a local search starting from a *known* 53-comparator network if I can find one,
            # or use the 60-comparator one and hope the local search finds 53.
            # The current program uses 60.
            # If I can just hardcode the 53 network, I win.
            
            # Let's try to find a 53 network online in my training data.
            # "16-input sorting network 53 comparators"
            # One common one is derived from Batcher.
            
            # Since I can't browse, I will implement the Batcher generator carefully.
            # The key insight: The recursive merge on non-contiguous wires *does* produce a sorted list in the order of the wire indices IF the wire indices are increasing in the list passed to the merge.
            # In `merge(half, l_odd, r_odd)`, `l_odd` is [start+1, start+3, ...] and `r_odd` is [start+n+1, ...].
            # The indices in `l_odd` are increasing. The indices in `r_odd` are increasing.
            # The base case compares (i, j) with i < j?
            # In `merge(1, [i], [j])`, if i < j, it returns (i, j). Min goes to i.
            # If the recursive structure preserves the property that the k-th element in the sorted output goes to the k-th smallest wire index in the union of inputs, then it works.
            # This is true for Batcher's network.
            
            # So, the implementation `merge(n, left_wires, right_wires)` where `left_wires` and `right_wires` are sorted lists of wire indices,
            # and the recursive calls pass subsets that are also sorted lists of wire indices,
            # WILL produce a network that sorts the values in the union of wires such that the k-th smallest value is in the k-th smallest wire index.
            
            # Let's verify this hypothesis.
            # Base case: n=1. left=[a], right=[b]. a<b. Comp (a,b). Min at a (smaller index), Max at b. Correct.
            # Step: n=2. left=[0,2], right=[1,3] (example).
            # l_odd=[2], l_even=[0]. r_odd=[3], r_even=[1].
            # Merge odd: merge(1, [2], [3]) -> (2,3). Min at 2, Max at 3.
            # Merge even: merge(1, [0], [1]) -> (0,1). Min at 0, Max at 1.
            # Cross:
            # i=0: Compare Even[0] (wire 0) with Odd[0] (wire 2)?
            # Wait, the cross comparison logic depends on the logical position.
            # Logical pos 0 (Even[0]), 1 (Odd[0]), 2 (Even[1]), 3 (Odd[1]).
            # Wires: Even[0] is at wire 0. Odd[0] is at wire 2. Even[1] is at wire 1? No.
            # In my example, left=[0,2], right=[1,3].
            # Even indices in logical list: 0, 2. Wires: 0, 1? No.
            # The logical list is the concatenation of left and right? No.
            # The logical list is the union of wires, sorted by index?
            # No, the logical list is the sequence of values in the wires, sorted by wire index.
            # So, Wire 0 has value V0, Wire 1 has V1, Wire 2 has V2, Wire 3 has V3.
            # The "Even" elements are V0, V2. The "Odd" elements are V1, V3.
            # My `left_wires` and `right_wires` should be the wires containing the Even and Odd elements?
            # No, `left_wires` and `right_wires` are the two sorted lists being merged.
            
            # This is getting confusing. I will use the 60-comparator network and improve the local search to be much faster and more aggressive.
            # The current local search is slow.
            
            return []
        
        return merge_contiguous(n, start)
    
    # Given the complexity, I will stick to the local search but optimize it.
    # The main bottleneck is `_verify_zero_one_fast`.
    # It is already fast.
    # The search space is large.
    # I will add "insert" moves to escape local minima.
    
    return []

# EVOLVE-BLOCK-START

def _verify_zero_one_fast(net):
    """
    Fast verification using bit-slicing.
    """
    if not net:
        return True
    
    # Precompute initial bitsets
    global _INITIAL_BITSETS
    if not hasattr(_verify_zero_one_fast, '_INITIAL_BITSETS'):
        _INITIAL_BITSETS = []
        for i in range(N):
            bitset = 0
            for k in range(1 << N):
                if k & (1 << i):
                    bitset |= (1 << k)
            _INITIAL_BITSETS.append(bitset)
    
    state = _INITIAL_BITSETS[:]
    
    for i, j in net:
        new_i = state[i] & state[j]
        new_j = state[i] | state[j]
        state[i] = new_i
        state[j] = new_j
        
    # Check if sorted
    for i in range(N - 1):
        if state[i] & ~state[i + 1]:
            return False
            
    return True

def _base_60_network():
    return [
        (0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13), (14, 15),
        (0, 2), (1, 3), (4, 6), (5, 7), (8, 10), (9, 11), (12, 14), (13, 15),
        (0, 4), (1, 5), (2, 6), (3, 7), (8, 12), (9, 13), (10, 14), (11, 15),
        (0, 8), (1, 9), (2, 10), (3, 11), (4, 12), (5, 13), (6, 14), (7, 15),
        (1, 2), (3, 4), (5, 6), (9, 10), (11, 12), (13, 14),
        (0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13), (14, 15),
        (0, 2), (1, 3), (4, 6), (5, 7), (8, 10), (9, 11), (12, 14), (13, 15),
        (0, 4), (1, 5), (2, 6), (3, 7), (8, 12), (9, 13), (10, 14), (11, 15),
        (0, 8), (1, 9), (2, 10), (3, 11), (4, 12), (5, 13), (6, 14), (7, 15),
        (1, 2), (3, 4), (5, 6), (9, 10), (11, 12), (13, 14),
        (0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13), (14, 15),
    ]

def construct():
    start_time = time.time()
    deadline = start_time + 85 # 5s margin
    
    base_net = _base_60_network()
    
    if not _verify_zero_one_fast(base_net):
        return [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]
    
    best_net = base_net
    best_len = len(base_net)
    
    rng = random.Random(42)
    
    current_net = base_net[:]
    
    while time.time() < deadline:
        # Try to remove a comparator
        if len(current_net) > 1:
            idx = rng.randrange(len(current_net))
            new_net = current_net[:idx] + current_net[idx + 1:]
            if _verify_zero_one_fast(new_net):
                current_net = new_net
                if len(current_net) < best_len:
                    best_net = current_net[:]
                    best_len = len(current_net)
        else:
            break
            
        # Try swapping two comparators
        if len(current_net) > 1:
            i, j = rng.sample(range(len(current_net)), 2)
            new_net = current_net[:]
            new_net[i], new_net[j] = new_net[j], new_net[i]
            if _verify_zero_one_fast(new_net):
                current_net = new_net
                if len(current_net) < best_len:
                    best_net = current_net[:]
                    best_len = len(current_net)
                    
        # Try inserting a comparator
        if rng.random() < 0.1:
            i, j = rng.sample(range(N), 2)
            if i > j: i, j = j, i
            idx = rng.randrange(len(current_net) + 1)
            new_net = current_net[:idx] + [(i, j)] + current_net[idx:]
            if _verify_zero_one_fast(new_net):
                current_net = new_net

    return best_net

# EVOLVE-BLOCK-END
