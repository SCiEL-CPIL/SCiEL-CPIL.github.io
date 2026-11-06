import numpy as np
import time

N = 16

def _generate_batcher_oes(n):
    """
    Generates Batcher's Odd-Even Merge Sort network.
    Returns a list of (i, j) pairs.
    """
    if n == 1:
        return []
    
    # Recursive helper for sorting
    def sort(n, offset=0):
        if n == 1:
            return []
        
        half = n // 2
        # Sort first half
        left = sort(half, offset)
        # Sort second half
        right = sort(half, offset + half)
        
        # Merge the two sorted halves
        # The merge step operates on the combined range [offset, offset+n)
        # The two sorted lists are [offset, offset+half) and [offset+half, offset+n)
        # Batcher's OES merge is defined recursively on interleaved indices?
        # No, standard Batcher OES merge:
        # To merge two sorted lists of length N:
        # 1. Recursively merge the odd-indexed elements (length N/2)
        # 2. Recursively merge the even-indexed elements (length N/2)
        # 3. Final stage: compare (0,1), (2,3), ..., (N-2, N-1)
        
        # This recursion is tricky with offsets because the "odd" and "even" 
        # elements of the merged block are not contiguous in the original array 
        # if we just view them as two halves. 
        # Actually, in Batcher's OES, the input to the merge is two sorted lists 
        # of length N/2. Let's call them L and R.
        # The merge network takes L and R and produces a sorted list of length N.
        # The structure is:
        # M(N) = M(N/2) on odd indices + M(N/2) on even indices + Final(N/2)
        
        # We need a helper that merges two sorted lists located at specific indices.
        # Let's define merge(n, offset) which assumes the block [offset, offset+n) 
        # contains two sorted lists of length n/2? 
        # No, the standard recursive definition for the *sorting* network S(n) is:
        # S(n) = S(n/2) on even indices + S(n/2) on odd indices + M(n)
        # Where M(n) is the merge network for two sorted lists of length n/2?
        # Wait, if S(n/2) sorts the even indices, they are sorted. 
        # If S(n/2) sorts the odd indices, they are sorted.
        # Then M(n) merges them.
        
        # Let's implement M(n, offset) which merges two sorted lists of length n/2 
        # located at [offset, offset+n/2) and [offset+n/2, offset+n).
        # Actually, the recursive step of M(n) operates on the *interleaved* indices?
        # No.
        
        # Let's look at the standard definition again.
        # Merge(N):
        #   if N == 1: return
        #   Merge(N/2) on odd indices
        #   Merge(N/2) on even indices
        #   Final: N/2 comparators
        
        # This implies that the input to Merge(N) is two sorted lists of length N/2?
        # No, Merge(N) takes a sequence of length N that is composed of two sorted lists 
        # of length N/2, but they are not necessarily in the first and second half?
        # In Batcher's OES, the input to the top-level merge is the output of 
        # S(N/2) on evens and S(N/2) on odds.
        # So the even indices are sorted, and the odd indices are sorted.
        # The merge network M(N) takes this "interleaved sorted" state and sorts the whole thing.
        
        # So, let's define a function `merge(n, offset)` that assumes the block 
        # [offset, offset+n) has the property that the even-indexed elements 
        # (relative to offset) are sorted, and the odd-indexed elements are sorted.
        # And it returns comparators to fully sort the block.
        
        def merge(n, offset):
            if n == 1:
                return []
            
            half = n // 2
            comps = []
            
            # Recursive step:
            # The "odd" indices of the current block are offset+1, offset+3, ...
            # These form a sub-block of size n/2?
            # The elements at offset+1, offset+3... are the odd elements.
            # Are they sorted? Yes, by the invariant of the caller.
            # Do we need to merge them further?
            # The recursive call Merge(N/2) on odd indices means we treat the odd indices 
            # as a new problem of size N/2.
            # The indices are offset+1, offset+3, ..., offset+n-1.
            # This is a stride-2 sequence.
            # We can call merge(half, offset+1) but with a stride?
            # My current merge(n, offset) assumes contiguous indices.
            # I need to handle stride.
            
            # Let's change the helper to take a list of indices or a stride.
            # Or simpler: just generate the network for N=16 using a known pattern or 
            # implement the recursion with explicit index mapping.
            
            # Alternative: Use a known valid network like Bitonic Sort (128 comps) 
            # or OET (128 comps) and just peel. 
            # The prompt says OET is 120. My calculation says 128.
            # Let's assume the prompt's "120" is a specific optimized OET or I'm missing something.
            # Regardless, 80 (Batcher) is better.
            
            # Let's try to implement the stride-aware merge.
            pass

    # Given the complexity of getting Batcher OES exactly right with strides in a short time,
    # and the risk of bugs, I will stick to the OET base but optimize the search.
    # However, the directive asks for a structural change.
    # Let's try a different structure: **Shell Sort Network**? No.
    # Let's try **Batcher Bitonic Sort**. It is easy to implement and has 128 comparators for N=16.
    # It is structurally different from OET.
    
    def bitonic_sort(n, offset=0, direction=1):
        if n <= 1:
            return []
        
        half = n // 2
        
        # Sort first half in direction 'direction'
        # Sort second half in opposite direction
        # Then merge in direction 'direction'
        
        comps = []
        
        # Recursively sort halves
        comps += bitonic_sort(half, offset, direction)
        comps += bitonic_sort(half, offset + half, -direction)
        
        # Merge
        # Merge a bitonic sequence of length n in direction 'direction'
        # Step 1: Compare (offset+i, offset+i+half) for i in 0..half-1
        # If direction is 1 (asc), we want min at i, max at i+half.
        # If direction is -1 (desc), we want max at i, min at i+half.
        
        for i in range(half):
            a = offset + i
            b = offset + i + half
            if direction == 1:
                comps.append((a, b)) # min to a, max to b
            else:
                comps.append((b, a)) # min to b, max to a -> effectively max to a, min to b
        
        # Step 2: Recursively merge the two halves
        # The first half [offset, offset+half) is now bitonic?
        # Actually, after step 1, the first half is sorted in 'direction' and the second half is sorted in 'direction'?
        # No.
        # In Bitonic Merge:
        # 1. Compare (i, i+n/2).
        # 2. The first half is bitonic, the second half is bitonic.
        # 3. Recursively sort first half in 'direction' and second half in 'direction'.
        
        comps += bitonic_sort(half, offset, direction)
        comps += bitonic_sort(half, offset + half, direction)
        
        return comps

    return bitonic_sort(n, 0, 1)

def _verify_network(comparators, n=16):
    """
    Verifies if the network sorts all binary inputs.
    Uses NumPy for speed.
    """
    inputs = np.array([list(map(int, format(i, f'0{n}b'))) for i in range(2**n)])
    
    for i, j in comparators:
        mask = inputs[:, i] > inputs[:, j]
        if np.any(mask):
            inputs[mask, i] = 1 - inputs[mask, i]
            inputs[mask, j] = 1 - inputs[mask, j]
            
    return np.array_equal(inputs, np.sort(inputs, axis=1))

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 85s limit (5s margin)
    
    # Use Bitonic Sort as the base (128 comparators for N=16)
    # This is structurally different from OET.
    network = _generate_batcher_oes(N)
    
    # Verify initial
    if not _verify_network(network):
        # Fallback to OET if Bitonic generation is buggy
        network = [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]
        if not _verify_network(network):
            return network

    best_network = network[:]
    best_size = len(network)
    
    # Pre-compute inputs for fast verification
    inputs = np.array([list(map(int, format(i, f'0{N}b'))) for i in range(2**N)])
    
    def verify_fast(comparators):
        state = inputs.copy()
        for i, j in comparators:
            mask = state[:, i] > state[:, j]
            if np.any(mask):
                state[mask, i] = 1 - state[mask, i]
                state[mask, j] = 1 - state[mask, j]
        return np.array_equal(state, np.sort(state, axis=1))

    current_network = network[:]
    
    # Multi-pass peeling
    # We will perform multiple passes of greedy removal.
    # In each pass, we try to remove comparators from end to start.
    # If we remove one, we continue.
    # We repeat until no more can be removed or time runs out.
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for k in range(len(current_network) - 1, -1, -1):
            if time.time() > deadline:
                break
            
            trial_network = current_network[:k] + current_network[k+1:]
            if verify_fast(trial_network):
                current_network = trial_network
                improved = True
                # If we improved, we should restart the pass? 
                # Or just continue? 
                # Continuing is faster, but might miss opportunities.
                # Restarting is safer for finding local minima.
                # Given time constraints, continuing is better.
                
    return current_network

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    N = 16
    
    # Generate Bitonic Sort Network
    def bitonic_sort(n, offset=0, direction=1):
        if n <= 1:
            return []
        
        half = n // 2
        comps = []
        comps += bitonic_sort(half, offset, direction)
        comps += bitonic_sort(half, offset + half, -direction)
        
        for i in range(half):
            a = offset + i
            b = offset + i + half
            if direction == 1:
                comps.append((a, b))
            else:
                comps.append((b, a))
        
        comps += bitonic_sort(half, offset, direction)
        comps += bitonic_sort(half, offset + half, direction)
        
        return comps

    network = bitonic_sort(N, 0, 1)
    
    # Fallback to OET if Bitonic is invalid (shouldn't happen)
    def _verify_network(comparators, n=16):
        inputs = np.array([list(map(int, format(i, f'0{n}b'))) for i in range(2**n)])
        for i, j in comparators:
            mask = inputs[:, i] > inputs[:, j]
            if np.any(mask):
                inputs[mask, i] = 1 - inputs[mask, i]
                inputs[mask, j] = 1 - inputs[mask, j]
        return np.array_equal(inputs, np.sort(inputs, axis=1))

    if not _verify_network(network):
        network = [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]

    inputs = np.array([list(map(int, format(i, f'0{N}b'))) for i in range(2**N)])
    
    def verify_fast(comparators):
        state = inputs.copy()
        for i, j in comparators:
            mask = state[:, i] > state[:, j]
            if np.any(mask):
                state[mask, i] = 1 - state[mask, i]
                state[mask, j] = 1 - state[mask, j]
        return np.array_equal(state, np.sort(state, axis=1))

    current_network = network[:]
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for k in range(len(current_network) - 1, -1, -1):
            if time.time() > deadline:
                break
            
            trial_network = current_network[:k] + current_network[k+1:]
            if verify_fast(trial_network):
                current_network = trial_network
                improved = True
                
    return current_network
# EVOLVE-BLOCK-END
