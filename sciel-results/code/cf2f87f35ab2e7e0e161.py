import numpy as np

N = 16

# EVOLVE-BLOCK-START
def construct():
    # Generate Odd-Even Transposition network
    comps = [(i, i + 1) for phase in range(N)
             for i in range(phase % 2, N - 1, 2)]
    
    # Optimization: Greedy removal of redundant comparators
    # Use NumPy to test all 2^16 inputs
    
    def is_valid(comparators):
        # Test all 2^16 boolean inputs
        # Represent inputs as a 16x65536 array of bits
        # Input mask k: wire i is bit i of k
        masks = np.arange(2**N, dtype=np.uint8)
        # Create a 16x65536 array
        # state[i, k] = bit i of k
        state = np.zeros((N, 2**N), dtype=np.uint8)
        for i in range(N):
            state[i] = (masks >> i) & 1
            
        for i, j in comparators:
            # min to i, max to j
            min_ij = np.minimum(state[i], state[j])
            max_ij = np.maximum(state[i], state[j])
            state[i] = min_ij
            state[j] = max_ij
            
        # Check if sorted: state[0] <= state[1] <= ... <= state[N-1]
        # This means for all k, state[0,k] <= state[1,k] ...
        # Equivalent to checking if state is sorted along axis 0
        # We can check if state[i] <= state[i+1] for all i
        for i in range(N - 1):
            if not np.all(state[i] <= state[i+1]):
                return False
        return True

    # Greedy removal
    # Try removing comparators from the end (often less critical)
    # We iterate through the list and try to remove each one
    # If removal breaks validity, we keep it
    
    # To be more efficient, we can try to remove in reverse order
    # and restart the check if we remove one?
    # Simple greedy: iterate from last to first.
    # If we remove one, the list changes, so indices shift.
    # Better: iterate through a copy, or iterate from last to first and pop.
    
    valid_comps = list(comps)
    
    # Iterate from last to first
    for idx in range(len(valid_comps) - 1, -1, -1):
        # Try removing valid_comps[idx]
        temp_comps = valid_comps[:idx] + valid_comps[idx+1:]
        if is_valid(temp_comps):
            valid_comps = temp_comps
            # Continue with the new list
        # Else, keep it
        
    return valid_comps
# EVOLVE-BLOCK-END
