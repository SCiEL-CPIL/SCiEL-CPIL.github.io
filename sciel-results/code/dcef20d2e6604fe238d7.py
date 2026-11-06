<<<<<<< SEARCH
def _is_valid(net):
    """
    Checks if the network sorts all 2^N binary inputs.
    Uses NumPy for vectorized simulation.
    """
    # Initialize all 2^N inputs
    # Using uint16 is sufficient for 17 bits
    inputs = np.arange(NUM_INPUTS, dtype=np.uint16)
    
    for i, j in net:
        # Extract bits i and j for all inputs
        bit_i = (inputs >> i) & 1
        bit_j = (inputs >> j) & 1
        
        # Compare-exchange: min to i, max to j
        # min(a, b) = a & b for bits
        # max(a, b) = a | b for bits
        new_bit_i = bit_i & bit_j
        new_bit_j = bit_i | bit_j
        
        # Update inputs
        # Clear bits i and j
        mask_i = ~(1 << i)
        mask_j = ~(1 << j)
        
        inputs = (inputs & mask_i) | (new_bit_i << i)
        inputs = (inputs & mask_j) | (new_bit_j << j)
        
    # Check if all outputs are sorted (non-decreasing)
    # Unpack bits into a matrix of shape (NUM_INPUTS, N)
    # This is the most expensive part, but vectorized.
    # mat[k, m] is bit m of input k
    mat = np.stack([(inputs >> i) & 1 for i in range(N)], axis=1)
    
    # Check if each row is non-decreasing
    # mat[:, :-1] <= mat[:, 1:]
    return np.all(mat[:, :-1] <= mat[:, 1:])
=======
def _is_valid(net, fast=False):
    """
    Checks if the network sorts all 2^N binary inputs.
    If fast=True, checks a random subset for speed.
    """
    if fast:
        # Use a subset of inputs for quick rejection
        num_samples = 10000
        inputs = np.random.randint(0, NUM_INPUTS, size=num_samples, dtype=np.uint16)
    else:
        inputs = np.arange(NUM_INPUTS, dtype=np.uint16)
    
    for i, j in net:
        # Extract bits i and j for all inputs
        bit_i = (inputs >> i) & 1
        bit_j = (inputs >> j) & 1
        
        # Compare-exchange: min to i, max to j
        new_bit_i = bit_i & bit_j
        new_bit_j = bit_i | bit_j
        
        # Update inputs
        # Use unsigned masks to avoid OverflowError
        # 0xFFFF is the max value for uint16
        mask_i = np.uint16(0xFFFF ^ (1 << i))
        mask_j = np.uint16(0xFFFF ^ (1 << j))
        
        inputs = (inputs & mask_i) | (new_bit_i << i)
        inputs = (inputs & mask_j) | (new_bit_j << j)
        
    # Check if all outputs are sorted (non-decreasing)
    # Unpack bits into a matrix of shape (NUM_INPUTS, N)
    mat = np.stack([(inputs >> i) & 1 for i in range(N)], axis=1)
    
    # Check if each row is non-decreasing
    return np.all(mat[:, :-1] <= mat[:, 1:])
>>>>>>> REPLACE

