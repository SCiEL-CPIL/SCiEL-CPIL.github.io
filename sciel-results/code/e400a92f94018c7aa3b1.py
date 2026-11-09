import time
import numpy as np

N = 18
_SEED_COMPARATORS = 153

def construct():
    """
    Construct a sorting network using a greedy heuristic that minimizes 
    the Hamming distance to the sorted state for all Boolean inputs.
    
    Returns:
        list of tuples: The sorting network as a list of (i, j) comparators.
    """
    start_time = time.time()
    deadline = 85.0  # Safety margin before 90s limit
    
    # 1. Setup: Bit-parallel representation of all 2^N inputs
    # Each wire is an integer. Bit k of wire w is 1 if input k has a 1 at wire w.
    # There are 2^N inputs, so we need integers with 2^N bits.
    # Python ints are arbitrary precision, but for speed we use NumPy uint64 
    # if possible? No, 2^18 > 64. 
    # We must use Python integers or a custom bit-vector structure.
    # However, the "inspiration" suggests NumPy simulation. 
    # Let's stick to the NumPy array of uint8/uint16 approach for the state 
    # but optimize the selection step.
    
    # Actually, the "Hamming distance" greedy method works best if we can 
    # quickly evaluate the effect of a comparator on the total error.
    
    # Let's use the standard NumPy vectorized simulation of all 2^N inputs.
    # State shape: (2^N, N)
    # This uses ~4.7 MB memory, which is fine.
    
    num_inputs = 2 ** N
    # Generate all binary inputs
    # inputs[i, j] is the j-th bit of input i
    # We want bit-parallelism? No, the standard approach is:
    # state[k, w] = bit w of input k
    
    # Efficient generation:
    # arange(2^N) gives integers 0..2^N-1
    # We need to unpack bits.
    inputs = np.zeros((num_inputs, N), dtype=np.uint8)
    # Use broadcasting to extract bits
    # For wire j (0 to N-1), bit j is (input >> (N-1-j)) & 1
    # Let's generate the powers of 2 or just use bitwise ops on the range array
    range_arr = np.arange(num_inputs, dtype=np.uint32)
    for j in range(N):
        shift = N - 1 - j
        inputs[:, j] = (range_arr >> shift) & 1
        
    # Target sorted state:
    # For a given input, the sorted state has k ones followed by N-k zeros.
    # k is the number of ones in the input.
    # We can compute the sorted state for all inputs efficiently.
    # popcount of each input row
    # np.bitwise_count is not in standard numpy for uint8 arrays directly in older versions, 
    # but we can sum the bits.
    num_ones = np.sum(inputs, axis=1, dtype=np.uint8)
    
    target = np.zeros_like(inputs)
    # For each row, the first 'num_ones' entries should be 1, rest 0.
    # We can use a cumulative sum trick or broadcasting.
    # Create an index matrix
    idx = np.arange(N)
    # target[i, j] = 1 if j < num_ones[i] else 0
    target = (idx[None, :] < num_ones[:, None]).astype(np.uint8)
    
    # Current state
    state = inputs.copy()
    network = []
    
    # Precompute the "error" metric?
    # We want to minimize the Hamming distance between state and target.
    # Error = sum(state XOR target)
    
    def get_error(s):
        return np.sum(s ^ target, dtype=np.int32)
    
    current_error = get_error(state)
    
    # Greedy Construction
    # While not sorted:
    #   Find (i, j) that minimizes error after applying comparator
    #   Apply it
    
    # Optimization: 
    # Applying comparator (i, j) to state:
    # new_i = min(state_i, state_j)
    # new_j = max(state_i, state_j)
    # This is equivalent to:
    # swap_mask = state_i > state_j
    # new_i = state_i & ~swap_mask | state_j & swap_mask  (bitwise)
    # But since we are dealing with uint8 arrays of 0/1:
    # min is AND, max is OR.
    # state_i_new = state_i & state_j
    # state_j_new = state_i | state_j
    
    # We need to efficiently find the best (i, j).
    # Brute force checking all N*(N-1)/2 pairs is ~153 checks per step.
    # Each check involves creating temporary arrays or in-place updates.
    # With 2^18 inputs, each operation is on ~4.7MB.
    # 153 * 4.7MB ~ 700MB of data movement per step? That might be slow.
    
    # Let's try a smarter heuristic or limit the search space.
    # Or, just do it and see if it fits in time.
    # Estimated steps: ~80-100.
    # 100 steps * 153 pairs * (2 min/max ops + error calc)
    # Error calc is a sum over 2^18 * N elements.
    # This might be too slow in Python/NumPy if done naively.
    
    # Alternative: Use the "deficit" idea.
    # For each wire w, let deficit[w] = number of inputs where state[w] != target[w].
    # Total error = sum(deficit).
    # When we compare (i, j), how does deficit change?
    # This is complex to precompute.
    
    # Let's try a simplified greedy:
    # Instead of checking all pairs, check pairs that are "far" in value?
    # Or just check all pairs but optimize the NumPy ops.
    
    # Let's implement the full greedy but with a time check.
    # If it's too slow, we fall back to a seed + local search.
    
    max_steps = 200
    for step in range(max_steps):
        if time.time() > deadline - 5.0:
            break
            
        if current_error == 0:
            break
            
        best_pair = None
        best_new_error = current_error
        
        # Iterate over all pairs
        # To speed up, we can pre-compute the current columns? No, state changes.
        
        # We will simulate the effect of each candidate comparator.
        # To avoid copying the whole state array for each candidate,
        # we can calculate the error change analytically or with small slices.
        
        # Error = sum(state ^ target)
        # Let A = state[:, i], B = state[:, j]
        # Let T_i = target[:, i], T_j = target[:, j]
        # Current contribution to error from i and j:
        # err_ij = sum(A ^ T_i) + sum(B ^ T_j)
        
        # After comparator (min to i, max to j):
        # A_new = A & B
        # B_new = A | B
        # New contribution:
        # new_err_ij = sum(A_new ^ T_i) + sum(B_new ^ T_j)
        
        # Change in error = new_err_ij - err_ij
        # We want to minimize total error, so minimize change.
        
        # We can pre-compute err_ij for all pairs?
        # There are N*N pairs.
        # Let's compute the error contribution for each wire.
        # err_w = sum(state[:, w] ^ target[:, w])
        
        # But the interaction between i and j matters for the *new* values.
        
        # Let's just do the simulation for candidate pairs.
        # To make it fast, we only touch columns i and j.
        
        for i in range(N - 1):
            A = state[:, i]
            T_i = target[:, i]
            err_i = np.sum(A ^ T_i, dtype=np.int32)
            
            for j in range(i + 1, N):
                B = state[:, j]
                T_j = target[:, j]
                err_j = np.sum(B ^ T_j, dtype=np.int32)
                
                # Calculate new values
                A_new = A & B
                B_new = A | B
                
                new_err_i = np.sum(A_new ^ T_i, dtype=np.int32)
                new_err_j = np.sum(B_new ^ T_j, dtype=np.int32)
                
                new_total_err = current_error - (err_i + err_j) + (new_err_i + new_err_j)
                
                if new_total_err < best_new_error:
                    best_new_error = new_total_err
                    best_pair = (i, j)
                    
        if best_pair is None or best_new_error >= current_error:
            # No improvement found, break
            break
            
        i, j = best_pair
        # Apply the best comparator
        A = state[:, i]
        B = state[:, j]
        A_new = A & B
        B_new = A | B
        state[:, i] = A_new
        state[:, j] = B_new
        
        network.append((i, j))
        current_error = best_new_error
        
    # After greedy construction, we might have a valid network.
    # Let's verify it.
    if current_error != 0:
        # Greedy failed to sort. Fall back to seed + local search?
        # Or just return the seed?
        # The prompt says "Keep the function names...".
        # If greedy fails, it's better to return a valid network.
        # The seed is always valid.
        # Let's check if the greedy network is valid.
        # If not, we should probably just return the seed to ensure score >= 1.0.
        # But the inspiration says greedy finds ~80-90 comparators.
        # Let's assume it works. If not, we'll add a fallback.
        
        # Fallback: Seed network
        best_network = [(i, i + 1) for phase in range(N)
                        for i in range(phase % 2, N - 1, 2)]
        return best_network

    # Optional: Local search to remove redundant comparators
    # This can improve the score further.
    if time.time() < deadline - 2.0:
        # Verify current network
        def verify(net):
            s = inputs.copy()
            for i, j in net:
                A = s[:, i]
                B = s[:, j]
                s[:, i] = A & B
                s[:, j] = A | B
            return np.all(s == target)
            
        if verify(network):
            # Try to remove comparators
            improved = True
            while improved and time.time() < deadline - 1.0:
                improved = False
                for idx in range(len(network) - 1, -1, -1):
                    if time.time() > deadline - 1.0:
                        break
                    candidate = network[:idx] + network[idx+1:]
                    if verify(candidate):
                        network = candidate
                        improved = True
                        break
        else:
            # Greedy produced an invalid network?
            # This shouldn't happen if error was 0, but just in case.
            # Return seed
            best_network = [(i, i + 1) for phase in range(N)
                            for i in range(phase % 2, N - 1, 2)]
            return best_network

    return network

# Keep the required helper functions if any were expected, though construct is the main one.
# The prompt says "Keep the function names and signatures of the seed program".
# The seed program had:
# _apply_comparator
# _is_sorted_network_vectorized
# _count_comparators
# _depth
# construct

# I will include stubs or simple implementations to match the interface, 
# although only construct is strictly required by the evaluator.

def _apply_comparator(state, i, j):
    """Apply a compare-exchange operation to a state array."""
    if state[i] > state[j]:
        state[i], state[j] = state[j], state[i]
    return state

def _is_sorted_network_vectorized(comparators):
    """Verify that a network of comparators sorts all Boolean inputs using NumPy."""
    inputs = np.zeros((2**N, N), dtype=np.uint8)
    for i in range(N):
        inputs[:, i] = (np.arange(2**N) >> (N - 1 - i)) & 1
        
    state = inputs.copy()
    
    for i, j in comparators:
        min_val = np.minimum(state[:, i], state[:, j])
        max_val = np.maximum(state[:, i], state[:, j])
        state[:, i] = min_val
        state[:, j] = max_val
        
    diffs = np.diff(state, axis=1)
    if np.any(diffs < 0):
        return False
    return True

def _count_comparators(comparators):
    """Count the number of comparators in a network."""
    return len(comparators)

def _depth(comparators):
    """Calculate the parallel depth of a sorting network."""
    if not comparators:
        return 0
    
    wire_depth = np.zeros(N, dtype=int)
    depth = 0
    for i, j in comparators:
        wire_depth[i] = max(wire_depth[i], depth)
        wire_depth[j] = max(wire_depth[j], depth)
        depth = max(wire_depth[i], wire_depth[j]) + 1
    
    return int(depth)
