import time
import random
import numpy as np

N = 18

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin before 90s limit

    # Start with a valid dense network: Odd-Even Transposition
    # This is a standard sorting network with N*(N-1)/2 comparators
    # For N=18, that's 153 comparators
    def odd_even_transposition():
        comparators = []
        for phase in range(N):
            for i in range(phase % 2, N - 1, 2):
                comparators.append((i, i + 1))
        return comparators
    
    # Fast 0-1 principle checker using bitsets
    # Each wire state is tracked as a bitset of 2^N inputs
    # But 2^18 = 262144 bits = 32768 bytes per wire, 18 wires = ~589KB total
    # This is feasible for fast checking
    
    def is_valid_01(comparators):
        """Check if network sorts all 0-1 inputs using bitset representation."""
        # For each of 2^N input patterns, track the state
        # Use numpy uint8 arrays for speed
        # Each column is one input pattern (2^N = 262144 columns)
        # Each row is one wire (18 rows)
        
        num_inputs = 1 << N
        # Create input matrix: for each input pattern, set the bits
        # Input pattern k: wire i is 1 if bit i of k is set
        inputs = np.zeros((N, num_inputs), dtype=np.uint8)
        for i in range(N):
            # Wire i is 1 for inputs where bit i is set
            # Pattern: 00000000 00000000 00000001 00000001 ...
            # For N=18, we can use: (arange(num_inputs) >> i) & 1
            if i < 63:  # numpy shift works for reasonable sizes
                inputs[i] = (np.arange(num_inputs, dtype=np.uint64) >> i) & 1
            else:
                # Fallback for larger i (not needed for N=18)
                for k in range(num_inputs):
                    inputs[i, k] = (k >> i) & 1
        
        # Apply comparators
        for (i, j) in comparators:
            # min goes to i, max goes to j
            # min(a,b) = a & b, max(a,b) = a | b for 0-1 values
            old_i = inputs[i].copy()
            old_j = inputs[j].copy()
            inputs[i] = old_i & old_j  # min
            inputs[j] = old_i | old_j  # max
        
        # Check if sorted: wire 0 should be all 0s (min), wire N-1 should be all 1s (max)
        # Actually, for a valid sorting network, after sorting:
        # For any input with k ones, wires 0..(N-k-1) should be 0 and wires (N-k)..N-1 should be 1
        # Simpler check: the output should be non-decreasing in each column
        
        # Check: for each column, inputs[0] <= inputs[1] <= ... <= inputs[N-1]
        # This is equivalent to: inputs[i] <= inputs[i+1] for all i
        for i in range(N - 1):
            if np.any(inputs[i] > inputs[i + 1]):
                return False
        return True
    
    # Alternative faster check: just verify the 0-1 principle properly
    # A network is correct iff it sorts all 2^N binary inputs
    # We can check this more efficiently by tracking which inputs end up in which position
    
    def is_valid_fast(comparators):
        """Faster validity check using the 0-1 principle."""
        # For a sorting network, it suffices to check all 2^N binary inputs
        # Use a different approach: track the multiset of values
        # For 0-1 inputs, the sorted output is determined by the count of 1s
        # So we just need to verify that for each input, the number of 1s is preserved
        # AND the output is sorted (all 0s before all 1s)
        
        num_inputs = 1 << N
        
        # Process all inputs in batch using numpy
        # Create input matrix
        arange = np.arange(num_inputs, dtype=np.uint64)
        inputs = np.zeros((N, num_inputs), dtype=np.uint8)
        for i in range(N):
            inputs[i] = (arange >> i) & 1
        
        # Apply comparators
        for (i, j) in comparators:
            a = inputs[i]
            b = inputs[j]
            inputs[i] = np.minimum(a, b)
            inputs[j] = np.maximum(a, b)
        
        # Check sorted: for each column, values should be non-decreasing
        # i.e., inputs[i] <= inputs[i+1] for all i, all columns
        for i in range(N - 1):
            if np.any(inputs[i] > inputs[i + 1]):
                return False
        return True
    
    # Even faster: use the fact that for 0-1 inputs, we only need to track
    # the position of the boundary between 0s and 1s
    
    def is_valid_boundary(comparators):
        """Check validity by tracking the 0-1 boundary for each input count."""
        # For each possible number of 1s (0 to N), there are C(N, k) inputs
        # with exactly k ones. After sorting, the first (N-k) wires should be 0
        # and the last k wires should be 1.
        # So we can check: for each k, pick one representative input with k ones,
        # simulate, and check if the output has 0s in positions 0..N-k-1 and 1s in N-k..N-1
        # But we need to check ALL inputs, not just one per k.
        
        # Actually, the 0-1 principle says we must check all 2^N inputs.
        # Let's use a different optimization: process inputs in chunks.
        
        num_inputs = 1 << N
        chunk_size = 65536  # Process 64K inputs at a time
        
        arange_full = np.arange(num_inputs, dtype=np.uint64)
        
        for start in range(0, num_inputs, chunk_size):
            end = min(start + chunk_size, num_inputs)
            arange_chunk = arange_full[start:end]
            
            inputs = np.zeros((N, end - start), dtype=np.uint8)
            for i in range(N):
                inputs[i] = (arange_chunk >> i) & 1
            
            for (i, j) in comparators:
                a = inputs[i]
                b = inputs[j]
                inputs[i] = np.minimum(a, b)
                inputs[j] = np.maximum(a, b)
            
            for i in range(N - 1):
                if np.any(inputs[i] > inputs[i + 1]):
                    return False
        
        return True
    
    # Start with odd-even transposition
    network = odd_even_transposition()
    best_network = network[:]
    best_count = len(network)
    
    # Verify initial network
    if not is_valid_boundary(network):
        # Shouldn't happen, but fallback to bubble sort
        network = []
        for i in range(N - 1):
            for j in range(N - 1 - i):
                network.append((j, j + 1))
        best_network = network[:]
        best_count = len(network)
    
    # Local search: try to remove or swap comparators
    random.seed(42)
    
    def try_remove(comparators, idx):
        """Try removing comparator at index idx."""
        if len(comparators) <= 1:
            return None
        new_comparators = comparators[:idx] + comparators[idx+1:]
        if is_valid_boundary(new_comparators):
            return new_comparators
        return None
    
    def try_swap(comparators, i, j):
        """Try swapping comparators at indices i and j."""
        if i == j:
            return None
        new_comparators = comparators[:]
        new_comparators[i], new_comparators[j] = comparators[j], comparators[i]
        if is_valid_boundary(new_comparators):
            return new_comparators
        return None
    
    # Iterative local search
    improved = True
    while improved and time.time() < deadline:
        improved = False
        network_len = len(network)
        
        # Try removing each comparator
        for idx in range(network_len):
            if time.time() > deadline:
                break
            new_net = try_remove(network, idx)
            if new_net is not None:
                network = new_net
                improved = True
                break
        
        # If no removal worked, try some swaps to escape local optima
        if not improved and time.time() < deadline - 5:
            # Try random swaps
            for _ in range(50):
                if time.time() > deadline:
                    break
                if len(network) < 2:
                    break
                i = random.randint(0, len(network) - 1)
                j = random.randint(0, len(network) - 1)
                if i == j:
                    continue
                new_net = try_swap(network, i, j)
                if new_net is not None:
                    network = new_net
                    improved = True
                    break
    
    # Update best if we improved
    if len(network) < best_count:
        best_network = network[:]
        best_count = len(network)
    
    # Try a second pass with different random seed for diversity
    if time.time() < deadline - 10:
        random.seed(123)
        network2 = best_network[:]
        improved2 = True
        while improved2 and time.time() < deadline - 5:
            improved2 = False
            network2_len = len(network2)
            
            for idx in range(network2_len):
                if time.time() > deadline - 5:
                    break
                new_net = try_remove(network2, idx)
                if new_net is not None:
                    network2 = new_net
                    improved2 = True
                    break
            
            if not improved2 and time.time() < deadline - 10:
                for _ in range(30):
                    if time.time() > deadline - 5:
                        break
                    if len(network2) < 2:
                        break
                    i = random.randint(0, len(network2) - 1)
                    j = random.randint(0, len(network2) - 1)
                    if i == j:
                        continue
                    new_net = try_swap(network2, i, j)
                    if new_net is not None:
                        network2 = new_net
                        improved2 = True
                        break
        
        if len(network2) < best_count:
            best_network = network2[:]
            best_count = len(network2)
    
    # Final verification
    if is_valid_boundary(best_network):
        return best_network
    else:
        # Fallback: return the seed odd-even transposition
        return odd_even_transposition()
# EVOLVE-BLOCK-END
