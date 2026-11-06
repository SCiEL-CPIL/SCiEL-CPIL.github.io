import sys
import time

N = 18

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 90.0
    safety_margin = 5.0
    
    # Batcher's Odd-Merge Sorting Network
    def batcher(n):
        if n <= 1:
            return []
        
        n1 = (n + 1) // 2
        n2 = n // 2
        
        comps = []
        
        # 1. Recursively sort the first half
        comps.extend(batcher(n1))
        
        # 2. Recursively sort the second half
        half2_comps = batcher(n2)
        for i, j in half2_comps:
            comps.append((i + n1, j + n1))
            
        # 3. Merge step
        # Compare (i, i + n1) for i in 0..n2-1
        for i in range(n2):
            comps.append((i, i + n1))
            
        # 4. Recursively merge the even and odd wires
        # The even wires are 0, 2, ..., 2*n2 - 2
        # The odd wires are 1, 3, ..., 2*n2 - 1
        # Note: If n1 > n2, there are extra wires in the first half?
        # Actually, the merge of two sorted lists of size n1 and n2 (n1>=n2)
        # results in two sorted lists of size n2 on the even and odd positions
        # of the first 2*n2 wires. The remaining n1-n2 wires are already sorted?
        
        # Standard Batcher merge for n1, n2:
        # After step 3, the wires 0..2*n2-1 contain the 2*n2 smallest elements?
        # No.
        
        # Let's use the property:
        # The even indices 0, 2, ..., 2*n2-2 form a sorted sequence of size n2.
        # The odd indices 1, 3, ..., 2*n2-1 form a sorted sequence of size n2.
        # So we apply batcher(n2) to these two sets.
        
        even_wires = list(range(0, 2 * n2, 2))
        odd_wires = list(range(1, 2 * n2, 2))
        
        # Apply batcher(n2) to even wires
        for i, j in batcher(n2):
            comps.append((even_wires[i], even_wires[j]))
            
        # Apply batcher(n2) to odd wires
        for i, j in batcher(n2):
            comps.append((odd_wires[i], odd_wires[j]))
            
        return comps

    network = batcher(N)
    
    # Verify correctness on all Boolean inputs (0/1)
    # This is fast: 2^18 = 262144 inputs.
    # We can simulate the network on all 2^18 bitmasks.
    
    def simulate(network, n):
        # Precompute the effect of each comparator on a bitmask?
        # Or just iterate.
        # 262144 * 150 comparators = ~39 million operations.
        # Python might be slow.
        # Optimization: Use bit operations.
        
        # For each input mask:
        # For each comparator (i, j):
        #   if bit i is 1 and bit j is 0: swap.
        
        # We can check if the output is sorted (all 0s then all 1s).
        
        # To speed up, we can use the fact that the network is a DAG.
        # But 39M simple ops might take 1-2 seconds in Python.
        # 90s limit is plenty.
        
        for mask in range(1 << n):
            state = mask
            for i, j in network:
                bit_i = (state >> i) & 1
                bit_j = (state >> j) & 1
                if bit_i == 1 and bit_j == 0:
                    # Swap: set i to 0, j to 1
                    state ^= (1 << i) | (1 << j)
            # Check if sorted
            # Sorted means state is of form 0...01...1
            # i.e., state & (state + 1) == 0 is NOT correct for sorted.
            # Sorted means no '1' is to the left of a '0'.
            # i.e., if state has a 1 at pos k, all pos > k must be 1? No.
            # All 0s must be to the right of all 1s? No, min to wire i, max to j.
            # So 0s are at the bottom (low indices) and 1s at the top (high indices)?
            # "min to wire i and max to j". i < j.
            # So wire 0 gets the smallest, wire N-1 gets the largest.
            # So the output should be 0...0 1...1.
            # i.e., state should be of the form 0^k 1^(n-k).
            # This means state is a prefix of 0s and suffix of 1s.
            # Check: state & (state + 1) == 0 is for powers of 2.
            # Check: state == ((1 << n) - 1) ^ ((1 << k) - 1) for some k?
            # Easier: 
            # Find the first 1. All bits below must be 0? No, bits below index of first 1 are 0.
            # All bits above are 1.
            
            # If state is 0, sorted.
            # If state is (1<<n)-1, sorted.
            
            # Check if state has form 000...0111...1
            # This is equivalent to: state & (state + 1) == 0 is FALSE.
            # Correct check:
            # Let k be the number of 1s.
            # state must equal ((1 << n) - 1) ^ ((1 << k) - 1)
            # i.e., state must be a sequence of k 1s followed by n-k 0s?
            # No, min to low index. So 0s at low indices.
            # So state should be 0...0 1...1.
            # This means state is of the form ((1 << (n-k)) - 1) << k? No.
            # state = ((1 << n) - 1) ^ ((1 << k) - 1) where k is number of 0s?
            # If k zeros, then bits 0..k-1 are 0, bits k..n-1 are 1.
            # state = ((1 << n) - 1) ^ ((1 << k) - 1)
            
            # We can just check if state is sorted by verifying:
            # state & (state + 1) is not the right way.
            # Just check:
            # for i in range(n-1):
            #    if (state >> i) & 1 == 1 and (state >> (i+1)) & 1 == 0: return False
            
            # Faster:
            # state must be a "suffix" of 1s.
            # i.e., state == ((1 << n) - 1) ^ ((1 << k) - 1) for some k.
            # This is equivalent to: state | (state + 1) == (1 << n) - 1 ? No.
            
            # Let's just use the loop, it's only 18 iterations.
            sorted_flag = True
            for i in range(n - 1):
                if ((state >> i) & 1) == 1 and ((state >> (i + 1)) & 1) == 0:
                    sorted_flag = False
                    break
            if not sorted_flag:
                return False
        return True

    # Run verification
    if not simulate(network, N):
        # Fallback to seed if Batcher fails (should not happen)
        return [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]
    
    return network
# EVOLVE-BLOCK-END
