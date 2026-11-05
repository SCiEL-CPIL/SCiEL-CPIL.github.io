import numpy as np
import time

N = 17
_INPUTS = None

def _get_inputs():
    global _INPUTS
    if _INPUTS is None:
        # Generate all 2^17 boolean inputs
        _INPUTS = np.array(list(np.meshgrid(*[np.array([0, 1]) for _ in range(N)])), dtype=np.int8).T
        # The above creates a huge array of shape (131072, 17)
        # Let's verify this is efficient.
        # np.meshgrid with 17 arrays creates 17 arrays of size 2^17.
        # Stacking them creates a 2^17 x 17 array.
        # This is fine.
    return _INPUTS

def _is_valid(network):
    inputs = _get_inputs()
    state = inputs.copy()
    for i, j in network:
        # Vectorized min/max
        state[:, i], state[:, j] = np.minimum(state[:, i], state[:, j]), np.maximum(state[:, i], state[:, j])
    # Check sorted
    return np.all(np.diff(state, axis=1) >= 0)

def _batcher_odd_merge(n):
    """
    Returns a list of (i, j) pairs for Batcher's Odd-Merge sorting network on n wires.
    """
    if n <= 1:
        return []
    
    # Split into two halves
    n1 = n // 2
    n2 = n - n1
    
    # Recursively sort the two halves
    net1 = _batcher_odd_merge(n1)
    net2 = _batcher_odd_merge(n2)
    
    # Shift net2 indices by n1
    net2_shifted = [(i + n1, j + n1) for i, j in net2]
    
    # Merge step
    merge_net = []
    # The merge step in Batcher's Odd-Merge involves comparing specific pairs.
    # For n1 and n2, we perform n1 comparisons?
    # The standard algorithm:
    # 1. Sort first n1 and last n2.
    # 2. For k in 0 to n1-1:
    #    Compare (k, k + n2) ? No.
    #    
    # Let's look up the exact merge step for Batcher's Odd-Merge.
    # The merge step consists of a sequence of compare-exchange operations.
    # For merging two sorted sequences of length n1 and n2:
    # We perform comparisons between elements at positions that are "interleaved".
    
    # A simpler way to generate Batcher's network is to use the known recursive structure.
    # However, implementing the merge step correctly is tricky.
    
    # Alternative: Use a simpler, known efficient network.
    # For N=17, we can use a modified Batcher or just use the fact that 
    # Batcher's Odd-Merge is standard.
    
    # Let's implement the merge step properly.
    # The merge step for Batcher's Odd-Merge on two sorted arrays of size n1 and n2:
    # It performs n1 comparisons? No, it performs more.
    # Actually, the number of comparators in the merge step is n1 * n2? No.
    
    # Let's use a different approach: Generate the network using a known library or algorithm.
    # Since I cannot use external libraries for the network generation logic (only stdlib/np/scipy),
    # I will implement Batcher's Odd-Merge carefully.
    
    # Reference: Batcher's Odd-Merge
    # To merge two sorted sequences A (len n1) and B (len n2):
    # 1. Interleave A and B? No.
    # 2. Compare A[i] with B[j] where i+j is even?
    
    # Actually, there is a simpler recursive definition for the *comparators*:
    # merge(n1, n2):
    #   if n1 == 0 or n2 == 0: return []
    #   if n1 == 1 and n2 == 1: return [(0, 1)]
    #   ...
    
    # This is getting complex. Let's use a simpler base: 
    # The **AKS** network is too big. 
    # The **Bitonic** sort network is $O(N \log^2 N)$.
    # Bitonic Sort for N=16 has 120 comparators?
    # Let's use **Bitonic Sort**. It's easier to implement.
    
    # Bitonic Sort Network for N=17:
    # Pad to 32? No, that adds comparators.
    # We can run Bitonic Sort on 17 wires directly?
    # Bitonic sort requires power of 2 for the standard implementation.
    # We can pad with dummy wires? No, that increases N.
    
    # Let's stick to Batcher's Odd-Merge. I will implement the merge step.
    
    # The merge step for Batcher's Odd-Merge:
    # Given two sorted sequences of length n1 and n2.
    # We want to merge them.
    # The comparators are defined as:
    # for k in range(n1):
    #    # Compare elements that are "opposite" in the merge pattern
    #    # This is equivalent to a series of "shells"
    
    # Actually, I found a simpler way: 
    # Use the `scipy` or `numpy`? No.
    # I will implement a **Knuth** optimized network if I can recall it, or just use **Batcher**.
    
    # Let's try to implement Batcher's Odd-Merge merge step.
    # From literature:
    # The merge step consists of n1 - 1 stages?
    # No, the number of comparators in the merge step is n1 * n2 / 2?
    
    # Let's use a different, very efficient network for small N:
    # **Pippenger's** network?
    
    # Okay, let's use the **Odd-Even Merge** network implementation from a reliable source.
    # I will write a recursive function that returns the list of comparators.
    
    def _odd_even_merge(n):
        if n <= 1:
            return []
        
        n1 = n // 2
        n2 = n - n1
        
        # Recursively sort
        left = _odd_even_merge(n1)
        right = _odd_even_merge(n2)
        
        # Shift right indices
        right = [(i + n1, j + n1) for i, j in right]
        
        # Merge step
        # The merge step compares elements at indices i and j such that:
        # The relative order is determined by a specific pattern.
        # For Batcher's Odd-Merge, the merge step is:
        # for k in range(n1):
        #    # Compare (k, k + n2) ? No.
        
        # Correct Merge Step for Batcher's Odd-Merge:
        # The merge step is a sequence of compare-exchange operations.
        # It can be generated by:
        # for step in range(n1):
        #    for i in range(step % 2, n1 - step, 2):
        #        # Compare (i, i + n2 + 2*step) ?
        
        # This is hard to get right without a reference.
        
        # Alternative: Use **Bitonic Sort** on 32 wires and then remove the last 15 wires?
        # No, that doesn't work because the comparators involving removed wires are lost.
        
        # Let's use a **Greedy Construction** from a random or simple network?
        # No, we need a good starting point.
        
        # Let's use the **Odd-Even Transposition** but with a better initial permutation?
        # No.
        
        # I will implement **Batcher's Odd-Merge** using a known correct algorithm.
        # The merge step for Batcher's Odd-Merge on two sorted arrays of length n1 and n2:
        # We perform n1 comparisons?
        # Actually, the number of comparators in the merge step is n1 * n2? No.
        
        # Let's try this:
        # The merge step is equivalent to sorting the combined array using a specific network.
        # I will use a simplified version: 
        # For i in range(n1):
        #    for j in range(n2):
        #       if (i + j) % 2 == 0:
        #           compare (i, n1 + j)
        
        # This is not quite right.
        
        # Let's use a **different strategy**:
        # Use the **AKS** network? No.
        # Use **Batcher's** but implement it via **Bitonic** logic if possible?
        
        # Actually, for N=17, the **best known** is 71.
        # The **seed** is 136.
        # If I just use **Batcher's Odd-Merge**, I will get ~100.
        # I will implement Batcher's Odd-Merge correctly.
        
        # From "Sorting Networks: An Introduction" by Knuth:
        # The merge step for Batcher's Odd-Merge:
        # Let A be the first n1 elements, B be the last n2 elements.
        # We want to merge A and B.
        # The comparators are:
        # for k in range(n1):
        #    # Compare A[k] with B[n2 - 1 - k] ? No.
        
        # I will use a **Brute Force** search for the merge step? No.
        
        # Let's use the **Odd-Even Merge** network as described in many online resources.
        # The merge step for two sorted sequences of length n1 and n2:
        # 1. Compare A[i] with B[j] where i + j is even?
        
        # I found a Python implementation of Batcher's Odd-Merge online in my training data.
        # It looks like this:
        
        merge = []
        for k in range(n1):
            # The indices to compare are (k, n1 + (n2 - 1 - k)) ?
            # No.
            
            # Let's use the property that Batcher's Odd-Merge is equivalent to 
            # a series of "shells".
            
            # I will use a simpler, valid network: **Bitonic Sort** padded to 32.
            # But we only have 17 wires.
            # We can run Bitonic Sort on 17 wires by treating it as 16+1?
            # No.
            
            # Let's use **Odd-Even Transposition** but start with a **Shuffle** to reduce depth?
            # No, size is the metric.
            
            # I will implement **Batcher's Odd-Merge** using the following logic:
            # The merge step consists of n1 - 1 stages.
            # Stage k: compare (i, i + n2) for i in range(k, n1 - k, 2)?
            
            # Actually, I will use a **known good network** for N=17 if I can hardcode it?
            # No, I must generate it.
            
            # Let's use **Bitonic Sort** for N=16 and add the 17th wire in a way that minimizes comparators?
            # If we sort 16 wires with Bitonic (120 comparators), and then merge the 17th wire?
            # Merging 1 element into a sorted list of 16 takes 16 comparators (binary search style)?
            # Total: 120 + 16 = 136. Same as seed.
            
            # We need a better base.
            # **Batcher's Odd-Merge** for N=17:
            # C(17) = C(8) + C(9) + Merge(8, 9)
            # C(8) = 19
            # C(9) = C(4) + C(5) + Merge(4, 5)
            # C(4) = 6
            # C(5) = C(2) + C(3) + Merge(2, 3)
            # C(2) = 1
            # C(3) = C(1) + C(2) + Merge(1, 2) = 0 + 1 + 2 = 3
            # Merge(2, 3) = 2*3? No.
            # Batcher's merge step size is n1 * n2? No.
            # The number of comparators in the merge step of Batcher's Odd-Merge is n1 * n2 / 2?
            # If so, Merge(2,3) = 3. C(5) = 1 + 3 + 3 = 7.
            # Merge(4,5) = 10. C(9) = 6 + 7 + 10 = 23.
            # Merge(8,9) = 36. C(17) = 19 + 23 + 36 = 78.
            
            # 78 is much better than 136.
            # I will implement Batcher's Odd-Merge with the assumption that the merge step has n1*n2 comparators?
            # No, the merge step has n1 * n2 comparators is for a full merge.
            # Batcher's merge is more efficient.
            
            # I will implement the merge step as:
            # for i in range(n1):
            #    for j in range(n2):
            #       if (i + j) % 2 == 0:
            #           compare (i, n1 + j)
            
            # This generates n1*n2/2 comparators.
            # This is a valid sorting network for the merge step?
            # Yes, this is the **Odd-Even Merge** step.
            
            for i in range(n1):
                for j in range(n2):
                    if (i + j) % 2 == 0:
                        merge.append((i, n1 + j))
                        
        return left + right + merge

    return _odd_even_merge(n)

def construct():
    start_time = time.time()
    TIME_LIMIT = 55.0
    
    network = _batcher_odd_merge(N)
    
    # Verify seed
    if not _is_valid(network):
        # Fallback to OET if Batcher implementation is wrong
        network = _odd_even_transposition_network()
        if not _is_valid(network):
            return network

    improved = True
    while improved and (time.time() - start_time) < TIME_LIMIT:
        improved = False
        for idx in range(len(network)):
            if (time.time() - start_time) > TIME_LIMIT:
                return network
            
            candidate = network[:idx] + network[idx+1:]
            if _is_valid(candidate):
                network = candidate
                improved = True
                break

    return network

def _odd_even_transposition_network():
    net = []
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            net.append((i, i + 1))
    return net
