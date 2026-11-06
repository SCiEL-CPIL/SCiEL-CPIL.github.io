"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 18

# EVOLVE-BLOCK-START
import time

def batcher_merge_sort(n):
    """
    Constructs a Batcher odd-merge sorting network for n wires.
    Returns a list of (i, j) comparators.
    """
    if n <= 1:
        return []
    if n == 2:
        return [(0, 1)]
    if n == 3:
        return [(0, 1), (1, 2), (0, 1)]
    
    comparators = []
    
    # Split into odd and even indices
    # Odd indices: 0, 2, 4, ...
    # Even indices: 1, 3, 5, ...
    # The number of elements in each sub-network is ceil(n/2) and floor(n/2)
    
    # To simplify, we map the odd indices to 0..n_odd-1 and even to 0..n_even-1
    # But Batcher's algorithm usually assumes n is even or handles it by padding?
    # Batcher's odd-merge works for any n.
    # Let's define the recursive calls on the subsets.
    
    # Sub-network 1: indices 0, 2, 4, ...
    # Sub-network 2: indices 1, 3, 5, ...
    
    # We need to generate comparators for the sub-networks on their local indices,
    # then map them back to global indices.
    
    # Local index k in sub-network 1 maps to global index 2*k
    # Local index k in sub-network 2 maps to global index 2*k + 1
    
    n_odd = (n + 1) // 2
    n_even = n // 2
    
    # Recursively sort the odd-indexed wires
    sub_odd = batcher_merge_sort(n_odd)
    for i, j in sub_odd:
        comparators.append((2*i, 2*j))
        
    # Recursively sort the even-indexed wires
    sub_even = batcher_merge_sort(n_even)
    for i, j in sub_even:
        comparators.append((2*i + 1, 2*j + 1))
        
    # Now merge the two sorted sub-sequences
    # The merge step for Batcher's odd-merge:
    # Compare (i, i + n/2) for i in 0, 2, ..., n/2 - 2 ?
    # Actually, the merge is done by comparing (i, i + M) where M is the size of the smaller sequence?
    # Standard description:
    # Let the two sorted sequences be A (length n_odd) and B (length n_even).
    # They are interleaved in the wires: A0, B0, A1, B1, ...
    # The merge compares (i, i + M) where M is the size of the "step"?
    
    # Let's use the standard recursive merge definition:
    # merge(A, B) where A and B are sorted.
    # 1. Compare (i, i + M) for i = 0, 2, ..., M-2 (if M is even)
    #    Here M is the length of the shorter sequence? Or n/2?
    #    In Batcher's scheme for n=2m:
    #    Merge step: Compare (i, i+m) for i=0, 2, ..., m-2.
    #    Then recursively merge the even and odd indices of the result.
    
    # For general n:
    # Let m = n // 2
    # Compare (i, i + m) for i = 0, 2, ..., 2*(m//2 - 1) ?
    
    # Let's stick to the case where n is even first, then handle odd.
    # If n is even, m = n/2.
    # Compare (i, i + m) for i = 0, 2, ..., m - 2.
    # Then recursively merge the subsequences of length m formed by even and odd indices.
    
    if n % 2 == 0:
        m = n // 2
        # Compare (i, i + m) for i = 0, 2, ..., m - 2
        for i in range(0, m, 2):
            comparators.append((i, i + m))
        
        # Now the array is "2-sorted" in some sense.
        # We need to recursively merge the even indices (0, 2, ...) and odd indices (1, 3, ...).
        # The even indices contain the merged result of the even parts?
        # Actually, the recursive merge is on the subsequences of length m.
        # Subsequence 1: indices 0, 2, ..., m-1? No, indices 0, 2, ..., n-2?
        # The recursive merge calls batcher_merge_sort(m) on the even indices and odd indices of the current state.
        
        # Map local indices 0..m-1 to global 0, 2, ..., 2m-2
        sub_merge_odd = batcher_merge_sort(m)
        for i, j in sub_merge_odd:
            comparators.append((2*i, 2*j))
            
        # Map local indices 0..m-1 to global 1, 3, ..., 2m-1
        sub_merge_even = batcher_merge_sort(m)
        for i, j in sub_merge_even:
            comparators.append((2*i + 1, 2*j + 1))
            
    else:
        # n is odd.
        # This is more complex. Batcher's algorithm for odd n:
        # Sort n-1 elements, then insert the last one?
        # Or use the general recursive definition.
        # For n=3: (0,1), (1,2), (0,1)
        # For n=5:
        # Sort 0,2,4 (size 3) and 1,3 (size 2).
        # Then merge.
        # Merge of size 3 and 2:
        # Compare (0, 2)? (1, 3)?
        # Let's just use a simpler method for odd n if needed, or rely on the fact that 18 is even.
        # Since N=18 is even, the base case for odd n might not be hit in the top-level call,
        # but recursive calls might hit odd sizes (e.g., 9, 5, 3).
        
        # For odd n, we can compare (i, i + n//2) for i = 0, 2, ..., n//2 - 1?
        # Let's implement the standard odd-merge for odd n.
        # Reference: Knuth, TAOCP Vol 3.
        # Merge of A (length p) and B (length q), p >= q.
        # 1. Compare (i, i+q) for i=0, 2, ..., 2*floor(q/2)-2?
        # This is getting complicated.
        # Alternative: Use a known small network for n=3, 5, 7, 9?
        # Or just handle n=3 explicitly and for n>=5 odd, use a generic method?
        
        # Let's try to implement the merge for general n.
        # If n is odd, let m = n // 2.
        # Compare (i, i + m) for i = 0, 2, ..., 2*m - 2?
        # For n=5, m=2. Compare (0,2), (1,3)?
        # Then recursively merge even (0,2,4) and odd (1,3)?
        # Size of even subsequence is 3, odd is 2.
        
        m = n // 2
        for i in range(0, 2*m, 2):
            # i goes 0, 2, ..., 2m-2
            # i+m goes m, m+2, ..., 3m-2
            # But max index is 2m.
            # If i+m > 2m-1, skip.
            if i + m < n:
                comparators.append((i, i + m))
        
        # Recursively merge even and odd indices
        # Even indices: 0, 2, ..., n-1 (if n odd, last is even? No, 0..n-1, n is odd, so n-1 is even).
        # Number of even indices: (n+1)//2
        # Number of odd indices: n//2
        
        sub_merge_odd = batcher_merge_sort((n + 1) // 2)
        for i, j in sub_merge_odd:
            comparators.append((2*i, 2*j))
            
        sub_merge_even = batcher_merge_sort(n // 2)
        for i, j in sub_merge_even:
            comparators.append((2*i + 1, 2*j + 1))
            
    return comparators

def construct():
    # Use Batcher's odd-merge sort
    # Add a deadline guard just in case, though construction is fast.
    start_time = time.time()
    deadline = 80  # 80 seconds
    
    net = batcher_merge_sort(N)
    
    # Verify correctness on a few random inputs?
    # The task says correctness is certified by the evaluator.
    # But we should ensure it's a valid sorting network.
    # For safety, we can do a quick check on a few permutations.
    
    def simulate(net, inputs):
        arr = list(inputs)
        for i, j in net:
            if arr[i] > arr[j]:
                arr[i], arr[j] = arr[j], arr[i]
        return arr
    
    import random
    # Quick check on 100 random inputs
    for _ in range(100):
        inputs = list(range(N))
        random.shuffle(inputs)
        sorted_out = simulate(net, inputs)
        if sorted_out != list(range(N)):
            # If it fails, fall back to a known correct network (odd-even)
            # But Batcher should be correct.
            return [(i, i + 1) for phase in range(N)
                    for i in range(phase % 2, N - 1, 2)]
            
    if time.time() > deadline:
        # Should not happen
        pass
        
    return net
# EVOLVE-BLOCK-END
