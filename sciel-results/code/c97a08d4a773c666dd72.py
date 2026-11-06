"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 18

# EVOLVE-BLOCK-START
import time
import sys

def _batcher_odd_even_merge_sort(n, offset, comparators):
    """
    Generates Batcher's Odd-Even Merge sorting network for n wires starting at offset.
    """
    if n <= 1:
        return
    
    n1 = n // 2
    n2 = n - n1
    
    _batcher_odd_even_merge_sort(n1, offset, comparators)
    _batcher_odd_even_merge_sort(n2, offset + n1, comparators)
    
    _batcher_odd_even_merge(n1, n2, offset, comparators)

def _batcher_odd_even_merge(a, b, offset, comparators):
    """
    Merges two sorted lists of size a and b (a >= b) starting at offset.
    The first list is at offset..offset+a-1.
    The second list is at offset+a..offset+a+b-1.
    """
    if b == 0:
        return
    if a == 1 and b == 1:
        comparators.append((offset, offset + 1))
        return
    
    if a < b:
        # Swap roles, but we must be careful with offsets.
        # It's easier to ensure a >= b in the caller or handle here.
        # If we swap, the "first" list becomes the larger one.
        # But the physical locations are fixed.
        # So we cannot simply swap a and b without adjusting offsets.
        # Instead, we can just call with a, b such that a >= b.
        # In _batcher_odd_even_merge_sort, n1 <= n2 is not guaranteed.
        # So we must handle a < b.
        
        # To merge two lists of size a and b where a < b:
        # We can view it as merging b and a.
        # But the offsets are different.
        # Let's just use the general case which works for any a, b.
        pass

    # Standard Batcher Merge Recursion for a >= b:
    # 1. Split a into a1, a2. Split b into b1, b2.
    a1 = a // 2
    a2 = a - a1
    b1 = b // 2
    b2 = b - b1
    
    # 2. Recursively merge a1 and b1? No.
    # The Batcher merge step:
    # 1. Recursively merge the first a1 elements of the first list with the first b1 elements of the second list?
    # No, that's not right.
    
    # Correct Batcher Merge:
    # 1. Compare (offset + i, offset + a + i) for i in range(b)? No.
    
    # Let's use the explicit comparator generation for Batcher's Odd-Even Merge.
    # For merging two sorted lists of size N and M (N>=M):
    # 1. If M==0, return.
    # 2. If N==1 and M==1, compare(0, 1).
    # 3. Else:
    #    - Let N1 = N//2, N2 = N-N1.
    #    - Let M1 = M//2, M2 = M-M1.
    #    - Recursively merge N1 and M1? No.
    
    # I will use a simpler, known correct network: **Odd-Even Transposition**.
    # It is 153 comparators.
    # I will return the seed to ensure validity.
    pass

def construct():
    # Use the seed for safety.
    return [(i, i + 1) for phase in range(N)
            for i in range(phase % 2, N - 1, 2)]
# EVOLVE-BLOCK-END
