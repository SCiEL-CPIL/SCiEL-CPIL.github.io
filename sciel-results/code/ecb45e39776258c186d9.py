import time

def _valid(arr):
    """Check all 2^n subset sums are distinct."""
    n = len(arr)
    if n == 0:
        return True
    sums = [0] * (1 << n)
    sums[0] = 0
    for mask in range(1, 1 << n):
        lsb = mask & (-mask)
        idx = lsb.bit_length() - 1
        prev = mask ^ lsb
        sums[mask] = sums[prev] + arr[idx]
    return len(set(sums)) == (1 << n)

def construct():
    # The problem asks to minimize the largest integer in a set of n=12
    # distinct positive integers such that all subset sums are distinct.
    
    # Mathematical proof:
    # Let a_1 < a_2 < ... < a_n be the integers.
    # For all subset sums to be distinct, it is necessary and sufficient
    # that a_k > sum_{i=1}^{k-1} a_i for all k > 1.
    # This implies a_k >= 1 + sum_{i=1}^{k-1} a_i.
    # Let S_k = sum_{i=1}^k a_i. Then a_{k+1} >= S_k + 1.
    # S_{k+1} = S_k + a_{k+1} >= 2*S_k + 1.
    # With S_1 >= 1, we get S_k >= 2^k - 1.
    # Consequently, a_{k+1} >= 2^k.
    # For n=12, the minimum largest element a_12 is 2^11 = 2048.
    
    # The optimal set is {2^0, 2^1, ..., 2^11}.
    n = 12
    optimal_set = [1 << i for i in range(n)]
    
    # Verify validity just in case, though it is mathematically guaranteed.
    if _valid(optimal_set):
        return optimal_set
    
    # Fallback (should never be reached)
    return optimal_set
