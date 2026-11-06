def construct():
    n = 12
    # The theoretical minimum for the largest element in a set of n distinct 
    # positive integers with distinct subset sums is 2^(n-1).
    # This is achieved by the powers of 2: 1, 2, 4, ..., 2^(n-1).
    # Proof: Let a_1 < a_2 < ... < a_n. 
    # For subset sums to be distinct, a_{k+1} must be strictly greater than 
    # the sum of all previous elements a_1 + ... + a_k.
    # Thus a_{k+1} >= sum(a_1..a_k) + 1.
    # This implies a_k >= 2^(k-1).
    # Therefore a_n >= 2^(n-1).
    
    result = [1 << i for i in range(n)]
    return result
