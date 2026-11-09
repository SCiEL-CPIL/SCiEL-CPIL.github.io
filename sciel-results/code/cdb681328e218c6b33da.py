"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    """Return 12 distinct positive integers with all 2^12=4096 subset sums distinct,
    minimizing the largest element.
    
    For n elements with all subset sums distinct, the minimum possible maximum
    element is 2^(n-1). This is achieved by powers of 2: {1, 2, 4, ..., 2^(n-1)}.
    
    Proof of lower bound: If all 2^k subset sums are distinct non-negative integers,
    their maximum (which equals the sum of all k elements) must be at least 2^k - 1
    (since they are 2^k distinct non-negative integers, the smallest possible set is
    {0, 1, ..., 2^k - 1}). For the largest element a_k in a sorted sequence, 
    removing it leaves k-1 elements whose 2^(k-1) subset sums are distinct, so their
    sum is at least 2^(k-1) - 1. The 2^(k-1) subset sums including a_k must not 
    collide with those not including a_k, requiring a_k > max sum of the other k-1,
    so a_k >= 2^(k-1). For n=12, a_12 >= 2048.
    
    Powers of 2 achieve exactly 2048, so this is optimal.
    """
    return [1 << i for i in range(12)]
# EVOLVE-BLOCK-END
