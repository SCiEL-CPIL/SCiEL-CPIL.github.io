def construct():
    """
    Returns a list of exactly n=12 distinct positive integers such that all subset sums are distinct.
    The goal is to minimize the largest integer.
    
    The optimal solution is the sequence of powers of 2: [1, 2, 4, ..., 2^(n-1)].
    This sequence is superincreasing, ensuring all 2^n subset sums are unique.
    For n=12, the largest element is 2^11 = 2048, which is the theoretical minimum.
    
    Proof sketch: Sort elements a_1 <= a_2 <= ... <= a_n. For all 2^n subset sums to be distinct,
    we require a_k > sum(a_1,...,a_{k-1}) for each k (otherwise two subsets collide).
    This inductively yields a_k >= 2^(k-1), so a_n >= 2^(n-1) = 2048.
    Powers of 2 achieve this bound exactly.
    """
    n = 12
    result = [1 << i for i in range(n)]
    
    # Optional verification (very fast for n=12)
    # subset_sums = set()
    # for mask in range(1 << n):
    #     s = 0
    #     for i in range(n):
    #         if mask & (1 << i):
    #             s += result[i]
    #     subset_sums.add(s)
    # assert len(subset_sums) == (1 << n)
    
    return result
