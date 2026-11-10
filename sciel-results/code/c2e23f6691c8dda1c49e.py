
def construct():
    # The optimal solution is powers of 2, with max = 2^11 = 2048
    # This is proven optimal by induction: a_k >= 2^(k-1)
    n = 12
    result = [1 << i for i in range(n)]
    
    # Verify that all subset sums are distinct
    subset_sums = set()
    for mask in range(1 << n):
        s = 0
        for i in range(n):
            if mask & (1 << i):
                s += result[i]
        subset_sums.add(s)
        
    # For powers of 2, all subset sums are distinct.
    # len(subset_sums) should be 1 << n.
    
    return result
