def construct():
    # The problem requires 12 distinct positive integers such that all 
    # 2^12 = 4096 subset sums are distinct.
    # This is equivalent to finding a superincreasing sequence.
    # To minimize the largest element, we must minimize each element 
    # greedily: a_i > sum(a_1...a_{i-1}).
    # The minimal such sequence is powers of 2: 1, 2, 4, ..., 2^11.
    # The largest element is 2^11 = 2048.
    # This is the theoretical minimum for the largest element.
    
    n = 12
    result = []
    current_sum = 0
    for i in range(n):
        # The next element must be strictly greater than the sum of all previous elements
        # to ensure all subset sums remain distinct.
        # To minimize the largest element, we pick the smallest possible integer.
        next_val = current_sum + 1
        result.append(next_val)
        current_sum += next_val
        
    return result
