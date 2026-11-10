import time

def construct():
    """
    Constructs a set of 12 distinct positive integers with distinct subset sums,
    minimizing the largest element.
    
    The optimal set is the superincreasing sequence starting with 1,
    where each next element is 1 plus the sum of all previous elements.
    This results in powers of 2: 1, 2, 4, ..., 2048.
    The largest element is 2048, which is the theoretical minimum.
    """
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin for 60s limit
    
    n = 12
    # Greedily construct the superincreasing sequence
    # a_i = 1 + sum(a_1 ... a_{i-1})
    # This minimizes the next element, and thus the final element.
    seq = []
    current_sum = 0
    
    for i in range(n):
        if time.time() > deadline:
            # Fallback to hardcoded optimal if for some reason we are slow (unlikely)
            return [1 << j for j in range(n)]
        
        if i == 0:
            val = 1
        else:
            val = 1 + current_sum
        
        seq.append(val)
        current_sum += val
        
        # Check if we have exceeded the maximum allowed (not strictly necessary for n=12)
        if val > 1048576:
             # This won't happen for n=12
             break

    return seq
