import time

def construct():
    """
    Constructs a set of 12 distinct positive integers such that all 4096 subset 
    sums are distinct, and the largest integer is minimized.
    
    The optimal solution is the superincreasing sequence 1, 2, 4, ..., 2^(n-1).
    This sequence minimizes the largest element for a given n.
    We construct it iteratively.
    """
    n = 12
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin
    
    # The greedy choice a_k = sum(a_1...a_{k-1}) + 1 is always valid and optimal.
    # a_1 = 1
    # a_2 = 1 + 1 = 2
    # a_3 = 1 + 2 + 1 = 4
    # ...
    # a_k = 2^(k-1)
    
    elements = []
    current_sum = 0
    
    for i in range(n):
        # Check time just in case, though this loop is O(n)
        if time.time() > deadline:
            break
            
        next_elem = current_sum + 1
        elements.append(next_elem)
        current_sum += next_elem
        
    # If we didn't finish due to time (highly unlikely), fallback
    if len(elements) < n:
        return [1 << i for i in range(n)]
        
    return elements

# EVOLVE-BLOCK-END
