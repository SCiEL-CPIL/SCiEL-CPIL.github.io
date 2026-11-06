def construct():
    p = 1009
    n = 1008
    
    # Precompute modular inverses
    inv = [0] * (p + 1)
    for x in range(1, p):
        inv[x] = pow(x, p - 2, p)
    
    best_start = 1
    best_length = 0
    left = 1
    
    for right in range(1, n + 1):
        # If 'right' is self-inverse, it cannot be part of any valid interval
        if inv[right] == right:
            left = right + 1
            continue
        
        # Check if the inverse of 'right' is in the current window [left, right-1]
        # If so, we must move 'left' past it to maintain validity
        if left <= inv[right] <= right - 1:
            left = inv[right] + 1
        
        # The interval [left, right] is now valid
        current_length = right - left + 1
        if current_length > best_length:
            best_length = current_length
            best_start = left
    
    return [best_start, best_length]
