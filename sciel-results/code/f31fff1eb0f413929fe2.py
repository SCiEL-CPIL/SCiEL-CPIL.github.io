import time

def construct():
    p = 1009
    # Precompute modular inverses
    inv = [0] * p
    for i in range(1, p):
        inv[i] = pow(i, p - 2, p)
        
    best_start = 1
    best_len = 0
    
    # Iterate over all possible starts
    for s in range(1, p):
        # Iterate over all possible ends
        for e in range(s, p):
            # Check if adding e to the interval [s, e-1] creates a forbidden pair
            # A forbidden pair exists if inv[e] is in [s, e]
            if s <= inv[e] <= e:
                break
            # If we get here, [s, e] is valid
            current_len = e - s + 1
            if current_len > best_len:
                best_len = current_len
                best_start = s
                
    return [best_start, best_len]
