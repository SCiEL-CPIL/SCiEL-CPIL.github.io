import time

def construct():
    p = 1009
    # Precompute inverses
    inv = [0] * (p + 1)
    for x in range(1, p):
        inv[x] = pow(x, p - 2, p)
    
    best_start = 2
    best_length = 1  # [2,2] is valid since 2*2=4 != 1 mod 1009
    
    deadline = time.time() + 80
    
    # Two-pointer: for each left, find max right
    for left in range(1, p):
        if time.time() > deadline:
            break
        # Window must not contain 1 or 1008
        if left == 1 or left == 1008:
            continue
        right = left - 1
        # Extend right as far as possible
        while right + 1 < p:  # right+1 <= 1008
            candidate = right + 1
            if candidate == 1 or candidate == 1008:
                right = candidate  # skip this position, can't include it
                continue
            # Check if [left, candidate] is valid
            valid = True
            for x in range(left, candidate + 1):
                ix = inv[x]
                if left <= ix <= candidate:
                    valid = False
                    break
            if valid:
                right = candidate
                length = right - left + 1
                if length > best_length:
                    best_length = length
                    best_start = left
            else:
                # Can't extend further
                break
        # Also check the window ending before 1008 if we stopped early
    
    return [best_start, best_length]
