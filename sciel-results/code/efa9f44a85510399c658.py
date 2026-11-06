def construct():
    MOD = 1009
    N = 1008
    
    # Precompute modular inverses for 1..N
    # inv[i] is the inverse of i mod MOD
    inv = [0] * (N + 1)
    for i in range(1, N + 1):
        inv[i] = pow(i, MOD - 2, MOD)
    
    # Identify self-inverse elements: x such that x * x % MOD == 1
    # x^2 = 1 => (x-1)(x+1) = 0 mod 1009 => x=1 or x=1008
    self_inverse = {1, 1008}
    
    best_start = 1
    best_length = 0
    left = 1
    
    # Sliding window approach
    for right in range(1, N + 1):
        # If 'right' is self-inverse, it cannot be in any valid interval
        if right in self_inverse:
            left = right + 1
            continue
        
        # Check if the inverse of 'right' is in the current window [left, right-1]
        r_inv = inv[right]
        
        # If r_inv is in [left, right-1], we must move 'left' past r_inv
        # Note: r_inv != right because right is not self-inverse
        if left <= r_inv < right:
            left = r_inv + 1
            
        # Now [left, right] is a valid interval
        current_length = right - left + 1
        if current_length > best_length:
            best_length = current_length
            best_start = left
            
    return [best_start, best_length]
