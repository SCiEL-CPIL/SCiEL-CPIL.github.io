import numpy as np

def construct():
    MOD = 1009
    N = 1008
    
    # Precompute modular inverses
    inv = [0] * (MOD)
    for x in range(1, MOD):
        inv[x] = pow(x, MOD - 2, MOD)
    
    # Convert to numpy array for fast slicing and operations
    # Using int32 is sufficient and faster
    inv_arr = np.array(inv, dtype=np.int32)
    
    best_len = 0
    best_start = 2
    
    # The interval must be contained in [1, 1008].
    # Self-inverse elements are 1 and 1008 (since 1008 = -1 mod 1009).
    # If 1 is in the interval, x=1, y=1 gives 1*1=1, which is invalid.
    # If 1008 is in the interval, x=1008, y=1008 gives 1008*1008 = 1, invalid.
    # Thus, the interval cannot contain 1 or 1008.
    # So L >= 2 and R <= 1007.
    
    for L in range(2, 1008):
        # Binary search for the maximum valid R for this L
        lo = L
        hi = 1007
        max_valid_R = L - 1
        
        while lo <= hi:
            mid = (lo + hi) // 2
            
            # Get inverses of all x in [L, mid]
            # inv_arr is 1-indexed logically (index 0 is 0, index 1 is inv(1), etc.)
            # So slice [L, mid] corresponds to indices L to mid in the array
            x_vals = inv_arr[L:mid+1]
            
            # Check if any inverse falls within [L, mid]
            # If inv[x] is in [L, mid], then x and inv[x] are both in the interval,
            # which violates the condition.
            # Condition for validity: No x in [L, mid] has inv[x] in [L, mid].
            
            # Using numpy boolean operations
            # (x_vals >= L) & (x_vals <= mid) finds the "conflicts"
            # If any exist, the interval is invalid.
            
            if not np.any((x_vals >= L) & (x_vals <= mid)):
                max_valid_R = mid
                lo = mid + 1
            else:
                hi = mid - 1
        
        if max_valid_R >= L:
            length = max_valid_R - L + 1
            if length > best_len:
                best_len = length
                best_start = L
                
    return [best_start, best_len]
