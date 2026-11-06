import time

def construct():
    MOD = 1009
    N = 1008
    
    # Precompute modular inverses
    inv = [0] * (N + 1)
    for i in range(1, N + 1):
        inv[i] = pow(i, MOD - 2, MOD)
        
    max_len = 0
    best_start = 1
    
    # Two-pointer sliding window
    # left is the start of the window
    # right is the end of the window
    # We maintain the invariant that [left, right] is valid.
    # Actually, the standard two pointer for "longest subarray satisfying property" 
    # usually moves left and right.
    # Here, for a fixed left, we find max right.
    # We can optimize by noting that if [s, e] is valid, [s, e-1] is valid.
    # If [s, e] is invalid, [s+1, e] might be valid.
    
    # Let's use the "next conflict" idea.
    # For a given start s, the interval is broken at the first e 
    # such that inv[e] is in [s, e] OR there exists x in [s, e-1] with inv[x]=e.
    # The latter is equivalent to inv[e] in [s, e-1].
    # So the condition for e to be invalid in [s, e] is s <= inv[e] <= e.
    
    # This is exactly what the previous code did.
    # I will implement it with a slight optimization: 
    # Instead of restarting the inner loop for every s, 
    # we can observe that if [s, e] is the max valid interval, 
    # then for s+1, the max valid interval must end at least at e 
    # (actually, removing the left element cannot introduce new conflicts, 
    # so the right boundary can only stay or move right).
    
    right = 1
    for left in range(1, N + 1):
        if right < left:
            right = left
            
        # Extend right as far as possible
        while right <= N:
            # Check if adding 'right' to [left, right-1] is valid
            # Conflict if inv[right] is in [left, right]
            if left <= inv[right] <= right:
                break
            right += 1
            
        # The valid interval for 'left' is [left, right-1]
        current_len = (right - 1) - left + 1
        if current_len > max_len:
            max_len = current_len
            best_start = left
            
        # If the interval is empty (length 0), we must move left
        # But we also need to ensure we don't get stuck if right doesn't move
        # However, if right == left and it's invalid, right stays, left increments.
        # If right > left, right stays or moves right.
        
    return [best_start, max_len]
