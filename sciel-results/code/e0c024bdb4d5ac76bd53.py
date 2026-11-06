import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    MOD = 1009
    N = 1008

    # Precompute modular inverses
    # inv[i] is the modular inverse of i modulo 1009
    inv = [0] * (N + 1)
    for i in range(1, N + 1):
        inv[i] = pow(i, MOD - 2, MOD)

    # Self-inverse elements: x^2 == 1 mod 1009 => x=1, x=1008
    # These elements conflict with themselves.
    
    # Sliding window approach
    # in_window[i] is True if i is in the current window [L, R]
    in_window = [False] * (N + 1)
    
    # conflicts counts the number of pairs {x, inv[x]} that are fully contained in the window.
    # If x == inv[x], it counts as 1 conflict if x is in the window.
    # If x != inv[x], it counts as 1 conflict if both x and inv[x] are in the window.
    # We want conflicts == 0.
    
    conflicts = 0
    L = 1
    best_length = 0
    best_start = 1
    
    for R in range(1, N + 1):
        # Add R to the window
        in_window[R] = True
        
        # Check if adding R creates a new conflict
        # Case 1: R is self-inverse
        if R == inv[R]:
            conflicts += 1
        else:
            # Case 2: R's inverse is already in the window
            if in_window[inv[R]]:
                conflicts += 1
        
        # If there are conflicts, shrink the window from the left until valid
        while conflicts > 0:
            # Remove L from the window
            # Check if removing L resolves a conflict
            if L == inv[L]:
                # L was self-conflicting, removing it removes that conflict
                conflicts -= 1
            else:
                # L's inverse is in the window (excluding L itself, since we're removing L)
                # If inv[L] is in [L+1, R], then the pair {L, inv[L]} was a conflict, 
                # and removing L breaks it.
                if in_window[inv[L]]:
                    conflicts -= 1
            
            in_window[L] = False
            L += 1
        
        # Now the window [L, R] is valid
        current_length = R - L + 1
        if current_length > best_length:
            best_length = current_length
            best_start = L
            
        # Time check (though O(N) is very fast, this is a safety net)
        if time.time() > deadline:
            break

    if best_length == 0:
        # Fallback: any single element that is not self-inverse
        for x in range(1, N + 1):
            if x != 1 and x != 1008:
                best_start = x
                best_length = 1
                break
        if best_length == 0:
            # Should not happen for N=1008, but just in case
            best_start = 1
            best_length = 1

    return [best_start, best_length]
