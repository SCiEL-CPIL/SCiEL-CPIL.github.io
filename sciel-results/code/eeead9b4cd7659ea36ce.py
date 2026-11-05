import time

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill

    MOD = 1009
    N = 1008

    # Precompute modular inverses
    inv = [0] * (MOD + 1)
    for x in range(1, MOD):
        inv[x] = pow(x, MOD - 2, MOD)

    # Self-inverse elements (x^2 == 1 mod 1009): {1, 1008}
    self_inv = set()
    for x in range(1, MOD):
        if x * x % MOD == 1:
            self_inv.add(x)

    best_len = 0
    best_start = 1
    
    L = 1
    for R in range(1, MOD):
        # If R is self-inverse, it cannot be in any valid interval
        if R in self_inv:
            L = R + 1
            continue
        
        partner = inv[R]
        
        # If partner is in the current window [L, R-1], we have a conflict.
        # We must move L to exclude the partner.
        if L <= partner < R:
            L = partner + 1
            
        # The window [L, R] is now valid.
        current_len = R - L + 1
        if current_len > best_len:
            best_len = current_len
            best_start = L

        # Deadline guard
        if time.time() > DEADLINE:
            break

    return [best_start, best_len]
