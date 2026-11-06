import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s wall-clock limit

    MOD = 1009
    N = 1008

    # Precompute modular inverses: inv[i] = i^(-1) mod MOD
    inv = [0] * (N + 1)
    for i in range(1, N + 1):
        inv[i] = pow(i, MOD - 2, MOD)

    best_start = 1
    best_len = 0
    s = 1

    for e in range(1, N + 1):
        ie = inv[e]
        
        # Check for conflict with the new element e
        if ie == e:
            # Self-inverse: e cannot be in the interval
            s = e + 1
        elif s <= ie < e:
            # Conflict: inv[e] is in the current window [s, e-1]
            # Move start past inv[e]
            s = ie + 1
        # If ie > e or ie < s, no conflict introduced by e with the window [s, e-1]
        # Note: if ie > e, it's outside the window to the right.
        # Note: if ie < s, it's outside the window to the left.

        if s <= e:
            length = e - s + 1
            if length > best_len:
                best_len = length
                best_start = s

        # Check deadline
        if time.time() > deadline:
            break

    if best_len == 0:
        # Fallback: Should not happen, but just in case
        # Find any valid single element
        for i in range(1, N + 1):
            if inv[i] != i:
                return [i, 1]
        return [1, 1]

    return [best_start, best_len]
