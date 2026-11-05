"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    MOD = 1009
    N = 1008

    # Precompute modular inverses for 1..1008
    inv = [0] * (MOD + 1)
    for x in range(1, MOD):
        inv[x] = pow(x, MOD - 2, MOD)

    left = 1
    best_start = 1
    best_len = 0

    for right in range(1, N + 1):
        if right == 1:
            # 1 is self-inverse, cannot be in interval
            left = max(left, 2)
        elif right == 1008:
            # 1008 is self-inverse, cannot be in interval
            left = max(left, 1009)
        else:
            # right is not 1 or 1008
            # If inv[right] is in the current window [left, right-1], we have a conflict
            # inv[right] is in [left, right-1] iff left <= inv[right] < right
            if inv[right] < right and inv[right] >= left:
                left = inv[right] + 1
            
            # Ensure 1 is not in the window
            # If right > 1, 1 could be in [left, right] if left <= 1
            if right > 1 and left <= 1:
                left = 2
            
            # Ensure 1008 is not in the window
            # If right < 1008, 1008 is not in [left, right]
            # If right == 1008, handled above
            # So no check needed for 1008 here

        if left <= right:
            length = right - left + 1
            if length > best_len:
                best_len = length
                best_start = left

    return [best_start, best_len]
# EVOLVE-BLOCK-END
