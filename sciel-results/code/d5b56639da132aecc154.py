"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

# EVOLVE-BLOCK-START

def _verify(start, length, p=1009):
    end = start + length - 1
    if start < 1 or end > 1008 or length <= 0:
        return False
    for x in range(start, end + 1):
        if (x * x) % p == 1:
            return False
        inv = pow(x, p - 2, p)
        if start <= inv <= end:
            return False
    return True

def construct():
    deadline = time.time() + 85.0
    p = 1009

    # Precompute inverse for each residue in [1, 1008]
    inv = [0] * 1009
    for x in range(1, 1009):
        inv[x] = pow(x, p - 2, p)

    # For each x, determine the "conflict partner":
    #   - if x is self-inverse (x*x % p == 1), then x conflicts with itself
    #   - otherwise, x conflicts with inv[x]
    # A window [L, R] is valid iff no x in [L,R] has its conflict partner also in [L,R].

    # Build arrays:
    #   self_inv[x] = True if x*x % p == 1
    #   partner[x] = inv[x] if not self_inv[x], else None
    self_inv = [False] * 1009
    partner = [0] * 1009
    for x in range(1, 1009):
        if inv[x] == x:
            self_inv[x] = True
        else:
            partner[x] = inv[x]

    # Two-pointer / sliding window:
    # Maintain window [L, R]. For each R, add it, then shrink L until no conflicts.
    # Track for each value v in [1,1008] whether it's in the window.
    # Track number of conflicts: a conflict exists when some x in window has partner[x] in window,
    # or x is self-inverse and in window.

    L = 1
    in_window = [False] * 1009
    conflicts = 0  # count of x in window such that (self_inv[x] or partner[x] in window)

    best_start = 2
    best_len = 1

    for R in range(1, 1009):
        # Add R
        in_window[R] = True
        if self_inv[R]:
            conflicts += 1
        elif in_window[partner[R]]:
            conflicts += 1

        # Shrink L until conflicts == 0
        while conflicts > 0 and L <= R:
            # Remove L
            if self_inv[L]:
                conflicts -= 1
            elif in_window[partner[L]]:
                conflicts -= 1
            in_window[L] = False
            L += 1

        # Now [L, R] is valid
        cur_len = R - L + 1
        if cur_len > best_len:
            best_len = cur_len
            best_start = L

        if time.time() > deadline:
            break

    # Verify and return
    if _verify(best_start, best_len):
        return [best_start, best_len]

    # Fallback
    return [2, 1]

# EVOLVE-BLOCK-END
