import time

def construct():
    MOD = 1009
    N = 1008
    deadline = time.time() + 60

    # Precompute modular inverses
    inv = [0] * (MOD + 1)
    for x in range(1, MOD):
        inv[x] = pow(x, MOD - 2, MOD)

    # For each R, compute the minimal L such that [L, R] is valid.
    # A window [L, R] is valid iff for every x in [L, R], inv[x] is NOT in [L, R].
    # Equivalently: for every x in [L, R], either inv[x] < L or inv[x] > R.
    #
    # Key insight: as R increases, the constraint for each existing x only gets
    # harder (inv[x] > R becomes false sooner). But actually, the constraint is:
    # inv[x] must not be in [L, R]. As R grows, the forbidden range [L, R] grows,
    # making it harder. So the minimal L for a given R is non-decreasing in R.
    #
    # For a fixed R, the minimal valid L is:
    #   L_min(R) = max over x in [L, R] of (inv[x] + 1) if inv[x] <= R, else L
    # This is circular. Instead, use the sliding window:
    #
    # Maintain L. For R from 1 to N:
    #   If inv[R] == R: L = R + 1 (self-inverse, can't include R)
    #   If L <= inv[R] < R: L = inv[R] + 1 (R's inverse is in window, must exclude it)
    #   But we also need to check: for all x in [L, R-1], is inv[x] in [L, R]?
    #   The new forbidden value is R. So we need: for all x in [L, R-1], inv[x] != R.
    #   inv[x] == R iff x == inv[R]. So we need inv[R] not in [L, R-1], which is the same check.
    #   So the sliding window is correct.

    L = 1
    best_len = 0
    best_start = 1

    for R in range(1, N + 1):
        if time.time() > deadline - 1:
            break

        if inv[R] == R:
            L = R + 1
            continue

        if L <= inv[R] < R:
            L = inv[R] + 1

        current_len = R - L + 1
        if current_len > best_len:
            best_len = current_len
            best_start = L

    return [best_start, best_len]
