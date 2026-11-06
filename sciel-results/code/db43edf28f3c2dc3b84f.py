import time

def construct():
    start_time = time.time()
    deadline = start_time + 45.0

    MOD = 1009
    N = 1008

    # Precompute modular inverses via Fermat's little theorem
    inv = [0] * (N + 1)
    for i in range(1, N + 1):
        inv[i] = pow(i, MOD - 2, MOD)

    best_start = 1
    best_len = 0

    left = 1

    for right in range(1, N + 1):
        if time.time() > deadline:
            break

        x = right
        y = inv[x]

        if x == y:
            # Self-inverse residues (1 and 1008) cannot appear in any valid interval
            left = right + 1
            continue

        # If y lies strictly inside the current window [left, right-1],
        # the pair (y, x) would violate the constraint; advance left past y.
        if left <= y < right:
            left = y + 1

        current_len = right - left + 1
        if current_len > best_len:
            best_len = current_len
            best_start = left

    return [best_start, best_len]
