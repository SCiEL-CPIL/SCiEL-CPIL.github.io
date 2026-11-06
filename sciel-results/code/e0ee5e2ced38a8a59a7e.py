import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    MOD = 1009
    N = 1008

    # Precompute inverses
    inv = [0] * (N + 1)
    for i in range(1, N + 1):
        inv[i] = pow(i, MOD - 2, MOD)

    # Self-inverse elements: x^2 % MOD == 1
    self_inv = set()
    for i in range(1, N + 1):
        if i * i % MOD == 1:
            self_inv.add(i)

    best_start = 2
    best_len = 1

    e = 0

    for s in range(1, N + 1):
        if time.time() > deadline:
            break
        if s in self_inv:
            continue
        if e < s:
            e = s

        # Extend e as far as possible
        while e < N:
            if time.time() > deadline:
                break
            new_e = e + 1
            if new_e in self_inv:
                break
            # Check if inv[new_e] falls within [s, new_e]
            if s <= inv[new_e] <= new_e:
                break
            e = new_e

        length = e - s + 1
        if length > best_len:
            best_len = length
            best_start = s

    return [best_start, best_len]
