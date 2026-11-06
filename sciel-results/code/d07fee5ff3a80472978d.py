import time


def construct():
    p = 1009
    deadline = time.time() + 85.0

    # Precompute modular inverses for 1..1008
    inv = [0] * (p + 1)
    is_self_inverse = [False] * (p + 1)
    for x in range(1, p):
        inv[x] = pow(x, p - 2, p)
        if (x * x) % p == 1:
            is_self_inverse[x] = True

    best_start = 2
    best_length = 1

    # For each possible start, greedily extend the interval as far as valid
    for start in range(1, p):
        if p - start <= best_length:
            break
        if time.time() > deadline:
            break

        present = [0] * (p + 1)
        length = 0

        for x in range(start, p):
            if is_self_inverse[x]:
                break
            if present[inv[x]]:
                break
            present[x] = 1
            length += 1

        if length > best_length:
            best_length = length
            best_start = start

    return [best_start, best_length]
