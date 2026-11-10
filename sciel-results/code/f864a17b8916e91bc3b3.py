import numpy as np

def construct():
    import time
    start_time = time.time()
    deadline = start_time + 85

    n_max = 10000
    is_prime = bytearray(b'\x01') * (n_max + 1)
    is_prime[0] = is_prime[1] = 0
    for i in range(2, int(n_max**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))

    primes = np.array([p for p in range(2, n_max + 1) if is_prime[p]], dtype=np.int32)
    n = len(primes)
    max_gap = n_max

    f = np.zeros((n, max_gap + 1), dtype=np.int32)
    F = np.zeros((n, max_gap + 1), dtype=np.int32)
    parent = np.full((n, max_gap + 1), -1, dtype=np.int32)

    for j in range(n):
        if j == 0:
            F[0] = 1
            continue
        for i in range(j):
            g = int(primes[j] - primes[i])
            if g <= max_gap:
                candidate = 1 + int(F[i][g])
                if candidate > f[j][g]:
                    f[j][g] = candidate
                    parent[j][g] = i
        F[j] = np.maximum.accumulate(f[j])
        F[j] = np.maximum(F[j], 1)

    flat_idx = np.argmax(f)
    best_j, best_g = np.unravel_index(flat_idx, f.shape)
    best_len = int(f[best_j, best_g])

    if best_len <= 1:
        return [int(primes[0])]

    result = []
    j = int(best_j)
    g = int(best_g)
    while j >= 0:
        result.append(int(primes[j]))
        p = int(parent[j, g])
        if p == -1:
            break
        prev_g = int(primes[j] - primes[p])
        j = p
        g = prev_g
    result.reverse()
    return result
