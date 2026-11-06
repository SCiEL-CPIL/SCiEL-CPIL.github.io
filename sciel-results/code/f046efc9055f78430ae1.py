import time
from math import isqrt
import bisect

def construct():
    start_time = time.time()
    deadline = start_time + 50.0

    N_LIMIT = 10000

    # 1. Sieve of Eratosthenes
    is_prime = [True] * (N_LIMIT + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, isqrt(N_LIMIT) + 1):
        if is_prime[i]:
            for j in range(i*i, N_LIMIT + 1, i):
                is_prime[j] = False

    primes = [i for i, v in enumerate(is_prime) if v]
    n_primes = len(primes)

    if n_primes < 2:
        return primes

    # 2. DP with suffix max + two-pointer window
    # D[i][j] = max length of valid sequence ending with (primes[i], primes[j])
    D = [[0] * n_primes for _ in range(n_primes)]
    prev = [[-1] * n_primes for _ in range(n_primes)]

    best_len = 2
    best_i = 0
    best_j = 1

    for i in range(n_primes - 1):
        if time.time() > deadline:
            break

        # Build suffix max arrays for column i (rows 0..i-1)
        if i > 0:
            M_val = [0] * i
            M_idx = [-1] * i
            M_val[i - 1] = D[i - 1][i]
            M_idx[i - 1] = i - 1
            for k in range(i - 2, -1, -1):
                if D[k][i] > M_val[k + 1]:
                    M_val[k] = D[k][i]
                    M_idx[k] = k
                else:
                    M_val[k] = M_val[k + 1]
                    M_idx[k] = M_idx[k + 1]
        else:
            M_val = None
            M_idx = None

        p_i = primes[i]

        if i > 0:
            # Two-pointer: for each j, find smallest k such that
            # primes[k] >= p_i - (primes[j] - p_i)  i.e. primes[i] - primes[k] <= primes[j] - primes[i]
            k_min = 0
            for j in range(i + 1, n_primes):
                gap = primes[j] - p_i
                threshold = p_i - gap

                while k_min < i and primes[k_min] < threshold:
                    k_min += 1

                if k_min < i:
                    best_prev_len = M_val[k_min]
                    best_k = M_idx[k_min]
                    new_len = best_prev_len + 1
                    D[i][j] = new_len
                    prev[i][j] = best_k
                    if new_len > best_len:
                        best_len = new_len
                        best_i = i
                        best_j = j
                else:
                    D[i][j] = 2
                    prev[i][j] = -1
        else:
            for j in range(i + 1, n_primes):
                D[i][j] = 2
                prev[i][j] = -1

    # Reconstruct sequence
    if best_len < 2:
        return primes[:best_len]

    seq = [primes[best_j], primes[best_i]]
    cur_i = best_i
    cur_j = best_j

    while prev[cur_i][cur_j] != -1:
        k = prev[cur_i][cur_j]
        seq.append(primes[k])
        cur_j = cur_i
        cur_i = k

    seq.reverse()
    return seq
