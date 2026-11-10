import time
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = 55.0  # Safety margin before 60s limit

    limit = 1000000
    max_val = limit + 1

    # Sieve to compute the largest prime factor (LPF) for every number up to max_val.
    # lpf[i] will hold the largest prime factor of i.
    # We iterate p from 2 to max_val; if lpf[p] is still 0, p is prime,
    # and we set lpf[j] = p for all multiples j of p.
    # Since we go in increasing p, the last assignment gives the largest prime factor.
    lpf = np.zeros(max_val + 1, dtype=np.int64)

    for p in range(2, max_val + 1):
        if lpf[p] == 0:  # p is prime
            # Use NumPy for faster assignment to multiples
            lpf[p::p] = p

        # Periodic time check to avoid exceeding deadline
        if p % 100000 == 0:
            if time.time() - start_time > deadline:
                break

    # Convert to Python list for faster indexing in the scan loop
    lpf_list = lpf.tolist()

    # Scan for the minimum max(LPF(n), LPF(n+1)) over n in [10000, 1000000]
    best_n = 10000
    min_score = max(lpf_list[10000], lpf_list[10001])

    for n in range(10001, limit + 1):
        a = lpf_list[n]
        b = lpf_list[n + 1]
        score = a if a > b else b
        if score < min_score:
            min_score = score
            best_n = n

        # Periodic time check to avoid exceeding deadline
        if (n - 10001) % 50000 == 0:
            if time.time() - start_time > deadline:
                break

    return best_n
