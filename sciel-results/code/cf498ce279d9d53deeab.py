import numpy as np
import time


def prime_number_theorem_partial_function():
    start_time = time.time()
    deadline = start_time + 550

    K = 200000
    budget = 0.10001

    S_list = []
    remaining = budget

    for k in range(K, 1, -1):
        inv_k = 1.0 / k
        if inv_k <= remaining:
            S_list.append(k)
            remaining -= inv_k
        if time.time() > deadline - 30:
            break

    if not S_list:
        domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
        values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
        return domain, values

    S = np.array(S_list, dtype=np.int64)
    S_sum = np.sum(1.0 / S)
    f1 = 10.0 * S_sum

    if f1 > 1.0001:
        domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
        values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
        return domain, values

    domain = np.concatenate(([1], S))
    values = np.concatenate(([f1], np.full(len(S), -10.0)))

    return domain, values
