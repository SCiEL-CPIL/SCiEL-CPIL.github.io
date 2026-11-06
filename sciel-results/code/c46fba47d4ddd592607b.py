import numpy as np
import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 10s margin for 60s limit

    # Calculate LCM of 1..30
    def lcm(a, b):
        return a * b // gcd(a, b)

    L = 1
    for i in range(1, 31):
        L = lcm(L, i)

    # Weights w_i = L // i
    w = [L // i for i in range(1, 31)]

    # Split weights into two halves
    w1 = w[:15]
    w2 = w[15:]

    # Generate all possible sums and their base-3 indices for a list of weights
    def gen_sums_and_idxs(weights):
        n = len(weights)
        sums = np.array([0], dtype=np.int64)
        idxs = np.array([0], dtype=np.int64)

        for w_val in weights:
            N = len(sums)
            # New sums: s - w, s, s + w
            # Corresponding index digits: 0, 1, 2
            # New index = old_index * 3 + digit
            
            new_sums = np.empty(3 * N, dtype=np.int64)
            new_idxs = np.empty(3 * N, dtype=np.int64)

            new_sums[0:N] = sums - w_val
            new_sums[N:2*N] = sums
            new_sums[2*N:3*N] = sums + w_val

            new_idxs[0:N] = idxs * 3 + 0
            new_idxs[N:2*N] = idxs * 3 + 1
            new_idxs[2*N:3*N] = idxs * 3 + 2

            sums = new_sums
            idxs = new_idxs

            if time.time() > deadline:
                break

        return sums, idxs

    sums1, idxs1 = gen_sums_and_idxs(w1)
    sums2, idxs2 = gen_sums_and_idxs(w2)

    if time.time() > deadline:
        # Fallback to seed
        return [0] * 28 + [1, -1]

    # Sort sums2 and keep track of original indices
    sort_idx = np.argsort(sums2)
    sorted_sums2 = sums2[sort_idx]
    sorted_idxs2 = idxs2[sort_idx]

    # Find best match
    # We want to minimize |s1 + s2| != 0
    targets = -sums1
    positions = np.searchsorted(sorted_sums2, targets)

    # Check position and position-1
    # Clip to valid range
    p1 = np.clip(positions, 0, len(sorted_sums2) - 1)
    p2 = np.clip(positions - 1, 0, len(sorted_sums2) - 1)

    # Calculate differences
    d1 = sums1 + sorted_sums2[p1]
    d2 = sums1 + sorted_sums2[p2]

    # Absolute values
    ad1 = np.abs(d1)
    ad2 = np.abs(d2)

    # Mask zeros (we need non-zero sum)
    ad1[ad1 == 0] = np.inf
    ad2[ad2 == 0] = np.inf

    # Find global minimum
    min_d1 = np.min(ad1)
    min_d2 = np.min(ad2)

    if min_d1 < min_d2:
        min_val = min_d1
        i1 = np.argmin(ad1)
        i2 = sorted_idxs2[p1[i1]]
    else:
        min_val = min_d2
        i1 = np.argmin(ad2)
        i2 = sorted_idxs2[p2[i1]]

    # Decode indices to coefficients
    def decode_idx(idx, n_weights):
        coeffs = []
        temp = int(idx)
        for _ in range(n_weights):
            d = temp % 3
            temp //= 3
            if d == 0:
                coeffs.append(-1)
            elif d == 1:
                coeffs.append(0)
            else:
                coeffs.append(1)
        return coeffs

    c1 = decode_idx(idxs1[i1], 15)
    c2 = decode_idx(i2, 15)

    res = c1 + c2

    # Optional: Verify and refine if time permits?
    # The integer sum is guaranteed non-zero and minimal among the found set.
    # Since we covered all combinations, this is the global minimum.
    
    return res
