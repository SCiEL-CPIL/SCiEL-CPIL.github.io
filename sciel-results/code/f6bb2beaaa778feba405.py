import numpy as np
import time
import random

def construct():
    n = 256
    start_time = time.time()
    deadline = start_time + 50

    # Generate all APs within [0, n)
    aps = []
    for d in range(1, n):
        for a in range(d):
            seq = list(range(a, n, d))
            if len(seq) >= 2:
                aps.append(seq)

    m = len(aps)

    # Build membership matrix: member[j, i] = 1 if index i is in AP j
    # This allows O(1) flip impact evaluation via matrix-row access
    member = np.zeros((m, n), dtype=np.int8)
    for j, ap in enumerate(aps):
        for idx in ap:
            member[j, idx] = 1

    # Greedy initialization
    signs = np.zeros(n, dtype=np.int8)
    ap_sums = np.zeros(m, dtype=np.int16)

    for i in range(n):
        best_sign = 1
        best_local_max = float('inf')
        col = member[:, i]
        affected = np.nonzero(col)[0]
        if len(affected) == 0:
            signs[i] = 1
            continue
        for s in [1, -1]:
            new_sums = ap_sums[affected] + s
            local_max = np.max(np.abs(new_sums))
            if local_max < best_local_max:
                best_local_max = local_max
                best_sign = s
        signs[i] = best_sign
        ap_sums[affected] += best_sign

    current_max = int(np.max(np.abs(ap_sums)))
    best_signs = signs.copy()
    best_max = current_max

    # Simulated Annealing
    T = 3.0
    T_min = 0.1
    cooling = 0.9997

    while time.time() < deadline - 3:
        if T < T_min:
            T = T_min

        i = random.randint(0, n - 1)
        col = member[:, i]
        affected = np.nonzero(col)[0]
        if len(affected) == 0:
            continue

        old_sign = signs[i]
        new_sign = -old_sign
        signs[i] = new_sign
        ap_sums[affected] += 2 * new_sign

        new_max = int(np.max(np.abs(ap_sums)))
        delta = new_max - current_max

        if delta < 0:
            current_max = new_max
            if current_max < best_max:
                best_max = current_max
                best_signs = signs.copy()
        else:
            # Accept with probability exp(-delta / T)
            if delta == 0:
                accept = True
            else:
                accept = random.random() < np.exp(-delta / T)
            if not accept:
                ap_sums[affected] -= 2 * new_sign
                signs[i] = old_sign
                continue

        T *= cooling

    return best_signs.tolist()
