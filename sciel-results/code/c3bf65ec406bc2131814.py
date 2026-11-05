import time
import numpy as np
import random

def construct():
    start_time = time.time()
    deadline = start_time + 135  # 15s safety margin before 150s kill

    # Generate primes <= 1000
    def get_primes(limit):
        sieve = bytearray([1]) * (limit + 1)
        sieve[0] = sieve[1] = 0
        for i in range(2, int(limit**0.5) + 1):
            if sieve[i]:
                sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
        return [i for i in range(limit + 1) if sieve[i]]

    primes = get_primes(1000)
    n_primes = len(primes)
    assert n_primes == 168

    # Search horizon
    L_max = 20000

    # Precompute positions for each prime and residue
    positions = []
    for p in primes:
        pos_list = []
        for r in range(p):
            if r == 0:
                pos_list.append(np.arange(p, L_max + 1, p))
            else:
                pos_list.append(np.arange(r, L_max + 1, p))
        positions.append(pos_list)

    def find_L(coverage):
        """Find largest L such that coverage[x] >= 2 for all x in [1, L]."""
        arr = coverage[1:L_max + 1]
        if arr.min() >= 2:
            return L_max
        first_gap = np.argmax(arr < 2)
        return int(first_gap)

    def fast_find_L(coverage, current_L):
        """Find largest L such that coverage[x] >= 2 for all x in [1, L], starting from current_L."""
        if current_L >= L_max:
            return L_max
        # Check if current_L still valid
        if coverage[1:current_L + 1].min() < 2:
            return find_L(coverage)
        # Extend from current_L
        x = current_L + 1
        while x <= L_max and coverage[x] >= 2:
            x += 1
        return x - 1

    # Greedy initialization
    residues = [0] * n_primes
    coverage = np.zeros(L_max + 1, dtype=np.int32)

    for i, p in enumerate(primes):
        best_r = 0
        best_gain = -1
        for r in range(p):
            pos = positions[i][r]
            mask = pos <= 500
            sub_pos = pos[mask]
            gain = np.sum(coverage[sub_pos] < 2)
            if gain > best_gain:
                best_gain = gain
                best_r = r
        residues[i] = best_r
        coverage[positions[i][best_r]] += 1

    current_L = find_L(coverage)
    best_L = current_L
    best_residues = residues[:]

    # Simulated Annealing
    T = 15.0
    T_min = 0.01
    alpha = 0.9998
    rng = random.Random(42)

    while time.time() < deadline:
        if T < T_min:
            break

        i = rng.randrange(n_primes)
        p = primes[i]
        r_old = residues[i]
        r_new = rng.randrange(p)
        if r_old == r_new:
            continue

        old_pos = positions[i][r_old]
        new_pos = positions[i][r_new]

        # Apply
        coverage[old_pos] -= 1
        coverage[new_pos] += 1
        residues[i] = r_new

        new_L = fast_find_L(coverage, current_L)

        if new_L >= current_L:
            current_L = new_L
            if new_L > best_L:
                best_L = new_L
                best_residues = residues[:]
        else:
            delta = current_L - new_L
            accept = False
            if delta < T * 10:
                if rng.random() < np.exp(-delta / T):
                    accept = True
            if not accept:
                coverage[old_pos] += 1
                coverage[new_pos] -= 1
                residues[i] = r_old

        T *= alpha

    # Final check
    cur_L = find_L(coverage)
    if cur_L > best_L:
        best_L = cur_L
        best_residues = residues[:]

    return [int(r) for r in best_residues]
