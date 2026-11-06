import time
import math
import random
import numpy as np
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 145

    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(math.sqrt(n)) + 1):
            if is_prime[i]:
                for j in range(i*i, n + 1, i):
                    is_prime[j] = False
        return [i for i, prime in enumerate(is_prime) if prime]

    primes = sieve(1000)

    candidate_sets = [
        [3, 5] + [p for p in primes if p >= 13],
        [2] + [p for p in primes if p >= 67],
        [3, 5, 7] + [p for p in primes if p >= 150],
        [2, 3] + [p for p in primes if p >= 330]
    ]

    valid_sets = []
    for P in candidate_sets:
        s = sum(Fraction(1, p) for p in P)
        if s <= 1:
            valid_sets.append(P)

    if not valid_sets:
        primes_seed = [p for p in primes if p >= 13]
        return [[p, (i + 1) % p] for i, p in enumerate(primes_seed)]

    best_L_global = 0
    best_result_global = []

    time_per_set = 30
    L_MAX = 10000

    for P in valid_sets:
        if time.time() > deadline - 5:
            break

        set_start = time.time()
        set_deadline = set_start + time_per_set

        P_sorted = sorted(P)
        n_primes = len(P_sorted)

        residues = {p: 0 for p in P_sorted}
        cover_count = np.zeros(L_MAX + 1, dtype=np.int32)

        def update_cover(p, r, delta):
            if r == 0:
                start = p
            else:
                start = r
            if start > L_MAX:
                return
            cover_count[start:L_MAX+1:p] += delta

        def get_L():
            zeros = np.nonzero(cover_count[1:L_MAX+1] == 0)[0]
            if len(zeros) == 0:
                return L_MAX
            else:
                return zeros[0]

        # Greedy initialization
        assigned_primes = set()
        for n in range(1, L_MAX + 1):
            if cover_count[n] == 0:
                chosen_p = None
                for p in P_sorted:
                    if p not in assigned_primes:
                        chosen_p = p
                        break
                if chosen_p is None:
                    break
                r = n % chosen_p
                residues[chosen_p] = r
                assigned_primes.add(chosen_p)
                update_cover(chosen_p, r, 1)
            if time.time() > set_deadline:
                break

        current_L = get_L()
        best_set_L = current_L
        best_set_residues = residues.copy()

        def adjust_L(L):
            while L < L_MAX and cover_count[L + 1] > 0:
                L += 1
            while L > 0 and cover_count[L] == 0:
                L -= 1
            return L

        T = 5.0
        T_min = 0.1
        cooling = 0.999
        no_improve_steps = 0
        max_no_improve = 2000

        while time.time() < set_deadline:
            p = random.choice(P_sorted)
            r_old = residues[p]
            r_new = random.randint(0, p - 1)

            if r_new == r_old:
                continue

            update_cover(p, r_old, -1)
            update_cover(p, r_new, 1)

            new_L = adjust_L(current_L)

            if new_L > current_L:
                current_L = new_L
                residues[p] = r_new
                no_improve_steps = 0
                if current_L > best_set_L:
                    best_set_L = current_L
                    best_set_residues = residues.copy()
            else:
                delta = new_L - current_L
                if delta == 0:
                    if random.random() < 0.5:
                        residues[p] = r_new
                        current_L = new_L
                        no_improve_steps += 1
                else:
                    prob = math.exp(delta / T)
                    if random.random() < prob:
                        residues[p] = r_new
                        current_L = new_L
                        no_improve_steps += 1
                    else:
                        update_cover(p, r_new, -1)
                        update_cover(p, r_old, 1)
                        no_improve_steps += 1

            T *= cooling
            if T < T_min:
                T = T_min

            if no_improve_steps > max_no_improve:
                T = 5.0
                no_improve_steps = 0

        if best_set_L > best_L_global:
            best_L_global = best_set_L
            best_result_global = [[p, best_set_residues[p]] for p in best_set_residues]

    if not best_result_global:
        primes_seed = [p for p in primes if p >= 13]
        return [[p, (i + 1) % p] for i, p in enumerate(primes_seed)]

    return best_result_global
