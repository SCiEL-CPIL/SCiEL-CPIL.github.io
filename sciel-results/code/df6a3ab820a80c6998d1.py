import time
import numpy as np
from fractions import Fraction
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    def get_primes(limit):
        sieve = bytearray(b'\x01') * (limit + 1)
        sieve[0] = sieve[1] = 0
        for i in range(2, int(limit**0.5) + 1):
            if sieve[i]:
                for j in range(i*i, limit + 1, i):
                    sieve[j] = 0
        return [i for i in range(2, limit + 1) if sieve[i]]

    primes = get_primes(1000)

    N = 10000

    # Generate diverse candidate prime sets
    candidate_sets = []

    # Systematic: try all subsets of small primes that fit budget
    small_primes = [p for p in primes if p <= 50]
    from itertools import combinations
    for size in range(1, 8):
        for combo in combinations(small_primes, size):
            s = Fraction(0, 1)
            for p in combo:
                s += Fraction(1, p)
            if s <= 1:
                candidate_sets.append(list(combo))

    # Greedy variants
    for skip in [set(), {2}, {3}, {2, 3}, {5}, {7}]:
        ps = []
        s = Fraction(0, 1)
        for p in primes:
            if p in skip:
                continue
            if s + Fraction(1, p) <= 1:
                ps.append(p)
                s += Fraction(1, p)
        if ps:
            candidate_sets.append(ps)

    # Random sets
    random.seed(42)
    for _ in range(20):
        ps = []
        s = Fraction(0, 1)
        shuffled = primes[:]
        random.shuffle(shuffled)
        for p in shuffled:
            if s + Fraction(1, p) <= 1:
                ps.append(p)
                s += Fraction(1, p)
        if ps:
            candidate_sets.append(ps)

    # Deduplicate
    unique_sets = []
    seen = set()
    for ps in candidate_sets:
        key = tuple(sorted(ps))
        if key not in seen:
            seen.add(key)
            unique_sets.append(ps)

    best_L = 0
    best_classes = []

    def optimize_residues_numpy(prime_list, time_limit, rng_seed=0):
        if not prime_list:
            return 0, []

        k = len(prime_list)
        rng = random.Random(rng_seed)

        # Precompute index arrays using numpy
        # For each prime i and residue r, the indices are arange(r if r>0 else p, N+1, p)
        # Store as list of numpy arrays
        prime_indices = []
        for p in prime_list:
            indices_for_p = []
            for r in range(p):
                if r == 0:
                    idxs = np.arange(p, N + 1, p)
                else:
                    idxs = np.arange(r, N + 1, p)
                indices_for_p.append(idxs)
            prime_indices.append(indices_for_p)

        # Initialize residues
        residues = [rng.randint(0, p - 1) for p in prime_list]

        # Build coverage using numpy
        counts = np.zeros(N + 1, dtype=np.int32)
        for i, p in enumerate(prime_list):
            r = residues[i]
            idxs = prime_indices[i][r]
            if len(idxs) > 0:
                counts[idxs] += 1

        covered = (counts > 0).astype(np.int8)

        def find_L(covered_arr):
            # Find first zero in covered_arr[1:]
            zeros = np.where(covered_arr[1:] == 0)[0]
            if len(zeros) == 0:
                return N
            return zeros[0]

        L = find_L(covered)
        best_L_local = L
        best_residues = residues[:]

        T = 10.0
        T_min = 0.01
        cooling = 0.999
        steps = 0

        while time.time() < time_limit and T > T_min:
            i = rng.randint(0, k - 1)
            p = prime_list[i]
            old_r = residues[i]

            # Targeted move: try to cover L+1
            if L < N and rng.random() < 0.85:
                new_r = (L + 1) % p
            else:
                new_r = rng.randint(0, p - 1)

            if new_r == old_r:
                continue

            # Apply move using numpy
            old_idxs = prime_indices[i][old_r]
            new_idxs = prime_indices[i][new_r]

            if len(old_idxs) > 0:
                counts[old_idxs] -= 1
            if len(new_idxs) > 0:
                counts[new_idxs] += 1

            # Update covered
            # Positions where count went from 0 to 1 or 1 to 0
            # Simpler: recompute covered for affected positions
            affected = np.unique(np.concatenate([old_idxs, new_idxs])) if (len(old_idxs) > 0 or len(new_idxs) > 0) else np.array([], dtype=np.int32)
            if len(affected) > 0:
                covered[affected] = (counts[affected] > 0).astype(np.int8)

            residues[i] = new_r

            new_L = find_L(covered)

            if new_L > L or rng.random() < math.exp((new_L - L) / T):
                L = new_L
                if L > best_L_local:
                    best_L_local = L
                    best_residues = residues[:]
            else:
                # Revert
                if len(new_idxs) > 0:
                    counts[new_idxs] -= 1
                if len(old_idxs) > 0:
                    counts[old_idxs] += 1
                affected = np.unique(np.concatenate([old_idxs, new_idxs])) if (len(old_idxs) > 0 or len(new_idxs) > 0) else np.array([], dtype=np.int32)
                if len(affected) > 0:
                    covered[affected] = (counts[affected] > 0).astype(np.int8)
                residues[i] = old_r

            T *= cooling
            steps += 1

        classes = [[prime_list[i], best_residues[i]] for i in range(k)]
        return best_L_local, classes

    # Run optimization for each set with time allocation
    n_sets = len(unique_sets)
    time_per_set = (deadline - start_time - 5.0) / n_sets

    for idx, ps in enumerate(unique_sets):
        if time.time() > deadline - 5:
            break
        t_limit = time.time() + min(time_per_set, deadline - time.time() - 5)
        L, classes = optimize_residues_numpy(ps, t_limit, rng_seed=idx * 1000 + 42)
        if L > best_L:
            best_L = L
            best_classes = classes

    # Final verification
    if best_classes:
        s = Fraction(0, 1)
        for p, r in best_classes:
            s += Fraction(1, p)
        if s <= 1:
            return best_classes
    return [[2, 0]]
