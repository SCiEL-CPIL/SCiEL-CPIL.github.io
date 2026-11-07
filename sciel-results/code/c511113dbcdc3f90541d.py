import time
import sys
import random
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    N = 256
    candidates = list(range(2, N + 1))
    num_candidates = len(candidates)

    # Precompute prime factorization
    factors = [None] * num_candidates
    for i, v in enumerate(candidates):
        x = v
        f = {}
        d = 2
        while d * d <= x:
            while x % d == 0:
                f[d] = f.get(d, 0) + 1
                x //= d
            d += 1
        if x > 1:
            f[x] = f.get(x, 0) + 1
        factors[i] = f

    # Precompute conflict: conflict[i] is a set of frozensets {j,k} such that i | j*k
    conflict = [set() for _ in range(num_candidates)]
    val_to_idx = {v: i for i, v in enumerate(candidates)}

    for j in range(num_candidates):
        for k in range(j + 1, num_candidates):
            prod = candidates[j] * candidates[k]
            for i in range(num_candidates):
                if prod % candidates[i] == 0:
                    conflict[i].add(frozenset([j, k]))

    # For each i, precompute the set of candidates that would conflict with i
    # if added to a set containing two other candidates
    # can_add(i, S) checks all pairs in S
    
    def can_add(idx_new, S_indices):
        n = len(S_indices)
        # Check: for all pairs (j, k) in S, is (j,k) in conflict[idx_new]?
        # Also check: for all j in S, is there k in S such that (idx_new, k) in conflict[j]?
        # And: for all k in S, is there j in S such that (idx_new, j) in conflict[k]?
        
        # Check new | j*k for all pairs (j,k) in S
        for i_idx in range(n):
            j = S_indices[i_idx]
            for k_idx in range(i_idx + 1, n):
                k = S_indices[k_idx]
                if frozenset([j, k]) in conflict[idx_new]:
                    return False
                # Check j | new*k
                if frozenset([idx_new, k]) in conflict[j]:
                    return False
                # Check k | new*j
                if frozenset([idx_new, j]) in conflict[k]:
                    return False
        return True

    def is_valid_set(S_indices):
        n = len(S_indices)
        for i_idx in range(n):
            i = S_indices[i_idx]
            for j_idx in range(i_idx + 1, n):
                j = S_indices[j_idx]
                for k_idx in range(j_idx + 1, n):
                    k = S_indices[k_idx]
                    if frozenset([j, k]) in conflict[i]:
                        return False
        return True

    best_S = []
    best_size = 0

    # Heuristic ordering
    sorted_indices_by_risk = sorted(range(num_candidates), key=lambda i: len(conflict[i]))
    sorted_indices_desc = sorted(range(num_candidates), key=lambda i: candidates[i], reverse=True)

    def greedy_construct(order):
        S = []
        for idx in order:
            if can_add(idx, S):
                S.append(idx)
        return S

    def local_search(S_indices, max_time=None):
        if max_time is None:
            max_time = deadline - 5.0

        S_set = set(S_indices)
        not_in_S = [i for i in range(num_candidates) if i not in S_set]

        improved = True
        while improved and time.time() < max_time:
            improved = False

            # 1. Try Additions
            random.shuffle(not_in_S)
            for idx in not_in_S:
                if can_add(idx, S_indices):
                    S_indices.append(idx)
                    S_set.add(idx)
                    not_in_S.remove(idx)
                    improved = True
                    break

            if improved:
                continue

            # 2. Try Single Swaps
            if S_indices and not_in_S:
                attempts = 0
                max_attempts = 100
                while attempts < max_attempts and time.time() < max_time:
                    i = random.choice(S_indices)
                    j = random.choice(not_in_S)

                    S_minus_i = [x for x in S_indices if x != i]
                    if can_add(j, S_minus_i):
                        S_indices = S_minus_i + [j]
                        S_set.remove(i)
                        S_set.add(j)
                        not_in_S.remove(j)
                        not_in_S.append(i)
                        improved = True
                        break
                    attempts += 1

            if improved:
                continue

            # 3. Try Double Swaps
            if len(S_indices) >= 2 and len(not_in_S) >= 2:
                attempts = 0
                max_attempts = 50
                while attempts < max_attempts and time.time() < max_time:
                    i, k = random.sample(S_indices, 2)
                    j, l = random.sample(not_in_S, 2)

                    if len({i, k, j, l}) < 4:
                        attempts += 1
                        continue

                    S_minus_ik = [x for x in S_indices if x != i and x != k]

                    if not can_add(j, S_minus_ik):
                        attempts += 1
                        continue

                    S_minus_ik_plus_j = S_minus_ik + [j]
                    if not can_add(l, S_minus_ik_plus_j):
                        attempts += 1
                        continue

                    S_indices = S_minus_ik + [j, l]
                    S_set.remove(i)
                    S_set.remove(k)
                    S_set.add(j)
                    S_set.add(l)

                    not_in_S = [x for x in not_in_S if x != j and x != l]
                    not_in_S.append(i)
                    not_in_S.append(k)

                    improved = True
                    break
                    attempts += 1

        return S_indices

    # Run multiple restarts
    rng = random.Random(42)
    num_restarts = 200

    for r in range(num_restarts):
        if time.time() > deadline - 15:
            break

        if r % 4 == 0:
            order = sorted_indices_by_risk[:]
            k = min(50, len(order))
            rng.shuffle(order[:k])
        elif r % 4 == 1:
            order = sorted_indices_desc[:]
            k = min(50, len(order))
            rng.shuffle(order[:k])
        elif r % 4 == 2:
            order = list(range(num_candidates))
            rng.shuffle(order)
        else:
            order = sorted(range(num_candidates), key=lambda i: (len(conflict[i]), -candidates[i]))
            k = min(50, len(order))
            rng.shuffle(order[:k])

        S = greedy_construct(order)

        time_left = deadline - time.time()
        search_time = max(1.0, time_left / 2.0)
        S = local_search(S, max_time=start_time + search_time)

        if len(S) > best_size:
            best_size = len(S)
            best_S = S[:]

    if not is_valid_set(best_S):
        fallback = []
        for i in sorted_indices_desc:
            if can_add(i, fallback):
                fallback.append(i)
        return [candidates[i] for i in fallback]

    result = [candidates[i] for i in best_S]
    return result

if __name__ == "__main__":
    res = construct()
    print(len(res))
