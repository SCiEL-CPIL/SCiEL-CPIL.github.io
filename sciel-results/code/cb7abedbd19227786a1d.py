import random
import time
import numpy as np


def _greedy_build(N, order, deadline):
    """Build a B2[2] set using greedy with given order."""
    S = set()
    counts = np.zeros(2 * N + 1, dtype=np.int32)

    for x in order:
        if time.time() > deadline:
            break
        # Check if x can be added
        valid = True
        if counts[2 * x] >= 2:
            valid = False
        else:
            # Vectorized check: x + y for all y in S
            if S:
                S_arr_tmp = np.array(list(S), dtype=np.int32)
                sums = x + S_arr_tmp
                if np.any(counts[sums] >= 2):
                    valid = False
        if valid:
            S.add(x)
            counts[2 * x] += 1
            if S:
                S_arr_tmp = np.array(list(S), dtype=np.int32)
                # x + y for y in S, y != x
                mask = S_arr_tmp != x
                if np.any(mask):
                    sums = x + S_arr_tmp[mask]
                    counts[sums] += 1

    return S, counts


def _local_search(S, counts, N, deadline, seed=42):
    """Perform local search to improve the B2[2] set using vectorized checks."""
    random.seed(seed)
    current_S = set(S)
    S_arr = np.array(sorted(current_S), dtype=np.int32)
    current_counts = counts.copy()
    best_S = frozenset(current_S)
    best_size = len(current_S)

    # Track which elements are in S for O(1) membership
    in_S = np.zeros(N + 1, dtype=bool)
    for x in current_S:
        in_S[x] = True

    # Candidate pool: all elements not in S
    # We'll sample from a precomputed list of non-members

    it = 0
    while time.time() < deadline:
        it += 1
        if it % 1000 == 0 and time.time() > deadline:
            break

        # Try to add a random element from candidates
        available_mask = ~in_S
        available_mask[0] = False  # 0 is not in 1..N
        available_indices = np.where(available_mask)[0]
        if len(available_indices) == 0:
            break

        y = int(random.choice(available_indices))

        # Check if y can be added: 2*y and y+z for z in S
        valid = True
        if current_counts[2 * y] >= 2:
            valid = False
        elif len(S_arr) > 0:
            sums = y + S_arr
            if np.any(current_counts[sums] >= 2):
                valid = False

        if valid:
            # Accept add
            current_counts[2 * y] += 1
            if len(S_arr) > 0:
                sums = y + S_arr
                current_counts[sums] += 1
            current_S.add(y)
            in_S[y] = True
            S_arr = np.append(S_arr, y)

            if len(current_S) > best_size:
                best_S = frozenset(current_S)
                best_size = len(current_S)
        else:
            # Try swap: remove a random element x, then add y
            if len(current_S) > 0:
                x = int(random.choice(S_arr))

                # Remove x from current state
                current_counts[2 * x] -= 1
                if len(S_arr) > 0:
                    mask = S_arr != x
                    if np.any(mask):
                        sums = x + S_arr[mask]
                        current_counts[sums] -= 1
                current_S.remove(x)
                in_S[x] = False
                S_arr = S_arr[S_arr != x]

                # Now try to add y
                valid2 = True
                if current_counts[2 * y] >= 2:
                    valid2 = False
                elif len(S_arr) > 0:
                    sums = y + S_arr
                    if np.any(current_counts[sums] >= 2):
                        valid2 = False

                if valid2:
                    # Accept swap
                    current_counts[2 * y] += 1
                    if len(S_arr) > 0:
                        sums = y + S_arr
                        current_counts[sums] += 1
                    current_S.add(y)
                    in_S[y] = True
                    S_arr = np.append(S_arr, y)

                    if len(current_S) > best_size:
                        best_S = frozenset(current_S)
                        best_size = len(current_S)
                else:
                    # Rollback swap: put x back
                    current_counts[2 * x] += 1
                    if len(S_arr) > 0:
                        sums = x + S_arr
                        current_counts[sums] += 1
                    current_S.add(x)
                    in_S[x] = True
                    S_arr = np.append(S_arr, x)

    return set(best_S)


def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin before 120s limit
    N = 1000

    best_S = set()
    best_size = 0

    # Try multiple greedy orderings
    orderings = [
        range(1, N + 1),  # ascending
        range(N, 0, -1),  # descending
        list(range(1, N + 1, 2)),  # odd first
        list(range(2, N + 1, 2)),  # even first
        list(range(1, N + 1, 3)),  # every 3rd
        list(range(1, N + 1, 4)),  # every 4th
    ]

    for i, order in enumerate(orderings):
        if time.time() > deadline:
            break

        S, counts = _greedy_build(N, order, deadline)

        if len(S) > best_size:
            best_S = S
            best_size = len(S)

        # Local search on this solution
        if time.time() < deadline - 15:  # Leave time for more orderings
            improved_S = _local_search(S, counts, N, deadline, seed=42 + i)
            if len(improved_S) > best_size:
                best_S = improved_S
                best_size = len(improved_S)

    # Final local search on best solution found
    if time.time() < deadline - 5 and best_S:
        # Rebuild counts for final search
        counts = np.zeros(2 * N + 1, dtype=np.int32)
        S_list = sorted(best_S)
        S_arr_final = np.array(S_list, dtype=np.int32)
        for idx, x in enumerate(S_list):
            counts[2 * x] += 1
            # x + y for y in S, y > x to avoid double counting
            # Actually, we need all pairs. Let's do it properly.
        # Rebuild properly
        counts = np.zeros(2 * N + 1, dtype=np.int32)
        for x in best_S:
            counts[2 * x] += 1
        for i in range(len(S_list)):
            for j in range(i + 1, len(S_list)):
                counts[S_list[i] + S_list[j]] += 1

        final_S = _local_search(best_S, counts, N, deadline, seed=100)
        if len(final_S) > best_size:
            best_S = final_S
            best_size = len(final_S)

    return list(best_S)
