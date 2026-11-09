"""Combined structural + efficient local search for the 3-pair-sum-free set task."""
import time
import random


def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill
    N = 128

    def time_left():
        return deadline - time.time()

    # --- Precompute pair-sum membership bitmasks ---
    # For each v in 1..128: pairs_of_sum[v] = set of (i,j) with i<j, i+j=v
    # We use bitmasks over positions 1..128 (bit i set iff i in S)
    # sum_mask[v] = bitmask of x in 1..128 such that v-x in 1..128 and x != v-x
    # i.e., for a candidate x, x and v-x form a pair summing to v

    # Precompute for each v, the list of (x, v-x) pairs with x < v-x, both in [1,128]
    pairs_of_sum = [[] for _ in range(129)]
    for v in range(2, 129):
        for x in range(1, v // 2 + 1):
            y = v - x
            if y <= 128 and y > x:
                pairs_of_sum[v].append((x, y))

    # --- Conflict helpers (L = S ∩ [1,64], U = S ∩ [65,128]) ---
    # A forbidden triple is (a,b,c) distinct with a+b, a+c, b+c all in S.
    # Since U-U sums > 128, at most one of {a,b,c} is in U.
    # Type A: a,b,c all in L.
    # Type B: a,b in L, c in U. Need a+b in S, a+c in S, b+c in S.

    def has_conflict(L_set, U_set, S_set):
        """Check if there's any forbidden triple. Returns True if conflict exists."""
        L_list = sorted(L_set)
        k = len(L_list)
        # Type A: three from L
        for i in range(k):
            a = L_list[i]
            for j in range(i + 1, k):
                b = L_list[j]
                ab = a + b
                if ab not in S_set:
                    continue
                for m in range(j + 1, k):
                    c = L_list[m]
                    if (a + c) in S_set and (b + c) in S_set:
                        return True
        # Type B: two from L, one from U
        for i in range(k):
            a = L_list[i]
            for j in range(i + 1, k):
                b = L_list[j]
                ab = a + b
                if ab not in S_set:
                    continue
                # Need c in U such that a+c in S and b+c in S
                # a+c and b+c must be in S (which means in U since >64)
                for c in U_set:
                    if (a + c) in S_set and (b + c) in S_set:
                        return True
        return False

    def count_conflicts(L_set, U_set, S_set):
        """Count forbidden triples."""
        L_list = sorted(L_set)
        k = len(L_list)
        count = 0
        # Type A
        for i in range(k):
            a = L_list[i]
            for j in range(i + 1, k):
                b = L_list[j]
                ab = a + b
                if ab not in S_set:
                    continue
                for m in range(j + 1, k):
                    c = L_list[m]
                    if (a + c) in S_set and (b + c) in S_set:
                        count += 1
        # Type B
        for i in range(k):
            a = L_list[i]
            for j in range(i + 1, k):
                b = L_list[j]
                ab = a + b
                if ab not in S_set:
                    continue
                for c in U_set:
                    if (a + c) in S_set and (b + c) in S_set:
                        count += 1
        return count

    def objective(L_set, U_set, S_set):
        """Higher is better."""
        if has_conflict(L_set, U_set, S_set):
            return len(S_set) - count_conflicts(L_set, U_set, S_set) * 5
        return len(S_set)

    # --- Initial solution: full U, empty L ---
    U_set = set(range(65, 129))
    L_set = set()
    S_set = U_set | L_set

    best_S = set(S_set)
    best_size = len(S_set)

    # --- Phase 1: Greedy add elements from L candidates ---
    # Try adding each x in [1,64] if it doesn't create a conflict
    for x in range(1, 65):
        if time.time() > deadline:
            break
        L_new = L_set | {x}
        S_new = S_set | {x}
        if not has_conflict(L_new, U_set, S_new):
            L_set = L_new
            S_set = S_new

    if len(S_set) > best_size:
        best_S = set(S_set)
        best_size = len(S_set)

    # --- Phase 2: Local search with simulated annealing ---
    rng = random.Random(12345)
    all_small = list(range(1, 65))
    all_large = list(range(65, 129))

    cur_L = set(L_set)
    cur_U = set(U_set)
    cur_S = set(S_set)
    cur_obj = objective(cur_L, cur_U, cur_S)

    # Track best valid solution
    best_valid_S = set(cur_S)
    best_valid_size = len(cur_S)
    if cur_obj == best_valid_size:  # valid
        best_valid_S = set(cur_S)

    temperature = 3.0
    cooling = 0.9995

    while time.time() < deadline:
        # Choose a move
        move_type = rng.random()

        if move_type < 0.35:
            # Add a small element
            candidates = [x for x in all_small if x not in cur_L]
            if not candidates:
                continue
            x = rng.choice(candidates)
            new_L = cur_L | {x}
            new_U = set(cur_U)
            new_S = cur_S | {x}
        elif move_type < 0.55:
            # Remove a small element
            if not cur_L:
                continue
            x = rng.choice(list(cur_L))
            new_L = cur_L - {x}
            new_U = set(cur_U)
            new_S = cur_S - {x}
        elif move_type < 0.70:
            # Add a large element (from [65,128] not in U)
            candidates = [x for x in all_large if x not in cur_U]
            if not candidates:
                continue
            x = rng.choice(candidates)
            new_L = set(cur_L)
            new_U = cur_U | {x}
            new_S = cur_S | {x}
        elif move_type < 0.85:
            # Remove a large element
            if not cur_U:
                continue
            x = rng.choice(list(cur_U))
            new_L = set(cur_L)
            new_U = cur_U - {x}
            new_S = cur_S - {x}
        else:
            # Swap: remove one small, add another small
            if not cur_L:
                continue
            x = rng.choice(list(cur_L))
            candidates = [y for y in all_small if y not in cur_L]
            if not candidates:
                continue
            y = rng.choice(candidates)
            new_L = (cur_L - {x}) | {y}
            new_U = set(cur_U)
            new_S = (cur_S - {x}) | {y}

        new_obj = objective(new_L, new_U, new_S)
        delta = new_obj - cur_obj

        if delta >= 0:
            cur_L, cur_U, cur_S, cur_obj = new_L, new_U, new_S, new_obj
            if new_obj == len(new_S):  # valid
                if len(new_S) > best_valid_size:
                    best_valid_S = set(new_S)
                    best_valid_size = len(new_S)
                    best_S = set(new_S)
                    best_size = len(new_S)
        else:
            # Simulated annealing acceptance
            if temperature > 0.1:
                if rng.random() < (2.718281828 ** (delta / temperature)):
                    cur_L, cur_U, cur_S, cur_obj = new_L, new_U, new_S, new_obj
                    if new_obj == len(new_S):
                        if len(new_S) > best_valid_size:
                            best_valid_S = set(new_S)
                            best_valid_size = len(new_S)
                            best_S = set(new_S)
                            best_size = len(new_S)

        temperature *= cooling

    # --- Phase 3: Final greedy pass on best valid set ---
    S_final = set(best_valid_S)
    L_final = S_final & set(range(1, 65))
    U_final = S_final & set(range(65, 129))

    # Try adding small elements
    for x in range(1, 65):
        if time.time() > deadline:
            break
        if x in S_final:
            continue
        L_new = L_final | {x}
        S_new = S_final | {x}
        if not has_conflict(L_new, U_final, S_new):
            L_final = L_new
            S_final = S_new

    # Try adding large elements
    for x in range(65, 129):
        if time.time() > deadline:
            break
        if x in S_final:
            continue
        U_new = U_final | {x}
        S_new = S_final | {x}
        if not has_conflict(L_final, U_new, S_new):
            U_final = U_new
            S_final = S_new

    # Final validity check
    L_check = S_final & set(range(1, 65))
    U_check = S_final & set(range(65, 129))
    if has_conflict(L_check, U_check, S_final):
        # Fallback to best known valid
        S_final = best_valid_S

    result = sorted(S_final)
    result = [int(x) for x in result]
    return result
