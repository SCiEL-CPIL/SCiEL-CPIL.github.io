import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin for 90s limit

    # --- Fast bitmask-based can_add ---
    # Bit i corresponds to number i+1 (1..128)

    def set_to_mask(S):
        mask = 0
        for x in S:
            mask |= (1 << (x - 1))
        return mask

    def mask_to_set(mask):
        return [i + 1 for i in range(128) if (mask >> i) & 1]

    def can_add_mask(mask, x):
        """Check if adding x (1-indexed) to set represented by mask is valid."""
        idx = x - 1
        if (mask >> idx) & 1:
            return False  # Already in set

        # Find all b in S such that x+b is also in S
        max_b = 128 - x
        if max_b < 1:
            return True

        A = []
        for b in range(1, max_b + 1):
            b_idx = b - 1
            if (mask >> b_idx) & 1:
                sum_idx = (x + b) - 1
                if (mask >> sum_idx) & 1:
                    A.append(b)

        if len(A) < 2:
            return True

        # For each pair (b, c) in A, check if b+c is in S or b+c == x
        n = len(A)
        for i in range(n):
            b = A[i]
            for j in range(i + 1, n):
                c = A[j]
                s = b + c
                if s == x:
                    return False
                s_idx = s - 1
                if s <= 128 and (mask >> s_idx) & 1:
                    return False
        return True

    def can_add_set(S, x):
        """Set-based can_add for clarity."""
        V = [v for v in S if (x + v) <= 128 and (x + v) in S]
        n = len(V)
        if n < 2:
            return True
        for i in range(n):
            b = V[i]
            for j in range(i + 1, n):
                c = V[j]
                s = b + c
                if s == x or s in S:
                    return False
        return True

    def is_valid(S):
        """Full validity check."""
        S_list = sorted(S)
        n = len(S_list)
        S_set = set(S_list)
        for i in range(n):
            a = S_list[i]
            for j in range(i + 1, n):
                b = S_list[j]
                if (a + b) not in S_set:
                    continue
                for k in range(n):
                    if k == i or k == j:
                        continue
                    c = S_list[k]
                    if (a + c) in S_set and (b + c) in S_set:
                        return False
        return True

    # --- Try MILP first ---
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
        import scipy.sparse as sp

        # Generate all "cores": triples (a,b,c) with a<b<c and a+b,a+c,b+c all <= 128
        cores = []
        for a in range(1, 129):
            for b in range(a + 1, 129):
                max_c = 128 - b
                if max_c <= b:
                    continue
                for c in range(b + 1, max_c + 1):
                    s1 = a + b
                    s2 = a + c
                    s3 = b + c
                    # All six values must be in 1..128
                    if s1 <= 128 and s2 <= 128 and s3 <= 128:
                        cores.append((a, b, c, s1, s2, s3))

        num_cores = len(cores)
        if num_cores == 0:
            return list(range(1, 129))

        c_obj = np.ones(128) * -1

        rows = []
        cols = []
        data = []
        for idx, core in enumerate(cores):
            for val in core:
                rows.append(idx)
                cols.append(val - 1)
                data.append(1)

        A_sparse = sp.csr_matrix((data, (rows, cols)), shape=(num_cores, 128))
        b_ub = np.ones(num_cores) * 5
        bounds = Bounds(lb=0, ub=1)
        integrality = np.ones(128)
        constraint = LinearConstraint(A_sparse, ub=b_ub)

        remaining_time = deadline - time.time()
        if remaining_time < 10:
            raise Exception("Not enough time")

        res = milp(c_obj, constraints=constraint, integrality=integrality,
                   bounds=bounds, options={'time_limit': remaining_time - 5})

        if res.x is not None:
            sol = np.round(res.x).astype(int)
            S = set()
            for i, val in enumerate(sol):
                if val == 1:
                    S.add(i + 1)

            if is_valid(S):
                return [int(x) for x in S]
            else:
                # Cleanup: remove elements to fix violations
                changed = True
                while changed and time.time() < deadline:
                    changed = False
                    for core in cores:
                        if all(v in S for v in core):
                            # Remove the element with fewest other core memberships
                            # Simple: remove first
                            S.remove(core[0])
                            changed = True
                            break
                if is_valid(S):
                    return [int(x) for x in S]

    except Exception:
        pass

    # --- Bitmask-based local search ---
    best_set = set()
    best_size = 0

    random.seed(42)

    def greedy_random_order():
        candidates = list(range(1, 129))
        random.shuffle(candidates)
        S = set()
        mask = 0
        for x in candidates:
            if can_add_mask(mask, x):
                S.add(x)
                mask |= (1 << (x - 1))
        return S

    def greedy_ascending():
        S = set()
        mask = 0
        for x in range(1, 129):
            if can_add_mask(mask, x):
                S.add(x)
                mask |= (1 << (x - 1))
        return S

    def greedy_descending():
        S = set()
        mask = 0
        for x in range(128, 0, -1):
            if can_add_mask(mask, x):
                S.add(x)
                mask |= (1 << (x - 1))
        return S

    def odd_numbers():
        return set(range(1, 129, 2))

    def high_numbers():
        return set(range(65, 129))

    def low_numbers():
        return set(range(1, 65))

    strategies = [
        greedy_ascending,
        greedy_descending,
        odd_numbers,
        high_numbers,
        low_numbers,
    ]

    for strategy in strategies:
        if time.time() > deadline:
            break
        S = strategy()
        if len(S) > best_size:
            best_size = len(S)
            best_set = set(S)

    # Local search with bitmask
    def local_search(S, max_iters=200):
        mask = set_to_mask(S)
        S_set = set(S)
        for iteration in range(max_iters):
            if time.time() > deadline:
                break

            # Try to add any element
            added = False
            for x in range(1, 129):
                if x not in S_set:
                    if can_add_mask(mask, x):
                        S_set.add(x)
                        mask |= (1 << (x - 1))
                        added = True
                        break

            if not added:
                # Try swaps: remove y, add x
                improved = False
                S_list = list(S_set)
                not_in_S = [x for x in range(1, 129) if x not in S_set]
                random.shuffle(S_list)
                random.shuffle(not_in_S)

                for x in not_in_S[:30]:
                    for y in S_list[:50]:
                        mask_temp = mask & ~(1 << (y - 1))
                        if can_add_mask(mask_temp, x):
                            S_set.discard(y)
                            S_set.add(x)
                            mask = mask_temp | (1 << (x - 1))
                            improved = True
                            break
                    if improved:
                        break

                if not improved:
                    # Try 2-for-1 swaps: remove two, add one (net -1, not helpful for size)
                    # Try 1-for-2: remove one, add two (net +1)
                    improved2 = False
                    random.shuffle(S_list)
                    random.shuffle(not_in_S)
                    for y in S_list[:30]:
                        mask_temp = mask & ~(1 << (y - 1))
                        # Try to add two elements
                        added_count = 0
                        new_elements = []
                        for x in not_in_S[:40]:
                            if can_add_mask(mask_temp, x):
                                mask_temp |= (1 << (x - 1))
                                new_elements.append(x)
                                added_count += 1
                                if added_count == 2:
                                    break
                        if added_count >= 2:
                            S_set.discard(y)
                            for x in new_elements:
                                S_set.add(x)
                            mask = mask_temp
                            improved2 = True
                            break
                    if not improved2:
                        break

        return S_set

    num_restarts = 30
    for restart in range(num_restarts):
        if time.time() > deadline:
            break

        S = greedy_random_order()
        S = local_search(S, max_iters=150)

        if len(S) > best_size:
            best_size = len(S)
            best_set = set(S)

    # Also try local search from the best fixed strategy result
    if time.time() < deadline:
        S = local_search(set(best_set), max_iters=200)
        if len(S) > best_size:
            best_size = len(S)
            best_set = set(S)

    # Final greedy fill
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for x in range(1, 129):
            if x in best_set:
                continue
            if can_add_set(best_set, x):
                best_set.add(x)
                changed = True

    # Verify
    if not is_valid(best_set):
        # Fallback: return a known valid set
        best_set = set(range(1, 129, 2))  # All odd numbers, size 64

    result = [int(x) for x in best_set]
    return result
