# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    n = 256
    t0 = time.time()
    deadline = t0 + 55.0  # Safety margin before 60s kill

    # Precompute all arithmetic progressions
    # For each progression, store (start, step, length)
    # We'll track partial sums as we assign signs

    def get_progressions():
        """Get all arithmetic progressions in [0, n-1] with step >= 1 and length >= 1."""
        progs = []
        for d in range(1, n):
            for s in range(d):
                # Length of progression starting at s with step d
                length = (n - 1 - s) // d + 1
                if length >= 1:
                    progs.append((s, d, length))
        return progs

    def evaluate(x):
        """
        Compute max absolute sum over all arithmetic progressions in x.
        x is a numpy array of length n with values -1, 1.
        """
        max_abs = 1  # singletons
        for d in range(1, n):
            L_max = (n - 1) // d + 1
            k_indices = np.arange(L_max)
            s_indices = np.arange(d)
            flat_indices = s_indices[:, None] + k_indices[None, :] * d
            valid_mask = flat_indices < n

            M = np.zeros((d, L_max), dtype=np.int8)
            M[valid_mask] = x[flat_indices[valid_mask]]

            P = np.cumsum(M, axis=1)

            p_mins = P.min(axis=1)
            p_maxs = P.max(axis=1)

            local_max = np.max(np.maximum(0, p_maxs) - np.minimum(0, p_mins))

            if local_max > max_abs:
                max_abs = int(local_max)

        return int(max_abs)

    def solve_csp(bound, time_limit):
        """
        CSP backtracking with forward checking.
        Try to find a sequence where all AP sums have abs <= bound.
        Returns (success, sequence) or (False, None).
        """
        t_start = time.time()
        x = np.zeros(n, dtype=np.int8)  # 0 = unassigned
        assigned = np.zeros(n, dtype=bool)

        # For each AP, track current sum and which positions are assigned
        # We need efficient tracking. Let's use a different approach:
        # For each position, maintain the set of APs that contain it.
        # When we assign x[i], we update all APs containing i.

        # Precompute: for each position i, which APs contain it?
        # AP is identified by (s, d) where s = i % d, and i >= s
        # Actually, let's think differently.

        # Alternative: Use a more direct approach.
        # For each AP (s, d, L), we track:
        # - sum_so_far: sum of assigned elements in this AP
        # - assigned_count: how many elements are assigned
        # - total_length: L
        # The constraint is: |sum_so_far + remaining_sum| <= bound
        # where remaining_sum is in [-(L - assigned_count), (L - assigned_count)]
        # So we need: |sum_so_far| - (L - assigned_count) <= bound
        # i.e., sum_so_far <= bound + (L - assigned_count)
        # and sum_so_far >= -bound - (L - assigned_count)

        # Number of APs: sum over d=1 to n-1 of d = n(n-1)/2 = 256*255/2 = 32640
        # That's a lot to track. Let's be smarter.

        # Actually, for forward checking, we only need to check APs that are "active"
        # (have at least one assigned element but not all).

        # Let's use a different strategy: iterative deepening on the bound.
        # Start with a reasonable bound and try to satisfy it.

        # For efficiency, let's use a randomized backtracking with restarts.

        best_x = None

        def backtrack(pos, current_bound):
            nonlocal best_x
            if time.time() - t_start > time_limit:
                return False

            if pos == n:
                # All assigned, check if valid
                # We should have maintained constraints, but verify
                x_copy = x.copy()
                if evaluate(x_copy) <= current_bound:
                    best_x = x_copy.copy()
                    return True
                return False

            # Try both values
            for val in [1, -1]:
                x[pos] = val
                assigned[pos] = True

                # Forward checking: verify all APs containing pos are still satisfiable
                valid = True
                for d in range(1, n):
                    s = pos % d
                    if s > pos:
                        continue
                    # Check if pos is in AP (s, d)
                    if (pos - s) % d != 0:
                        continue

                    # Compute current sum of assigned elements in this AP
                    # and count remaining
                    ap_sum = 0
                    ap_assigned = 0
                    ap_total = (n - 1 - s) // d + 1
                    k = 0
                    while s + k * d < n:
                        idx = s + k * d
                        if assigned[idx]:
                            ap_sum += x[idx]
                            ap_assigned += 1
                        k += 1

                    remaining = ap_total - ap_assigned
                    # Constraint: |ap_sum + remaining_sum| <= current_bound
                    # remaining_sum in [-remaining, remaining]
                    # So we need: ap_sum - remaining <= current_bound
                    # and ap_sum + remaining >= -current_bound
                    if ap_sum - remaining > current_bound or ap_sum + remaining < -current_bound:
                        valid = False
                        break

                if not valid:
                    x[pos] = 0
                    assigned[pos] = False
                    continue

                if backtrack(pos + 1, current_bound):
                    return True

                x[pos] = 0
                assigned[pos] = False

            return False

        # Try with the given bound
        if backtrack(0, bound):
            return True, best_x

        return False, None

    # Strategy: Start with a reasonable bound and try to solve
    # Known results suggest bound 3 or 4 might be achievable for n=256

    best_overall = None
    best_overall_val = 999

    # First, get a baseline from Thue-Morse
    def thue_morse(n):
        seq = [1]
        while len(seq) < n:
            seq = seq + [-s for s in seq]
        return np.array(seq[:n], dtype=np.int8)

    x_tm = thue_morse(n)
    val_tm = evaluate(x_tm)
    best_overall = x_tm.copy()
    best_overall_val = val_tm

    # Try CSP with decreasing bounds
    # Start from the Thue-Morse value and try to improve
    current_bound = best_overall_val

    # Try bounds from min(3, current_bound) up to current_bound
    for b in range(min(3, current_bound), current_bound + 1):
        if time.time() > deadline - 5:
            break

        time_remaining = deadline - 5 - time.time()
        if time_remaining < 2:
            break

        # Try multiple restarts for this bound
        for restart in range(3):
            if time.time() > deadline - 2:
                break

            success, x_sol = solve_csp(b, time_remaining / 3)
            if success:
                best_overall = x_sol
                best_overall_val = b
                break  # Found a solution with this bound, try lower bound

    # If we found a good solution, try to polish with local search
    if best_overall_val > 3 and time.time() < deadline - 5:
        # Try local search to improve
        x = best_overall.copy()
        current_val = best_overall_val

        t_start = time.time()
        max_time = min(deadline - 3 - time.time(), 10)

        improved = True
        while improved and (time.time() - t_start) < max_time:
            improved = False
            indices = np.arange(n)
            np.random.shuffle(indices)
            for i in indices:
                if time.time() > t_start + max_time:
                    break
                old = x[i]
                x[i] = -old
                new_val = evaluate(x)
                if new_val < current_val:
                    current_val = new_val
                    improved = True
                    if current_val < best_overall_val:
                        best_overall_val = current_val
                        best_overall = x.copy()
                else:
                    x[i] = old

    return [int(v) for v in best_overall]

# EVOLVE-BLOCK-END
