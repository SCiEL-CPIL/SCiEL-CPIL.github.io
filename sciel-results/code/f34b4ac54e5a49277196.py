import numpy as np
import time
import random

def construct():
    MIN_SUM = 128
    MAX_SUM = 256
    NUM_SUMS = MAX_SUM - MIN_SUM + 1  # 129
    N_MAX = 128  # max element value

    start_time = time.time()
    DEADLINE = 85.0  # safety margin

    def calc_mult(A_list):
        """A_list must be sorted. Returns multiplicity array."""
        M = np.zeros(NUM_SUMS, dtype=np.int32)
        n = len(A_list)
        for i in range(n):
            a = A_list[i]
            s = 2 * a
            if MIN_SUM <= s <= MAX_SUM:
                M[s - MIN_SUM] += 1
            for j in range(i + 1, n):
                b = A_list[j]
                s = a + b
                if s > MAX_SUM:
                    break
                if s >= MIN_SUM:
                    M[s - MIN_SUM] += 1
        return M

    def is_valid(M):
        return bool(np.all(M >= 1))

    def objective(M):
        """Smooth surrogate: peak + small variance penalty."""
        peak = int(np.max(M))
        # Sum of squares as a flatness measure
        ss = int(np.sum(M.astype(np.int64) ** 2))
        return peak * 10000 + ss

    # Start with a good initial set
    # Use a structured starting point
    A = list(range(64, 129))  # [64, 65, ..., 128]
    M = calc_mult(A)
    best_max = int(np.max(M))
    best_obj = objective(M)
    best_A = list(A)

    # Phase 1: Greedy removal to minimize peak
    for iteration in range(10000):
        if time.time() - start_time > DEADLINE:
            break
        if not is_valid(M):
            break
        current_max = int(np.max(M))
        current_obj = objective(M)
        improved = False

        # Try removing each element, prefer those that reduce objective
        best_candidate = None
        best_candidate_obj = current_obj
        for idx in range(len(A)):
            new_A = A[:idx] + A[idx+1:]
            if not new_A:
                continue
            new_M = calc_mult(new_A)
            if is_valid(new_M):
                new_obj = objective(new_M)
                if new_obj < best_candidate_obj:
                    best_candidate_obj = new_obj
                    best_candidate = (idx, new_A, new_M)

        if best_candidate is not None:
            idx, new_A, new_M = best_candidate
            A = new_A
            M = new_M
            current_max = int(np.max(M))
            if current_max < best_max:
                best_max = current_max
                best_obj = objective(M)
                best_A = list(A)
            improved = True

        if not improved:
            # Try random removals that don't increase peak
            found = False
            for _ in range(20):
                if time.time() - start_time > DEADLINE:
                    break
                idx = random.randint(0, len(A)-1)
                new_A = A[:idx] + A[idx+1:]
                if not new_A:
                    continue
                new_M = calc_mult(new_A)
                if is_valid(new_M) and int(np.max(new_M)) <= current_max:
                    new_obj = objective(new_M)
                    if new_obj <= current_obj:
                        A = new_A
                        M = new_M
                        if new_obj < best_obj:
                            best_obj = new_obj
                            best_A = list(A)
                        found = True
                        break
            if not found:
                break

    # Phase 2: Targeted local search - focus on bottleneck sums
    for iteration in range(50000):
        if time.time() - start_time > DEADLINE:
            break
        current_max = int(np.max(M))
        current_obj = objective(M)

        if len(A) == 0:
            break

        A_set = set(A)
        available = [x for x in range(N_MAX + 1) if x not in A_set]

        # Identify bottleneck sums (highest multiplicity)
        bottleneck_sums = np.where(M == current_max)[0] + MIN_SUM

        # Try swaps that target bottleneck sums
        found_improvement = False

        # Strategy 1: Remove an element that contributes to bottleneck sums
        for idx in range(len(A)):
            if time.time() - start_time > DEADLINE:
                break
            elem = A[idx]
            # Check if removing this helps
            new_A = A[:idx] + A[idx+1:]
            if not new_A:
                continue
            new_M = calc_mult(new_A)
            if is_valid(new_M):
                new_max = int(np.max(new_M))
                new_obj = objective(new_M)
                if new_obj < current_obj:
                    A = new_A
                    M = new_M
                    if new_max < best_max:
                        best_max = new_max
                        best_obj = new_obj
                        best_A = list(A)
                    found_improvement = True
                    break

        if found_improvement:
            continue

        # Strategy 2: Swap - remove one, add another
        if available:
            # Prioritize adding elements that cover bottleneck sums
            for idx in range(len(A)):
                if time.time() - start_time > DEADLINE:
                    break
                elem_remove = A[idx]
                new_A_base = A[:idx] + A[idx+1:]

                # Try adding elements that might help
                for elem_add in random.sample(available, min(5, len(available))):
                    # Insert in sorted position
                    pos = 0
                    while pos < len(new_A_base) and new_A_base[pos] < elem_add:
                        pos += 1
                    new_A = new_A_base[:pos] + [elem_add] + new_A_base[pos:]

                    new_M = calc_mult(new_A)
                    if is_valid(new_M):
                        new_max = int(np.max(new_M))
                        new_obj = objective(new_M)
                        if new_obj < current_obj:
                            A = new_A
                            M = new_M
                            if new_max < best_max:
                                best_max = new_max
                                best_obj = new_obj
                                best_A = list(A)
                            found_improvement = True
                            break
                if found_improvement:
                    break

        if not found_improvement:
            # Simulated annealing style: accept some neutral moves
            if available and random.random() < 0.1:
                idx_remove = random.randint(0, len(A) - 1)
                elem_add = random.choice(available)
                new_A_base = A[:idx_remove] + A[idx_remove+1:]
                pos = 0
                while pos < len(new_A_base) and new_A_base[pos] < elem_add:
                    pos += 1
                new_A = new_A_base[:pos] + [elem_add] + new_A_base[pos:]

                new_M = calc_mult(new_A)
                if is_valid(new_M) and int(np.max(new_M)) <= current_max:
                    A = new_A
                    M = new_M

    # Phase 3: Final polish - exhaustive removal check
    for iteration in range(10000):
        if time.time() - start_time > DEADLINE:
            break
        if not is_valid(M):
            break
        current_max = int(np.max(M))
        current_obj = objective(M)
        improved = False

        for idx in range(len(A)):
            new_A = A[:idx] + A[idx+1:]
            if not new_A:
                continue
            new_M = calc_mult(new_A)
            if is_valid(new_M):
                new_max = int(np.max(new_M))
                new_obj = objective(new_M)
                if new_obj < current_obj:
                    A = new_A
                    M = new_M
                    if new_max < best_max:
                        best_max = new_max
                        best_obj = new_obj
                        best_A = list(A)
                    improved = True
                    break

        if not improved:
            break

    # Return the best solution found
    if time.time() - start_time > DEADLINE:
        return best_A

    # Final check: ensure validity
    final_M = calc_mult(best_A)
    if is_valid(final_M):
        return best_A
    else:
        # Fallback to a known valid solution
        return list(range(64, 129))

construct()
