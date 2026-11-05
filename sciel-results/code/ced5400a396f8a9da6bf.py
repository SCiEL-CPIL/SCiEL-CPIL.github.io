"""Rewritten program using exact ILP for maximum sum-free subsets."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds


def construct():
    start_time = time.time()
    DEADLINE = 145.0

    def remaining():
        return DEADLINE - (time.time() - start_time)

    def instance(size, seed, limit=10000):
        rng = random.Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x != 0:
                values.add(x)
        return sorted(values)

    def is_sum_free(S):
        Ss = set(S)
        for a in S:
            for b in S:
                if a + b in Ss:
                    return False
        return True

    def solve_ilp(arr):
        """Exact ILP: maximize sum(x_i) subject to x_i + x_j + x_k <= 2 for all conflicts."""
        n = len(arr)
        if n == 0:
            return []
        
        # Find all conflict triples (i, j, k) where arr[i] + arr[j] == arr[k]
        # We need: for all such triples, x_i + x_j + x_k <= 2
        # This ensures at most 2 of the three are selected, preventing a+b=c
        
        idx_map = {v: i for i, v in enumerate(arr)}
        
        # Build constraint matrix
        # For each pair (i, j), check if arr[i]+arr[j] is in arr
        # If arr[i]+arr[j] == arr[k], add constraint x_i + x_j + x_k <= 2
        
        constraints = []
        rhs = []
        
        seen = set()
        for i in range(n):
            for j in range(n):
                s = arr[i] + arr[j]
                if s in idx_map:
                    k = idx_map[s]
                    # Constraint: x_i + x_j + x_k <= 2
                    # Use sorted tuple to avoid duplicates
                    key = tuple(sorted([i, j, k]))
                    if key not in seen:
                        seen.add(key)
                        constraints.append(key)
                        rhs.append(2)
        
        m = len(constraints)
        
        if m == 0:
            # No conflicts, take all
            return list(arr)
        
        # Build A matrix
        A = np.zeros((m, n), dtype=np.float64)
        for idx, (i, j, k) in enumerate(constraints):
            A[idx, i] = 1.0
            A[idx, j] = 1.0
            A[idx, k] = 1.0
        
        c = -np.ones(n)  # minimize -sum(x_i) = maximize sum(x_i)
        
        # Variables: x_i in {0, 1}
        integrality = np.ones(n)
        lb = np.zeros(n)
        ub = np.ones(n)
        
        cons = LinearConstraint(A, lb=0, ub=np.array(rhs, dtype=np.float64))
        bounds = Bounds(lb, ub)
        
        try:
            res = milp(c=c, constraints=cons, integrality=integrality, bounds=bounds,
                       options={"time_limit": 140.0, "presolve": True})
            if res.success and res.x is not None:
                selected = [arr[i] for i in range(n) if res.x[i] > 0.5]
                if is_sum_free(selected):
                    return selected
        except Exception:
            pass
        
        return []

    def greedy_fallback(arr):
        """Fast greedy fallback if ILP fails or times out."""
        chosen = []
        chosen_set = set()
        for x in sorted(arr, key=lambda v: (abs(v), v)):
            ok = True
            for z in chosen_set:
                if (x + z) in chosen_set or (x - z) in chosen_set:
                    ok = False
                    break
            if ok:
                chosen.append(x)
                chosen_set.add(x)
        return chosen

    def local_search(arr, initial, time_budget):
        """Hill-climbing add/swap to improve a feasible solution."""
        current = list(initial)
        current_set = set(current)
        arr_set = set(arr)
        not_in = list(arr_set - current_set)
        best = list(current)
        best_len = len(current)
        
        def can_add(x):
            for z in current_set:
                if (x + z) in current_set or (x - z) in current_set:
                    return False
            return True
        
        rng = random.Random(42)
        while time_budget() > 0.5 and not_in:
            # Try to add
            added = False
            for x in not_in[:]:
                if can_add(x):
                    current.append(x)
                    current_set.add(x)
                    not_in.remove(x)
                    if len(current) > best_len:
                        best = list(current)
                        best_len = len(current)
                    added = True
            if not added:
                # Try swaps: remove one, add one
                swapped = False
                for x in not_in[:]:
                    for idx in range(len(current)):
                        removed = current[idx]
                        current.pop(idx)
                        current_set.discard(removed)
                        if can_add(x):
                            current.append(x)
                            current_set.add(x)
                            not_in.remove(x)
                            not_in.append(removed)
                            if len(current) > best_len:
                                best = list(current)
                                best_len = len(current)
                            swapped = True
                            break
                        else:
                            current.insert(idx, removed)
                            current_set.add(removed)
                    if swapped:
                        break
                if not swapped:
                    break
        return best

    def solve_instance(arr):
        n = len(arr)
        if n <= 1:
            return list(arr)
        
        # Try ILP for exact solution
        if remaining() > 10.0:
            ilp_result = solve_ilp(arr)
            if ilp_result and is_sum_free(ilp_result):
                # Try to polish with local search in case ILP found non-maximal (shouldn't happen)
                if remaining() > 3.0:
                    polished = local_search(arr, ilp_result, remaining)
                    if len(polished) > len(ilp_result):
                        return polished
                return ilp_result
        
        # Fallback: greedy + local search
        best = greedy_fallback(arr)
        if remaining() > 2.0:
            best = local_search(arr, best, remaining)
        
        if not is_sum_free(best):
            best = greedy_fallback(arr)
        return best

    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        arr = instance(size, seed)
        results.append(solve_instance(arr))
    return results
# EVOLVE-BLOCK-END
