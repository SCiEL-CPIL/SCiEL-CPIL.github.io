import time
from math import lcm

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill

    denominators = list(range(2, 49))
    n = len(denominators)

    # Calculate LCM of all denominators for exact integer arithmetic
    L = 1
    for d in denominators:
        L = lcm(L, d)

    weights = [L // d for d in denominators]

    # Known good: 1/2+1/3+1/7+1/43, deficit = 1/1806
    known_sum = L // 2 + L // 3 + L // 7 + L // 43
    best_sum = known_sum
    best_indices = [0, 1, 5, 41]

    # ============================================================
    # BRANCH-AND-BOUND with sorted weights and tight pruning
    # ============================================================
    # Sort indices by weight descending (largest reciprocals first)
    sorted_order = sorted(range(n), key=lambda i: weights[i], reverse=True)
    sorted_weights = [weights[i] for i in sorted_order]

    # Precompute suffix sums for upper bound
    suffix_sum = [0] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix_sum[i] = suffix_sum[i + 1] + sorted_weights[i]

    nodes_explored = 0
    sys.setrecursionlimit(10000)

    def dfs(depth, current_sum, chosen):
        nonlocal best_sum, best_indices, nodes_explored
        if time.time() > deadline - 2.0:
            return

        # Pruning: even taking all remaining can't beat best
        if current_sum + suffix_sum[depth] <= best_sum:
            return

        if depth == n:
            return

        # Try including sorted_order[depth]
        idx = sorted_order[depth]
        new_sum = current_sum + sorted_weights[depth]
        if new_sum < L:  # Strictly below 1
            if new_sum > best_sum:
                best_sum = new_sum
                best_indices = list(chosen) + [idx]
            # Recurse with item included
            nodes_explored += 1
            chosen.append(idx)
            dfs(depth + 1, new_sum, chosen)
            chosen.pop()

        # Try excluding sorted_order[depth]
        nodes_explored += 1
        dfs(depth + 1, current_sum, chosen)

    # Run B&B
    dfs(0, 0, [])

    # ============================================================
    # LOCAL SEARCH REFINEMENT
    # ============================================================
    if time.time() < deadline - 5.0:
        best_set = set(best_indices)
        improved = True
        while improved and time.time() < deadline - 3.0:
            improved = False
            # Try adding one element
            for i in range(n):
                if i not in best_set:
                    new_sum = best_sum + weights[i]
                    if new_sum < L and new_sum > best_sum:
                        best_sum = new_sum
                        best_set.add(i)
                        improved = True
                        break
            if time.time() > deadline - 3.0:
                break
            # Try swapping: remove one, add one
            for i in list(best_set):
                if time.time() > deadline - 3.0:
                    break
                for j in range(n):
                    if j not in best_set:
                        new_sum = best_sum - weights[i] + weights[j]
                        if new_sum < L and new_sum > best_sum:
                            best_sum = new_sum
                            best_set.remove(i)
                            best_set.add(j)
                            improved = True
                            break
                    if time.time() > deadline - 3.0:
                        break
                if time.time() > deadline - 3.0:
                    break

    if best_indices is not None and len(best_indices) > 0:
        total_check = sum(weights[i] for i in best_indices)
        if total_check < L and total_check > 0:
            return best_indices

    return [0, 1, 5, 41]
