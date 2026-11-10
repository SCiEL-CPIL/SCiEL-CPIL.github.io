"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    from fractions import Fraction
    import time

    deadline = time.time() + 80.0  # safety margin before 90s kill
    denominators = list(range(2, 49))
    n = len(denominators)
    # Precompute reciprocals as Fractions
    recips = [Fraction(1, d) for d in denominators]

    # Greedy initial solution: take in order while sum < 1
    best_indices = []
    best_sum = Fraction(0)
    for i in range(n):
        if best_sum + recips[i] < 1:
            best_indices.append(i)
            best_sum += recips[i]
    best_deficit = 1 - best_sum

    # Branch and bound: try to improve
    # Sort by value descending (largest fractions first) for tighter bounds
    # But we need to track original indices
    order = sorted(range(n), key=lambda i: -recips[i])
    # Precompute suffix sums of sorted reciprocals (as Fractions) for upper bound
    suffix = [Fraction(0)] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix[i] = suffix[i + 1] + recips[order[i]]

    best_sum_val = best_sum
    best_idx_set = set(best_indices)

    # DFS with branch and bound
    # State: position in sorted order, current sum, current indices
    def dfs(pos, current_sum, current_indices):
        nonlocal best_sum_val, best_idx_set, best_deficit

        if time.time() > deadline:
            return

        # Prune: even if we take all remaining, can't beat current best deficit
        # i.e., current_sum + suffix[pos] < best_sum_val  => can't improve
        if current_sum + suffix[pos] <= best_sum_val:
            return

        # Update best if current sum is valid and better
        if current_sum > best_sum_val and current_sum < 1:
            best_sum_val = current_sum
            best_idx_set = set(current_indices)
            best_deficit = 1 - best_sum_val

        if pos == n:
            return

        # Option 1: skip order[pos]
        dfs(pos + 1, current_sum, current_indices)

        # Option 2: take order[pos] if it keeps sum < 1
        idx = order[pos]
        new_sum = current_sum + recips[idx]
        if new_sum < 1:
            dfs(pos + 1, new_sum, current_indices + [idx])

    # Start DFS with empty set
    dfs(0, Fraction(0), [])

    if best_idx_set:
        return sorted(best_idx_set)
    return best_indices
# EVOLVE-BLOCK-END
