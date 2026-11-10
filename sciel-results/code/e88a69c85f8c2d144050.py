"""Optimized program: multi-start greedy + targeted backtracking to maximize sequence length."""
import time

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 78  # safety margin before 90s kill

    def greedy():
        values, sums = [], set()
        for x in range(1, 257):
            tails, total = [x], x
            for a in reversed(values):
                total += a
                tails.append(total)
            if not sums.intersection(tails):
                values.append(x)
                sums.update(tails)
        return values

    best = greedy()
    best_len = len(best)

    def check_add(values, sums, x):
        """Check if x can be appended; return list of new sums or None."""
        total = x
        new_sums = [total]
        for a in reversed(values):
            total += a
            new_sums.append(total)
        for s in new_sums:
            if s in sums:
                return None
        return new_sums

    def bt(target, vals, sm, lo, sub_deadline):
        """Backtracking to build sequence of target length starting from vals."""
        if time.time() > sub_deadline:
            return None
        if len(vals) == target:
            return list(vals)

        # Heuristic ordering: try smaller values first, but also consider
        # skipping some to allow more room later
        for x in range(lo, 257):
            if time.time() > sub_deadline:
                return None
            ns = check_add(vals, sm, x)
            if ns is None:
                continue
            vals.append(x)
            sm2 = sm | set(ns)
            res = bt(target, vals, sm2, x + 1, sub_deadline)
            if res is not None:
                return res
            vals.pop()
        return None

    # Try to find longer sequences via backtracking
    # Use multiple starting strategies:
    # 1. Fresh start from empty
    # 2. Use greedy prefix of various lengths, then backtrack the rest

    # Strategy 1: Fresh backtracking search
    for target_len in range(best_len + 1, best_len + 12):
        if time.time() > deadline:
            break
        sub_deadline = min(deadline, time.time() + 4)
        result = bt(target_len, [], set(), 1, sub_deadline)
        if result is not None and len(result) > best_len:
            best = result
            best_len = len(result)

    # Strategy 2: Use greedy prefix, then try to extend with backtracking
    # Take first k elements of greedy, then try to add more
    for k in range(1, min(best_len, 15)):
        if time.time() > deadline:
            break
        prefix = best[:k]
        # Compute sums for the prefix
        prefix_sums = set()
        for i in range(len(prefix)):
            s = 0
            for j in range(i, len(prefix)):
                s += prefix[j]
                prefix_sums.add(s)
        
        # Try to extend from this prefix to various target lengths
        for target_len in range(best_len + 1, best_len + 8):
            if time.time() > deadline:
                break
            sub_deadline = min(deadline, time.time() + 2)
            result = bt(target_len, list(prefix), prefix_sums, prefix[-1] + 1, sub_deadline)
            if result is not None and len(result) > best_len:
                best = result
                best_len = len(result)
            if time.time() > deadline:
                break

    return best
# EVOLVE-BLOCK-END
