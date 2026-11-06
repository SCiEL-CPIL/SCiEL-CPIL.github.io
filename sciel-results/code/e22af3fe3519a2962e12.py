"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 140.0

    target_diffs = set(range(1, 31))

    def check_sidon(a):
        """Check if sorted list a is a Sidon set. Returns True if valid."""
        n = len(a)
        diff_set = set()
        for i in range(n):
            for j in range(i + 1, n):
                d = a[j] - a[i]
                if d in diff_set:
                    return False
                diff_set.add(d)
        return True

    def covers_targets(a):
        """Check if differences of a cover all target diffs."""
        n = len(a)
        diff_set = set()
        for i in range(n):
            for j in range(i + 1, n):
                diff_set.add(a[j] - a[i])
        return target_diffs.issubset(diff_set)

    def is_valid(a):
        return check_sidon(a) and covers_targets(a)

    best_solution = None
    best_diameter = float('inf')

    # Try to find a 9-element Sidon set {0, a1, ..., a8} covering diffs 1-30
    # Use backtracking with pruning

    def backtrack_search(max_val, time_limit):
        """Search for 9-element Sidon set in [0, max_val] covering diffs 1-30."""
        a = [0]
        used_diffs = set()

        def dfs(a, used_diffs):
            if time.time() - start_time > time_limit:
                return None
            if len(a) == 9:
                if target_diffs.issubset(used_diffs):
                    return list(a)
                return None

            # Determine range for next element
            lo = a[-1] + 1 if a else 1
            # Upper bound: we need to fit remaining elements
            remaining = 9 - len(a)
            # Heuristic upper bound
            hi = min(max_val, a[-1] + 50 if a else 50)

            for x in range(lo, hi + 1):
                # Check if adding x maintains Sidon property
                valid = True
                new_used = set(used_diffs)
                for y in a:
                    d = x - y
                    if d in new_used:
                        valid = False
                        break
                    new_used.add(d)

                if not valid:
                    continue

                # Pruning: check if we can still cover all target diffs
                # with remaining elements
                if len(a) >= 2:
                    # Check if remaining needed diffs can be covered
                    # This is a weak pruning but helps
                    pass

                a.append(x)
                result = dfs(a, new_used)
                a.pop()
                if result is not None:
                    return result
            return None

        return dfs(a, used_diffs)

    # Try increasing max_val
    for max_val in range(30, 80):
        if time.time() - start_time > DEADLINE - 15:
            break
        result = backtrack_search(max_val, DEADLINE - 15)
        if result is not None:
            best_solution = result
            best_diameter = max(result)
            break

    # If backtracking didn't find anything, try randomized local search
    if best_solution is None:
        random.seed(42)
        for restart in range(500):
            if time.time() - start_time > DEADLINE - 10:
                break
            # Random 9-element set including 0
            a = [0] + sorted(random.sample(range(1, 150), 8))
            if is_valid(a):
                if max(a) < best_diameter:
                    best_diameter = max(a)
                    best_solution = a

    # Final fallback
    if best_solution is None:
        # Return a known valid Sidon set that covers 1-30
        # {0, 1, 4, 10, 12, 17, 21, 25, 28}
        candidate = [0, 1, 4, 10, 12, 17, 21, 25, 28]
        if is_valid(candidate):
            return candidate
        # Larger fallback
        candidate = [0, 1, 4, 9, 13, 16, 22, 25, 28]
        if is_valid(candidate):
            return candidate
        # Absolute fallback: construct greedily
        a = [0]
        for d in range(1, 31):
            added = False
            for x in range(1, 500):
                if x in a:
                    continue
                trial = sorted(a + [x])
                if is_valid(trial):
                    a = trial
                    added = True
                    break
            if not added:
                # Add a large element
                y = 2 * max(a) + 3 * d + 1
                while not is_valid(sorted(a + [y, y + d])):
                    y += 1
                a = sorted(a + [y, y + d])
        return a

    return best_solution
# EVOLVE-BLOCK-END
