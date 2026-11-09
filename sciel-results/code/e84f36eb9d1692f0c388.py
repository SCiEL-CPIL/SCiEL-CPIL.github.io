import time
import random
import math
import sys


def construct():
    start = time.time()
    DEADLINE = start + 145.0  # 5s safety margin before 150s kill

    prescribed = [1, 2, 5, 11]
    pres_set = set(prescribed)
    N = 284

    def remaining():
        return DEADLINE - time.time()

    def is_valid_set(s):
        if len(s) != len(set(s)):
            return False
        if not pres_set.issubset(set(s)):
            return False
        for v in s:
            if v < 1 or v > N:
                return False
        seen = set()
        n = len(s)
        for i in range(n):
            for j in range(i, n):
                v = s[i] + s[j]
                if v in seen:
                    return False
                seen.add(v)
        return True

    # ---------- Backtracking with constraint propagation ----------
    def backtrack_search(target_size, time_budget):
        """
        Systematic backtracking search for a Sidon set of given size.
        Uses constraint propagation to prune infeasible branches early.
        """
        end_time = time.time() + time_budget
        best_found = None

        # Prescribed elements must be included
        base = sorted(prescribed)
        base_set = set(base)

        # Track which sums are already used
        used_sums = set()
        for i in range(len(base)):
            for j in range(i, len(base)):
                used_sums.add(base[i] + base[j])

        # Track which values are used
        used_values = set(base)

        # For each candidate value, track which sums it would create
        # and whether those sums conflict with existing ones

        def can_add(value, current_arr, current_sums, current_values):
            """Check if value can be added without creating conflicts."""
            if value in current_values:
                return False
            # Check all pairs (value, x) for x in current_arr
            for x in current_arr:
                s = value + x
                if s in current_sums:
                    return False
            # Check value + value
            if (value + value) in current_sums:
                return False
            return True

        def get_conflicts(value, current_arr, current_sums):
            """Get set of sums that would be created by adding value."""
            conflicts = set()
            for x in current_arr:
                conflicts.add(value + x)
            conflicts.add(value + value)
            return conflicts

        # Recursive backtracking with memoization of failed states
        failed_states = set()  # (frozenset of used values, frozenset of used sums) -> too large, use hash
        # Instead, we'll use a simpler approach: track depth and prune

        def search(arr, sums, values, depth, target):
            nonlocal best_found
            if time.time() > end_time:
                return False

            if len(arr) == target:
                best_found = list(arr)
                return True

            # If we can't reach target with remaining slots, prune
            remaining_needed = target - len(arr)
            available = N - len(values)
            if remaining_needed > available:
                return False

            # Try adding each candidate value in order
            # Use heuristic: try values that create fewer new sum conflicts
            candidates = []
            for v in range(1, N + 1):
                if v not in values:
                    if can_add(v, arr, sums, values):
                        # Score: prefer values that leave more flexibility
                        new_sums_count = len(get_conflicts(v, arr, sums))
                        candidates.append((new_sums_count, v))

            # Sort by fewest new sums (greedy heuristic)
            candidates.sort()

            for _, v in candidates:
                if time.time() > end_time:
                    return False

                # Add v
                new_arr = arr + [v]
                new_sums = sums | get_conflicts(v, arr, sums)
                new_values = values | {v}

                if search(new_arr, new_sums, new_values, depth + 1, target):
                    return True

            return False

        # Try to find a solution
        arr = list(base)
        if search(arr, used_sums, base_set, 0, target_size):
            return best_found

        return None

    # ---------- Greedy baseline ----------
    def greedy_build():
        arr = list(prescribed)
        used = set(arr)
        sums = set()
        for a in arr:
            for b in arr:
                if a <= b:
                    sums.add(a + b)
        for x in range(1, N + 1):
            if x in used:
                continue
            ok = True
            new_sums = []
            for a in arr:
                s = a + x
                if s in sums:
                    ok = False
                    break
                new_sums.append(s)
            if ok and (x + x) in sums:
                ok = False
            if ok:
                sums.update(new_sums)
                sums.add(x + x)
                used.add(x)
                arr.append(x)
        return arr

    def greedy_build_random(seed_val):
        rng = random.Random(seed_val)
        arr = list(prescribed)
        used = set(arr)
        sums = set()
        for a in arr:
            for b in arr:
                if a <= b:
                    sums.add(a + b)
        pool = [x for x in range(1, N + 1) if x not in used]
        rng.shuffle(pool)
        for x in pool:
            ok = True
            new_sums = []
            for a in arr:
                s = a + x
                if s in sums:
                    ok = False
                    break
                new_sums.append(s)
            if ok and (x + x) in sums:
                ok = False
            if ok:
                sums.update(new_sums)
                sums.add(x + x)
                used.add(x)
                arr.append(x)
        return arr

    try:
        # Get greedy baseline
        g1 = greedy_build()
        g2 = greedy_build_random(123)
        g3 = greedy_build_random(456)
        best = max([g1, g2, g3], key=len)
        best_size = len(best)

        # ---------- Incremental growth using backtracking ----------
        for target in range(best_size + 1, 21):
            if remaining() < 15:
                break

            # Allocate time for backtracking: more time for larger targets
            time_budget = min(remaining() - 10, 30.0)
            if remaining() < 40:
                time_budget = min(remaining() - 10, 20.0)

            found = backtrack_search(target, time_budget)
            if found is not None and is_valid_set(found):
                best = found
                best_size = target
            else:
                # Try with more time if available
                if remaining() > 50:
                    time_budget2 = min(remaining() - 15, 60.0)
                    found = backtrack_search(target, time_budget2)
                    if found is not None and is_valid_set(found):
                        best = found
                        best_size = target
                break

        # Final validation
        if not is_valid_set(best):
            best = greedy_build()

        return best

    except Exception:
        try:
            return greedy_build()
        except Exception:
            return [1, 2, 5, 11]
