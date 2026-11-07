import time
from math import gcd
from functools import reduce

DIVISORS = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
N = len(DIVISORS)
MAX_VAL = 1000000
PERIOD = reduce(lambda a, b: a * b // gcd(a, b), DIVISORS)  # 60060

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill

    n = N
    divisors = DIVISORS
    max_val = MAX_VAL
    period = PERIOD

    def find_assignment(l, W):
        """Find assignment in window [l, l+W) using backtracking with MRV."""
        if W < n:
            return None
        if l + W > max_val:
            return None

        # Build candidate lists (offsets within window)
        cands = []
        for i in range(n):
            d = divisors[i]
            rem = (-l) % d
            if rem >= W:
                return None
            c = list(range(rem, W, d))
            cands.append(c)

        # MRV ordering: sort by number of candidates
        order = sorted(range(n), key=lambda i: len(cands[i]))
        if len(cands[order[0]]) == 0:
            return None

        assignment = [None] * n
        used = [False] * W

        def backtrack(idx):
            if idx == n:
                return True

            # Dynamic MRV: pick unassigned variable with fewest available candidates
            best_var = -1
            best_count = W + 1
            for vi in range(idx, n):
                var = order[vi]
                if assignment[var] is not None:
                    continue
                avail = sum(1 for j in cands[var] if not used[j])
                if avail == 0:
                    return False
                if avail < best_count:
                    best_count = avail
                    best_var = vi

            if best_var == -1:
                return True

            var = order[best_var]
            for j in cands[var]:
                if not used[j]:
                    used[j] = True
                    assignment[var] = l + j
                    if backtrack(idx + 1):
                        return True
                    used[j] = False
                    assignment[var] = None
            return False

        if backtrack(0):
            result = [assignment[i] for i in range(n)]
            for i in range(n):
                if not (isinstance(result[i], int) and result[i] % divisors[i] == 0 and 1 <= result[i] <= max_val):
                    return None
            return result
        return None

    def feasible(l, W):
        """Quick feasibility check."""
        if W < n:
            return False
        if l + W > max_val:
            return False
        for i in range(n):
            d = divisors[i]
            rem = (-l) % d
            if rem >= W:
                return False
        return True

    best_diff = float('inf')
    best_result = None

    # Phase 1: Sliding window approach
    # Generate all multiples of each divisor up to MAX_VAL
    # Then sort all values and use sliding window to find minimal span
    # containing at least one multiple for each divisor

    # For efficiency, we'll generate multiples for each divisor and merge them
    # with a sliding window approach

    # First, let's try a more targeted approach:
    # For each possible starting position L (1 to period), find the minimal W
    # using binary search + feasibility check

    # Phase 1a: Try small W values first with fine-grained L scan
    for W in range(n - 1, n - 1 + 20):
        if time.time() > deadline - 30:
            break
        # Scan L in steps for this W
        step = max(1, period // 5000)
        for L in range(1, period + 1, step):
            if time.time() > deadline - 30:
                break
            if feasible(L, W):
                result = find_assignment(L, W)
                if result:
                    if W < best_diff:
                        best_diff = W
                        best_result = result
                    break  # Found for this W, try smaller W

    # Phase 1b: If found, try to improve with finer L scan for smaller W
    if best_diff < float('inf'):
        for W in range(n - 1, best_diff):
            if time.time() > deadline - 15:
                break
            # Finer scan for smaller W
            for L in range(1, period + 1):
                if time.time() > deadline - 15:
                    break
                if feasible(L, W):
                    result = find_assignment(L, W)
                    if result:
                        if W < best_diff:
                            best_diff = W
                            best_result = result
                        break  # Found for this W, try smaller W

    # Phase 2: Binary search on W for each L
    if time.time() < deadline - 5:
        for L in range(1, period + 1):
            if time.time() > deadline - 3:
                break

            hi = min(best_diff - 1, max_val - L) if best_diff < float('inf') else min(50, max_val - L)
            lo = n - 1

            if lo > hi:
                continue

            found_width = None
            while lo <= hi:
                if time.time() > deadline - 3:
                    break
                mid = (lo + hi) // 2
                if feasible(L, mid):
                    result = find_assignment(L, mid)
                    if result:
                        found_width = mid
                        hi = mid - 1
                    else:
                        lo = mid + 1
                else:
                    lo = mid + 1

            if found_width is not None and found_width < best_diff:
                best_diff = found_width
                best_result = find_assignment(L, found_width)

    # Phase 3: Sliding window approach on merged sorted list of all multiples
    # This is a different strategy: generate all multiples, sort, slide window
    if time.time() < deadline - 2:
        # Generate all multiples for each divisor
        all_multiples = []  # (value, divisor_index)
        for i, d in enumerate(divisors):
            for m in range(d, max_val + 1, d):
                all_multiples.append((m, i))

        all_multiples.sort()

        # Sliding window: find shortest window containing all 10 divisor types
        left = 0
        counts = [0] * n
        unique_count = 0
        best_sw_diff = float('inf')
        best_sw_left = -1
        best_sw_right = -1

        for right in range(len(all_multiples)):
            d_idx = all_multiples[right][1]
            counts[d_idx] += 1
            if counts[d_idx] == 1:
                unique_count += 1

            while unique_count == n:
                # Check if we have at least one of each divisor
                span = all_multiples[right][0] - all_multiples[left][0]
                if span < best_sw_diff:
                    best_sw_diff = span
                    best_sw_left = left
                    best_sw_right = right

                # Try to shrink from left
                d_idx_left = all_multiples[left][1]
                counts[d_idx_left] -= 1
                if counts[d_idx_left] == 0:
                    unique_count -= 1
                left += 1

        # If sliding window found a better solution, extract the assignment
        if best_sw_diff < best_diff:
            # Extract the values from the window
            window_vals = [all_multiples[i][0] for i in range(best_sw_left, best_sw_right + 1)]
            window_divs = [all_multiples[i][1] for i in range(best_sw_left, best_sw_right + 1)]

            # We need to pick one value for each divisor
            # Use the same backtracking approach on this window
            L = window_vals[0]
            W = window_vals[-1] - L + 1

            result = find_assignment(L, W)
            if result and (result[-1] - result[0]) <= best_sw_diff:
                if (result[-1] - result[0]) < best_diff:
                    best_diff = result[-1] - result[0]
                    best_result = result

    if best_result:
        return best_result

    return [divisors[i] for i in range(n)]
