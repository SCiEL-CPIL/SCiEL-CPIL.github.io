"""Rewritten program for cardinality-separated subset sums."""
import time

def construct():
    start_time = time.time()
    deadline = start_time + 50  # safety margin

    def elapsed():
        return time.time() - start_time

    # Known good baseline: powers of 2 (unique subset sums, hence cardinality-separated)
    baseline = [1, 2, 4, 8, 16, 32, 64, 128, 256]

    def is_valid_consecutive(a):
        """Check if list a (sorted) satisfies the consecutive-block separation condition.
        For each k >= 1: sum of k smallest > sum of (k-1) largest."""
        n = len(a)
        if n == 0:
            return True
        prefix = [0] * (n + 1)
        for i in range(n):
            prefix[i + 1] = prefix[i] + a[i]
        # For k elements: min_k_sum = prefix[k], max_(k-1)_sum = prefix[n] - prefix[n-(k-1)]
        for k in range(1, n + 1):
            min_k = prefix[k]
            if k - 1 >= 1:
                max_km1 = prefix[n] - prefix[n - (k - 1)]
                if min_k <= max_km1:
                    return False
        return True

    def try_add(a, x):
        """Try adding x to sorted list a. Returns True if consecutive condition still holds."""
        n = len(a)
        if n == 0:
            return True
        # New list would be a with x inserted in sorted position
        # Check the condition for the new list
        # We can check efficiently: the condition is
        # for each k: sum of k smallest of new > sum of (k-1) largest of new
        # Let's just build the new list and check
        new_a = list(a)
        # Insert x in sorted position
        lo, hi = 0, len(new_a)
        while lo < hi:
            mid = (lo + hi) // 2
            if new_a[mid] < x:
                lo = mid + 1
            else:
                hi = mid
        new_a.insert(lo, x)
        return is_valid_consecutive(new_a)

    def greedy_smallest(a, deadline):
        """Greedily add the smallest valid next number."""
        a = list(a)
        while True:
            if elapsed() > deadline:
                break
            # Try smallest valid next number
            start = a[-1] + 1 if a else 1
            found = None
            # We need to search for the smallest x >= start such that adding x works
            # But we also need x <= 256
            # Let's search from start upward
            # Optimization: the smallest valid x can be found by checking sequentially
            # but we can skip many values. The condition is:
            # Let new list have n+1 elements. For k=n+1: min_{n+1} = total sum > max_n = total - min_element
            #   => min_element > 0, always true.
            # For k=n: min_n (sum of n smallest of new) > max_{n-1} (sum of n-1 largest of new)
            #   The n smallest of new: if x is the largest, they're the old n elements: sum = total_old
            #   The n-1 largest of new: depends on where x fits.
            # This is complex, so let's just check sequentially but with a jump.
            
            # Actually, let's just check from start to 256, but break early
            # To speed up, note that if x is very large, it's more likely valid.
            # But we want the SMALLEST valid x to keep options open.
            # Let's check sequentially.
            for x in range(start, 257):
                if elapsed() > deadline:
                    break
                if try_add(a, x):
                    found = x
                    break
            if found is None:
                break
            # Add found
            a.append(found)
            a.sort()
        return a

    def greedy_largest(a, deadline):
        """Greedily add the largest valid next number (to leave room for smaller ones later? No, we want max size).
        Actually, adding largest first might block smaller ones. Let's try smallest-first which is better for maximizing count."""
        return greedy_smallest(a, deadline)

    # Strategy 1: Start from empty, greedily add smallest valid
    result1 = greedy_smallest([], deadline)
    best = result1
    best_size = len(best)

    # Strategy 2: Start from a small seed and extend
    # Try different seeds
    for seed in [[1], [2], [1, 2], [1, 3], [2, 4], [1, 2, 4]]:
        if elapsed() > deadline:
            break
        result_s = greedy_smallest(seed, deadline)
        if len(result_s) > best_size:
            best = result_s
            best_size = len(size_s) if False else len(result_s)

    # Strategy 3: Local search - try to improve by replacing elements
    # For the best set, try removing one element and adding a different one
    if elapsed() < deadline - 10:
        improved = True
        while improved and elapsed() < deadline - 5:
            improved = False
            best = list(best)
            for i in range(len(best)):
                if elapsed() > deadline:
                    break
                # Remove best[i]
                old = best[i]
                without = [x for x in best if x != old]
                # Try to add a different number
                for x in range(1, 257):
                    if elapsed() > deadline:
                        break
                    if x == old or x in without:
                        continue
                    if try_add(without, x):
                        new_set = without + [x]
                        new_set.sort()
                        # Check if this improves (it won't in size, but might allow future growth)
                        # For now, just accept if valid
                        # Actually, we want to see if this leads to a larger final set
                        extended = greedy_smallest(new_set, deadline - 5)
                        if len(extended) > best_size:
                            best = extended
                            best_size = len(extended)
                            improved = True
                            break
                if improved:
                    break

    # Final validation
    if not is_valid_consecutive(best):
        # Fallback
        best = baseline

    # Ensure distinct ints in 1..256
    best = sorted(set(int(x) for x in best if 1 <= x <= 256))

    # Final sanity check
    if not is_valid_consecutive(best):
        best = baseline

    return best
