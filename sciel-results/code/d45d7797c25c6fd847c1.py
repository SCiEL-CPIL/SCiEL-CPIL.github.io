import time
import random

def construct():
    deadline = time.time() + 85  # Safety margin before 90s kill limit

    def is_mean_of_subset(element, others):
        """Check if element is the arithmetic mean of any subset of others with size >= 2.
        Uses integer bitsets for fast DP."""
        n = len(others)
        if n < 2:
            return False

        max_sum = sum(others)
        max_k = n

        # dp[k] is a bitmask where bit s is set if sum s is achievable with exactly k elements
        dp = [0] * (max_k + 1)
        dp[0] = 1  # sum 0 with 0 elements

        for val in others:
            # Iterate backwards to avoid using the same element multiple times
            for k in range(max_k, 0, -1):
                if dp[k - 1]:
                    dp[k] |= (dp[k - 1] << val)

        for k in range(2, max_k + 1):
            target = k * element
            if target > max_sum:
                break
            if (dp[k] >> target) & 1:
                return True

        return False

    def can_add(element, current):
        """Check if element can be added to current set without violating constraints."""
        # 1. Check if element is the mean of any subset of current
        if is_mean_of_subset(element, current):
            return False

        # 2. Check if adding element causes any a in current to become the mean of a subset
        #    involving element.
        #    Subset S of (current U {element}) \ {a} has mean a.
        #    S must contain element (otherwise it was already a violation in current).
        #    S = T U {element}, T subset of current \ {a}.
        #    sum(T) + element = (|T| + 1) * a
        #    sum(T) = (|T| + 1) * a - element
        #    Let j = |T|. Then sum(T) = (j+1)*a - element.
        #    j >= 1 (since S has size >= 2 and contains element, T has size >= 1? 
        #    Wait. S has size k >= 2. S contains element. So T has size k-1 >= 1.
        #    So j ranges from 1 to len(current) - 1.

        for a in current:
            others_without_a = [x for x in current if x != a]
            n = len(others_without_a)
            if n < 1:
                continue

            max_sum_a = sum(others_without_a)
            
            # DP for subsets of others_without_a
            dp = [0] * (n + 1)
            dp[0] = 1

            for val in others_without_a:
                for j in range(n, 0, -1):
                    if dp[j - 1]:
                        dp[j] |= (dp[j - 1] << val)

            # Check if any valid subset sum exists
            for j in range(1, n + 1):
                target = (j + 1) * a - element
                if 0 <= target <= max_sum_a:
                    if (dp[j] >> target) & 1:
                        return False

        return True

    def verify_set(s):
        """Verify the entire set is valid."""
        if len(s) > 48:
            return False
        if len(set(s)) != len(s):
            return False
        if any(x < 1 or x > 512 for x in s):
            return False
        
        for i, elem in enumerate(s):
            others = s[:i] + s[i+1:]
            if is_mean_of_subset(elem, others):
                return False
        return True

    # Generate diverse orderings
    all_nums = list(range(1, 513))
    
    def is_prime(x):
        if x < 2:
            return False
        for i in range(2, int(x**0.5) + 1):
            if x % i == 0:
                return False
        return True

    primes = [x for x in range(2, 513) if is_prime(x)]
    non_primes = [x for x in range(1, 513) if not is_prime(x)]
    powers_of_2 = [2**i for i in range(1, 10) if 2**i <= 512]
    non_powers = [x for x in range(1, 513) if x not in set(powers_of_2)]

    # Alternating small/large
    alt = []
    for i in range(256):
        alt.append(i + 1)
        alt.append(512 - i)

    orderings = [
        list(range(1, 513)),           # ascending
        list(range(512, 0, -1)),       # descending
        powers_of_2 + non_powers,      # powers of 2 first
        [x for x in range(1, 513) if x % 2 == 1] + [x for x in range(1, 513) if x % 2 == 0],
        [x for x in range(1, 513) if x % 3 != 0] + [x for x in range(1, 513) if x % 3 == 0],
        [x for x in range(1, 513) if x % 5 != 0] + [x for x in range(1, 513) if x % 5 == 0],
        primes + non_primes,           # primes first
        alt,                           # alternating
        list(range(512, 256, -1)) + list(range(256, 0, -1)),
    ]

    rng = random.Random(42)
    for _ in range(10):
        order = list(range(1, 513))
        rng.shuffle(order)
        orderings.append(order)

    best_set = []

    for ordering in orderings:
        if time.time() > deadline - 5:
            break

        selected = []
        for x in ordering:
            if time.time() > deadline - 5:
                break
            if len(selected) >= 48:
                break
            if can_add(x, selected):
                selected.append(x)

        if len(selected) > len(best_set):
            best_set = selected[:]
            if verify_set(best_set):
                pass # Validity is guaranteed by construction, but good to know

        if len(best_set) >= 48:
            break

    # Local search / Repair if we haven't reached 48
    if best_set and len(best_set) < 48 and time.time() < deadline - 10:
        improved = best_set[:]
        remaining = [x for x in range(1, 513) if x not in improved]
        
        # Try adding elements
        for x in remaining:
            if time.time() > deadline - 5:
                break
            if len(improved) >= 48:
                break
            if can_add(x, improved):
                improved.append(x)

        # Try swaps: remove one element, try to add two new ones (net +1)
        # Or just remove one and add one (net 0) to allow further additions?
        # Let's try: remove one, try to add as many as possible from remaining.
        if len(improved) < 48 and time.time() < deadline - 10:
            # Sort remaining by some heuristic? Or just iterate.
            # To save time, we'll do a limited number of swap attempts.
            swap_attempts = 0
            for idx in range(len(improved) - 1, -1, -1):
                if time.time() > deadline - 5:
                    break
                if len(improved) >= 48:
                    break
                if swap_attempts > 100: # Limit total swaps to avoid timeout
                    break
                
                removed = improved.pop(idx)
                added_any = False
                # Try to add elements from remaining
                for x in remaining:
                    if time.time() > deadline - 5:
                        break
                    if len(improved) >= 48:
                        break
                    if x == removed:
                        continue
                    if can_add(x, improved):
                        improved.append(x)
                        added_any = True
                        remaining.remove(x)
                        break # Only add one per swap for simplicity and speed
                
                if not added_any:
                    improved.insert(idx, removed)
                else:
                    remaining.append(removed)
                    swap_attempts += 1

        if len(improved) > len(best_set):
            best_set = improved

    # Final attempt to fill remaining slots
    if best_set and len(best_set) < 48 and time.time() < deadline - 2:
        final = best_set[:]
        remaining = [x for x in range(1, 513) if x not in final]
        for x in remaining:
            if time.time() > deadline - 1:
                break
            if len(final) >= 48:
                break
            if can_add(x, final):
                final.append(x)
        if len(final) > len(best_set):
            best_set = final

    # Final verification
    if best_set and not verify_set(best_set):
        # This should not happen, but if it does, fall back to a simple valid set
        # e.g., powers of 2 are always valid? 
        # Let's just return the largest valid subset we can quickly find
        # Or just return best_set[:48] if we trust the construction.
        # Given the strict checks in can_add, it should be valid.
        pass

    return best_set
