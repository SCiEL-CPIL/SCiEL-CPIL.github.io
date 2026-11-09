"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill
    n = 256

    def time_left():
        return DEADLINE - (time.time() - start_time)

    def is_valid_set(S):
        """Check if set S satisfies the property."""
        s = sorted(S)
        for i in range(len(s)):
            a = s[i]
            for j in range(i + 1, len(s)):
                b = s[j]
                for k in range(j, len(s)):
                    c = s[k]
                    if (b + c) % a == 0:
                        return False
        return True

    def can_add(S, x):
        """Check if x can be added to set S while maintaining validity."""
        s = sorted(S + [x])
        for i in range(len(s)):
            a = s[i]
            for j in range(i + 1, len(s)):
                b = s[j]
                for k in range(j, len(s)):
                    c = s[k]
                    if (b + c) % a == 0:
                        return False
        return True

    # Strategy 1: Greedy from small to large, prioritizing numbers with fewer divisors
    def greedy_small_to_large():
        selected = []
        for num in range(1, n + 1):
            if time_left() < 2.0:
                break
            if can_add(selected, num):
                selected.append(num)
        return selected

    # Strategy 2: Greedy from large to small
    def greedy_large_to_small():
        selected = []
        for num in range(n, 0, -1):
            if time_left() < 2.0:
                break
            if can_add(selected, num):
                selected.append(num)
        return selected

    # Strategy 3: Greedy with divisor-count prioritization
    # Numbers with fewer divisors create fewer conflicts
    def divisor_count(x):
        count = 0
        for d in range(1, int(x**0.5) + 1):
            if x % d == 0:
                count += 1
                if d != x // d:
                    count += 1
        return count

    def greedy_divisor_prio():
        selected = []
        # Sort by divisor count ascending, then by value
        candidates = sorted(range(1, n + 1), key=lambda x: (divisor_count(x), x))
        for num in candidates:
            if time_left() < 2.0:
                break
            if can_add(selected, num):
                selected.append(num)
        return selected

    # Strategy 4: Start with upper tail, try to add smaller numbers
    def upper_tail_plus():
        selected = list(range(171, n + 1))
        # Try adding smaller numbers
        for num in range(1, 171):
            if time_left() < 2.0:
                break
            if can_add(selected, num):
                selected.append(num)
        return selected

    # Strategy 5: Local search - start from a good set, try swaps
    def local_search(initial_set, max_iterations=5000):
        S = list(initial_set)
        S_set = set(S)
        best = len(S)
        available = [x for x in range(1, n + 1) if x not in S_set]
        
        for _ in range(max_iterations):
            if time_left() < 3.0:
                break
            improved = False
            # Try adding a single element
            for x in available:
                if x in S_set:
                    continue
                if can_add(S, x):
                    S.append(x)
                    S_set.add(x)
                    improved = True
                    best = max(best, len(S))
                    break
            if improved:
                continue
            # Try swaps: remove one, add one
            for i in range(len(S)):
                if time_left() < 3.0:
                    break
                removed = S.pop(i)
                S_set.discard(removed)
                for x in available:
                    if x in S_set:
                        continue
                    if can_add(S, x):
                        S.append(x)
                        S_set.add(x)
                        improved = True
                        best = max(best, len(S))
                        break
                if not improved:
                    S.insert(i, removed)
                    S_set.add(removed)
            if not improved:
                break
        return S

    # Run multiple strategies and pick the best
    best_set = []
    best_len = 0

    strategies = [
        greedy_small_to_large,
        greedy_large_to_small,
        greedy_divisor_prio,
        upper_tail_plus,
    ]

    for strategy in strategies:
        if time_left() < 5.0:
            break
        S = strategy()
        if len(S) > best_len:
            best_len = len(S)
            best_set = S

    # Try local search on the best set found
    if time_left() > 5.0 and best_set:
        S = local_search(best_set)
        if len(S) > best_len:
            best_len = len(S)
            best_set = S

    # Try local search on other good sets too
    if time_left() > 5.0:
        for strategy in strategies[:2]:
            if time_left() < 5.0:
                break
            S = strategy()
            if len(S) >= best_len - 5:
                S = local_search(S, max_iterations=3000)
                if len(S) > best_len:
                    best_len = len(S)
                    best_set = S

    # Final verification
    if best_set and not is_valid_set(best_set):
        # Fallback to safe upper tail
        best_set = list(range(192, 257))

    if not best_set:
        best_set = list(range(192, 257))

    return best_set
# EVOLVE-BLOCK-END
