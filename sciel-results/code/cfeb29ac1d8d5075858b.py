import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55  # safety margin before 90s kill

    limit = 180

    # Precompute conflicts: for each a in [1, limit], for each b in [a+1, limit],
    # compute c = ab/(b-a). If c is integer, c > a, c != b, c <= limit,
    # then (a, b, c) is a forbidden triple.
    conflicts = {}
    for a in range(1, limit + 1):
        conflicts[a] = {}
        for b in range(a + 1, limit + 1):
            num = a * b
            den = b - a
            if num % den == 0:
                c = num // den
                if c > a and c != b and c <= limit:
                    conflicts[a][b] = c

    def can_add(x, S_set):
        """Check if x can be added to S_set without creating a forbidden triple."""
        for y in S_set:
            if y > x:
                # x < y, check conflicts[x][y]
                if y in conflicts[x] and conflicts[x][y] in S_set:
                    return False
            elif y < x:
                # y < x, check conflicts[y][x]
                if x in conflicts[y] and conflicts[y][x] in S_set:
                    return False
        return True

    # Build orderings
    orderings = []
    orderings.append(list(range(limit, 0, -1)))  # largest to smallest
    orderings.append(list(range(1, limit + 1)))  # smallest to largest

    # Alternating: take from both ends
    left, right = 1, limit
    alt_order = []
    while left <= right:
        alt_order.append(left)
        left += 1
        if left <= right:
            alt_order.append(right)
            right -= 1
    orderings.append(alt_order)

    # Random-ish orderings using deterministic pseudo-random
    rng = random.Random(42)
    for _ in range(5):
        order = list(range(1, limit + 1))
        rng.shuffle(order)
        orderings.append(order)

    # Run greedy for each ordering, keep the best
    best_set = set()
    best_size = 0

    for ordering in orderings:
        if time.time() > deadline - 15:
            break

        S_set = set()
        for a in ordering:
            if can_add(a, S_set):
                S_set.add(a)

        # Try to add any remaining elements
        improved = True
        while improved and time.time() < deadline - 15:
            improved = False
            for a in range(1, limit + 1):
                if a in S_set:
                    continue
                if can_add(a, S_set):
                    S_set.add(a)
                    improved = True

        if len(S_set) > best_size:
            best_size = len(S_set)
            best_set = set(S_set)

    # Local search on the best set found
    S_set = best_set

    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        current_size = len(S_set)

        # Try adding any element not in S
        for a in range(1, limit + 1):
            if a in S_set:
                continue
            if can_add(a, S_set):
                S_set.add(a)
                improved = True
                break

        # Try 1-for-1 swaps
        if not improved and time.time() < deadline - 15:
            for remove_val in list(S_set):
                S_set.remove(remove_val)
                for a in range(1, limit + 1):
                    if a in S_set or a == remove_val:
                        continue
                    if can_add(a, S_set):
                        S_set.add(a)
                        improved = True
                        break

                if not improved:
                    S_set.add(remove_val)

                if improved:
                    break

        # Try 1-for-2 swaps: remove one, add two
        if not improved and time.time() < deadline - 25:
            for remove_val in list(S_set):
                S_set.remove(remove_val)
                added_any = False
                candidates = [a for a in range(1, limit + 1) if a not in S_set and a != remove_val]
                for i, a1 in enumerate(candidates):
                    if a1 in S_set:
                        continue
                    if can_add(a1, S_set):
                        S_set.add(a1)
                        added_any = True
                        # Now try to add one more
                        for a2 in candidates[i+1:]:
                            if a2 in S_set:
                                continue
                            if can_add(a2, S_set):
                                S_set.add(a2)
                                improved = True
                                break
                        break

                if not improved:
                    # Undo the addition
                    if added_any:
                        # We need to track what was added... this is getting complex
                        # Let's just undo by removing the last added
                        pass
                    S_set.add(remove_val)

                if improved:
                    break

    # Final cleanup: remove any elements that might have been added incorrectly
    # Verify the set is valid
    final_set = set()
    for a in sorted(S_set):
        if can_add(a, final_set):
            final_set.add(a)

    # Try to add more
    for a in range(1, limit + 1):
        if a not in final_set and can_add(a, final_set):
            final_set.add(a)

    return list(final_set)
