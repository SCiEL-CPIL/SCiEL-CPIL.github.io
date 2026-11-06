import time
import random
import bisect

N = 256

def construct():
    start_time = time.time()
    deadline = 80.0  # safety margin before 90s limit

    def is_valid_set(S):
        """Check if set S satisfies the property."""
        S = sorted(S)
        for i in range(len(S)):
            a = S[i]
            for j in range(i + 1, len(S)):
                b = S[j]
                for k in range(j, len(S)):
                    c = S[k]
                    if (b + c) % a == 0:
                        return False
        return True

    def can_add(S, S_sorted, x):
        """Check if x can be added to set S (with S_sorted being sorted list) while maintaining validity."""
        if x in S:
            return True
        pos = bisect.bisect_left(S_sorted, x)
        n = len(S_sorted)

        # x as 'a': for all b,c in S with x<b<=c, check x ∤ (b+c)
        for i in range(pos, n):
            b = S_sorted[i]
            if b <= x:
                continue
            target = (-b) % x
            for j in range(i, n):
                c = S_sorted[j]
                if c < b:
                    continue
                if c % x == target:
                    return False

        # x as 'b': for all a,c in S with a<x<=c, check a ∤ (x+c)
        for i in range(pos):
            a = S_sorted[i]
            # Check c = x: (x + x) % a
            if (2 * x) % a == 0:
                return False
            # Check c in S_sorted[pos:]
            for j in range(pos, n):
                c = S_sorted[j]
                if (x + c) % a == 0:
                    return False

        # x as 'c': for all a,b in S with a<b<=x, check a ∤ (b+x)
        for i in range(n):
            a = S_sorted[i]
            for j in range(i + 1, n):
                b = S_sorted[j]
                if b > x:
                    break
                if (b + x) % a == 0:
                    return False

        return True

    def greedy_extend(S, S_sorted, candidates):
        """Try to add elements from candidates to S greedily."""
        S = set(S)
        S_sorted = sorted(S)
        for x in candidates:
            if x in S:
                continue
            if can_add(S, S_sorted, x):
                S.add(x)
                pos = bisect.bisect_left(S_sorted, x)
                S_sorted.insert(pos, x)
        return S, S_sorted

    # Strategy 1: Start with upper tail and extend
    # Find the best starting point
    best_set = []
    best_sorted = []

    for min_start in [171, 180, 192, 200, 210]:
        if time.time() - start_time > deadline:
            break

        # Start with all elements from min_start to N
        S = set(range(min_start, N + 1))
        S_sorted = list(range(min_start, N + 1))

        # Try to extend by adding smaller elements
        candidates = list(range(1, min_start))

        # Try ascending
        S1, S1_sorted = greedy_extend(S, S_sorted, candidates)

        # Try descending
        S2, S2_sorted = greedy_extend(S, S_sorted, sorted(candidates, reverse=True))

        # Try random shuffle
        random.shuffle(candidates)
        S3, S3_sorted = greedy_extend(S, S_sorted, candidates)

        # Pick the best
        best_candidate = max([(S1, S1_sorted), (S2, S2_sorted), (S3, S3_sorted)], key=lambda x: len(x[0]))

        if len(best_candidate[0]) > len(best_set):
            best_set = best_candidate[0]
            best_sorted = best_candidate[1]

    # Strategy 2: Local search / hill climbing
    current_set = set(best_set)
    current_sorted = sorted(current_set)
    current_size = len(current_set)

    # WalkSAT-style local search
    p_flip = 0.4
    max_steps = 3000

    for step in range(max_steps):
        if time.time() - start_time > deadline:
            break

        if current_size <= 1:
            break

        # Try to add elements
        others = [x for x in range(1, N + 1) if x not in current_set]
        random.shuffle(others)

        improved = False
        for candidate in others[:15]:
            if can_add(current_set, current_sorted, candidate):
                current_set.add(candidate)
                pos = bisect.bisect_left(current_sorted, candidate)
                current_sorted.insert(pos, candidate)
                current_size += 1
                if current_size > len(best_set):
                    best_set = list(current_sorted)
                improved = True
                break

        if not improved:
            # Remove a random element to allow addition
            idx = random.randint(0, current_size - 1)
            removed = current_sorted.pop(idx)
            current_set.discard(removed)
            current_size -= 1

            # Try to add back or add new elements
            others = [x for x in range(1, N + 1) if x not in current_set]
            random.shuffle(others)

            added_count = 0
            for candidate in others[:10]:
                if added_count >= 2:
                    break
                if can_add(current_set, current_sorted, candidate):
                    current_set.add(candidate)
                    pos = bisect.bisect_left(current_sorted, candidate)
                    current_sorted.insert(pos, candidate)
                    current_size += 1
                    added_count += 1

                    if current_size > len(best_set):
                        best_set = list(current_sorted)

    # Final validation
    if not is_valid_set(best_set):
        # Fallback to a known valid set
        best_set = list(range(171, 257))

    return best_set
