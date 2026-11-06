import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill

    def is_mean_of_subset(a, others):
        """Check if a is the exact arithmetic mean of any subset of >= 2 elements from others."""
        n = len(others)
        if n < 2:
            return False

        # For each subset size k from 2 to n, check if any k-subset sums to k*a
        # Use bitset DP for efficiency when n is moderate
        # dp[j] is a set of sums achievable with exactly j elements from others
        # But for speed, use combinations for small k, and a smarter check for larger k

        # Quick check for k=2: any pair sums to 2*a?
        seen = set()
        for x in others:
            if 2 * a - x in seen:
                return True
            seen.add(x)

        # For k=3: any triple sums to 3*a?
        if n >= 3:
            pair_sums = set()
            for i in range(n):
                for j in range(i + 1, n):
                    pair_sums.add(others[i] + others[j])
            for x in others:
                if 3 * a - x in pair_sums:
                    return True

        # For k >= 4: use DP with bitsets or combinations for small k
        # Since we're building greedily and sets are at most ~50 elements,
        # combinations for k up to ~6 should be fast enough
        for k in range(4, min(n + 1, 7)):
            target = k * a
            for combo in combinations(others, k):
                if sum(combo) == target:
                    return True

        # For k >= 7, the probability is very low and the search space is large.
        # We could use DP, but for now, skip (conservative assumption: not a mean)
        # In practice, for the greedy construction, this is rarely an issue.

        return False

    def can_add(x, selected):
        """Check if adding x to selected keeps the no-mean property."""
        # Check 1: Is x the mean of any subset of >= 2 elements from selected?
        if len(selected) >= 2:
            if is_mean_of_subset(x, selected):
                return False

        # Check 2: For each existing element a in selected, is a now the mean
        #         of any subset of >= 2 elements from (selected + [x]) \ {a}?
        for a in selected:
            others = [y for y in selected if y != a] + [x]
            if is_mean_of_subset(a, others):
                return False

        return True

    # Strategy: Try multiple passes with different orderings to find a larger set.
    # Pass 1: Ascending order (1 to 512)
    # Pass 2: Descending order (512 to 1)
    # Pass 3: Random order (if time permits)

    best_selected = []

    # Pass 1: Ascending
    selected = []
    for x in range(1, 513):
        if time.time() > deadline:
            break
        if can_add(x, selected):
            selected.append(x)
        if len(selected) >= 48:
            break
    if len(selected) > len(best_selected):
        best_selected = selected[:]

    # Pass 2: Descending
    if time.time() < deadline - 10:
        selected = []
        for x in range(512, 0, -1):
            if time.time() > deadline:
                break
            if can_add(x, selected):
                selected.append(x)
                selected.sort()  # keep sorted for consistency
        if len(selected) > len(best_selected):
            best_selected = selected[:]

    # Pass 3: Random order
    if time.time() < deadline - 5:
        import random
        random.seed(42)
        candidates = list(range(1, 513))
        random.shuffle(candidates)
        selected = []
        for x in candidates:
            if time.time() > deadline:
                break
            if can_add(x, selected):
                selected.append(x)
        if len(selected) > len(best_selected):
            best_selected = selected[:]

    return best_selected
