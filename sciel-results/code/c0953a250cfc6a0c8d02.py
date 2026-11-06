"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    deadline = time.time() + 80.0  # safety margin before 90s kill

    def subset_sum_exists(others, k, target):
        """Check if any k-subset of others sums to exactly target.
        Uses bitset DP for efficiency when k is small, combinations otherwise."""
        n = len(others)
        if k < 2 or k > n:
            return False
        if k == 2:
            seen = set()
            for x in others:
                if target - x in seen:
                    return True
                seen.add(x)
            return False
        if k == 3:
            pair_sums = set()
            for i in range(n):
                for j in range(i + 1, n):
                    pair_sums.add(others[i] + others[j])
            for x in others:
                if target - x in pair_sums:
                    return True
            return False
        # For k >= 4, use DP with sets of achievable sums
        # dp[j] = set of sums achievable with j elements
        if k <= 6:
            for combo in combinations(others, k):
                if sum(combo) == target:
                    return True
            return False
        # For larger k, use bitset-style DP
        # dp[j] is a set of achievable sums with exactly j elements
        dp = [set() for _ in range(k + 1)]
        dp[0].add(0)
        for x in others:
            for j in range(min(k, n), 0, -1):
                for s in dp[j - 1]:
                    ns = s + x
                    if ns <= target:
                        dp[j].add(ns)
                    if ns > target:
                        break
            if target in dp[k]:
                return True
        return False

    def is_valid_mean(a, others):
        """Check if a is the mean of any subset of size >= 2 from others."""
        n = len(others)
        for k in range(2, n + 1):
            target = k * a
            if subset_sum_exists(others, k, target):
                return True
        return False

    def is_valid_addition(x, selected):
        """Check if adding x to selected keeps the property."""
        # Check 1: Is x the mean of any subset of selected with size >= 2?
        if len(selected) >= 2:
            if is_valid_mean(x, selected):
                return False

        # Check 2: For each existing element a, does adding x make a the mean
        # of some subset of (selected + [x]) \ {a} with size >= 2?
        for a in selected:
            # Subsets that don't include x: already valid by induction
            # Subsets that include x: need U ⊆ selected\{a}, |U| >= 1,
            # such that sum(U) + x = (|U| + 1) * a
            # i.e., sum(U) = (|U| + 1) * a - x, with |U| from 1 to len(selected)-1
            others_without_a = [y for y in selected if y != a]
            m = len(others_without_a)
            for k in range(1, m + 1):
                target = (k + 1) * a - x
                if target <= 0:
                    continue
                if subset_sum_exists(others_without_a, k, target):
                    return False
        return True

    def greedy_build(order):
        """Build a valid set by trying elements in the given order."""
        selected = []
        for x in order:
            if time.time() > deadline:
                break
            if len(selected) >= 48:
                break
            if is_valid_addition(x, selected):
                selected.append(x)
        return selected

    # Try multiple strategies to find the largest valid set
    best = []

    # Strategy 1: Ascending order
    best = greedy_build(range(1, 513))

    # Strategy 2: Descending order
    if time.time() < deadline:
        candidate = greedy_build(range(512, 0, -1))
        if len(candidate) > len(best):
            best = candidate

    # Strategy 3: Randomized orders (multiple attempts)
    random.seed(42)
    attempts = 0
    while time.time() < deadline - 5.0 and attempts < 20:
        order = list(range(1, 513))
        random.shuffle(order)
        candidate = greedy_build(order)
        if len(candidate) > len(best):
            best = candidate
        attempts += 1

    # Strategy 4: Local search - try to improve by swapping
    if time.time() < deadline - 10.0:
        current = list(best)
        improved = True
        while improved and time.time() < deadline - 10.0:
            improved = False
            for i, x in enumerate(current):
                # Try removing x and adding a different element
                remaining = [y for y in current if y != x]
                for y in range(1, 513):
                    if y in current:
                        continue
                    # Check if we can swap x for y
                    # First check if y is valid to add to remaining
                    if not is_valid_addition(y, remaining):
                        continue
                    new_set = remaining + [y]
                    # Verify the whole set is valid
                    valid = True
                    for a in new_set:
                        others = [b for b in new_set if b != a]
                        if is_valid_mean(a, others):
                            valid = False
                            break
                    if valid:
                        current = new_set
                        improved = True
                        break
                if improved:
                    break
        if len(current) > len(best):
            best = current

    return best
# EVOLVE-BLOCK-END
