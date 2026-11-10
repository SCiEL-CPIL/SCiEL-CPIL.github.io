"""Rewritten program: structural change to a bitset-based constraint solver with randomized local search."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start = time.time()
    deadline = start + 85.0  # safety margin before 90s kill

    N = 512
    MAX_K = 48

    def is_valid(set_vals):
        """Check if no element is the mean of any subset of >=2 other elements.
        Uses bitset DP: for each a in set, check if k*a is achievable by k others."""
        m = len(set_vals)
        if m < 3:
            return True
        vals = sorted(set_vals)
        
        for a in vals:
            others = [v for v in vals if v != a]
            n_others = len(others)
            max_sum = n_others * N
            # dp[k] = bitset of sums achievable with exactly k elements from others
            dp = [0] * (n_others + 1)
            dp[0] = 1  # bit 0 set: sum 0 with 0 elements
            for v in others:
                for k in range(n_others, 0, -1):
                    dp[k] |= dp[k - 1] << v
            for k in range(2, n_others + 1):
                target = k * a
                if target <= max_sum and (dp[k] >> target) & 1:
                    return False
        return True

    def count_conflicts(set_vals):
        """Count how many (a, k) pairs violate the constraint. 0 means valid."""
        m = len(set_vals)
        if m < 3:
            return 0
        vals = sorted(set_vals)
        conflicts = 0
        for a in vals:
            others = [v for v in vals if v != a]
            n_others = len(others)
            max_sum = n_others * N
            dp = [0] * (n_others + 1)
            dp[0] = 1
            for v in others:
                for k in range(n_others, 0, -1):
                    dp[k] |= dp[k - 1] << v
            for k in range(2, n_others + 1):
                target = k * a
                if target <= max_sum and (dp[k] >> target) & 1:
                    conflicts += 1
        return conflicts

    # Greedy construction with incremental checking
    def greedy_construct():
        selected = []
        for x in range(1, N + 1):
            if len(selected) >= MAX_K:
                break
            candidate = selected + [x]
            if is_valid(candidate):
                selected = candidate
        return selected

    # Try multiple greedy strategies with different orderings
    best_set = []
    best_size = 0

    # Strategy 1: ascending order
    s1 = greedy_construct()
    if len(s1) > best_size:
        best_size = len(s1)
        best_set = s1[:]
    
    if time.time() > deadline - 30:
        return best_set

    # Strategy 2: descending order
    selected = []
    for x in range(N, 0, -1):
        if len(selected) >= MAX_K:
            break
        candidate = selected + [x]
        if is_valid(candidate):
            selected = candidate
    if len(selected) > best_size:
        best_size = len(selected)
        best_set = selected[:]

    if time.time() > deadline - 20:
        return best_set

    # Strategy 3: random orderings
    random.seed(42)
    for trial in range(5):
        if time.time() > deadline - 15:
            break
        order = list(range(1, N + 1))
        random.shuffle(order)
        selected = []
        for x in order:
            if len(selected) >= MAX_K:
                break
            candidate = selected + [x]
            if is_valid(candidate):
                selected = candidate
        if len(selected) > best_size:
            best_size = len(selected)
            best_set = selected[:]

    if best_size >= MAX_K:
        return best_set

    # Local search: try to improve by adding elements or swapping
    current = set(best_set)
    current_conflicts = count_conflicts(list(current))
    
    for iteration in range(50000):
        if time.time() > deadline - 3:
            break
        
        # Try adding a random element
        if len(current) < MAX_K:
            candidates = [x for x in range(1, N + 1) if x not in current]
            if candidates:
                # Try a few random candidates
                for _ in range(min(3, len(candidates))):
                    x = random.choice(candidates)
                    new_set = list(current | {x})
                    nc = count_conflicts(new_set)
                    if nc == 0:
                        current = current | {x}
                        current_conflicts = 0
                        if len(current) > best_size:
                            best_size = len(current)
                            best_set = list(current)
                        break
                    elif nc < current_conflicts:
                        current = current | {x}
                        current_conflicts = nc
        
        # Try swapping: remove one, add one
        if current:
            a = random.choice(list(current))
            candidates = [x for x in range(1, N + 1) if x not in current]
            if candidates:
                b = random.choice(candidates)
                new_set = list((current - {a}) | {b})
                nc = count_conflicts(new_set)
                if nc == 0:
                    current = (current - {a}) | {b}
                    current_conflicts = 0
                    if len(current) > best_size:
                        best_size = len(current)
                        best_set = list(current)
                elif nc < current_conflicts:
                    current = (current - {a}) | {b}
                    current_conflicts = nc

        # Occasionally try to remove an element and re-add
        if current and len(current) > 5:
            a = random.choice(list(current))
            reduced = current - {a}
            candidates = [x for x in range(1, N + 1) if x not in reduced]
            if candidates:
                b = random.choice(candidates)
                new_set = list(reduced | {b})
                nc = count_conflicts(new_set)
                if nc == 0:
                    current = reduced | {b}
                    current_conflicts = 0
                    if len(current) > best_size:
                        best_size = len(current)
                        best_set = list(current)

    return best_set
# EVOLVE-BLOCK-END
