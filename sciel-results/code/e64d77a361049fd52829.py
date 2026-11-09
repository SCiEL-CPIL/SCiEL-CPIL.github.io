"""Rewritten program: maximize subset of [1,128] with no equal-gcd triples."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin
    
    N = 128
    nums = list(range(1, N + 1))
    
    # Precompute GCDs
    gcds = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            gcds[i][j] = gcd(i, j)
            gcds[j][i] = gcds[i][j]

    def is_valid_subset(subset):
        """Check if the subset has no bad triples."""
        if len(subset) < 3:
            return True
        # Sort for consistent indexing if needed, but set is fine
        s = list(subset)
        n = len(s)
        for i in range(n):
            a = s[i]
            for j in range(i + 1, n):
                b = s[j]
                gab = gcds[a][b]
                for k in range(j + 1, n):
                    c = s[k]
                    if gab == gcds[a][c] == gcds[b][c]:
                        return False
        return True

    def can_add(subset, x):
        """Check if adding x to subset creates a bad triple."""
        # Only need to check triples involving x
        s = list(subset)
        n = len(s)
        for i in range(n):
            a = s[i]
            ga = gcds[a][x]
            for j in range(i + 1, n):
                b = s[j]
                gb = gcds[b][x]
                gab = gcds[a][b]
                if ga == gb == gab:
                    return False
        return True

    def can_swap(subset, remove_x, add_y):
        """Check if replacing remove_x with add_y keeps the set valid."""
        new_subset = set(subset)
        new_subset.discard(remove_x)
        new_subset.add(add_y)
        return is_valid_subset(new_subset)

    # Greedy initialization
    def greedy_init(order):
        selected = set()
        for x in order:
            if can_add(selected, x):
                selected.add(x)
        return selected

    # Generate several initial sets
    best_set = set()
    
    # Order 1: Powers of 2
    p2 = [2**i for i in range(8) if 2**i <= 128]
    s1 = greedy_init(p2)
    if len(s1) > len(best_set): best_set = s1

    # Order 2: All numbers sorted by value
    s2 = greedy_init(nums)
    if len(s2) > len(best_set): best_set = s2

    # Order 3: All numbers sorted by number of divisors (ascending)
    def num_divisors(n):
        count = 0
        for i in range(1, int(n**0.5)+1):
            if n % i == 0:
                count += 2
                if i != n//i:
                    count -= 1
        return count
    s3 = greedy_init(sorted(nums, key=num_divisors))
    if len(s3) > len(best_set): best_set = s3

    # Order 4: Random permutations
    for _ in range(20):
        if time.time() > deadline - 10: break
        random_order = nums[:]
        random.shuffle(random_order)
        s_rand = greedy_init(random_order)
        if len(s_rand) > len(best_set):
            best_set = s_rand

    # Local Search / Hill Climbing
    # We try to add elements or swap elements to increase size or improve potential for adding
    current_set = set(best_set)
    
    while time.time() < deadline - 5:
        improved = False
        # Try to add any element not in set
        for x in nums:
            if x not in current_set:
                if can_add(current_set, x):
                    current_set.add(x)
                    improved = True
                    break
        if improved:
            if len(current_set) > len(best_set):
                best_set = set(current_set)
            continue
            
        # If no direct addition, try swaps
        # Randomly select an element to remove and one to add
        if current_set:
            remove_candidates = list(current_set)
            add_candidates = [x for x in nums if x not in current_set]
            if add_candidates:
                # Try a few random swaps
                for _ in range(100):
                    if time.time() > deadline - 5: break
                    rx = random.choice(remove_candidates)
                    ax = random.choice(add_candidates)
                    if can_swap(current_set, rx, ax):
                        current_set.remove(rx)
                        current_set.add(ax)
                        # Note: swap doesn't increase size, but might allow future additions
                        # We only keep best_set if size increases, but current_set evolves
                        break
                # Check if current_set has grown? No, swaps keep size same.
                # But maybe we can add after a swap? The loop above handles adding.
    
    return list(best_set)
# EVOLVE-BLOCK-END
