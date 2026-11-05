import time
import random
from math import gcd
from collections import defaultdict

def construct():
    deadline = time.time() + 85
    n = 128
    nums = list(range(1, n + 1))
    
    # Precompute GCD table
    gcd_table = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(i, n + 1):
            g = gcd(i, j)
            gcd_table[i][j] = g
            gcd_table[j][i] = g

    def can_add_fast(x, S_set, pair_gcd_cache):
        """Check if x can be added to set S without creating a bad triple.
        Uses the fact that for a bad triple (a,b,x), we need gcd(a,b)=gcd(a,x)=gcd(b,x).
        """
        s_list = list(S_set)
        m = len(s_list)
        for i in range(m):
            a = s_list[i]
            g_ax = gcd_table[a][x]
            if g_ax == 1:
                # gcd(a,x)=1, so for bad triple we'd need gcd(a,b)=1 and gcd(b,x)=1
                # i.e., b is coprime to both a and x
                for j in range(i + 1, m):
                    b = s_list[j]
                    if gcd_table[a][b] == 1 and gcd_table[b][x] == 1:
                        return False
            else:
                for j in range(i + 1, m):
                    b = s_list[j]
                    if gcd_table[a][b] == g_ax and gcd_table[b][x] == g_ax:
                        return False
        return True

    def greedy_build(order):
        S = set()
        for x in order:
            if time.time() > deadline:
                break
            if can_add_fast(x, S, None):
                S.add(x)
        return S

    def local_search_swaps(S, remaining):
        """Try swaps: remove one element from S, add one from remaining."""
        improved = False
        s_list = list(S)
        rem_list = list(remaining)
        for a in s_list:
            if time.time() > deadline - 2:
                break
            S_minus_a = S - {a}
            for x in rem_list:
                if x in S:
                    continue
                if can_add_fast(x, S_minus_a, None):
                    S.add(x)
                    S.discard(a)
                    improved = True
                    return S, True
        return S, improved

    best_size = 0
    best_set = set()
    
    # Generate diverse orderings
    def make_orderings():
        orderings = []
        # Pure random
        for _ in range(3):
            shuffled = nums[:]
            random.shuffle(shuffled)
            orderings.append(shuffled)
        # Sort by number of prime factors (more factors = more constraints, add later)
        def num_prime_factors(x):
            count = 0
            d = 2
            while d * d <= x:
                while x % d == 0:
                    count += 1
                    x //= d
                d += 1
            if x > 1:
                count += 1
            return count
        orderings.append(sorted(nums, key=num_prime_factors))  # fewest factors first
        orderings.append(sorted(nums, key=num_prime_factors, reverse=True))  # most factors first
        # Sort by value
        orderings.append(sorted(nums))
        orderings.append(sorted(nums, reverse=True))
        # Random with some structure: group by gcd with small numbers
        for _ in range(2):
            shuffled = nums[:]
            random.shuffle(shuffled)
            # Perturb: move some elements
            for _ in range(10):
                i, j = random.randint(0, len(shuffled)-1), random.randint(0, len(shuffled)-1)
                shuffled[i], shuffled[j] = shuffled[j], shuffled[i]
            orderings.append(shuffled)
        return orderings

    # Run initial batch of diverse orderings
    initial_orderings = make_orderings()
    for order in initial_orderings:
        if time.time() > deadline - 5:
            break
        S = greedy_build(order)
        if len(S) > best_size:
            best_size = len(S)
            best_set = set(S)

    # Run many random restarts
    num_restarts = 8000
    for i in range(num_restarts):
        if time.time() > deadline - 3:
            break
        shuffled = nums[:]
        random.shuffle(shuffled)
        S = greedy_build(shuffled)
        if len(S) > best_size:
            best_size = len(S)
            best_set = set(S)

    # Final local search: try to add any remaining numbers
    improved = True
    while improved and time.time() < deadline - 3:
        improved = False
        for x in range(1, n + 1):
            if x not in best_set:
                if can_add_fast(x, best_set, None):
                    best_set.add(x)
                    best_size += 1
                    improved = True
                    break

    # Swap-based local search
    if time.time() < deadline - 2:
        remaining = set(range(1, n + 1)) - best_set
        for _ in range(5):
            if time.time() > deadline - 2:
                break
            best_set, improved = local_search_swaps(best_set, remaining)
            if not improved:
                break
            # After a swap, try to add more
            remaining = set(range(1, n + 1)) - best_set
            improved2 = True
            while improved2 and time.time() < deadline - 2:
                improved2 = False
                for x in range(1, n + 1):
                    if x not in best_set:
                        if can_add_fast(x, best_set, None):
                            best_set.add(x)
                            improved2 = True
                            break

    return list(best_set)
