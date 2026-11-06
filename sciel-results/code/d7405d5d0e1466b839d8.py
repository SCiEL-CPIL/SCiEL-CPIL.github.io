import time
from math import gcd
import random

def construct():
    start_time = time.time()
    limit = 512
    deadline = start_time + 85
    n = limit

    compat = [[False] * (n + 1) for _ in range(n + 1)]
    for a in range(1, n + 1):
        for b in range(a, n + 1):
            l = (a * b) // gcd(a, b)
            c = l <= limit
            compat[a][b] = c
            compat[b][a] = c

    # Bitsets for divisors
    div_bitsets = [0] * (n + 1)
    for d in range(1, n + 1):
        bit = 1 << (d - 1)
        for m in range(d, n + 1, d):
            div_bitsets[m] |= bit

    def get_union_bitset(M):
        u = 0
        for m in M:
            u |= div_bitsets[m]
        return u

    def popcount(x):
        return bin(x).count('1')

    def is_valid_maximal_set(M):
        M_list = list(M)
        for i in range(len(M_list)):
            for j in range(i + 1, len(M_list)):
                if not compat[M_list[i]][M_list[j]]:
                    return False
        return True

    def local_search_maximal(M, time_limit):
        M = frozenset(M)
        if not is_valid_maximal_set(M):
            M_list = list(M)
            M_valid = set()
            for m in M_list:
                if all(compat[m][o] for o in M_valid):
                    M_valid.add(m)
            M = frozenset(M_valid)
        
        current_bitset = get_union_bitset(M)
        current_size = popcount(current_bitset)
        
        # Precompute compatibility bitmasks for fast checking
        # compat_mask[x] has bit i set if x is compatible with (i+1)
        compat_mask = [0] * (n + 1)
        for x in range(1, n + 1):
            mask = 0
            for y in range(1, n + 1):
                if compat[x][y]:
                    mask |= (1 << (y - 1))
            compat_mask[x] = mask
        
        # M_mask has bit i set if (i+1) is in M
        def compute_M_mask(M):
            mask = 0
            for m in M:
                mask |= (1 << (m - 1))
            return mask
        
        M_mask = compute_M_mask(M)
        
        # Check if x can be added to M (compatible with all, and not a multiple of any)
        def can_add(x, M_set, M_mask_val):
            if x in M_set:
                return False
            # Check compatibility: all elements in M must be compatible with x
            # Elements in M are those with bits set in M_mask_val
            # We need: for all y in M, compat[x][y] is True
            # i.e., M_mask_val & ~compat_mask[x] == 0
            if M_mask_val & ~compat_mask[x]:
                return False
            # Check x is not a multiple of any element in M
            for m in M_set:
                if x % m == 0:
                    return False
            return True
        
        # Check if x can replace e in M (compatible with all of M\{e}, not multiple of any in M\{e})
        def can_replace(x, e, M_set, M_mask_val):
            if x in M_set and x != e:
                return False
            M_minus_mask = M_mask_val & ~(1 << (e - 1))
            if M_minus_mask & ~compat_mask[x]:
                return False
            for m in M_set:
                if m != e and x % m == 0:
                    return False
            return True
        
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            
            # 1. Try adding a new maximal element
            for x in range(1, n + 1):
                if time.time() > time_limit:
                    break
                if x not in M:
                    if can_add(x, M, M_mask):
                        M2 = M | {x}
                        S2 = get_union_bitset(M2)
                        if popcount(S2) > current_size:
                            M = M2
                            M_mask = compute_M_mask(M)
                            current_bitset = S2
                            current_size = popcount(S2)
                            improved = True
                            break

            # 2. 1-for-1 Swap
            for e in list(M):
                if time.time() > time_limit:
                    break
                for x in range(1, n + 1):
                    if time.time() > time_limit:
                        break
                    if x not in M and x != e:
                        if can_replace(x, e, M, M_mask):
                            M2 = (M - {e}) | {x}
                            S2 = get_union_bitset(M2)
                            if popcount(S2) > current_size:
                                M = M2
                                M_mask = compute_M_mask(M)
                                current_bitset = S2
                                current_size = popcount(S2)
                                improved = True
                                break

            # 3. 1-for-2 Swap (Remove 1, Add 2)
            for e in list(M):
                if time.time() > time_limit:
                    break
                best_pair_size = -1
                best_pair = None
                for x in range(1, n + 1):
                    if time.time() > time_limit:
                        break
                    if x in M or x == e:
                        continue
                    # x must be compatible with all of M\{e} and not multiple of any
                    M_minus_mask_x = M_mask & ~(1 << (e - 1))
                    if M_minus_mask_x & ~compat_mask[x]:
                        continue
                    x_is_multiple = False
                    for m in M:
                        if m != e and x % m == 0:
                            x_is_multiple = True
                            break
                    if x_is_multiple:
                        continue
                    for y in range(x + 1, n + 1):
                        if time.time() > time_limit:
                            break
                        if y in M or y == e:
                            continue
                        if not compat[x][y]:
                            continue
                        # y must be compatible with all of M\{e} and not multiple of any
                        if M_minus_mask_x & ~compat_mask[y]:
                            continue
                        y_is_multiple = False
                        for m in M:
                            if m != e and y % m == 0:
                                y_is_multiple = True
                                break
                        if y_is_multiple:
                            continue
                        M2 = (M - {e}) | {x, y}
                        S2 = get_union_bitset(M2)
                        size2 = popcount(S2)
                        if size2 > best_pair_size:
                            best_pair_size = size2
                            best_pair = (x, y)
                if best_pair and best_pair_size > current_size:
                    M = (M - {e}) | set(best_pair)
                    M_mask = compute_M_mask(M)
                    current_bitset = get_union_bitset(M)
                    current_size = best_pair_size
                    improved = True

            # 4. 2-for-2 Swap (Remove 2, Add 2)
            if time.time() < time_limit:
                M_list = list(M)
                for i in range(len(M_list)):
                    if time.time() > time_limit:
                        break
                    for j in range(i + 1, len(M_list)):
                        if time.time() > time_limit:
                            break
                        e1 = M_list[i]
                        e2 = M_list[j]
                        M_minus = M - {e1, e2}
                        M_minus_mask = M_mask & ~(1 << (e1 - 1)) & ~(1 << (e2 - 1))
                        best_pair_size = -1
                        best_pair = None
                        for x in range(1, n + 1):
                            if time.time() > time_limit:
                                break
                            if x in M:
                                continue
                            if M_minus_mask & ~compat_mask[x]:
                                continue
                            x_is_multiple = False
                            for m in M_minus:
                                if x % m == 0:
                                    x_is_multiple = True
                                    break
                            if x_is_multiple:
                                continue
                            for y in range(x + 1, n + 1):
                                if time.time() > time_limit:
                                    break
                                if y in M:
                                    continue
                                if not compat[x][y]:
                                    continue
                                if M_minus_mask & ~compat_mask[y]:
                                    continue
                                y_is_multiple = False
                                for m in M_minus:
                                    if y % m == 0:
                                        y_is_multiple = True
                                        break
                                if y_is_multiple:
                                    continue
                                M2 = M_minus | {x, y}
                                S2 = get_union_bitset(M2)
                                size2 = popcount(S2)
                                if size2 > best_pair_size:
                                    best_pair_size = size2
                                    best_pair = (x, y)
                        if best_pair and best_pair_size > current_size:
                            M = M_minus | set(best_pair)
                            M_mask = compute_M_mask(M)
                            current_bitset = get_union_bitset(M)
                            current_size = best_pair_size
                            improved = True
                            break
                    if improved:
                        break

        return get_union_bitset(M)

    best_bitset = 0
    best_size = 0
    
    # Better seeds: include all singletons that are "rich" (have many divisors)
    seeds_maximal = []
    
    # Start with highly composite-like numbers
    rich_numbers = [512, 480, 420, 360, 336, 288, 240, 210, 180, 168, 144, 120, 96, 72, 60, 48, 36, 24, 12, 6, 4, 2]
    for m in rich_numbers:
        if m <= limit:
            seeds_maximal.append(frozenset({m}))
    
    # Add pairs of compatible numbers
    for a in range(2, n + 1):
        for b in range(a + 1, n + 1):
            if compat[a][b]:
                seeds_maximal.append(frozenset({a, b}))
                if len(seeds_maximal) > 300:
                    break
        if len(seeds_maximal) > 300:
            break
    
    # Add initial ranges
    for upper in [4, 6, 8, 12, 16, 24, 32, 48, 64]:
        seeds_maximal.append(frozenset(range(1, upper + 1)))
    
    # Add sets of small numbers
    for upper in [5, 7, 10, 15, 20]:
        seeds_maximal.append(frozenset(range(1, upper + 1)))
    
    for seed in seeds_maximal:
        if time.time() > deadline - 20:
            break
        S_bitset = local_search_maximal(seed, deadline - 20)
        S_size = popcount(S_bitset)
        if S_size > best_size:
            best_size = S_size
            best_bitset = S_bitset

    # Random restarts
    rng = random.Random(42)
    for _ in range(300):
        if time.time() > deadline - 5:
            break
        M = set()
        candidates = list(range(1, n + 1))
        rng.shuffle(candidates)
        for x in candidates:
            if all(compat[x][m] for m in M) and not any(x % m == 0 for m in M):
                M.add(x)
        if len(M) > 0:
            S_bitset = local_search_maximal(frozenset(M), deadline - 5)
            S_size = popcount(S_bitset)
            if S_size > best_size:
                best_size = S_size
                best_bitset = S_bitset

    # Convert bitset to list
    best = []
    for i in range(1, n + 1):
        if best_bitset & (1 << (i - 1)):
            best.append(i)
            
    # Final polish: try to add any compatible number
    for x in range(1, n + 1):
        if x not in best and all(compat[x][y] for y in best):
            best.append(x)
            best_bitset |= (1 << (x - 1))

    return sorted(best)
