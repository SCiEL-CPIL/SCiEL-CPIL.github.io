import time
import random
import sys

def construct():
    """
    Find a large subset of [1,256] such that for every distinct a,b in the subset,
    (a*b) % (a+b) != 0.
    """
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill

    N = 256
    all_nums = list(range(1, N + 1))

    # Precompute conflict graph using bitmasks for fast operations
    # conflict_mask[x] is a bitmask where bit y is set if x and y conflict
    conflict_mask = [0] * (N + 1)
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            if (i * j) % (i + j) == 0:
                conflict_mask[i] |= (1 << j)
                conflict_mask[j] |= (1 << i)

    def is_valid_set(S):
        """Check if set S is valid (no conflicts)."""
        S_list = list(S)
        for i in range(len(S_list)):
            a = S_list[i]
            for j in range(i + 1, len(S_list)):
                b = S_list[j]
                if conflict_mask[a] & (1 << b):
                    return False
        return True

    def build_mask(S):
        """Build bitmask from set."""
        mask = 0
        for y in S:
            mask |= (1 << y)
        return mask

    def count_conflicts_mask(x, S_mask):
        """Count conflicts using precomputed mask."""
        return bin(conflict_mask[x] & S_mask).count('1')

    def greedy_fill(S, unselected, S_mask):
        """Greedily add elements from unselected that have zero conflicts with S."""
        remaining = list(unselected)
        remaining.sort(key=lambda x: count_conflicts_mask(x, S_mask))
        for x in remaining:
            if (conflict_mask[x] & S_mask) == 0:
                S.add(x)
                S_mask |= (1 << x)
                unselected.remove(x)
        return S, unselected, S_mask

    def local_search_1for1(S, unselected, S_mask, time_budget):
        """Try removing one element and adding more."""
        improved = True
        while improved and time.time() < time_budget:
            improved = False
            s_list = list(S)
            random.shuffle(s_list)
            for x in s_list:
                if time.time() > time_budget:
                    break
                S_temp = set(S)
                S_temp.remove(x)
                S_temp_mask = build_mask(S_temp)
                u_list = list(unselected)
                u_list.sort(key=lambda y: count_conflicts_mask(y, S_temp_mask))
                for y in u_list:
                    if (conflict_mask[y] & S_temp_mask) == 0:
                        S_temp.add(y)
                        S_temp_mask |= (1 << y)
                if len(S_temp) > len(S):
                    S = S_temp
                    unselected = set(all_nums) - S
                    S_mask = build_mask(S)
                    improved = True
                    break
        return S, unselected, S_mask

    def local_search_2for3(S, unselected, S_mask, time_budget):
        """Try removing two elements and adding three or more."""
        improved = True
        while improved and time.time() < time_budget:
            improved = False
            s_list = list(S)
            random.shuffle(s_list)
            for i in range(len(s_list)):
                if time.time() > time_budget:
                    break
                x1 = s_list[i]
                for j in range(i + 1, len(s_list)):
                    if time.time() > time_budget:
                        break
                    x2 = s_list[j]
                    S_temp = set(S)
                    S_temp.remove(x1)
                    S_temp.remove(x2)
                    S_temp_mask = build_mask(S_temp)
                    u_list = list(unselected)
                    u_list.sort(key=lambda y: count_conflicts_mask(y, S_temp_mask))
                    for y in u_list:
                        if (conflict_mask[y] & S_temp_mask) == 0:
                            S_temp.add(y)
                            S_temp_mask |= (1 << y)
                    if len(S_temp) > len(S):
                        S = S_temp
                        unselected = set(all_nums) - S
                        S_mask = build_mask(S)
                        improved = True
                        break
                if improved or time.time() > time_budget:
                    break
        return S, unselected, S_mask

    def polish(S, unselected, S_mask, time_budget):
        """Final polish: try to add any remaining elements with zero conflicts."""
        remaining = list(unselected)
        random.shuffle(remaining)
        for x in remaining:
            if time.time() > time_budget:
                break
            if (conflict_mask[x] & S_mask) == 0:
                S.add(x)
                S_mask |= (1 << x)
                unselected.remove(x)
        return S, unselected, S_mask

    # Multiple diverse starting cores
    cores = []
    
    # Core 1: All odd numbers
    odds = set(range(1, N + 1, 2))
    if is_valid_set(odds):
        cores.append(odds)
    else:
        current = set()
        for x in sorted(odds):
            if count_conflicts_mask(x, build_mask(current)) == 0:
                current.add(x)
        cores.append(current)
    
    # Core 2: Numbers coprime to 6
    coprime6 = set(x for x in range(1, N + 1) if x % 2 != 0 and x % 3 != 0)
    if is_valid_set(coprime6):
        cores.append(coprime6)
    else:
        current = set()
        for x in sorted(coprime6):
            if count_conflicts_mask(x, build_mask(current)) == 0:
                current.add(x)
        cores.append(current)
    
    # Core 3: Numbers congruent to 1 or 5 mod 6
    mod6_1_5 = set(x for x in range(1, N + 1) if x % 6 in (1, 5))
    if is_valid_set(mod6_1_5):
        cores.append(mod6_1_5)
    else:
        current = set()
        for x in sorted(mod6_1_5):
            if count_conflicts_mask(x, build_mask(current)) == 0:
                current.add(x)
        cores.append(current)
    
    # Core 4: Random subset of odds
    random.seed(42)
    random_odds = set(random.sample(range(1, N + 1, 2), len(range(1, N + 1, 2)) // 2))
    if is_valid_set(random_odds):
        cores.append(random_odds)
    else:
        current = set()
        for x in sorted(random_odds):
            if count_conflicts_mask(x, build_mask(current)) == 0:
                current.add(x)
        cores.append(current)
    
    # Core 5: Numbers with high 2-adic valuation (powers of 2 and near-powers)
    high_v2 = set()
    for x in range(1, N + 1):
        v2 = 0
        tmp = x
        while tmp % 2 == 0:
            tmp //= 2
            v2 += 1
        if v2 >= 3:  # divisible by 8
            high_v2.add(x)
    if is_valid_set(high_v2):
        cores.append(high_v2)
    else:
        current = set()
        for x in sorted(high_v2):
            if count_conflicts_mask(x, build_mask(current)) == 0:
                current.add(x)
        cores.append(current)

    best_set = set()
    best_size = 0

    for core in cores:
        if time.time() > deadline - 5:
            break
        
        current_set = set(core)
        unselected = set(all_nums) - current_set
        S_mask = build_mask(current_set)
        
        # Greedy fill
        current_set, unselected, S_mask = greedy_fill(current_set, unselected, S_mask)
        
        # Local search 1-for-1
        time_budget = min(time.time() + 10, deadline - 10)
        current_set, unselected, S_mask = local_search_1for1(current_set, unselected, S_mask, time_budget)
        
        # Local search 2-for-3
        time_budget = min(time.time() + 15, deadline - 5)
        current_set, unselected, S_mask = local_search_2for3(current_set, unselected, S_mask, time_budget)
        
        # Polish
        time_budget = min(time.time() + 2, deadline - 1)
        current_set, unselected, S_mask = polish(current_set, unselected, S_mask, time_budget)
        
        # Final greedy fill
        current_set, unselected, S_mask = greedy_fill(current_set, unselected, S_mask)
        
        if len(current_set) > best_size:
            best_set = set(current_set)
            best_size = len(current_set)

    # Final verification
    if not is_valid_set(best_set):
        best_set = set(range(1, N + 1, 2))
        valid_set = set()
        for x in sorted(best_set):
            if count_conflicts_mask(x, build_mask(valid_set)) == 0:
                valid_set.add(x)
        best_set = valid_set

    return list(best_set)

# Quick self-test
if __name__ == "__main__":
    result = construct()
    print(f"Cardinality: {len(result)}")
    valid = True
    for i in range(len(result)):
        for j in range(i+1, len(result)):
            a, b = result[i], result[j]
            if (a*b) % (a+b) == 0:
                valid = False
                print(f"Conflict: {a}, {b}")
    print(f"Valid: {valid}")
