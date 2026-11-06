import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 88.0
    n = 256
    
    # Precompute conflict tables: for each a, list of (b, c) pairs where a | (b+c), b <= c, a < b
    conflict_pairs = {}
    for a in range(1, n + 1):
        pairs = []
        for b in range(a + 1, n + 1):
            k_min = (2 * b + a - 1) // a
            k_max = (n + b) // a
            for k in range(k_min, k_max + 1):
                c = k * a - b
                if b <= c <= n:
                    pairs.append((b, c))
        conflict_pairs[a] = pairs
    
    # Precompute for each element x, which (a, b, c) triples it participates in
    element_conflicts = {x: [] for x in range(1, n + 1)}
    for a in range(1, n + 1):
        for b, c in conflict_pairs[a]:
            element_conflicts[a].append((b, c, a))
            element_conflicts[b].append((a, c, b))
            element_conflicts[c].append((a, b, c))
    
    def is_valid(S):
        if not S:
            return True
        S_list = sorted(S)
        for a in S_list:
            if a == 1:
                count_gt_1 = sum(1 for x in S_list if x > 1)
                if count_gt_1 >= 2:
                    return False
                continue
            for b, c in conflict_pairs[a]:
                if b in S and c in S:
                    return False
        return True
    
    def get_conflict(S, x):
        if x == 1:
            gt_1 = [y for y in S if y > 1]
            if len(gt_1) >= 2:
                return (1, gt_1[0], gt_1[1])
            return None
        if 1 in S:
            gt_1 = [y for y in S if y > 1]
            if len(gt_1) >= 1:
                return (1, gt_1[0], x)
        for b, c in conflict_pairs[x]:
            if b in S and c in S:
                return (x, b, c)
        for a, other in element_conflicts[x]:
            if a in S and other in S:
                if a < x:
                    if x <= other:
                        return (a, x, other)
                    else:
                        return (a, other, x)
                else:
                    if other <= x:
                        return (other, x, a)
                    else:
                        return (other, a, x)
        return None
    
    # Step 1: Find best starting interval with gap exploitation
    best_start = None
    best_start_size = 0
    
    for L in range(1, n + 1):
        if time.time() > deadline - 25:
            break
        S_test = set(range(L, n + 1))
        if is_valid(S_test):
            S_greedy = set(S_test)
            for x in range(1, L):
                if time.time() > deadline - 25:
                    break
                if x in S_greedy:
                    continue
                test = S_greedy | {x}
                if is_valid(test):
                    S_greedy.add(x)
            if len(S_greedy) > best_start_size:
                best_start_size = len(S_greedy)
                best_start = S_greedy
    
    # Also try structured sets: take elements in (n/3, n] and selectively add from (n/6, n/3]
    # Key insight: for a > n/3, 2a > 2n/3 and 3a > n, so a can only conflict if b+c is exactly 2a
    # For a in (n/6, n/3], 2a in (n/3, 2n/3] and 3a in (n/2, n]
    for L in range(n // 3 + 1, n + 1):
        if time.time() > deadline - 25:
            break
        S_test = set(range(L, n + 1))
        if not is_valid(S_test):
            continue
        S_greedy = set(S_test)
        # Try adding elements from (n/6, L)
        for x in range(max(1, n // 6), L):
            if time.time() > deadline - 25:
                break
            test = S_greedy | {x}
            if is_valid(test):
                S_greedy.add(x)
        if len(S_greedy) > best_start_size:
            best_start_size = len(S_greedy)
            best_start = S_greedy
    
    if best_start is None:
        best_start = {n}
    
    best_set = set(best_start)
    best_size = len(best_set)
    
    def greedy_with_swaps(S, candidates, time_limit):
        S = set(S)
        for x in candidates:
            if time.time() > time_limit:
                break
            if x in S:
                continue
            test = S | {x}
            if is_valid(test):
                S.add(x)
            else:
                if time.time() < time_limit - 0.5:
                    conflict = get_conflict(S, x)
                    if conflict:
                        a, b, c = conflict
                        for y in [a, b, c]:
                            if y == x or y not in S:
                                continue
                            test_swap = S | {x}
                            test_swap.remove(y)
                            if is_valid(test_swap):
                                S = test_swap
                                break
        return S
    
    # Step 2: Multi-start greedy with swaps
    candidates = list(range(1, n + 1))
    num_attempts = 50
    for attempt in range(num_attempts):
        if time.time() > deadline - 12:
            break
        random.shuffle(candidates)
        S = greedy_with_swaps(best_start, candidates, deadline - 12)
        if len(S) > best_size:
            best_size = len(S)
            best_set = S
    
    # Step 3: Local search with 1-swap and 2-swap moves
    current = set(best_set)
    
    def local_search_1swap(current, time_limit):
        improved = True
        remaining = [x for x in range(1, n + 1) if x not in current]
        random.shuffle(remaining)
        while improved and time.time() < time_limit:
            improved = False
            for x in remaining:
                if time.time() > time_limit:
                    break
                if x in current:
                    continue
                test = current | {x}
                if is_valid(test):
                    current.add(x)
                    improved = True
                else:
                    conflict = get_conflict(current, x)
                    if conflict:
                        a, b, c = conflict
                        for y in [a, b, c]:
                            if y == x or y not in current:
                                continue
                            test_swap = current | {x}
                            test_swap.remove(y)
                            if is_valid(test_swap):
                                current = test_swap
                                improved = True
                                break
        return current
    
    def local_search_2swap(current, time_limit):
        """Try removing two elements and adding one new element."""
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            remaining = [x for x in range(1, n + 1) if x not in current]
            random.shuffle(remaining)
            cur_list = sorted(current)
            random.shuffle(cur_list)
            for x in remaining:
                if time.time() > time_limit:
                    break
                if x in current:
                    continue
                # Try adding x by removing two elements from current
                found = False
                for i in range(len(cur_list)):
                    if time.time() > time_limit:
                        break
                    y1 = cur_list[i]
                    for j in range(i + 1, len(cur_list)):
                        if time.time() > time_limit:
                            break
                        y2 = cur_list[j]
                        test = (current - {y1, y2}) | {x}
                        if is_valid(test):
                            current = test
                            improved = True
                            found = True
                            break
                    if found:
                        break
                if found:
                    break
        return current
    
    # Run 1-swap local search
    current = local_search_1swap(current, deadline - 5)
    if len(current) > best_size:
        best_size = len(current)
        best_set = current
    
    # Run 2-swap local search
    current = local_search_2swap(current, deadline - 2)
    if len(current) > best_size:
        best_size = len(current)
        best_set = current
    
    # Final polish: try all single additions
    for x in range(1, n + 1):
        if time.time() > deadline - 0.5:
            break
        if x in best_set:
            continue
        test = best_set | {x}
        if is_valid(test):
            best_set.add(x)
    
    return list(best_set)

if __name__ == "__main__":
    res = construct()
    print(f"Size: {len(res)}")
