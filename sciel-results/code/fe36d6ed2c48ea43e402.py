import time
import random

def _can_sum_to_200(S):
    """Check if any subset of S sums to exactly 200 using bitset DP."""
    if not S:
        return False
    reachable = 1  # bit i set means sum i is achievable
    for x in S:
        if x > 200:
            continue
        reachable |= (reachable << x)
        reachable &= (1 << 201) - 1
    return bool(reachable & (1 << 200))

def _find_conflict_bitset(S):
    """Find a subset of S that sums to 200. Returns list or None."""
    if not S:
        return None
    if not _can_sum_to_200(S):
        return None
    
    S_list = list(S)
    reach = [False] * 201
    last_elem = [-1] * 201
    prev_sum = [-1] * 201
    reach[0] = True
    
    for i, x in enumerate(S_list):
        if x > 200:
            continue
        for s in range(200 - x, -1, -1):
            if reach[s] and not reach[s + x]:
                reach[s + x] = True
                last_elem[s + x] = i
                prev_sum[s + x] = s
    
    if not reach[200]:
        return None
    
    subset = []
    s = 200
    while s > 0:
        i = last_elem[s]
        if i == -1:
            break
        subset.append(S_list[i])
        s = prev_sum[s]
    
    return subset if subset else None

def construct():
    start_time = time.time()
    deadline = 50.0  # Safety margin before 60s kill
    
    best_set = None
    best_size = 0
    
    # Multiple greedy initializations
    for order in [range(100, 0, -1), range(1, 101), sorted(range(1, 101), key=lambda x: x % 7)]:
        selected = []
        for x in order:
            if not _can_sum_to_200(selected + [x]):
                selected.append(x)
        if len(selected) > best_size:
            best_size = len(selected)
            best_set = selected[:]
    
    # Randomized greedy
    while time.time() - start_time < deadline - 25.0:
        S = list(range(1, 101))
        random.shuffle(S)
        selected = []
        for x in S:
            if not _can_sum_to_200(selected + [x]):
                selected.append(x)
        if len(selected) > best_size:
            best_size = len(selected)
            best_set = selected[:]
    
    # Local search: add excluded elements
    if best_set is not None and time.time() - start_time < deadline - 8.0:
        current_set = set(best_set)
        excluded = [x for x in range(1, 101) if x not in current_set]
        improved = True
        while improved and time.time() - start_time < deadline - 5.0:
            improved = False
            for x in excluded[:]:
                if x in current_set:
                    continue
                if not _can_sum_to_200(current_set | {x}):
                    current_set.add(x)
                    excluded.remove(x)
                    improved = True
                    if len(current_set) > best_size:
                        best_size = len(current_set)
                        best_set = list(current_set)
    
    # Conflict resolution: add x, remove conflicting element
    if time.time() - start_time < deadline - 3.0 and best_set is not None:
        current_set = set(best_set)
        excluded = [x for x in range(1, 101) if x not in current_set]
        random.shuffle(excluded)
        for x in excluded:
            if time.time() - start_time > deadline - 1.0:
                break
            test_set = current_set | {x}
            if _can_sum_to_200(test_set):
                conflict = _find_conflict_bitset(list(test_set))
                if conflict:
                    for elem in conflict:
                        if elem == x:
                            continue
                        new_set = test_set - {elem}
                        if not _can_sum_to_200(new_set) and len(new_set) >= len(current_set):
                            current_set = new_set
                            if len(current_set) > best_size:
                                best_size = len(current_set)
                                best_set = list(current_set)
                            break
    
    return best_set
