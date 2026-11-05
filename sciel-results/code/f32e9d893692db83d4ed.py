"""B_3[1] set construction: maximize cardinality with unique unordered triple sums."""
import time
import random
from itertools import combinations_with_replacement

def _triple_sums_set(S):
    """Return the set of all unordered triple sums (with repetition) for sorted list S."""
    n = len(S)
    sums = set()
    for i in range(n):
        for j in range(i, n):
            for k in range(j, n):
                sums.add(S[i] + S[j] + S[k])
    return sums

def _new_sums_with_x(x, S):
    """Compute all triple sums that involve x, given existing set S."""
    n = len(S)
    sums = set()
    # (x, x, x)
    sums.add(3 * x)
    for i in range(n):
        # (x, x, S[i])
        sums.add(2 * x + S[i])
        # (x, S[i], S[i])
        sums.add(x + 2 * S[i])
    for i in range(n):
        for j in range(i + 1, n):
            # (x, S[i], S[j])
            sums.add(x + S[i] + S[j])
    return sums

def _greedy_build(order, n_max=4096, time_limit=10.0):
    """Greedy build following a given order of candidates."""
    S = []
    existing_sums = set()
    deadline = time.time() + time_limit
    for x in order:
        if time.time() > deadline:
            break
        new_sums = _new_sums_with_x(x, S)
        if new_sums & existing_sums:
            continue
        S.append(x)
        existing_sums |= new_sums
    S.sort()
    return S

def _local_search_add(S, existing_sums, n_max=4096, time_limit=20.0):
    """Try to add elements by scanning candidates and checking collisions."""
    S = list(S)
    existing_sums = set(existing_sums)
    deadline = time.time() + time_limit
    candidates = [x for x in range(1, n_max + 1) if x not in S]
    random.shuffle(candidates)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for x in candidates:
            if time.time() > deadline:
                break
            if x in S:
                continue
            new_sums = _new_sums_with_x(x, S)
            if new_sums & existing_sums:
                continue
            S.append(x)
            existing_sums |= new_sums
            improved = True
    S.sort()
    return S, existing_sums

def _local_search_swap(S, existing_sums, n_max=4096, time_limit=25.0):
    """Try to swap out elements to make room for more, or add new ones."""
    S = list(S)
    existing_sums = set(existing_sums)
    deadline = time.time() + time_limit
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try adding
        candidates = [x for x in range(1, n_max + 1) if x not in S]
        random.shuffle(candidates)
        for x in candidates:
            if time.time() > deadline:
                break
            new_sums = _new_sums_with_x(x, S)
            if not (new_sums & existing_sums):
                S.append(x)
                existing_sums |= new_sums
                improved = True
                break
        if not improved:
            # Try removing one element and then adding a new one
            for idx in range(len(S)):
                if time.time() > deadline:
                    break
                removed = S[idx]
                # Recompute sums without removed
                S_tmp = S[:idx] + S[idx+1:]
                sums_tmp = _triple_sums_set(S_tmp)
                # Try to add a new element
                candidates = [x for x in range(1, n_max + 1) if x not in S_tmp]
                random.shuffle(candidates)
                for x in candidates:
                    if time.time() > deadline:
                        break
                    new_sums = _new_sums_with_x(x, S_tmp)
                    if not (new_sums & sums_tmp):
                        S_tmp.append(x)
                        S = S_tmp
                        existing_sums = sums_tmp | new_sums
                        improved = True
                        break
                if improved:
                    break
    S.sort()
    return S, existing_sums

def construct():
    start_time = time.time()
    total_time_limit = 80.0  # safety margin before 90s kill
    
    best_S = []
    best_len = 0
    
    # Phase 1: Multiple greedy builds with different orderings
    greedy_deadline = start_time + 15.0
    attempts = 0
    while time.time() < greedy_deadline:
        attempts += 1
        # Different ordering strategies
        if attempts % 3 == 1:
            # Linear increasing
            order = list(range(1, 4097))
        elif attempts % 3 == 2:
            # Random shuffle
            order = list(range(1, 4097))
            random.shuffle(order)
        else:
            # Exponential-ish spacing then fill
            order = []
            val = 1
            while val <= 4096:
                order.append(val)
                val = int(val * 1.6) + 1
            remaining = [x for x in range(1, 4097) if x not in order]
            random.shuffle(remaining)
            order.extend(remaining)
        
        S = _greedy_build(order, n_max=4096, time_limit=3.0)
        if len(S) > best_len:
            best_S = S
            best_len = len(S)
    
    # Phase 2: Local search - add more elements
    if time.time() < start_time + total_time_limit:
        existing_sums = _triple_sums_set(best_S)
        remaining_time = start_time + total_time_limit - time.time()
        add_time = min(remaining_time * 0.4, 20.0)
        best_S, existing_sums = _local_search_add(best_S, existing_sums, n_max=4096, time_limit=add_time)
        if len(best_S) > best_len:
            best_len = len(best_S)
    
    # Phase 3: Local search - swaps
    if time.time() < start_time + total_time_limit:
        existing_sums = _triple_sums_set(best_S)
        remaining_time = start_time + total_time_limit - time.time()
        swap_time = min(remaining_time * 0.6, 30.0)
        best_S, existing_sums = _local_search_swap(best_S, existing_sums, n_max=4096, time_limit=swap_time)
        if len(best_S) > best_len:
            best_len = len(best_S)
    
    # Phase 4: Final add attempt
    if time.time() < start_time + total_time_limit - 2:
        existing_sums = _triple_sums_set(best_S)
        remaining_time = start_time + total_time_limit - time.time() - 1
        if remaining_time > 2:
            best_S, existing_sums = _local_search_add(best_S, existing_sums, n_max=4096, time_limit=remaining_time)
            if len(best_S) > best_len:
                best_len = len(best_S)
    
    # Ensure all elements are in [1, 4096] and distinct
    best_S = sorted(set(x for x in best_S if 1 <= x <= 4096))
    return best_S

if __name__ == "__main__":
    result = construct()
    print(f"Found {len(result)} elements")
    print(result)
