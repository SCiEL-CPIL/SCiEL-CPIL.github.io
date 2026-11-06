"""B2[2] set construction in [1, 1000] using incremental count arrays and targeted local search."""
import time
import random

MAX_VAL = 1000
MAX_SUM = 2 * MAX_VAL  # 2000

def _greedy_build():
    """Standard greedy: add x if it doesn't cause any sum to exceed 2 reps."""
    chosen = []
    counts = [0] * (MAX_SUM + 1)
    for x in range(1, MAX_VAL + 1):
        # Check if adding x would violate B2[2]
        ok = True
        for y in chosen:
            s = x + y
            if counts[s] >= 2:
                ok = False
                break
        if ok:
            s = 2 * x
            if counts[s] >= 2:
                ok = False
        if ok:
            # Commit: update counts
            for y in chosen:
                counts[x + y] += 1
            counts[2 * x] += 1
            chosen.append(x)
    return chosen

def _build_counts(set_list):
    """Build count array for a set."""
    counts = [0] * (MAX_SUM + 1)
    n = len(set_list)
    for i in range(n):
        a = set_list[i]
        for j in range(i, n):
            b = set_list[j]
            counts[a + b] += 1
    return counts

def _try_add_incremental(set_list, counts, x):
    """Try adding x. Returns (success, new_set_list, new_counts).
    If success, new_counts is the updated count array.
    """
    if x in set_list:
        return False, set_list, counts
    # Check all new sums
    ok = True
    for y in set_list:
        if counts[x + y] >= 2:
            ok = False
            break
    if ok and counts[2 * x] >= 2:
        ok = False
    if not ok:
        return False, set_list, counts
    # Update counts
    new_counts = list(counts)
    for y in set_list:
        new_counts[x + y] += 1
    new_counts[2 * x] += 1
    new_set = set_list + [x]
    return True, new_set, new_counts

def _remove_from_counts(set_list, counts, x):
    """Remove x from set_list and update counts. Returns (new_set, new_counts)."""
    new_counts = list(counts)
    for y in set_list:
        if y != x:
            new_counts[x + y] -= 1
    new_counts[2 * x] -= 1
    new_set = [v for v in set_list if v != x]
    return new_set, new_counts

def _local_search_incremental(initial, deadline, seed=42):
    """Local search maintaining incremental count arrays for speed."""
    rng = random.Random(seed)
    current = list(initial)
    current_set = set(current)
    counts = _build_counts(current)
    best = list(current)
    best_size = len(current)
    best_counts = list(counts)
    
    all_vals = list(range(1, MAX_VAL + 1))
    
    while time.time() < deadline:
        op = rng.random()
        
        if op < 0.25:
            # Pure add
            x = rng.choice(all_vals)
            if x in current_set:
                continue
            ok, new_set, new_counts = _try_add_incremental(current, counts, x)
            if ok:
                current = new_set
                current_set.add(x)
                counts = new_counts
                if len(current) > best_size:
                    best = list(current)
                    best_size = len(current)
                    best_counts = list(counts)
        elif op < 0.80:
            # Swap: remove one, add one
            if not current:
                continue
            remove_x = rng.choice(current)
            add_x = rng.choice(all_vals)
            if add_x == remove_x or add_x in current_set:
                continue
            # Remove remove_x
            temp_set, temp_counts = _remove_from_counts(current, counts, remove_x)
            # Try to add add_x
            ok, new_set, new_counts = _try_add_incremental(temp_set, temp_counts, add_x)
            if ok:
                current = new_set
                current_set = set(current)
                counts = new_counts
                if len(current) > best_size:
                    best = list(current)
                    best_size = len(current)
                    best_counts = list(counts)
        else:
            # Remove one (to explore)
            if current:
                remove_x = rng.choice(current)
                current, counts = _remove_from_counts(current, counts, remove_x)
                current_set = set(current)
    
    return best

def _targeted_local_search(initial, deadline, seed=42):
    """Targeted search: identify bottleneck sums and try to swap elements that conflict with them."""
    rng = random.Random(seed)
    current = list(initial)
    current_set = set(current)
    counts = _build_counts(current)
    best = list(current)
    best_size = len(current)
    
    all_vals = list(range(1, MAX_VAL + 1))
    
    while time.time() < deadline:
        # Find sums with exactly 2 representations (bottlenecks)
        bottleneck_sums = [s for s in range(2, MAX_SUM + 1) if counts[s] == 2]
        
        if not bottleneck_sums:
            # No bottlenecks, try pure adds
            x = rng.choice(all_vals)
            if x in current_set:
                continue
            ok, new_set, new_counts = _try_add_incremental(current, counts, x)
            if ok:
                current = new_set
                current_set.add(x)
                counts = new_counts
                if len(current) > best_size:
                    best = list(current)
                    best_size = len(current)
            continue
        
        # Pick a random bottleneck sum
        target_sum = rng.choice(bottleneck_sums)
        
        # Find pairs that produce target_sum
        pairs = []
        for i in range(len(current)):
            a = current[i]
            b = target_sum - a
            if b >= a and b in current_set:
                pairs.append((a, b))
        
        if len(pairs) < 2:
            continue
        
        # Try to remove one element from a pair and add a different element
        # that doesn't conflict with the target_sum constraint
        a, b = rng.choice(pairs)
        
        # Try removing 'a' or 'b' and adding something else
        for remove_x in [a, b]:
            if remove_x not in current:
                continue
            temp_set, temp_counts = _remove_from_counts(current, counts, remove_x)
            
            # Try adding random candidates
            for _ in range(3):
                add_x = rng.choice(all_vals)
                if add_x in temp_set or add_x == remove_x:
                    continue
                ok, new_set, new_counts = _try_add_incremental(temp_set, temp_counts, add_x)
                if ok:
                    current = new_set
                    current_set = set(current)
                    counts = new_counts
                    if len(current) > best_size:
                        best = list(current)
                        best_size = len(current)
                    break
            # If no add worked, try pure add after removal
            if len(current) == len(temp_set):
                x = rng.choice(all_vals)
                if x not in temp_set:
                    ok, new_set, new_counts = _try_add_incremental(temp_set, temp_counts, x)
                    if ok:
                        current = new_set
                        current_set = set(current)
                        counts = new_counts
                        if len(current) > best_size:
                            best = list(current)
                            best_size = len(current)
    
    return best

def _fast_is_valid(set_list):
    """Faster validity check using counting array."""
    n = len(set_list)
    if n == 0:
        return True
    s = sorted(set_list)
    counts = [0] * (MAX_SUM + 1)
    for i in range(n):
        for j in range(i, n):
            sm = s[i] + s[j]
            counts[sm] += 1
            if counts[sm] > 2:
                return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin: 150s limit, use 110s
    
    # Step 1: Greedy baseline
    greedy_set = _greedy_build()
    best = list(greedy_set)
    best_size = len(best)
    
    # Step 2: Incremental local search (fast, many iterations)
    remaining = deadline - time.time()
    if remaining > 10:
        chunk = remaining * 0.4
        result = _local_search_incremental(best, time.time() + chunk, seed=42)
        if len(result) > best_size:
            best = result
            best_size = len(result)
    
    # Step 3: Targeted local search (slower per iter but smarter)
    remaining = deadline - time.time()
    if remaining > 10:
        chunk = remaining * 0.4
        result = _targeted_local_search(best, time.time() + chunk, seed=123)
        if len(result) > best_size:
            best = result
            best_size = len(result)
    
    # Step 4: Another round of incremental search from improved state
    remaining = deadline - time.time()
    if remaining > 5:
        chunk = remaining * 0.8
        result = _local_search_incremental(best, time.time() + chunk, seed=789)
        if len(result) > best_size:
            best = result
            best_size = len(result)
    
    # Final validation
    if not _fast_is_valid(best):
        best = _greedy_build()
    
    return best
