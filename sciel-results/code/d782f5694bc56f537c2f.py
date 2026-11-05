import time
import random

def construct():
    start_time = time.time()
    deadline = 85.0  # 90s limit, leave 5s margin

    # Precompute: for each x in [1,100], compute the number of subsets of [1..100]\{x} that sum to 200-x
    # This is a proxy for "conflict density" - higher means x is more "dangerous"
    # Use bitset DP for speed
    conflict_density = {}
    for x in range(1, 101):
        target = 200 - x
        if target < 0:
            conflict_density[x] = 0
            continue
        # Compute reachable sums using all elements except x
        r = 1
        for y in range(1, 101):
            if y == x:
                continue
            r |= (r << y)
        # Count how many subsets sum to target
        # We can't easily count from bitset, but we can check if target is reachable
        # For heuristic ordering, just use whether it's reachable (binary) or approximate
        # Better: use the bitset to check reachability, and for density use a simple proxy
        conflict_density[x] = 1 if ((r >> target) & 1) else 0

    # Greedy initialization: sort by conflict density (ascending), then by value (descending)
    # Elements with lower conflict density are safer to include
    candidates = sorted(range(1, 101), key=lambda x: (conflict_density[x], -x))

    selected = []
    reachable = 1
    for x in candidates:
        new_reach = reachable | (reachable << x)
        if not (new_reach >> 200) & 1:
            selected.append(x)
            reachable = new_reach

    best_set = set(selected)
    best_size = len(best_set)

    # Local search with bitset reachability
    current_set = best_set.copy()
    current_reachable = reachable

    all_elements = list(range(1, 101))

    def can_add(x, R):
        return not ((R >> (200 - x)) & 1)

    def update_reachable(R, x):
        return R | (R << x)

    def is_valid_set(S):
        r = 1
        for x in S:
            r |= (r << x)
            if (r >> 200) & 1:
                return False
        return True

    improved = True
    while improved and time.time() < deadline:
        improved = False

        # Phase 1: Try to add any single missing element (prioritize by conflict density)
        missing = [x for x in all_elements if x not in current_set]
        missing.sort(key=lambda x: (conflict_density[x], -x))

        for x in missing:
            if time.time() > deadline:
                break
            if can_add(x, current_reachable):
                current_set.add(x)
                current_reachable = update_reachable(current_reachable, x)
                improved = True
                if len(current_set) > best_size:
                    best_set = current_set.copy()
                    best_size = len(current_set)

        # Phase 2: Directed 1-for-1 swaps
        if not improved and time.time() < deadline:
            curr_list = list(current_set)
            miss_list = [x for x in all_elements if x not in current_set]
            miss_list.sort(key=lambda x: (conflict_density[x], -x))

            found_swap = False
            for a in curr_list:
                if time.time() > deadline:
                    break
                for b in miss_list:
                    if time.time() > deadline:
                        break
                    # Quick check: if b can be added to (current_set - {a})
                    # Compute reachable for current_set - {a}
                    # This is expensive, so use a heuristic: try the swap and validate
                    new_set = current_set.copy()
                    new_set.remove(a)
                    new_set.add(b)
                    if is_valid_set(new_set):
                        current_set = new_set
                        current_reachable = 1
                        for x in current_set:
                            current_reachable |= (current_reachable << x)
                        improved = True
                        found_swap = True
                        if len(current_set) > best_size:
                            best_set = current_set.copy()
                            best_size = len(current_set)
                        break
                if found_swap or time.time() > deadline:
                    break

            if found_swap:
                continue

        # Phase 3: 1-for-2 swaps to increase size
        if not improved and time.time() < deadline:
            curr_list = list(current_set)
            miss_list = [x for x in all_elements if x not in current_set]
            miss_list.sort(key=lambda x: (conflict_density[x], -x))

            found_12 = False
            for a in curr_list:
                if time.time() > deadline:
                    break
                # Try pairs from miss_list
                for i in range(len(miss_list)):
                    if time.time() > deadline:
                        break
                    b = miss_list[i]
                    for j in range(i + 1, len(miss_list)):
                        if time.time() > deadline:
                            break
                        c = miss_list[j]
                        new_set = current_set.copy()
                        new_set.remove(a)
                        new_set.add(b)
                        new_set.add(c)
                        if is_valid_set(new_set):
                            current_set = new_set
                            current_reachable = 1
                            for x in current_set:
                                current_reachable |= (current_reachable << x)
                            improved = True
                            found_12 = True
                            if len(current_set) > best_size:
                                best_set = current_set.copy()
                                best_size = len(current_set)
                            break
                    if found_12 or time.time() > deadline:
                        break
                if found_12 or time.time() > deadline:
                    break

            if found_12:
                continue

    # Final high-precision polish: exhaustive single-add pass
    if time.time() < deadline:
        for x in range(100, 0, -1):
            if x in current_set:
                continue
            if can_add(x, current_reachable):
                current_set.add(x)
                current_reachable = update_reachable(current_reachable, x)
                if len(current_set) > best_size:
                    best_set = current_set.copy()
                    best_size = len(current_set)

    # Final 1-for-2 exhaustive polish
    if time.time() < deadline:
        curr_list = sorted(current_set, reverse=True)
        miss_list = [x for x in range(1, 101) if x not in current_set]
        found = True
        while found and time.time() < deadline:
            found = False
            for a in curr_list:
                if time.time() > deadline:
                    break
                new_set_without_a = current_set - {a}
                r_without_a = 1
                for x in new_set_without_a:
                    r_without_a |= (r_without_a << x)
                for i in range(len(miss_list)):
                    if time.time() > deadline:
                        break
                    b = miss_list[i]
                    if b in current_set:
                        continue
                    if can_add(b, r_without_a):
                        r_with_b = r_without_a | (r_without_a << b)
                        for j in range(i + 1, len(miss_list)):
                            if time.time() > deadline:
                                break
                            c = miss_list[j]
                            if c in current_set:
                                continue
                            if can_add(c, r_with_b):
                                new_set = new_set_without_a | {b, c}
                                if is_valid_set(new_set):
                                    current_set = new_set
                                    current_reachable = 1
                                    for x in current_set:
                                        current_reachable |= (current_reachable << x)
                                    if len(current_set) > best_size:
                                        best_set = current_set.copy()
                                        best_size = len(current_set)
                                    found = True
                                    break
                    if found or time.time() > deadline:
                        break
                if found or time.time() > deadline:
                    break

    return list(best_set)
