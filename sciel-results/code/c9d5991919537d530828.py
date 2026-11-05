import random
import time

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s limit

    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row
    lower_bound = total_diffs

    def get_max_entry(rows):
        return max(max(row) for row in rows)

    def get_row_diffs_set(row):
        diffs = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs

    def get_all_diffs_set(rows):
        diffs = set()
        for row in rows:
            diffs |= get_row_diffs_set(row)
        return diffs

    def get_row_diffs_bitset(row):
        diffs = 0
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                diffs |= (1 << d)
        return diffs

    def is_valid_row(row, global_used_set):
        row_diffs = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d in row_diffs:
                    return False, set()
                row_diffs.add(d)
        if row_diffs & global_used_set:
            return False, set()
        return True, row_diffs

    def greedy_row(max_val, global_used_set):
        row = [0]
        current_diffs = set()
        while len(row) < k + 1:
            last = row[-1]
            found = False
            for v in range(last + 1, max_val + 1):
                conflict = False
                new_diffs = set()
                for u in row:
                    d = v - u
                    if d in current_diffs:
                        conflict = True
                        break
                    new_diffs.add(d)
                if conflict:
                    continue
                if new_diffs & global_used_set:
                    continue
                row.append(v)
                current_diffs |= new_diffs
                found = True
                break
            if not found:
                return None
        return row

    def random_row(max_val, global_used_set, attempts=1):
        for _ in range(attempts):
            row = [0]
            current_diffs = set()
            success = True
            for _ in range(k):
                last = row[-1]
                candidates = list(range(last + 1, max_val + 1))
                random.shuffle(candidates)
                found = False
                for v in candidates:
                    conflict = False
                    new_diffs = set()
                    for u in row:
                        d = v - u
                        if d in current_diffs:
                            conflict = True
                            break
                        new_diffs.add(d)
                    if conflict:
                        continue
                    if new_diffs & global_used_set:
                        continue
                    row.append(v)
                    current_diffs |= new_diffs
                    found = True
                    break
                if not found:
                    success = False
                    break
            if success:
                valid, _ = is_valid_row(row, global_used_set)
                if valid:
                    return row
        return None

    def backtrack_row(max_val, global_used_set, depth_limit=10000):
        """
        Backtracking search for a single row with max entry <= max_val.
        Returns the row or None.
        """
        rows = [[0]]
        current_diffs = set()
        
        def dfs(idx, row, curr_diffs):
            if time.time() > deadline - 1.0:
                return None
            if idx == k + 1:
                return row
            
            last = row[-1] if row else -1
            # Generate candidates in increasing order
            for v in range(last + 1, max_val + 1):
                conflict = False
                new_diffs = set()
                for u in row:
                    d = v - u
                    if d in curr_diffs:
                        conflict = True
                        break
                    new_diffs.add(d)
                
                if conflict:
                    continue
                
                if new_diffs & global_used_set:
                    continue
                
                new_row = row + [v]
                new_diffs_set = curr_diffs | new_diffs
                
                result = dfs(idx + 1, new_row, new_diffs_set)
                if result is not None:
                    return result
            
            return None
        
        return dfs(0, [0], set())

    def validate(rows):
        all_diffs = set()
        for row in rows:
            if len(row) != k + 1:
                return False
            if row[0] != 0:
                return False
            for i in range(len(row) - 1):
                if row[i] >= row[i + 1]:
                    return False
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in all_diffs:
                        return False
                    all_diffs.add(d)
        return True

    # Initial Solution: Greedy with large scope
    init_scope = 250
    rows = [None] * n_rows
    success = False

    if time.time() < deadline - 20:
        for _ in range(5):
            if time.time() > deadline - 20:
                break
            rows = [None] * n_rows
            global_used = set()
            found = True
            for i in range(n_rows):
                row = greedy_row(init_scope, global_used)
                if row is None:
                    found = False
                    break
                rows[i] = row
                global_used |= get_row_diffs_set(row)
            if found:
                success = True
                break

    if not success:
        # Fallback: Very large scope with pure greedy
        rows = []
        global_used = set()
        for i in range(n_rows):
            row = greedy_row(500, global_used)
            if row is None:
                row = [0] + list(range(1, k + 1))
            rows.append(row)
            global_used |= get_row_diffs_set(row)

    best_rows = [list(r) for r in rows]
    best_max = get_max_entry(best_rows)
    current_rows = [list(r) for r in rows]
    current_max = best_max

    # Optimization Loop: Binary Search for Minimum Max
    low = lower_bound
    high = best_max

    search_start_time = time.time()
    remaining_time = deadline - search_start_time - 5.0

    if remaining_time > 10:
        while low < high and (time.time() - search_start_time) < remaining_time - 5:
            mid = (low + high) // 2
            check_time_limit = time.time() + (remaining_time - (time.time() - search_start_time)) * 0.5

            # Try to find a solution with max <= mid
            found_sol = None
            for _ in range(3):
                if time.time() > check_time_limit:
                    break
                sol_rows = []
                global_used = set()
                valid_sol = True
                for i in range(n_rows):
                    row = backtrack_row(mid, global_used)
                    if row is None:
                        # Fallback to random
                        row = random_row(mid, global_used, attempts=2)
                    if row is None:
                        valid_sol = False
                        break
                    sol_rows.append(row)
                    global_used |= get_row_diffs_set(row)
                if valid_sol:
                    if validate(sol_rows):
                        found_sol = sol_rows
                        break

            if found_sol is not None:
                best_rows = [list(r) for r in found_sol]
                best_max = get_max_entry(best_rows)
                high = mid
            else:
                low = mid + 1

    if best_rows:
        current_rows = best_rows
        current_max = best_max

    # Local Search to Polish
    step = 0
    no_improve_count = 0
    max_no_improve = 100

    while time.time() < deadline - 2.0:
        step += 1

        row_idx = random.randint(0, n_rows - 1)

        global_used_except = set()
        for i in range(n_rows):
            if i != row_idx:
                global_used_except |= get_row_diffs_set(current_rows[i])

        curr_row_max = current_rows[row_idx][-1]

        target = curr_row_max - 1

        if target < k:
            continue

        new_row = None

        # Try backtracking to find a smaller row
        if time.time() < deadline - 5.0:
            new_row = backtrack_row(target, global_used_except)

        if new_row is None:
            # Try randomized
            for _ in range(5):
                if time.time() > deadline - 2.0:
                    break
                new_row = random_row(target, global_used_except, attempts=1)
                if new_row is not None:
                    break

        if new_row is not None:
            valid, _ = is_valid_row(new_row, global_used_except)
            if valid:
                current_rows[row_idx] = new_row
                new_global_max = get_max_entry(current_rows)
                if new_global_max < best_max:
                    best_max = new_global_max
                    best_rows = [list(r) for r in current_rows]
                    no_improve_count = 0
                else:
                    no_improve_count += 1
        else:
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                # Try to change structure at same max to unblock
                new_row = random_row(curr_row_max, global_used_except, attempts=5)
                if new_row is not None and new_row != current_rows[row_idx]:
                    current_rows[row_idx] = new_row
                    no_improve_count = 0
                    max_no_improve += 50
                else:
                    no_improve_count = 0
                    max_no_improve += 100

    if not validate(best_rows):
        best_rows = []
        gu = set()
        for i in range(n_rows):
            r = greedy_row(1000, gu)
            if r is None:
                r = [0] + list(range(1, k + 1))
            best_rows.append(r)
            gu |= get_row_diffs_set(r)

    return best_rows
