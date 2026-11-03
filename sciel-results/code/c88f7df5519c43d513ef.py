import time
import random
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    # Lower bound: need n_rows * k * (k+1)/2 distinct differences in [1, M-1]
    total_diffs = n_rows * k * (k + 1) // 2
    lb = total_diffs + 1

    def is_valid(rows):
        if len(rows) != n_rows:
            return False
        used = set()
        for r in rows:
            if len(r) != k + 1:
                return False
            if r[0] != 0:
                return False
            for i in range(len(r)):
                for j in range(i + 1, len(r)):
                    d = r[j] - r[i]
                    if d in used:
                        return False
                    used.add(d)
        return True

    def get_used_diffs(rows):
        used = set()
        for r in rows:
            for i in range(len(r)):
                for j in range(i + 1, len(r)):
                    used.add(r[j] - r[i])
        return used

    def check_value(row_partial, x, used):
        """Check if x can be added to row_partial (sorted, starts at 0)."""
        for a in row_partial:
            if x - a in used:
                return False
        return True

    def add_value_diffs(row_partial, x, used):
        """Add all differences of x with row_partial to used."""
        for a in row_partial:
            used.add(x - a)

    def remove_value_diffs(row_partial, x, used):
        """Remove all differences of x with row_partial from used."""
        for a in row_partial:
            used.discard(x - a)

    def csp_construct(seed_val, max_scope):
        """
        Backtracking CSP with forward checking.
        Builds rows one by one; each row is built element by element.
        When choosing each element, tries values in a smart order.
        """
        random.seed(seed_val)
        used = set()
        rows = []

        for row_idx in range(n_rows):
            row = [0]
            # Build remaining k elements
            for pos in range(k):
                found = False
                last = row[-1]

                # Generate candidate values
                # Strategy: try values that are "available" - their diffs don't conflict
                # Use a window around last + offset
                # First try compact: last+1, last+2, ... up to some limit
                # Then random

                # Phase 1: try sequential values from last+1
                # Limit how far we go in sequential phase
                seq_limit = min(last + 30 + pos * 5, max_scope)
                for x in range(last + 1, seq_limit + 1):
                    if check_value(row, x, used):
                        row.append(x)
                        add_value_diffs(row, x, used)
                        found = True
                        break

                if not found:
                    # Phase 2: random search in [last+1, max_scope]
                    # Try random values
                    for _ in range(200):
                        x = random.randint(last + 1, max_scope)
                        if check_value(row, x, used):
                            row.append(x)
                            add_value_diffs(row, x, used)
                            found = True
                            break

                if not found:
                    # Phase 3: full sequential search
                    for x in range(last + 1, max_scope + 1):
                        if check_value(row, x, used):
                            row.append(x)
                            add_value_diffs(row, x, used)
                            found = True
                            break

                if not found:
                    return None, float('inf')

            rows.append(row)

        scope = max(max(r) for r in rows)
        return rows, scope

    def csp_construct_smart(seed_val, max_scope):
        """
        Improved CSP: for each row, try to build it compactly.
        Uses a priority: prefer smaller values that are valid.
        Also uses "lookahead" - if a value leads to a dead end later, backtrack.
        """
        random.seed(seed_val)
        used = set()
        rows = []

        def build_row(row_idx):
            """Recursively build a single row with backtracking."""
            row = [0]

            def backtrack(pos):
                if pos == k + 1:
                    return True
                last = row[-1]
                # Try values from last+1 upward, with some randomization
                # First try sequential
                max_try = min(last + 50 + pos * 10, max_scope)
                candidates = list(range(last + 1, max_try + 1))
                # Add some random candidates
                for _ in range(20):
                    candidates.append(random.randint(last + 1, max_scope))
                random.shuffle(candidates)

                for x in candidates:
                    if x in row:
                        continue
                    if check_value(row, x, used):
                        row.append(x)
                        add_value_diffs(row, x, used)
                        if backtrack(pos + 1):
                            return True
                        remove_value_diffs(row, x, used)
                        row.pop()

                # If sequential didn't work, try full range
                for x in range(last + 1, max_scope + 1):
                    if x in row:
                        continue
                    if check_value(row, x, used):
                        row.append(x)
                        add_value_diffs(row, x, used)
                        if backtrack(pos + 1):
                            return True
                        remove_value_diffs(row, x, used)
                        row.pop()

                return False

            if backtrack(1):
                rows.append(row)
                return True
            return False

        for ri in range(n_rows):
            if not build_row(ri):
                return None, float('inf')

        scope = max(max(r) for r in rows)
        return rows, scope

    def compact_solution(rows, max_time):
        """
        Aggressively compact a valid solution by trying to lower each element.
        Returns the compacted rows and scope.
        """
        rows = [list(r) for r in rows]
        used = get_used_diffs(rows)
        scope = max(max(r) for r in rows)
        deadline_compact = time.time() + max_time

        improved = True
        while improved and time.time() < deadline_compact:
            improved = False
            # Try to lower each element (except 0) to a smaller available value
            for r_idx in range(n_rows):
                if time.time() > deadline_compact:
                    break
                row = rows[r_idx]
                for p_idx in range(1, len(row)):
                    if time.time() > deadline_compact:
                        break
                    old_val = row[p_idx]
                    # Remove old_val's differences
                    for a in row:
                        if a != old_val:
                            used.discard(abs(a - old_val))

                    row.pop(p_idx)

                    # Try to find a smaller value that fits
                    # Try from 1 up to old_val - 1
                    new_val = None
                    for x in range(1, old_val):
                        if x in row:
                            continue
                        valid = True
                        for a in row:
                            if abs(x - a) in used:
                                valid = False
                                break
                        if valid:
                            new_val = x
                            break

                    if new_val is not None:
                        # Insert new_val in sorted position
                        pos = 0
                        while pos < len(row) and row[pos] < new_val:
                            pos += 1
                        row.insert(pos, new_val)
                        for a in row:
                            if a != new_val:
                                used.add(abs(a - new_val))
                        if new_val < old_val:
                            improved = True
                            new_scope = max(max(r) for r in rows)
                            if new_scope < scope:
                                scope = new_scope
                    else:
                        # Revert
                        pos = 0
                        while pos < len(row) and row[pos] < old_val:
                            pos += 1
                        row.insert(pos, old_val)
                        for a in row:
                            if a != old_val:
                                used.add(abs(a - old_val))

        return rows, scope

    def local_search(rows, max_time):
        """
        Local search: try to reduce scope by moving elements to smaller values.
        Uses a more sophisticated approach than simple compaction.
        """
        rows = [list(r) for r in rows]
        used = get_used_diffs(rows)
        scope = max(max(r) for r in rows)
        deadline_ls = time.time() + max_time
        random.seed(42)

        # Try to reduce the maximum element
        for iteration in range(200):
            if time.time() > deadline_ls:
                break

            # Find the row with the largest element
            max_row_idx = 0
            max_val = 0
            max_pos = 0
            for r_idx in range(n_rows):
                for p_idx in range(len(rows[r_idx])):
                    if rows[r_idx][p_idx] > max_val:
                        max_val = rows[r_idx][p_idx]
                        max_row_idx = r_idx
                        max_pos = p_idx

            row = rows[max_row_idx]
            old_val = row[max_pos]

            # Remove old_val
            for a in row:
                if a != old_val:
                    used.discard(abs(a - old_val))
            row.pop(max_pos)

            # Try to find a smaller value for this position
            # The new value must be greater than row[max_pos-1] (if max_pos > 0)
            # and less than old_val
            lower_bound = row[max_pos - 1] + 1 if max_pos > 0 else 1
            found = False
            for x in range(lower_bound, old_val):
                if x in row:
                    continue
                valid = True
                for a in row:
                    if abs(x - a) in used:
                        valid = False
                        break
                if valid:
                    row.insert(max_pos, x)
                    for a in row:
                        if a != x:
                            used.add(abs(a - x))
                    found = True
                    break

            if not found:
                # Revert
                row.insert(max_pos, old_val)
                for a in row:
                    if a != old_val:
                        used.add(abs(a - old_val))

        new_scope = max(max(r) for r in rows)
        return rows, new_scope

    # Main algorithm
    best_rows = None
    best_scope = float('inf')

    # Phase 1: Find a valid solution using CSP
    # Try different max_scope values, starting from a reasonable estimate
    # The lower bound is lb, but we need some slack
    initial_scope_estimates = [lb + 20, lb + 40, lb + 60, lb + 80, lb + 100, lb + 150, lb + 200]

    for scope_est in initial_scope_estimates:
        if time.time() > deadline - 60:
            break
        for seed in range(30):
            if time.time() > deadline - 60:
                break
            rows, scope = csp_construct_smart(seed, scope_est)
            if rows is not None and scope < best_scope:
                best_scope = scope
                best_rows = [list(r) for r in rows]
                # If we found a solution close to the lower bound, stop early
                if scope <= lb + 30:
                    break
        if best_rows is not None and best_scope <= lb + 30:
            break

    # Fallback: if no solution found, use simple greedy
    if best_rows is None:
        used = set()
        best_rows = []
        for _ in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                x = row[-1] + 1
                while True:
                    x += 1
                    valid = True
                    for a in row:
                        if (x - a) in used:
                            valid = False
                            break
                    if valid:
                        break
                row.append(x)
                for a in row:
                    if a != x:
                        used.add(x - a)
            best_rows.append(row)
        best_scope = max(max(r) for r in best_rows)

    # Phase 2: Compact the solution
    # Use remaining time for compaction
    remaining_time = deadline - time.time() - 5
    if remaining_time > 10:
        compacted_rows, compacted_scope = compact_solution(best_rows, remaining_time * 0.5)
        if is_valid(compacted_rows) and compacted_scope < best_scope:
            best_rows = compacted_rows
            best_scope = compacted_scope

    # Phase 3: Local search for further improvement
    remaining_time = deadline - time.time() - 5
    if remaining_time > 10:
        ls_rows, ls_scope = local_search(best_rows, remaining_time * 0.5)
        if is_valid(ls_rows) and ls_scope < best_scope:
            best_rows = ls_rows
            best_scope = ls_scope

    # Phase 4: Final compaction pass
    remaining_time = deadline - time.time() - 2
    if remaining_time > 5:
        final_rows, final_scope = compact_solution(best_rows, remaining_time)
        if is_valid(final_rows) and final_scope < best_scope:
            best_rows = final_rows
            best_scope = final_scope

    # Final validation
    if not is_valid(best_rows):
        # Emergency fallback
        used = set()
        best_rows = []
        for _ in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                x = row[-1] + 1
                while True:
                    x += 1
                    valid = True
                    for a in row:
                        if (x - a) in used:
                            valid = False
                            break
                    if valid:
                        break
                row.append(x)
                for a in row:
                    if a != x:
                        used.add(x - a)
            best_rows.append(row)

    return best_rows
