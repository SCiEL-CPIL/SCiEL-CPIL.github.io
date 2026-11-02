import time
import random
import sys
from itertools import combinations

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 150s kill

    total_diffs = n_rows * k * (k + 1) // 2
    lb = total_diffs + 1
    max_scope_bound = lb + 300  # upper bound for difference lookup

    best_rows = None
    best_scope = float('inf')

    def validate_rows(rows):
        """Check all differences are distinct across all rows."""
        used = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in used:
                        return False
                    used.add(d)
        return True

    def get_scope(rows):
        return max(max(r) for r in rows)

    def compute_used_set(rows):
        """Compute the set of all differences across all rows."""
        used = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    used.add(row[j] - row[i])
        return used

    def greedy_construct(seed, strategy='basic'):
        """
        Construct rows using greedy with randomization.
        strategy: 'basic', 'random_order', 'compact', 'spread'
        """
        rng = random.Random(seed)
        used = set()
        rows = []

        def try_add_element(row, last, search_start, search_end):
            """Try to find a valid next element for row."""
            for x in range(search_start, search_end):
                new_diffs = [x - a for a in row]
                if any(d in used for d in new_diffs):
                    continue
                row.append(x)
                used.update(new_diffs)
                return True
            return False

        if strategy == 'random_order':
            row_order = list(range(n_rows))
            rng.shuffle(row_order)
            row_slots = [None] * n_rows

            for row_idx in row_order:
                row = [0]
                while len(row) < k + 1:
                    last = row[-1]
                    found = False

                    if strategy == 'compact':
                        search_end = last + 30 + rng.randint(0, 20)
                    elif strategy == 'spread':
                        search_end = last + 50 + rng.randint(0, 40)
                    else:
                        search_end = last + 40 + rng.randint(0, 30)

                    search_end = min(search_end, lb + 200)

                    # Try candidates in slightly randomized order
                    candidates = list(range(last + 1, search_end))
                    if rng.random() < 0.6:
                        for i in range(len(candidates) - 1, 0, -1):
                            if rng.random() < 0.1:
                                candidates[i], candidates[i-1] = candidates[i-1], candidates[i]
                    else:
                        rng.shuffle(candidates)

                    for x in candidates:
                        new_diffs = [x - a for a in row]
                        if any(d in used for d in new_diffs):
                            continue
                        row.append(x)
                        used.update(new_diffs)
                        found = True
                        break

                    if not found:
                        # Wider search
                        for x in range(last + 1, lb + 300):
                            new_diffs = [x - a for a in row]
                            if any(d in used for d in new_diffs):
                                continue
                            row.append(x)
                            used.update(new_diffs)
                            found = True
                            break
                        if not found:
                            return None
                row_slots[row_idx] = row

            if any(r is None for r in row_slots):
                return None
            return row_slots
        else:
            # Build rows in order
            for row_idx in range(n_rows):
                row = [0]
                while len(row) < k + 1:
                    last = row[-1]
                    found = False

                    if strategy == 'compact':
                        search_end = last + 25 + rng.randint(0, 15)
                    elif strategy == 'spread':
                        search_end = last + 50 + rng.randint(0, 40)
                    else:
                        search_end = last + 35 + rng.randint(0, 25)

                    search_end = min(search_end, lb + 200)

                    candidates = list(range(last + 1, search_end))
                    if rng.random() < 0.5:
                        pass
                    else:
                        rng.shuffle(candidates)

                    for x in candidates:
                        new_diffs = [x - a for a in row]
                        if any(d in used for d in new_diffs):
                            continue
                        row.append(x)
                        used.update(new_diffs)
                        found = True
                        break

                    if not found:
                        for x in range(last + 1, lb + 300):
                            new_diffs = [x - a for a in row]
                            if any(d in used for d in new_diffs):
                                continue
                            row.append(x)
                            used.update(new_diffs)
                            found = True
                            break
                        if not found:
                            return None
                rows.append(row)
            return rows

    def local_improve(rows, seed, max_iters=200):
        """Try to reduce scope by shifting elements left."""
        rng = random.Random(seed)
        rows = [list(r) for r in rows]
        scope = get_scope(rows)

        for _ in range(max_iters):
            if time.time() > deadline - 2:
                break

            # Pick a random row and a random non-zero element
            row_idx = rng.randint(0, n_rows - 1)
            row = rows[row_idx]
            if len(row) < 2:
                continue

            elem_idx = rng.randint(1, len(row) - 1)
            old_val = row[elem_idx]
            if old_val <= 1:
                continue

            # Remove this element
            row.pop(elem_idx)

            # Compute all differences in current configuration (without this element)
            used_set = set()
            for i, r in enumerate(rows):
                for a in range(len(r)):
                    for b in range(a + 1, len(r)):
                        used_set.add(r[b] - r[a])

            # Try to re-insert at a smaller position
            found = False
            min_val = row[elem_idx - 1] + 1 if elem_idx > 0 else 1
            max_val = old_val - 1

            for new_val in range(max_val, min_val - 1, -1):
                # Check differences with all elements in this row
                valid = True
                # Differences with elements before insertion point
                for a in row[:elem_idx]:
                    d = new_val - a
                    if d in used_set:
                        valid = False
                        break
                if not valid:
                    continue
                # Differences with elements after insertion point
                for a in row[elem_idx:]:
                    d = a - new_val
                    if d in used_set:
                        valid = False
                        break
                if valid:
                    row.insert(elem_idx, new_val)
                    found = True
                    break

            if found:
                new_scope = get_scope(rows)
                if new_scope < scope:
                    scope = new_scope
            else:
                # Restore
                row.insert(elem_idx, old_val)

        return rows, scope

    def random_restart_construct(seed):
        """Build with completely random choices."""
        rng = random.Random(seed)
        used = set()
        rows = []

        for _ in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                last = row[-1]
                found = False
                # Random jump size
                jump = rng.randint(1, 60)
                x = last + jump
                # Try this and nearby
                for candidate in range(x - 10, x + 10):
                    if candidate <= last:
                        continue
                    new_diffs = [candidate - a for a in row]
                    if any(d in used for d in new_diffs):
                        continue
                    row.append(candidate)
                    used.update(new_diffs)
                    found = True
                    break
                if not found:
                    # Try wider
                    for candidate in range(last + 1, last + 200):
                        new_diffs = [candidate - a for a in row]
                        if any(d in used for d in new_diffs):
                            continue
                        row.append(candidate)
                        used.update(new_diffs)
                        found = True
                        break
                    if not found:
                        return None
            rows.append(row)
        return rows

    # Strategy 1: Many greedy attempts with different strategies
    strategies = ['basic', 'compact', 'spread', 'random_order']
    attempt_count = 0
    for i in range(3000):
        if time.time() > deadline - 5:
            break
        strategy = strategies[i % len(strategies)]
        rows = greedy_construct(i * 31 + 7, strategy)
        if rows is not None:
            scope = get_scope(rows)
            if scope < best_scope:
                best_scope = scope
                best_rows = [list(r) for r in rows]
                if best_scope == lb:
                    break
        attempt_count += 1

    # Strategy 2: Local improvement on best found
    if best_rows is not None:
        improved, imp_scope = local_improve(best_rows, 42, max_iters=1000)
        if imp_scope < best_scope:
            best_scope = imp_scope
            best_rows = improved
        if time.time() > deadline - 5:
            return best_rows

    # Strategy 3: Random restarts
    for i in range(800):
        if time.time() > deadline - 5:
            break
        rows = random_restart_construct(100000 + i * 97)
        if rows is not None:
            scope = get_scope(rows)
            if scope < best_scope:
                best_scope = scope
                best_rows = [list(r) for r in rows]
                if best_scope == lb:
                    break

    # Strategy 4: Local improvement on best found again
    if best_rows is not None:
        improved, imp_scope = local_improve(best_rows, 12345, max_iters=500)
        if imp_scope < best_scope:
            best_scope = imp_scope
            best_rows = improved

    # Strategy 5: Hill climbing - try to reduce any element
    if best_rows is not None:
        rng = random.Random(99999)
        current_rows = [list(r) for r in best_rows]
        current_scope = best_scope

        for iteration in range(500):
            if time.time() > deadline - 3:
                break

            # Try to shift an element in a row to a smaller value
            row_idx = rng.randint(0, n_rows - 1)
            row = current_rows[row_idx]
            if len(row) < 2:
                continue

            # Try to reduce a random element (not the first)
            elem_pos = rng.randint(1, len(row) - 1)
            old_val = row[elem_pos]
            if old_val <= 1:
                continue

            # Remove this element
            row.pop(elem_pos)

            # Compute all differences in current configuration (without this element)
            used_set = set()
            for i, r in enumerate(current_rows):
                for a in range(len(r)):
                    for b in range(a + 1, len(r)):
                        used_set.add(r[b] - r[a])

            # Try to re-insert at a smaller position
            found = False
            min_val = row[elem_pos - 1] + 1 if elem_pos > 0 else 1
            max_val = old_val - 1

            for new_val in range(max_val, min_val - 1, -1):
                # Check differences with all elements in this row
                valid = True
                for a in row[:elem_pos]:
                    d = new_val - a
                    if d in used_set:
                        valid = False
                        break
                if not valid:
                    continue
                for a in row[elem_pos:]:
                    d = a - new_val
                    if d in used_set:
                        valid = False
                        break
                if valid:
                    row.insert(elem_pos, new_val)
                    found = True
                    break

            if found:
                new_scope = get_scope(current_rows)
                if new_scope < current_scope:
                    current_scope = new_scope
                    if current_scope < best_scope:
                        best_scope = current_scope
                        best_rows = [list(r) for r in current_rows]
            else:
                # Restore
                row.insert(elem_pos, old_val)

    # Strategy 6: Final tight compaction pass - repeatedly try to reduce any element
    if best_rows is not None:
        current_rows = [list(r) for r in best_rows]
        current_scope = best_scope
        improved_any = True
        max_compaction_rounds = 50

        for round_num in range(max_compaction_rounds):
            if time.time() > deadline - 2:
                break
            if not improved_any:
                break
            improved_any = False

            # Systematically try to reduce each element in each row
            for row_idx in range(n_rows):
                if time.time() > deadline - 2:
                    break
                row = current_rows[row_idx]
                # Try from the last element backwards
                for elem_pos in range(len(row) - 1, 0, -1):
                    old_val = row[elem_pos]
                    if old_val <= 1:
                        continue

                    # Remove this element
                    row.pop(elem_pos)

                    # Compute all differences in current configuration (without this element)
                    used_set = set()
                    for i, r in enumerate(current_rows):
                        for a in range(len(r)):
                            for b in range(a + 1, len(r)):
                                used_set.add(r[b] - r[a])

                    # Try to re-insert at a smaller position
                    found = False
                    min_val = row[elem_pos - 1] + 1 if elem_pos > 0 else 1
                    max_val = old_val - 1

                    for new_val in range(max_val, min_val - 1, -1):
                        # Check differences with all elements in this row
                        valid = True
                        for a in row[:elem_pos]:
                            d = new_val - a
                            if d in used_set:
                                valid = False
                                break
                        if not valid:
                            continue
                        for a in row[elem_pos:]:
                            d = a - new_val
                            if d in used_set:
                                valid = False
                                break
                        if valid:
                            row.insert(elem_pos, new_val)
                            found = True
                            break

                    if found:
                        new_scope = get_scope(current_rows)
                        if new_scope < current_scope:
                            current_scope = new_scope
                            improved_any = True
                            if current_scope < best_scope:
                                best_scope = current_scope
                                best_rows = [list(r) for r in current_rows]
                    else:
                        # Restore
                        row.insert(elem_pos, old_val)

    return best_rows


# Quick self-test
if __name__ == "__main__":
    result = construct(4, 7)
    if result:
        print(f"Scope: {max(max(r) for r in result)}")
        print(f"Rows: {result}")
        # Validate
        used = set()
        valid = True
        for row in result:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in used:
                        valid = False
                        print(f"Duplicate diff {d}")
                    used.add(d)
        print(f"Valid: {valid}")
    else:
        print("No valid construction found")
