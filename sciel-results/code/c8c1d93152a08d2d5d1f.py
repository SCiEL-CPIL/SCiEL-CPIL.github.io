import time
import random
import sys

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    L = k + 1
    total_diffs = n_rows * k * (k + 1) // 2
    lb = total_diffs  # minimum possible scope (0-indexed, so scope >= total_diffs)

    best_rows = None
    best_scope = float('inf')

    def get_scope(rows):
        return max(max(r) for r in rows)

    def build_used(rows):
        used = set()
        for row in rows:
            for i in range(L):
                ri = row[i]
                for j in range(i + 1, L):
                    used.add(row[j] - ri)
        return used

    def validate(rows):
        if rows is None:
            return False
        used = set()
        for row in rows:
            if len(row) != L:
                return False
            if row[0] != 0:
                return False
            for i in range(L):
                for j in range(i + 1, L):
                    d = row[j] - row[i]
                    if d in used:
                        return False
                    used.add(d)
        return True

    def can_place(row, pos, new_val, used_set):
        """Check if setting row[pos]=new_val keeps all diffs distinct."""
        if pos > 0 and new_val <= row[pos - 1]:
            return False
        if pos < L - 1 and new_val >= row[pos + 1]:
            return False
        # Collect old diffs involving pos
        v_old = row[pos]
        old_diffs = set()
        for i in range(L):
            if i < pos:
                old_diffs.add(v_old - row[i])
            elif i > pos:
                old_diffs.add(row[i] - v_old)
        # Check new diffs against used_set minus old_diffs
        v = new_val
        for i in range(L):
            if i < pos:
                d = v - row[i]
                if d in used_set and d not in old_diffs:
                    return False
            elif i > pos:
                d = row[i] - v
                if d in used_set and d not in old_diffs:
                    return False
        return True

    def apply_set(row, pos, new_val, used_set):
        """Update used_set and row when setting row[pos]=new_val."""
        v_old = row[pos]
        for i in range(L):
            if i < pos:
                used_set.discard(v_old - row[i])
            elif i > pos:
                used_set.discard(row[i] - v_old)
        row[pos] = new_val
        v = new_val
        for i in range(L):
            if i < pos:
                used_set.add(v - row[i])
            elif i > pos:
                used_set.add(row[i] - v)

    def local_compact(rows, time_limit):
        """Iteratively compact each element to its smallest feasible value."""
        work = [list(r) for r in rows]
        used_set = build_used(work)
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            for r_idx in range(n_rows):
                row = work[r_idx]
                for pos in range(L - 1, 0, -1):
                    old_val = row[pos]
                    lower = row[pos - 1] + 1
                    if lower >= old_val:
                        continue
                    upper = row[pos + 1] if pos < L - 1 else old_val
                    for new_val in range(lower, min(old_val, upper)):
                        if can_place(row, pos, new_val, used_set):
                            apply_set(row, pos, new_val, used_set)
                            improved = True
                            break
        return work

    def greedy_construct(seed, strategy='basic'):
        """Build rows greedily with given strategy."""
        rng = random.Random(seed)
        used = set()
        rows = []

        if strategy == 'random_order':
            row_order = list(range(n_rows))
            rng.shuffle(row_order)
            row_slots = [None] * n_rows
            for row_idx in row_order:
                row = [0]
                while len(row) < L:
                    last = row[-1]
                    found = False
                    # Adaptive search window
                    window = 25 + rng.randint(0, 20) + (len(row) - 1) * 3
                    for x in range(last + 1, last + window + 1):
                        nd = [x - a for a in row]
                        if any(d in used for d in nd):
                            continue
                        row.append(x)
                        used.update(nd)
                        found = True
                        break
                    if not found:
                        for x in range(last + 1, last + 300):
                            nd = [x - a for a in row]
                            if any(d in used for d in nd):
                                continue
                            row.append(x)
                            used.update(nd)
                            found = True
                            break
                        if not found:
                            return None
                row_slots[row_idx] = row
            if any(r is None for r in row_slots):
                return None
            return row_slots
        else:
            for _ in range(n_rows):
                row = [0]
                while len(row) < L:
                    last = row[-1]
                    found = False
                    if strategy == 'compact':
                        window = 18 + rng.randint(0, 10) + (len(row) - 1) * 2
                    elif strategy == 'spread':
                        window = 40 + rng.randint(0, 30) + (len(row) - 1) * 4
                    else:
                        window = 28 + rng.randint(0, 18) + (len(row) - 1) * 3
                    candidates = list(range(last + 1, last + window + 1))
                    if rng.random() < 0.4:
                        rng.shuffle(candidates)
                    for x in candidates:
                        nd = [x - a for a in row]
                        if any(d in used for d in nd):
                            continue
                        row.append(x)
                        used.update(nd)
                        found = True
                        break
                    if not found:
                        for x in range(last + 1, last + 350):
                            nd = [x - a for a in row]
                            if any(d in used for d in nd):
                                continue
                            row.append(x)
                            used.update(nd)
                            found = True
                            break
                        if not found:
                            return None
                rows.append(row)
            return rows

    def hill_climb(rows_init, time_limit):
        """Hill climbing: try to reduce scope by moving elements."""
        rows = [list(r) for r in rows_init]
        used_set = build_used(rows)
        scope = get_scope(rows)
        
        for _ in range(5000):
            if time.time() > time_limit:
                break
            # Pick a random element to try to move
            r_idx = random.randrange(n_rows)
            pos = random.randrange(1, L)
            row = rows[r_idx]
            old_val = row[pos]
            
            lower = row[pos - 1] + 1
            upper = row[pos + 1] - 1 if pos < L - 1 else old_val
            
            if lower >= old_val:
                continue
            
            # Try to move to a smaller value
            target = random.randint(lower, min(old_val - 1, upper)) if old_val > lower else None
            if target is None or target >= upper:
                continue
            
            if can_place(row, pos, target, used_set):
                apply_set(row, pos, target, used_set)
                new_scope = get_scope(rows)
                if new_scope < scope:
                    scope = new_scope
                elif random.random() < 0.1:
                    # Accept some worsening moves for exploration
                    pass
                else:
                    # Revert if not improving
                    apply_set(row, pos, old_val, used_set)
        
        return rows

    def simulated_annealing(rows_init, time_limit):
        """Simulated annealing to escape local optima."""
        rows = [list(r) for r in rows_init]
        used_set = build_used(rows)
        scope = get_scope(rows)
        best_scope = scope
        best_rows = [list(r) for r in rows]
        
        temp = 50.0
        cooling = 0.995
        
        while time.time() < time_limit and temp > 0.5:
            r_idx = random.randrange(n_rows)
            pos = random.randrange(1, L)
            row = rows[r_idx]
            old_val = row[pos]
            
            lower = row[pos - 1] + 1
            upper = row[pos + 1] - 1 if pos < L - 1 else old_val
            
            if lower >= old_val:
                temp *= cooling
                continue
            
            # Random move
            if random.random() < 0.7:
                target = random.randint(lower, min(old_val - 1, upper)) if old_val > lower else None
            else:
                target = random.randint(max(lower, old_val + 1), upper) if upper > old_val else None
            
            if target is None or target < lower or (pos < L - 1 and target >= row[pos + 1]):
                temp *= cooling
                continue
            
            if can_place(row, pos, target, used_set):
                apply_set(row, pos, target, used_set)
                new_scope = get_scope(rows)
                delta = new_scope - scope
                if delta < 0 or random.random() < pow(2.718281828, -delta / temp):
                    scope = new_scope
                    if scope < best_scope:
                        best_scope = scope
                        best_rows = [list(r) for r in rows]
                else:
                    apply_set(row, pos, old_val, used_set)
            
            temp *= cooling
        
        return best_rows

    # Phase 1: Many greedy constructions
    strategies = ['basic', 'compact', 'spread', 'random_order']
    for i in range(5000):
        if time.time() > deadline - 10:
            break
        strategy = strategies[i % len(strategies)]
        rows = greedy_construct(i * 137 + 42, strategy)
        if rows is not None:
            scope = get_scope(rows)
            if scope < best_scope:
                best_scope = scope
                best_rows = [list(r) for r in rows]
                if best_scope <= lb:
                    break

    if best_rows is None:
        # Fallback: try harder
        for i in range(2000):
            if time.time() > deadline - 5:
                break
            rows = greedy_construct(999999 + i * 31, 'basic')
            if rows is not None:
                scope = get_scope(rows)
                if scope < best_scope:
                    best_scope = scope
                    best_rows = [list(r) for r in rows]

    if best_rows is None:
        # Last resort: return a trivial (possibly invalid) answer
        return [[i * 100 for i in range(L)] for _ in range(n_rows)]

    # Phase 2: Local compaction
    if time.time() < deadline - 8:
        improved = local_compact(best_rows, deadline - 7)
        imp_scope = get_scope(improved)
        if imp_scope < best_scope:
            best_scope = imp_scope
            best_rows = improved

    # Phase 3: Hill climbing on best
    if time.time() < deadline - 6:
        hc_rows = hill_climb(best_rows, deadline - 5)
        hc_scope = get_scope(hc_rows)
        if hc_scope < best_scope:
            best_scope = hc_scope
            best_rows = hc_rows

    # Phase 4: More greedy + compact cycles
    for i in range(3000):
        if time.time() > deadline - 5:
            break
        rows = greedy_construct(500000 + i * 71, strategies[i % len(strategies)])
        if rows is not None:
            rows = local_compact(rows, deadline - 4)
            scope = get_scope(rows)
            if scope < best_scope:
                best_scope = scope
                best_rows = [list(r) for r in rows]
                if best_scope <= lb:
                    break

    # Phase 5: Simulated annealing on best
    if best_rows is not None and time.time() < deadline - 3:
        sa_rows = simulated_annealing(best_rows, deadline - 2)
        sa_scope = get_scope(sa_rows)
        if sa_scope < best_scope:
            best_scope = sa_scope
            best_rows = sa_rows

    # Phase 6: Final compaction
    if best_rows is not None and time.time() < deadline - 1:
        improved = local_compact(best_rows, deadline - 0.5)
        imp_scope = get_scope(improved)
        if imp_scope < best_scope:
            best_scope = imp_scope
            best_rows = improved

    # Validate before returning
    if best_rows is not None and not validate(best_rows):
        # Try to fix
        fixed = local_compact(best_rows, deadline - 0.2)
        if validate(fixed):
            best_rows = fixed
        else:
            # Return as-is; evaluator will score 0 if invalid
            pass

    return best_rows


# Quick self-test
if __name__ == "__main__":
    result = construct(4, 7)
    if result:
        print(f"Scope: {max(max(r) for r in result)}")
        print(f"Rows: {result}")
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
