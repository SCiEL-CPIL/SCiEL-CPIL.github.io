import time
import random
import sys
from collections import Counter
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 150s kill

    # Each row has k+1 elements starting at 0.
    # Number of differences per row = (k+1)*k/2
    diffs_per_row = (k + 1) * k // 2
    total_diffs = n_rows * diffs_per_row

    def get_scope(rows):
        """Get the maximum entry across all rows."""
        return max(max(r) for r in rows)

    def is_valid(rows):
        """Check if the rows form a valid DTS."""
        used = set()
        for row in rows:
            if len(row) != k + 1:
                return False
            if row[0] != 0:
                return False
            for i in range(1, len(row)):
                if row[i] <= row[i - 1]:
                    return False
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in used:
                        return False
                    used.add(d)
        return True

    def count_conflicts(rows):
        """Count the number of duplicate differences across all rows."""
        diffs = []
        for r in rows:
            for i in range(len(r)):
                for j in range(i + 1, len(r)):
                    diffs.append(r[j] - r[i])
        counts = Counter(diffs)
        conflicts = 0
        for count in counts.values():
            if count > 1:
                conflicts += count - 1
        return conflicts

    def objective(rows):
        """Objective: minimize conflicts first, then scope."""
        conflicts = count_conflicts(rows)
        scope = get_scope(rows)
        return conflicts * 1000000 + scope

    def greedy_construct(scope_limit=None, randomize=False, seed=None):
        """
        Greedy construction with optional randomization.
        """
        if seed is not None:
            random.seed(seed)
        
        used_diffs = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                found = False
                last_val = row[-1]
                max_val = scope_limit if scope_limit is not None else 10000
                
                # Generate candidate values
                if randomize:
                    # Try random values first, then sequential
                    candidates = list(range(last_val + 1, max_val + 1))
                    random.shuffle(candidates)
                    # But we need to ensure we don't skip too many, so maybe limit shuffle
                    # Actually, let's just pick a random subset or use a bias
                    # Simpler: pick a random start point within a window
                    window = min(20, max_val - last_val)
                    start_offset = random.randint(0, window)
                    # Try from last_val + 1 + start_offset, but also keep some sequential
                    # Let's just shuffle the first part
                    if len(candidates) > 50:
                        candidates = candidates[:50] + candidates[50:]
                        random.shuffle(candidates[:50])
                else:
                    candidates = range(last_val + 1, max_val + 1)

                for x in candidates:
                    if x <= last_val:
                        continue
                    if scope_limit is not None and x > scope_limit:
                        break
                    valid = True
                    new_diffs = []
                    for y in row:
                        d = x - y
                        if d in used_diffs:
                            valid = False
                            break
                        new_diffs.append(d)
                    
                    # Check internal uniqueness (implicit if x-y are distinct, which they are since y are distinct)
                    # But we also need to ensure new_diffs don't conflict with each other? 
                    # No, x-y for distinct y are distinct.
                    
                    if valid:
                        row.append(x)
                        for d in new_diffs:
                            used_diffs.add(d)
                        found = True
                        break
                if not found:
                    return None
            rows.append(row)
        return rows

    def greedy_construct_v2(scope_limit=None, randomize=False, seed=None, lookahead=1):
        """
        Improved greedy with lookahead: when placing an element, check if it leaves
        enough room for the remaining elements.
        """
        if seed is not None:
            random.seed(seed)
        
        used_diffs = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                found = False
                last_val = row[-1]
                max_val = scope_limit if scope_limit is not None else 10000
                remaining = k + 1 - len(row)
                
                # Minimum value needed: last_val + 1
                # Maximum value allowed: scope_limit
                # We need 'remaining' more elements, each at least 1 apart
                # So the last element must be at most scope_limit - (remaining - 1)
                # Actually, we just need to ensure we can fit 'remaining' elements
                # in the range [last_val + 1, scope_limit]
                
                candidates = []
                if randomize:
                    # Use a mix of random and sequential
                    if max_val - last_val > 100:
                        # For large ranges, sample more densely near the start
                        # to avoid wasting space
                        candidates = list(range(last_val + 1, min(last_val + 50, max_val + 1)))
                        random.shuffle(candidates)
                        # Also add some random points further out
                        for _ in range(20):
                            x = random.randint(last_val + 1, max_val)
                            if x not in candidates:
                                candidates.append(x)
                        candidates.sort()
                    else:
                        candidates = list(range(last_val + 1, max_val + 1))
                        random.shuffle(candidates)
                else:
                    candidates = list(range(last_val + 1, max_val + 1))

                for x in candidates:
                    if x <= last_val:
                        continue
                    if scope_limit is not None and x > scope_limit:
                        break
                    
                    # Quick feasibility check: can we fit 'remaining' elements after x?
                    # The next element must be > x, and we need 'remaining - 1' more
                    # So we need x + (remaining - 1) <= scope_limit
                    if remaining > 1 and x + (remaining - 1) > max_val:
                        break  # No point trying larger x
                    
                    valid = True
                    new_diffs = []
                    for y in row:
                        d = x - y
                        if d in used_diffs:
                            valid = False
                            break
                        new_diffs.append(d)
                    
                    if valid:
                        row.append(x)
                        for d in new_diffs:
                            used_diffs.add(d)
                        found = True
                        break
                if not found:
                    return None
            rows.append(row)
        return rows

    def backtrack_construct():
        """
        Backtracking construction for small instances or as a fallback.
        """
        used_diffs = set()
        rows = []
        
        def dfs(row_idx, pos, current_row):
            if pos == k + 1:
                rows.append(current_row[:])
                return True
            if row_idx == n_rows:
                return False
            
            start = current_row[-1] + 1 if pos > 0 else 1
            # Heuristic: try smaller values first
            for x in range(start, 1000):
                valid = True
                new_diffs = []
                for y in current_row:
                    d = x - y
                    if d in used_diffs:
                        valid = False
                        break
                    new_diffs.append(d)
                
                if valid:
                    current_row.append(x)
                    for d in new_diffs:
                        used_diffs.add(d)
                    
                    if pos == k:
                        # Finished a row
                        if dfs(row_idx + 1, 0, [0]):
                            return True
                        # Undo
                        for d in new_diffs:
                            used_diffs.discard(d)
                        current_row.pop()
                    else:
                        if dfs(row_idx, pos + 1, current_row):
                            return True
                        for d in new_diffs:
                            used_diffs.discard(d)
                        current_row.pop()
        
        dfs(0, 0, [0])
        if rows:
            return rows
        return None

    # Step 1: Generate initial solutions
    # Try greedy with various scope limits and randomizations
    best_rows = None
    best_obj = float('inf')
    
    # First, try deterministic greedy with tight scope limits
    for scope_limit in [100, 110, 112, 115, 120, 130, 140, 160, 200, 300, 500]:
        if time.time() > deadline - 60:
            break
        rows = greedy_construct(scope_limit)
        if rows is not None:
            obj = objective(rows)
            if obj < best_obj:
                best_obj = obj
                best_rows = rows
            # If we found a valid solution with low scope, we can stop early?
            # No, we want to minimize scope. If scope is already close to bound, maybe stop.
            if count_conflicts(rows) == 0 and get_scope(rows) <= 115:
                break

    # Try greedy_construct_v2 with lookahead
    for scope_limit in [110, 115, 120, 130, 140, 160, 200]:
        if time.time() > deadline - 60:
            break
        rows = greedy_construct_v2(scope_limit)
        if rows is not None:
            obj = objective(rows)
            if obj < best_obj:
                best_obj = obj
                best_rows = rows
            if count_conflicts(rows) == 0 and get_scope(rows) <= 115:
                break

    # If no good solution, try randomized greedy
    if best_rows is None or count_conflicts(best_rows) > 0:
        for _ in range(50):
            if time.time() > deadline - 30:
                break
            seed = random.randint(0, 10000)
            rows = greedy_construct(scope_limit=200, randomize=True, seed=seed)
            if rows is not None:
                obj = objective(rows)
                if obj < best_obj:
                    best_obj = obj
                    best_rows = rows

    # If still no valid solution, use backtracking (might be slow, so limit time)
    if best_rows is None or count_conflicts(best_rows) > 0:
        t_start = time.time()
        rows = backtrack_construct()
        if rows is not None:
            obj = objective(rows)
            if obj < best_obj:
                best_obj = obj
                best_rows = rows
        if time.time() - t_start > 10:
            pass # Just a note, backtrack might have taken long

    # Fallback: simple greedy without scope limit
    if best_rows is None:
        best_rows = greedy_construct()
        best_obj = objective(best_rows)

    # Step 2: Simulated Annealing to minimize objective
    # We use a hybrid approach: 
    # - If conflicts > 0, try to reduce conflicts.
    # - If conflicts == 0, try to reduce scope.
    
    current_rows = [r[:] for r in best_rows]
    current_obj = best_obj
    best_rows_sa = [r[:] for r in best_rows]
    best_obj_sa = best_obj

    # Maintain state for SA with incremental diff counting
    def compute_diff_counts(rows):
        """Return a Counter of all differences."""
        diffs = []
        for r in rows:
            for i in range(len(r)):
                for j in range(i + 1, len(r)):
                    diffs.append(r[j] - r[i])
        return Counter(diffs)

    diff_counts = compute_diff_counts(current_rows)
    current_conflicts = sum(c - 1 for c in diff_counts.values() if c > 1)
    current_scope = get_scope(current_rows)
    current_obj_val = current_conflicts * 1000000 + current_scope
    
    best_conflicts = current_conflicts
    best_scope = current_scope
    best_obj_val = current_obj_val
    best_rows_state = [r[:] for r in current_rows]

    temp = 500.0
    cooling = 0.995
    no_improve_count = 0
    max_no_improve = 500

    while time.time() < deadline - 5:
        # Adaptive temperature: restart if stuck
        if no_improve_count > max_no_improve:
            temp = 500.0
            no_improve_count = 0
            max_no_improve += 200
        
        # Pick a random move
        row_idx = random.randint(0, n_rows - 1)
        pos = random.randint(1, k)
        
        old_val = current_rows[row_idx][pos]
        min_val = current_rows[row_idx][pos - 1] + 1
        max_val = current_rows[row_idx][pos + 1] - 1 if pos < k else current_scope + 10
        
        if min_val > max_val:
            continue
            
        # Try a few random new values
        new_val = random.randint(min_val, max_val)
        if new_val == old_val:
            continue
        
        # Check strictly increasing
        if new_val <= min_val or (pos < k and new_val >= current_rows[row_idx][pos + 1]):
            continue
        
        # Calculate change in conflicts incrementally
        # Old diffs involving pos
        old_diffs = []
        for i in range(k + 1):
            if i != pos:
                d = abs(new_val - current_rows[row_idx][i])  # placeholder, we'll compute properly
                old_diffs.append(abs(old_val - current_rows[row_idx][i]))
        
        # New diffs involving pos
        new_diffs = []
        for i in range(k + 1):
            if i != pos:
                new_diffs.append(abs(new_val - current_rows[row_idx][i]))
        
        # Calculate delta in conflicts
        delta_conflicts = 0
        
        # For each old diff, decrement its count
        for d in old_diffs:
            if d in diff_counts:
                if diff_counts[d] == 1:
                    del diff_counts[d]
                else:
                    diff_counts[d] -= 1
                    if diff_counts[d] == 1:
                        delta_conflicts -= 1  # one less conflict
                    elif diff_counts[d] == 2:
                        delta_conflicts -= 1  # was 2 (1 conflict), now 1 (0 conflicts)
        
        # For each new diff, increment its count
        for d in new_diffs:
            if d not in diff_counts:
                diff_counts[d] = 1
            else:
                if diff_counts[d] == 1:
                    delta_conflicts += 1  # was 0 conflicts, now 1
                elif diff_counts[d] == 2:
                    delta_conflicts += 1  # was 1 conflict, now 2
                diff_counts[d] += 1
        
        new_conflicts = current_conflicts + delta_conflicts
        new_scope = max(current_scope, new_val) if new_val > current_scope else current_scope
        
        # If new_val is the max, scope might decrease
        if new_val < current_scope:
            # Check if any other row has a larger max
            new_scope = max(max(r) for r in current_rows)
        
        new_obj_val = new_conflicts * 1000000 + new_scope
        delta = new_obj_val - current_obj_val
        
        if delta < 0 or random.random() < pow(2.718, -delta / temp):
            # Accept the move
            current_rows[row_idx][pos] = new_val
            current_conflicts = new_conflicts
            current_scope = new_scope
            current_obj_val = new_obj_val
            
            if current_obj_val < best_obj_val:
                best_obj_val = current_obj_val
                best_rows_state = [r[:] for r in current_rows]
                no_improve_count = 0
            else:
                no_improve_count += 1
        else:
            # Reject: undo the diff_counts changes
            for d in new_diffs:
                if d in diff_counts:
                    if diff_counts[d] == 1:
                        del diff_counts[d]
                    else:
                        diff_counts[d] -= 1
                        if diff_counts[d] == 2:
                            delta_conflicts -= 1
                        elif diff_counts[d] == 1:
                            delta_conflicts -= 1
            for d in old_diffs:
                if d not in diff_counts:
                    diff_counts[d] = 1
                else:
                    if diff_counts[d] == 1:
                        delta_conflicts += 1
                    elif diff_counts[d] == 2:
                        delta_conflicts += 1
                    diff_counts[d] += 1
        
        temp *= cooling

    # Return the best valid solution found
    # If best solution has conflicts, we might need to return the best valid one from the greedy phase?
    # The task requires a valid solution. If SA ends with conflicts, we should check if we have a valid one from before.
    
    # Let's check if best_rows_state is valid
    if is_valid(best_rows_state):
        return best_rows_state
    
    # If not, check if the initial best_rows was valid
    if is_valid(best_rows):
        return best_rows
    
    # If neither is valid, try a final repair attempt using greedy with the best scope
    best_scope_found = min(get_scope(best_rows_state), get_scope(best_rows))
    for scope_limit in [best_scope_found, best_scope_found + 5, best_scope_found + 10, 150, 200]:
        if time.time() > deadline - 2:
            break
        rows = greedy_construct_v2(scope_limit)
        if rows is not None and is_valid(rows):
            return rows
        rows = greedy_construct(scope_limit, randomize=True, seed=random.randint(0, 1000))
        if rows is not None and is_valid(rows):
            return rows
    
    # Last resort: return the one with fewer conflicts, hoping it's valid
    if count_conflicts(best_rows_state) <= count_conflicts(best_rows):
        return best_rows_state
    else:
        return best_rows
