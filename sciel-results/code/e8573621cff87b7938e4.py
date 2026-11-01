import time
import random
import sys

sys.setrecursionlimit(10000)

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit
    
    # Total distinct differences needed
    diffs_per_row = k * (k + 1) // 2
    total_diffs_needed = n_rows * diffs_per_row
    
    # Lower bound on max entry M
    # We need total_diffs_needed distinct differences from {1, ..., M}
    lower_bound_M = total_diffs_needed
    
    # Best known for (4,7) is 134. Lower bound is 112.
    # We search M from lower_bound_M upwards.
    max_M_to_try = lower_bound_M + 60

    def get_diffs(row):
        """Compute all positive differences in a row."""
        diffs = []
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.append(row[j] - row[i])
        return diffs

    def conflict_count(rows):
        """Count total number of duplicate differences across all rows."""
        seen = set()
        dupes = 0
        for row in rows:
            for d in get_diffs(row):
                if d in seen:
                    dupes += 1
                else:
                    seen.add(d)
        return dupes

    def find_conflicts(rows):
        """
        Returns a dictionary mapping difference -> list of (row_idx, i, j) pairs 
        that produce this difference. Only includes diffs that appear >1 time.
        """
        diff_map = {}
        for r_idx, row in enumerate(rows):
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d not in diff_map:
                        diff_map[d] = []
                    diff_map[d].append((r_idx, i, j))
        
        conflicts = {}
        for d, locs in diff_map.items():
            if len(locs) > 1:
                conflicts[d] = locs
        return conflicts

    def greedy_largest_first_row(M, used_diffs, k):
        """
        Construct a single row of k+1 elements starting at 0, ending <= M.
        Strategy: Place elements from largest to smallest (reverse greedy) to maximize gaps.
        a_0 = 0 is fixed. We choose a_k, a_{k-1}, ..., a_1.
        Actually, standard greedy is easier to reason about:
        Let's try a forward greedy but with a twist:
        We want to maximize the "spread".
        
        Alternative Greedy: 
        1. Start with [0].
        2. For next element, pick the LARGEST possible value v in [current_max+1, M] 
           such that all new diffs (v - existing) are not in used_diffs.
        3. If no such v exists, backtrack? No, this is a heuristic.
        4. If we get stuck, try a different strategy for this row.
        
        Let's implement a randomized greedy that tries to place large jumps first.
        """
        row = [0]
        current_max = 0
        
        # We need to pick k more elements.
        # To maximize spread, we should try to pick large values.
        # But we must ensure we can fit the remaining elements.
        
        for step in range(1, k + 1):
            # We need to place element at index 'step'.
            # It must be > current_max.
            # It must be < M (if not last) or <= M.
            # Also, we need to leave enough room for (k - step) more elements.
            # So max possible value for this step is M - (k - step).
            # Min possible value is current_max + 1.
            
            max_allowed = M - (k - step)
            min_allowed = current_max + 1
            
            # Try values from max_allowed down to min_allowed
            # This prioritizes large gaps (largest-first placement logic)
            found = False
            for v in range(max_allowed, min_allowed - 1, -1):
                # Check if v conflicts with existing row elements
                valid = True
                new_diffs = []
                for a in row:
                    d = v - a
                    if d in used_diffs:
                        valid = False
                        break
                    new_diffs.append(d)
                
                if valid:
                    row.append(v)
                    used_diffs.update(new_diffs)
                    current_max = v
                    found = True
                    break
            
            if not found:
                # Backtrack or fail. For heuristic, we just fail this row construction.
                return None
            
        return row

    def repair_row(row, M, used_diffs_except_this_row, k):
        """
        Try to repair a specific row to eliminate its conflicts with the global set.
        used_diffs_except_this_row is the set of diffs from ALL OTHER rows.
        We want to find a valid row that doesn't conflict with these.
        """
        # Simple approach: Re-generate this row using greedy, avoiding used_diffs_except_this_row.
        # But we also need to avoid internal conflicts (which greedy handles by construction).
        # So just call greedy_largest_first_row with the external used_diffs.
        return greedy_largest_first_row(M, set(used_diffs_except_this_row), k)

    def solve_cdr(M, time_limit):
        """
        Conflict-Directed Repair Solver.
        1. Initialize rows greedily.
        2. While conflicts exist and time remains:
           a. Identify the row involved in the most conflicts.
           b. Remove its diffs from global set.
           c. Re-generate that row greedily, avoiding the remaining global diffs.
           d. If regeneration fails, try a randomized variant or smaller M? No, M is fixed.
           e. Add new diffs to global set.
        """
        rows = []
        used_diffs = set()
        
        # Phase 1: Initial Greedy Construction
        for _ in range(n_rows):
            row = greedy_largest_first_row(M, used_diffs, k)
            if row is None:
                return None
            rows.append(row)
            used_diffs.update(get_diffs(row))
        
        # Phase 2: Repair Loop
        current_conflicts = conflict_count(rows)
        if current_conflicts == 0:
            return rows
        
        best_rows = [r[:] for r in rows]
        best_conflicts = current_conflicts
        
        while time.time() < time_limit:
            conflicts = find_conflicts(rows)
            if not conflicts:
                return rows
            
            # Find the row involved in the most conflicts
            row_conflict_count = {}
            for d, locs in conflicts.items():
                # Each conflict involves multiple rows.
                # We credit the conflict to each row involved.
                for (r_idx, i, j) in locs:
                    row_conflict_count[r_idx] = row_conflict_count.get(r_idx, 0) + 1
            
            # Pick the row with the highest conflict count
            if not row_conflict_count:
                break
                
            target_row_idx = max(row_conflict_count, key=row_conflict_count.get)
            
            # Remove this row's diffs from the global set
            row_to_remove = rows[target_row_idx]
            diffs_to_remove = set(get_diffs(row_to_remove))
            for d in diffs_to_remove:
                used_diffs.discard(d)
            
            # Try to re-generate this row
            # To add diversity, we can randomize the greedy choice slightly
            # or just use the deterministic greedy which avoids the current global set.
            new_row = greedy_largest_first_row(M, used_diffs, k)
            
            if new_row is None:
                # Greedy failed. Try a randomized repair.
                # Pick a random valid increasing sequence that doesn't conflict with used_diffs.
                # This is harder. Let's try a few random restarts for this row.
                success = False
                for _ in range(10):
                    # Random row generation
                    vals = random.sample(range(1, M), k)
                    vals.sort()
                    candidate = [0] + vals
                    # Check validity against used_diffs
                    valid = True
                    cand_diffs = get_diffs(candidate)
                    for d in cand_diffs:
                        if d in used_diffs:
                            valid = False
                            break
                    if valid:
                        new_row = candidate
                        success = True
                        break
                
                if not success:
                    # Cannot repair this row. Revert and try a different strategy?
                    # For now, revert and break or try next M?
                    # Let's just revert and break out of this M attempt.
                    used_diffs.update(diffs_to_remove)
                    rows[target_row_idx] = row_to_remove
                    break
            
            # Update global state
            rows[target_row_idx] = new_row
            used_diffs.update(get_diffs(new_row))
            
            # Check improvement
            new_conflicts = conflict_count(rows)
            if new_conflicts < best_conflicts:
                best_conflicts = new_conflicts
                best_rows = [r[:] for r in rows]
                if new_conflicts == 0:
                    return best_rows
            
            # If we haven't improved in a while, we might be stuck in a loop.
            # Simple stagnation check: if best_conflicts hasn't changed in last 50 iterations?
            # We don't track iteration count here, but time limit will cut it off.
            # To avoid infinite loops on the same state, we can randomize the target row selection
            # if the top candidate keeps failing or not improving.
            # For simplicity, we rely on the deterministic greedy changing the state.
            # If the greedy produces the same row, we are stuck.
            # Let's add a check: if new_row == row_to_remove, we are stuck.
            if new_row == row_to_remove:
                # Stuck. Try randomizing the target selection or breaking.
                # Pick a different row with conflicts
                other_rows = [r for r in row_conflict_count if r != target_row_idx]
                if other_rows:
                    target_row_idx = random.choice(other_rows)
                    # Re-run the removal/regeneration for this new target
                    # This is a bit complex to inline. Let's just break to save time.
                    break
                else:
                    break

        return best_rows if best_conflicts == 0 else None

    # Main Loop: Try M from lower bound up
    for M in range(lower_bound_M, max_M_to_try + 1):
        if time.time() > deadline - 5:
            break
            
        # Give CDR a time budget
        # CDR is deterministic-ish but fast. Give it more time than SA per M.
        time_budget = 10.0
        sol = solve_cdr(M, start_time + time_budget)
        
        if sol is not None:
            # Verify
            all_diffs = set()
            valid = True
            for row in sol:
                for i in range(len(row)):
                    for j in range(i + 1, len(row)):
                        d = row[j] - row[i]
                        if d in all_diffs:
                            valid = False
                            break
                        all_diffs.add(d)
                if not valid:
                    break
            
            if valid:
                return sol
    
    # Fallback: Greedy valid solution (large scope)
    used_diffs = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        current_max = 0
        while len(row) < k + 1:
            x = current_max + 1
            while True:
                valid = True
                for a in row:
                    d = x - a
                    if d in used_diffs:
                        valid = False
                        break
                if valid:
                    row.append(x)
                    used_diffs.update(x - a for a in row[:-1])
                    current_max = x
                    break
                x += 1
        rows.append(row)
        
    return rows

if __name__ == "__main__":
    # Test with the specific task parameters
    res = construct(4, 7)
    print(res)
    max_val = max(max(r) for r in res)
    print(f"Max entry: {max_val}")
    
    # Verify
    diffs = set()
    valid = True
    for r in res:
        for i in range(len(r)):
            for j in range(i+1, len(r)):
                d = r[j] - r[i]
                if d in diffs:
                    valid = False
                diffs.add(d)
    print(f"Valid: {valid}")
