import time
import random
import sys

_START = None

def construct(n_rows, k):
    global _START
    if _START is None:
        _START = time.time()
    deadline = _START + 140.0  # Safety margin

    # d is number of differences per row
    d = k * (k + 1) // 2
    
    def greedy_construct(rng=None):
        """
        Constructs a valid set of rows using a greedy approach.
        Returns (rows, global_diffs, row_diffs)
        """
        rows = []
        global_diffs = set()
        row_diffs = []
        
        for r_idx in range(n_rows):
            row = [0]
            r_diffs = set()
            last = 0
            # We need to pick k more elements
            for _ in range(k):
                # Try next integer
                candidate = last + 1
                while True:
                    # Check differences with all existing elements in this row
                    valid = True
                    new_diffs = []
                    for a in row:
                        diff = candidate - a
                        if diff in global_diffs or diff in r_diffs:
                            valid = False
                            break
                        new_diffs.append(diff)
                    
                    if valid:
                        row.append(candidate)
                        r_diffs.update(new_diffs)
                        global_diffs.update(new_diffs)
                        last = candidate
                        break
                    else:
                        candidate += 1
                        # Optional: If candidate gets too large, we might want to backtrack, 
                        # but for greedy init, we just push forward.
                        if candidate > 1000: # Safety break
                             return None, None, None
            rows.append(row)
            row_diffs.append(r_diffs)
            
        return rows, global_diffs, row_diffs

    # Initial construction
    rows, global_diffs, row_diffs = greedy_construct()
    if rows is None:
        # Fallback to a simple wide spread if greedy fails (unlikely for small k)
        rows = [[i * (j + 1) for j in range(k + 1)] for i in range(n_rows)]
        # Rebuild diffs
        global_diffs = set()
        row_diffs = []
        for r in rows:
            rd = set()
            for i in range(len(r)):
                for j in range(i+1, len(r)):
                    rd.add(r[j] - r[i])
            row_diffs.append(rd)
            global_diffs.update(rd)

    best_rows = [r[:] for r in rows]
    best_scope = max(max(r) for r in best_rows)
    best_global_diffs = set(global_diffs)
    best_row_diffs = [set(d) for d in row_diffs]

    rng = random.Random(42)

    # Local Search
    # We try to reduce the scope by decreasing elements.
    # Move: Pick a row r, an index i (1..k), and try to decrease r[i].
    
    def try_decrease(r_idx, i, rows, global_diffs, row_diffs):
        """
        Tries to decrease rows[r_idx][i] to the smallest possible valid value > rows[r_idx][i-1].
        Returns True if a change was made.
        """
        old_val = rows[r_idx][i]
        min_val = rows[r_idx][i-1] + 1
        
        # We want to find the smallest new_val >= min_val that is valid.
        # We don't necessarily have to go to min_val, just any valid smaller value.
        # To be efficient, we can check if min_val is valid. If not, we might stop 
        # or check a few more. But usually, if min_val is invalid, larger ones might be valid.
        # However, for scope reduction, we only care if we can go LOWER than old_val.
        
        # Optimization: If old_val is already min_val, we can't decrease.
        if old_val == min_val:
            return False

        # Remove old differences involving index i
        row = rows[r_idx]
        old_diffs = set()
        for j in range(k + 1):
            if j == i: continue
            diff = abs(row[i] - row[j])
            old_diffs.add(diff)
        
        # Temporarily remove from global and row sets
        for diff in old_diffs:
            global_diffs.discard(diff)
            row_diffs[r_idx].discard(diff)

        # Try new values from min_val up to old_val - 1
        # We want the smallest valid one to minimize scope? 
        # Actually, any valid one that is < old_val reduces the scope if i is the max index,
        # or reduces the "tightness".
        # Let's try to find the smallest valid new_val.
        
        found_new_val = None
        for new_val in range(min_val, old_val):
            valid = True
            new_diffs = set()
            for j in range(k + 1):
                if j == i: continue
                diff = abs(new_val - row[j])
                if diff in global_diffs or diff in row_diffs[r_idx]:
                    valid = False
                    break
                new_diffs.add(diff)
            
            if valid:
                found_new_val = new_val
                # Store new_diffs for restoration
                break
        
        if found_new_val is not None:
            # Commit the change
            rows[r_idx][i] = found_new_val
            for diff in new_diffs:
                global_diffs.add(diff)
                row_diffs[r_idx].add(diff)
            return True
        else:
            # Revert: re-add old diffs
            for diff in old_diffs:
                global_diffs.add(diff)
                row_diffs[r_idx].add(diff)
            return False

    # Main Loop
    # We prioritize rows with larger max elements.
    # We prioritize indices with larger values (especially the last one).
    
    while time.time() < deadline:
        # Find the row with the largest max element
        max_scope = -1
        target_row = -1
        for i in range(n_rows):
            if max(rows[i]) > max_scope:
                max_scope = max(rows[i])
                target_row = i
        
        if target_row == -1:
            break
            
        # Try to decrease elements in target_row, from the end to the beginning
        # Decreasing the last element is most impactful for scope.
        changed = False
        for i in range(k, 0, -1):
            if try_decrease(target_row, i, rows, global_diffs, row_diffs):
                changed = True
                # If we decreased the last element, scope might drop immediately
                # We can break and re-evaluate max_scope
                if i == k:
                    break
        
        if not changed:
            # If we couldn't decrease in the max row, try others?
            # Or try a more aggressive move?
            # For now, if we can't decrease the max row, we might be stuck in a local minimum.
            # We can try to "shake" by increasing a small element to allow a larger decrease later?
            # This is complex. Let's try decreasing elements in other rows to see if it frees up differences.
            # Actually, decreasing elements in other rows doesn't directly help the max row's scope,
            # unless it frees up a difference that was blocking a decrease in the max row.
            
            # Let's try a random row and random index decrease.
            r_idx = rng.randint(0, n_rows - 1)
            i_idx = rng.randint(1, k)
            if try_decrease(r_idx, i_idx, rows, global_diffs, row_diffs):
                changed = True
                # Update best if scope improved
                curr_scope = max(max(r) for r in rows)
                if curr_scope < best_scope:
                    best_scope = curr_scope
                    best_rows = [r[:] for r in rows]
                    best_global_diffs = set(global_diffs)
                    best_row_diffs = [set(d) for d in row_diffs]
            else:
                # No move found anywhere? 
                # We might need to accept a worse solution temporarily? No, we only accept improvements.
                # If we can't make any decreasing move, we are at a local minimum.
                # We should break or try a different strategy.
                # Since we start with a valid greedy solution, and only move if valid, 
                # if no decreasing move is valid, we can't improve the scope with single-element decreases.
                
                # To escape, we could try a "swap" or "shift" move.
                # But for now, let's just break if we've made no progress in a while.
                # Actually, just breaking is safe.
                break

        # Update best
        curr_scope = max(max(r) for r in rows)
        if curr_scope < best_scope:
            best_scope = curr_scope
            best_rows = [r[:] for r in rows]
            best_global_diffs = set(global_diffs)
            best_row_diffs = [set(d) for d in row_diffs]

    return best_rows
