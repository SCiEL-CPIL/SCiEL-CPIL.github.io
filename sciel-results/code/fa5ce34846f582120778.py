import time
import random
import sys

_START = None

def construct(n_rows, k):
    global _START
    if _START is None:
        _START = time.time()
    deadline = _START + 138.0

    d = k * (k + 1) // 2

    def get_scope(rows):
        return max(max(r) for r in rows)

    def greedy_init():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
            rows.append(row)
        return rows

    def compute_diffs(rows):
        global_diffs = set()
        row_diffs = []
        for row in rows:
            r_diffs = set()
            n = len(row)
            for i in range(n):
                ri = row[i]
                for j in range(i + 1, n):
                    diff = row[j] - ri
                    r_diffs.add(diff)
            row_diffs.append(r_diffs)
            global_diffs.update(r_diffs)
        return global_diffs, row_diffs

    rows = greedy_init()
    best_rows = [r[:] for r in rows]
    best_scope = get_scope(rows)
    best_global_diffs, best_row_diffs = compute_diffs(best_rows)

    rng = random.Random(42)

    def check_and_apply_move(cur, global_diffs, row_diffs, ri, pos, new_val):
        """
        Checks if moving cur[ri][pos] to new_val is valid.
        If valid, applies the move and returns (True, new_scope).
        If invalid, returns (False, None).
        """
        old_val = cur[ri][pos]
        if new_val == old_val:
            return False, None
        
        row = cur[ri]
        if pos > 0 and new_val <= row[pos-1]:
            return False, None
        if pos < len(row) - 1 and new_val >= row[pos+1]:
            return False, None
            
        diffs_to_remove = set()
        diffs_to_add = set()
        
        for i in range(pos):
            diffs_to_remove.add(old_val - row[i])
            diffs_to_add.add(new_val - row[i])
            
        for j in range(pos + 1, len(row)):
            diffs_to_remove.add(row[j] - old_val)
            diffs_to_add.add(row[j] - new_val)
            
        # Check conflicts
        for d in diffs_to_add:
            if d in global_diffs and d not in diffs_to_remove:
                return False, None
        
        # Apply
        for d in diffs_to_remove:
            global_diffs.discard(d)
        for d in diffs_to_add:
            global_diffs.add(d)
            
        row_diffs[ri].difference_update(diffs_to_remove)
        row_diffs[ri].update(diffs_to_add)
        
        cur[ri][pos] = new_val
        
        new_scope = max(max(r) for r in cur)
        return True, new_scope

    def revert_move(cur, global_diffs, row_diffs, ri, pos, old_val, new_val):
        """
        Reverts a move from old_val to new_val.
        """
        row = cur[ri]
        diffs_to_remove = set() # Diffs involving new_val
        diffs_to_add = set()    # Diffs involving old_val
        
        for i in range(pos):
            diffs_to_remove.add(new_val - row[i])
            diffs_to_add.add(old_val - row[i])
            
        for j in range(pos + 1, len(row)):
            diffs_to_remove.add(row[j] - new_val)
            diffs_to_add.add(row[j] - old_val)
            
        for d in diffs_to_remove:
            global_diffs.discard(d)
        for d in diffs_to_add:
            global_diffs.add(d)
            
        row_diffs[ri].difference_update(diffs_to_remove)
        row_diffs[ri].update(diffs_to_add)
        
        cur[ri][pos] = old_val

    max_restarts = 1000
    steps_per_restart = 200000

    for restart in range(max_restarts):
        if time.time() > deadline:
            break

        if restart == 0:
            cur = [r[:] for r in best_rows]
            global_diffs, row_diffs = compute_diffs(cur)
            cur_scope = best_scope
        else:
            cur = [r[:] for r in best_rows]
            global_diffs, row_diffs = compute_diffs(cur)
            cur_scope = best_scope
            
            for _ in range(rng.randint(1, 3)):
                ri = rng.randrange(n_rows)
                pos = rng.randint(1, len(cur[ri]) - 1)
                lo = cur[ri][pos - 1] + 1
                hi = cur[ri][pos + 1] - 1 if pos < len(cur[ri]) - 1 else best_scope + 15
                if hi > lo:
                    new_val = rng.randint(lo, hi)
                    success, new_scope = check_and_apply_move(cur, global_diffs, row_diffs, ri, pos, new_val)
                    if success:
                        cur_scope = new_scope

        if cur_scope < best_scope:
            best_scope = cur_scope
            best_rows = [r[:] for r in cur]
            best_global_diffs, best_row_diffs = compute_diffs(best_rows)

        T = max(0.05, 1.0 - restart / max_restarts)
        for step in range(steps_per_restart):
            if time.time() > deadline:
                break

            ri = rng.randrange(n_rows)
            pos = rng.randint(1, len(cur[ri]) - 1)
            old_val = cur[ri][pos]
            lo = cur[ri][pos - 1] + 1
            hi = cur[ri][pos + 1] - 1 if pos < len(cur[ri]) - 1 else best_scope + 25
            if hi <= lo:
                continue
            new_val = rng.randint(lo, hi)
            if new_val == old_val:
                continue

            success, new_scope = check_and_apply_move(cur, global_diffs, row_diffs, ri, pos, new_val)
            
            if success:
                if new_scope < best_scope:
                    best_scope = new_scope
                    best_rows = [r[:] for r in cur]
                    best_global_diffs, best_row_diffs = compute_diffs(best_rows)
                
                delta = new_scope - cur_scope
                if delta <= 0 or rng.random() < (2.718281828 ** (-delta / max(T, 0.01))):
                    cur_scope = new_scope
                else:
                    revert_move(cur, global_diffs, row_diffs, ri, pos, old_val, new_val)
            else:
                pass

            T *= 0.99998

    return best_rows
