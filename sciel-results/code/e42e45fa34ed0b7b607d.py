import time
import random
import sys

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 110.0
    
    num_diffs = n_rows * k * (k + 1) // 2
    lb = num_diffs
    
    best_rows = None
    best_scope = float('inf')
    
    # Fast Greedy
    def greedy():
        used = 0
        rows = []
        for _ in range(n_rows):
            row = [0]
            for _ in range(k):
                last = row[-1]
                found = False
                # Try to find a valid x
                # Heuristic: try small x first
                for x in range(last + 1, lb + 50):
                    mask = 0
                    ok = True
                    for a in row:
                        d = x - a
                        bit = 1 << d
                        if used & bit:
                            ok = False
                            break
                        mask |= bit
                    if ok:
                        row.append(x)
                        used |= mask
                        found = True
                        break
                if not found:
                    x = last + 1
                    while True:
                        mask = 0
                        ok = True
                        for a in row:
                            d = x - a
                            bit = 1 << d
                            if used & bit:
                                ok = False
                                break
                            mask |= bit
                        if ok:
                            row.append(x)
                            used |= mask
                            break
                        x += 1
            rows.append(row)
        return rows

    rows = greedy()
    if rows:
        best_scope = max(max(r) for r in rows)
        best_rows = [list(r) for r in rows]

    # Backtracking
    def backtrack(target_scope):
        rows = []
        used_mask = 0
        k1 = k + 1
        
        def build_row(pos, last_val, row_so_far, min_max_constraint):
            nonlocal used_mask
            if time.time() > deadline - 10:
                return False
            
            if pos == k1:
                if row_so_far[-1] < min_max_constraint:
                    return False
                rows.append(list(row_so_far))
                return True
            
            # Pruning
            max_x = target_scope - (k1 - pos - 1)
            if max_x <= last_val:
                return False
            
            start_x = last_val + 1
            if pos == k1 - 1:
                start_x = max(start_x, min_max_constraint)
                
            for x in range(start_x, max_x + 1):
                mask = 0
                ok = True
                for a in row_so_far:
                    d = x - a
                    bit = 1 << d
                    if used_mask & bit:
                        ok = False
                        break
                    mask |= bit
                if ok:
                    used_mask |= mask
                    row_so_far.append(x)
                    if build_row(pos + 1, x, row_so_far, min_max_constraint):
                        return True
                    row_so_far.pop()
                    used_mask ^= mask
            return False

        def solve_rows(row_idx, min_max_val):
            nonlocal used_mask
            if time.time() > deadline - 10:
                return False
            if row_idx == n_rows:
                return True
            
            # Try to build row
            # We need to save the state of 'rows' to rollback if this row fails?
            # No, build_row appends to 'rows' only on success.
            # But if build_row fails, it doesn't append.
            # However, used_mask is modified during build_row.
            # If build_row returns False, used_mask is restored to state before call?
            # Yes, because we do used_mask ^= mask on backtrack.
            # But if build_row fails at a deep level, it backtracks up.
            # The top-level call to build_row will have restored used_mask to the state
            # at the beginning of the call (which is the state after previous rows were added).
            # Wait, if build_row returns False, it means it tried all x and failed.
            # In each failure branch, it restored used_mask.
            # So when build_row returns False, used_mask is back to the state
            # before the first x was tried in this call.
            # This is correct.
            
            if build_row(1, 0, [0], min_max_val):
                if solve_rows(row_idx + 1, rows[-1][-1]):
                    return True
                # If solve_rows fails, we need to remove the last row
                # and restore used_mask to the state before this row was built.
                # But build_row already restored used_mask to the state
                # before it started building this row?
                # No. build_row returns True if it succeeds.
                # If it succeeds, it appends to rows and used_mask contains the diffs.
                # If solve_rows fails later, we need to undo this row.
                
                # So we need to capture used_mask and rows length before build_row?
                # Or, we can just not append to 'rows' in build_row,
                # but return the row list, and let solve_rows handle it.
                
                # Let's refactor: build_row returns the row list or None.
                # This is cleaner.
                
                pass # Logic needs adjustment
                
            return False
            
        # Refactored Backtracking
        def build_row_ret(pos, last_val, row_so_far, min_max_constraint):
            nonlocal used_mask
            if time.time() > deadline - 10:
                return None
            
            if pos == k1:
                if row_so_far[-1] < min_max_constraint:
                    return None
                return list(row_so_far)
            
            max_x = target_scope - (k1 - pos - 1)
            if max_x <= last_val:
                return None
            
            start_x = last_val + 1
            if pos == k1 - 1:
                start_x = max(start_x, min_max_constraint)
                
            for x in range(start_x, max_x + 1):
                mask = 0
                ok = True
                for a in row_so_far:
                    d = x - a
                    bit = 1 << d
                    if used_mask & bit:
                        ok = False
                        break
                    mask |= bit
                if ok:
                    used_mask |= mask
                    row_so_far.append(x)
                    res = build_row_ret(pos + 1, x, row_so_far, min_max_constraint)
                    if res is not None:
                        return res
                    row_so_far.pop()
                    used_mask ^= mask
            return None

        def solve_rows_ret(row_idx, min_max_val):
            nonlocal used_mask
            if time.time() > deadline - 10:
                return None
            if row_idx == n_rows:
                return [] # Success, no more rows
            
            row = build_row_ret(1, 0, [0], min_max_val)
            if row is None:
                return None
            
            # Row is valid and added to used_mask
            # Recurse
            res = solve_rows_ret(row_idx + 1, row[-1])
            if res is not None:
                return [row] + res
            
            # Backtrack: remove row's diffs from used_mask
            for i in range(len(row)):
                for j in range(i+1, len(row)):
                    used_mask ^= (1 << (row[j]-row[i]))
            return None

        res = solve_rows_ret(0, 0)
        if res is not None:
            return res
        return None

    if time.time() < deadline - 20:
        sol = backtrack(lb)
        if sol:
            best_scope = lb
            best_rows = sol

    return best_rows

if __name__ == "__main__":
    result = construct(4, 7)
    if result:
        print(f"Scope: {max(max(r) for r in result)}")
    else:
        print("No solution")
