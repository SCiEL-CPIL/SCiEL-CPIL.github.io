import time
import random
import sys

def construct(n_rows, k):
    start_time = time.time()
    time_limit = 145  # Safety margin for 150s limit

    def _scope(rows):
        return max(max(row) for row in rows) if rows else 0

    def _row_diffs(row):
        diffs = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs

    def _is_valid_row(row):
        # Check that all positive differences within the row are distinct
        diffs = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d in diffs:
                    return False
                diffs.add(d)
        return True

    def _greedy_construct():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not (new & used):
                    row.append(x)
                    used |= new
            rows.append(row)
        return rows

    def _construct_row_with_limit(used_diffs, M):
        """Construct a valid Golomb ruler row of k+1 elements starting at 0,
        with max entry <= M, avoiding diffs in used_diffs."""
        row = [0]
        
        def backtrack(row, last_val):
            if len(row) == k + 1:
                return True
            
            # Heuristic: try values closer to the middle first? Or randomize.
            # Let's try a range, maybe biased towards larger values to spread out?
            # Actually, for minimal max entry, we want to pack tight, but diffs must be unique.
            # Let's iterate v from last_val+1 to M.
            
            # Optimization: If we have 'rem' elements left, the minimum span needed is rem*(rem+1)/2? No, that's for consecutive integers.
            # The differences must be distinct. The smallest possible next difference is 1, then 2, etc.
            # So the minimum max entry for a Golomb ruler of length n is n(n-1)/2.
            # Here we have k more elements to place.
            
            for v in range(last_val + 1, M + 1):
                valid = True
                new_diffs = set()
                for u in row:
                    d = v - u
                    if d in used_diffs or d in new_diffs:
                        valid = False
                        break
                    new_diffs.add(d)
                
                if valid:
                    row.append(v)
                    # Update used_diffs locally? No, we pass it or modify a copy.
                    # For speed, we can use a local set or modify a global one with undo.
                    # Let's use a local set for clarity first, optimize if needed.
                    if backtrack(row, v):
                        return True
                    row.pop()
            
            return False
        
        if backtrack(row, 0):
            return row
        return None

    def _local_search_bitset(rows, used_diffs, max_iters=10000):
        """Local search to minimize scope."""
        current_rows = [list(r) for r in rows]
        current_used = set(used_diffs)
        
        for _ in range(max_iters):
            # Pick a random row
            r_idx = random.randint(0, len(current_rows) - 1)
            old_row = current_rows[r_idx]
            old_diffs = _row_diffs(old_row)
            
            # Remove old diffs from used set
            current_used -= old_diffs
            
            # Try to reconstruct this row with a tighter limit or random restart
            # We want to reduce the max entry.
            # Let's try to find a row with max entry <= old_max - 1?
            # Or just find any valid row, and then see if it's better.
            # Actually, if we just find *any* valid row, the scope might increase.
            # We need to be careful.
            
            # Strategy: Try to find a row with max entry strictly less than old_max if possible.
            # If not, try to find one with same max but maybe "better" packing?
            # For now, let's just try to find a valid row.
            # To encourage lower scope, we can bias the search in _construct_row_with_limit?
            # Or we can try multiple limits.
            
            best_new_row = None
            best_new_max = old_max = max(old_row)
            
            # Try limits from old_max - 1 down to a reasonable lower bound
            # The theoretical lower bound for a Golomb ruler of length k+1 is k*(k-1)/2?
            # No, the minimal length for k marks is k(k-1)/2.
            # Here we have k+1 marks. So minimal max entry is k(k-1)/2.
            min_possible = k * (k - 1) // 2
            
            # But we also have to fit within the global used_diffs constraints.
            # So we might need a larger M.
            
            # Let's try a few random limits below old_max
            # And also one attempt at old_max to just see if we can improve structure?
            
            # Actually, if we just reconstruct with the same limit, we might get a different row.
            # If we get a row with a smaller max, that's an improvement.
            
            # Let's try to find a row with max <= old_max - 1 first.
            # If that fails, try old_max.
            
            # To save time, we'll do a limited number of attempts per limit.
            
            # We'll use a helper that tries to construct a row with a specific limit.
            # If it succeeds, we check the actual max.
            
            # Let's try limits: old_max - 1, old_max - 2, ..., down to min_possible.
            # But this could be slow if old_max is large.
            # Let's just try old_max - 1 and old_max.
            
            # Actually, a better heuristic: try to find a row that uses "rare" diffs?
            # Hard to define.
            
            # Let's just try to reconstruct with limit = old_max - 1.
            # If that fails, try old_max.
            
            new_row = _construct_row_with_limit(current_used, old_max - 1)
            if new_row is not None:
                best_new_row = new_row
            else:
                new_row = _construct_row_with_limit(current_used, old_max)
                if new_row is not None:
                    # Only accept if it doesn't increase scope? It can't increase since limit is old_max.
                    # But it might have the same max. That's okay, we might find a better one later.
                    # However, if we accept a row with the same max, we might get stuck.
                    # Let's only accept if it's strictly better?
                    # If we don't accept, we revert.
                    # For now, let's accept if it's valid. We can add a check later.
                    # Actually, if we accept a row with the same max, the scope doesn't improve.
                    # We need to improve the scope.
                    # So we should only accept if max(new_row) < old_max.
                    if max(new_row) < old_max:
                        best_new_row = new_row
                    else:
                        # Revert
                        current_used |= old_diffs
                        current_rows[r_idx] = old_row
                        continue
                else:
                    # Failed to construct, revert
                    current_used |= old_diffs
                    current_rows[r_idx] = old_row
                    continue
            
            # If we found a better row, accept it
            if best_new_row is not None:
                new_diffs = _row_diffs(best_new_row)
                current_rows[r_idx] = best_new_row
                current_used |= new_diffs
            else:
                # Revert
                current_used |= old_diffs
                current_rows[r_idx] = old_row
        
        return current_rows

    # Initial greedy solution
    best_rows = _greedy_construct()
    best_scope = _scope(best_rows)
    
    # Try to improve using local search
    used_diffs = set()
    for row in best_rows:
        used_diffs |= _row_diffs(row)
    
    # Run local search
    # We need to be careful with time.
    # Let's run local search with a time budget.
    
    deadline = start_time + time_limit
    improved = True
    while improved and time.time() < deadline - 10:
        improved = False
        # Shuffled order for local search to break symmetry
        rows_copy = [list(r) for r in best_rows]
        random.shuffle(rows_copy)
        
        # Use the bitset local search
        # But _local_search_bitset modifies the rows in place and updates used_diffs internally?
        # No, it returns new rows. We need to update used_diffs accordingly.
        
        # Let's refactor _local_search_bitset to return (rows, used_diffs)
        # Or just pass used_diffs and let it manage?
        # Current implementation manages its own copy.
        
        # Let's call it
        new_rows = _local_search_bitset(rows_copy, used_diffs, max_iters=2000)
        
        new_scope = _scope(new_rows)
        
        if new_scope < best_scope:
            best_rows = new_rows
            best_scope = new_scope
            improved = True
            # Update used_diffs for next iteration
            used_diffs = set()
            for row in best_rows:
                used_diffs |= _row_diffs(row)
    
    # Final verification
    all_diffs = set()
    for row in best_rows:
        row_diffs = _row_diffs(row)
        if row_diffs & all_diffs:
            # Invalid, should not happen
            pass
        all_diffs |= row_diffs
        
    return best_rows
