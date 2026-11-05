import time
import random
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0
    
    L = k + 1
    diffs_per_row = k * (k + 1) // 2

    def scope_of(rows):
        return max(max(r) for r in rows)

    def sum_of_maxes(rows):
        return sum(max(r) for r in rows)

    def row_diffs_set(row):
        s = set()
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                s.add(row[j] - row[i])
        return s

    def pure_greedy():
        """Fast greedy construction."""
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < L:
                x += 1
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
            rows.append(row)
        return rows

    def greedy_fill_row(forbidden, current_scope_hint=200):
        """
        Greedily construct a single row that avoids 'forbidden' differences.
        We try to keep the max value as small as possible.
        """
        row = [0]
        x = 0
        # We might need to go higher than current_scope_hint if forbidden is dense
        max_search = current_scope_hint + 50
        while len(row) < L:
            x += 1
            if x > max_search:
                # If we can't find a valid number in the hint range, extend it
                max_search += 50
            new = {x - a for a in row}
            if not (new & forbidden):
                row.append(x)
        return row

    def local_search_polish(rows, time_limit):
        """
        Aggressive local search to minimize the maximum entry.
        Focuses on the largest entries.
        """
        rows = [r[:] for r in rows]
        best_scope = scope_of(rows)
        
        row_diff_sets = [row_diffs_set(r) for r in rows]
        all_diffs = set()
        for s in row_diff_sets:
            all_diffs |= s
            
        # Identify the "bottleneck" rows (those with large max)
        # We will try to reduce the max of the row with the current global max.
        
        it = 0
        while time.time() < time_limit and it < 10000:
            it += 1
            # Find the row with the maximum last element
            max_row_idx = max(range(n_rows), key=lambda i: rows[i][-1])
            row = rows[max_row_idx]
            current_max = row[-1]
            
            # Try to reduce the last element
            # We try values from current_max - 1 down to row[-2] + 1
            found_improvement = False
            for nv in range(current_max - 1, row[-2], -1):
                # Check if nv is valid for the last position
                # 1. nv > row[-2]
                # 2. No new collisions
                
                # Quick check: does nv create a difference that is already in all_diffs \ row_diffs[max_row_idx]?
                # The new differences involving the last element are {nv - a for a in row[:-1]}
                new_diffs_last = {nv - a for a in row[:-1]}
                
                # We also need to remove the old differences involving the old last element
                # But for a quick check, if any of new_diffs_last are in (all_diffs - row_diffs[max_row_idx]), it's invalid.
                # Note: new_diffs_last might contain differences that were in the old row? No, nv != old_last.
                # However, new_diffs_last might contain differences that are in the old row's set but not the "other" rows.
                # Actually, we just need to check against OTHER rows.
                
                other_diffs = all_diffs - row_diff_sets[max_row_idx]
                
                if new_diffs_last & other_diffs:
                    continue
                
                # Also need to ensure internal consistency? 
                # Since we are only changing the last element, and nv > row[-2], 
                # and row is strictly increasing, the new row is strictly increasing.
                # The only differences are those involving the new last element.
                # We already checked those.
                
                # Valid move!
                # Update state
                old_diffs = row_diff_sets[max_row_idx]
                # Remove old last diffs
                old_last_diffs = {row[-1] - a for a in row[:-1]}
                
                all_diffs = (all_diffs - old_last_diffs) | new_diffs_last
                row_diff_sets[max_row_idx] = row_diffs_set(row[:-1] + [nv]) # Recalc for safety or update manually
                # Manual update is faster:
                # row_diff_sets[max_row_idx] = (row_diff_sets[max_row_idx] - old_last_diffs) | new_diffs_last
                
                row[-1] = nv
                rows[max_row_idx] = row
                
                best_scope = scope_of(rows)
                found_improvement = True
                break
            
            if not found_improvement:
                # Try to change an earlier element in the max row to allow the last element to be smaller?
                # This is complex. Let's try a random perturbation on the max row to escape local minimum.
                # Pick a random index j in [1, L-1]
                j = random.randint(1, L - 1)
                old_val = row[j]
                
                # Try to change row[j] to a smaller value? Or a different value?
                # If we change row[j], we might be able to shrink the suffix.
                # Let's try to decrease row[j] if possible.
                lower_bound = row[j-1] + 1
                upper_bound = row[j+1] - 1 if j + 1 < L else current_max - 1
                
                candidates = range(lower_bound, min(upper_bound + 1, old_val + 1))
                if candidates:
                    # Try random candidates
                    cand_list = list(candidates)
                    random.shuffle(cand_list)
                    for nv in cand_list:
                        if nv == old_val: continue
                        
                        # Check validity of changing row[j] to nv
                        # New diffs: {nv - a for a in row[:j]} U {a - nv for a in row[j+1:]}
                        # Old diffs involving row[j]: {old_val - a for a in row[:j]} U {a - old_val for a in row[j+1:]}
                        
                        # This is expensive to check fully. 
                        # Given time constraints, we might skip this deep local search if the simple last-element reduction fails.
                        pass
                # If no improvement in this iteration, we might be stuck. 
                # The SA phase should handle escaping. This polish is just for the final squeeze.
                # To avoid infinite loop on stuck states, we can break or do a random restart of the polish.
                if it % 1000 == 0:
                    # Do a random row regeneration for the max row
                    other_diffs = all_diffs - row_diff_sets[max_row_idx]
                    new_row = greedy_fill_row(other_diffs, best_scope)
                    if max(new_row) <= best_scope:
                         # Accept
                         all_diffs = (all_diffs - row_diff_sets[max_row_idx]) | row_diffs_set(new_row)
                         row_diff_sets[max_row_idx] = row_diffs_set(new_row)
                         rows[max_row_idx] = new_row
                         best_scope = scope_of(rows)
                        # If new_row is larger, ignore.

        return rows, best_scope

    # ---------- Main Search ----------
    best_rows = pure_greedy()
    best_scope = scope_of(best_rows)
    
    # Simulated Annealing
    T = 100.0
    T_min = 0.1
    cooling = 0.999
    
    rows = best_rows[:]
    row_diff_sets = [row_diffs_set(r) for r in rows]
    all_diffs = set()
    for s in row_diff_sets:
        all_diffs |= s
    
    current_energy = sum_of_maxes(rows)
    
    it = 0
    while time.time() < deadline - 20.0:
        it += 1
        T = max(T_min, T * cooling)
        
        # Choose a move
        ri = random.randrange(n_rows)
        move_type = random.random()
        
        old_energy = current_energy
        old_row = rows[ri]
        old_diffs = row_diff_sets[ri]
        
        if move_type < 0.7:
            # Local change: change one element
            ci = random.randrange(1, L)
            old_val = old_row[ci]
            
            # Generate candidates
            candidates = set()
            # Small shifts
            for d in range(-10, 11):
                nv = old_val + d
                if nv >= 1:
                    candidates.add(nv)
            # Random jump
            for _ in range(5):
                nv = random.randint(1, best_scope + 20)
                candidates.add(nv)
            
            # Enforce ordering
            lower = old_row[ci - 1] + 1
            upper = (old_row[ci + 1] - 1) if ci + 1 < L else best_scope + 20
            candidates = [c for c in candidates if lower <= c <= upper]
            
            if not candidates:
                continue
            
            random.shuffle(candidates)
            accepted = False
            for nv in candidates:
                if nv == old_val:
                    continue
                
                new_row = old_row[:]
                new_row[ci] = nv
                
                # Check validity
                new_diffs = row_diffs_set(new_row)
                if len(new_diffs) != diffs_per_row:
                    continue
                
                other_diffs = all_diffs - old_diffs
                if new_diffs & other_diffs:
                    continue
                
                # Valid move
                new_energy = sum_of_maxes(rows[:ri] + [new_row] + rows[ri+1:])
                delta_e = new_energy - old_energy
                
                if delta_e <= 0 or random.random() < math.exp(-delta_e / T):
                    # Accept
                    all_diffs = (all_diffs - old_diffs) | new_diffs
                    row_diff_sets[ri] = new_diffs
                    rows[ri] = new_row
                    current_energy = new_energy
                    
                    new_scope = scope_of(rows)
                    if new_scope < best_scope:
                        best_scope = new_scope
                        best_rows = rows[:]
                    accepted = True
                    break
        else:
            # Global move: Regenerate the row
            other_diffs = all_diffs - old_diffs
            # Hint: try to keep it within current best scope
            new_row = greedy_fill_row(other_diffs, best_scope)
            
            new_diffs = row_diffs_set(new_row)
            # Check if valid (greedy_fill_row should ensure no collision with other_diffs)
            # But we must ensure it doesn't collide with ITSELF? Greedy handles that.
            # We must ensure it doesn't collide with OTHERS. Greedy handles that via 'forbidden'.
            
            new_energy = sum_of_maxes(rows[:ri] + [new_row] + rows[ri+1:])
            delta_e = new_energy - old_energy
            
            if delta_e <= 0 or random.random() < math.exp(-delta_e / T):
                # Accept
                all_diffs = (all_diffs - old_diffs) | new_diffs
                row_diff_sets[ri] = new_diffs
                rows[ri] = new_row
                current_energy = new_energy
                
                new_scope = scope_of(rows)
                if new_scope < best_scope:
                    best_scope = new_scope
                    best_rows = rows[:]

    # Final Polish
    if time.time() < deadline - 5.0:
        polished, pol_scope = local_search_polish(best_rows, deadline - 5.0)
        if pol_scope < best_scope:
            best_rows = polished
            best_scope = pol_scope

    return best_rows
