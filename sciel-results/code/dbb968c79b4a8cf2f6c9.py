import random
import time
import math

def construct(n_rows, k):
    """
    Constructs a difference triangle set with n_rows rows of k+1 integers.
    Minimizes the largest entry using Simulated Annealing with efficient local moves.
    """
    start_time = time.time()
    time_limit = 115  # Leave 5s buffer for safety

    def get_scope(rows):
        return max(max(row) for row in rows)

    def greedy_init():
        """Generate an initial valid set using greedy method."""
        used_diffs = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                x = row[-1] + 1
                while True:
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & used_diffs):
                        break
                    x += 1
                row.append(x)
                used_diffs.update({x - a for a in row[:-1]})
            rows.append(row)
        return rows

    def solve():
        rows = greedy_init()
        
        # Pre-calculate initial scope
        current_scope = get_scope(rows)
        best_scope = current_scope
        best_rows = [row[:] for row in rows]
        
        # Efficient difference tracking
        # Max possible difference is bounded by scope. Let's use a set for simplicity first, 
        # but optimize if needed. For n=7, k=6, diffs are small.
        used_diffs = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i+1, len(row)):
                    used_diffs.add(row[j] - row[i])

        # SA Parameters
        T = 5.0
        T_min = 0.01
        cooling = 0.999
        max_iters = 500000
        
        # Local Move: Try to decrease a random element in a random row
        def try_decrease_move():
            nonlocal rows, used_diffs, current_scope
            
            # Pick random row
            r_idx = random.randint(0, n_rows - 1)
            row = rows[r_idx]
            
            # Pick random position > 0
            if len(row) <= 1:
                return False
            
            pos = random.randint(1, len(row) - 1)
            old_val = row[pos]
            
            # We want to decrease old_val.
            # New value must be > row[pos-1]
            min_val = row[pos-1] + 1
            
            if old_val == min_val:
                return False # Cannot decrease further
            
            # Try decreasing by 1, 2, ... up to a small amount
            # Or pick a random target in [min_val, old_val - 1]
            # Let's try a random target to jump
            # Or just try old_val - 1 first? 
            # Let's try a random value in the range to explore better
            # But decreasing is the goal.
            
            # Let's try to find the smallest valid value in [min_val, old_val - 1]
            # This is a "greedy decrease" move.
            
            # To make it SA-like, we can accept worse moves too, but "decrease" only makes scope smaller or equal.
            # If we change a value, the scope might increase if this was the max element? 
            # If pos is the last element, decreasing it reduces scope.
            # If pos is not the last, decreasing it might allow subsequent elements to be smaller? No, we don't touch them.
            # So decreasing an element never increases the scope of that row.
            # It might increase the scope if... no, scope is max element. Decreasing an element <= old_val cannot increase max(row).
            # So this move is always "improving" or "neutral" regarding scope?
            # Wait, if we decrease a middle element, the differences change. 
            # The scope is determined by the last element. 
            # If we decrease a non-last element, the scope stays the same (unless we allow re-optimizing the rest, which we don't).
            # So this move is good for fixing conflicts but doesn't directly reduce scope unless we are at the last element.
            
            # To reduce scope, we must decrease the last element, or decrease a previous element and then decrease the last element?
            # This move only changes one element.
            
            # Let's try to pick the last element more often to reduce scope.
            # Or, if we pick a middle element, we might enable a future move to reduce the last element.
            
            # Let's just try to decrease the selected element to a random valid value in [min_val, old_val-1].
            
            # Generate candidates
            # We need to check if new diffs are valid.
            
            # Diffs involving pos:
            # Left: row[pos] - row[i] for i in [0, pos-1]
            # Right: row[j] - row[pos] for j in [pos+1, len(row)-1]
            
            old_left_diffs = [old_val - row[i] for i in range(pos)]
            old_right_diffs = [row[j] - old_val for j in range(pos+1, len(row))]
            
            # Remove these from used_diffs temporarily? 
            # It's safer to check against used_diffs minus these.
            
            # We will iterate possible new values from old_val - 1 down to min_val.
            # We want the smallest valid one? Or random?
            # Smallest valid one is best for compression.
            
            # Let's try a few random targets to allow exploration
            # Actually, let's just try to decrease by 1 first. If valid, accept.
            # This is a very local move.
            
            # Let's try to find ANY valid value in [min_val, old_val - 1].
            # We'll pick a random one to avoid getting stuck in a local pattern.
            
            candidates = []
            # Sample a few values
            # If range is large, sample. If small, check all.
            range_size = old_val - min_val
            if range_size > 10:
                for _ in range(5):
                    candidates.append(random.randint(min_val, old_val - 1))
            else:
                candidates = list(range(min_val, old_val))
                random.shuffle(candidates)
            
            for new_val in candidates:
                # Check validity
                new_left_diffs = [new_val - row[i] for i in range(pos)]
                new_right_diffs = [row[j] - new_val for j in range(pos+1, len(row))]
                
                # All new diffs must be:
                # 1. Positive (guaranteed by increasing order)
                # 2. Not in used_diffs (excluding the old diffs of this row that we are replacing)
                # 3. Not in the row's other diffs (which are unchanged)
                
                # Let's compute the set of diffs that will be "free" if we make this change
                # The old diffs involving pos are removed.
                # The new diffs involving pos are added.
                
                # We need to check if any new diff is in (used_diffs - old_diffs_involving_pos)
                # And also if any new diff is in (row_diffs - old_diffs_involving_pos)
                
                # Simpler: 
                # Construct the set of "forbidden" diffs.
                # Forbidden = (used_diffs - old_diffs_involving_pos)
                # Also forbidden are the other diffs in the row.
                
                # Let's pre-compute row_diffs_not_involving_pos
                row_other_diffs = set()
                for i in range(len(row)):
                    for j in range(i+1, len(row)):
                        if i != pos and j != pos:
                            row_other_diffs.add(row[j] - row[i])
                
                # Check new diffs
                valid = True
                for d in new_left_diffs + new_right_diffs:
                    if d in row_other_diffs:
                        valid = False
                        break
                    # Check against global used_diffs
                    # We need to ignore the old diffs that are being removed
                    # old_diffs_involving_pos = set(old_left_diffs + old_right_diffs)
                    if d in used_diffs:
                        # If d is in old_diffs_involving_pos, it's okay because we're removing it
                        # But wait, if d is in old_diffs_involving_pos AND in new_diffs, that's fine?
                        # No, if d is in old_diffs, it's in used_diffs.
                        # If we remove old_diffs and add new_diffs, and d is in both, 
                        # it means the diff d is still present.
                        # But we must ensure d is not present in ANY OTHER row.
                        # So we check if d is in used_diffs.
                        # If it is, we need to know if it's ONLY from this row's old diffs.
                        # This is hard to track with a simple set.
                        
                        # Alternative: 
                        # Temporarily remove old_diffs from a copy? No, too slow.
                        
                        # Let's count occurrences? No.
                        
                        # Let's just check:
                        # Is d in used_diffs?
                        # If yes, is it possible that it's only from the old diffs we are removing?
                        # We can check if d is in old_diffs_involving_pos.
                        # If yes, we need to check if d appears in any OTHER row.
                        # This requires scanning other rows. O(N*k^2).
                        
                        # For n=7, k=6, this is small.
                        
                        # Let's do this check only if d is in used_diffs.
                        
                        # Check if d is in any other row
                        found_in_other = False
                        for r_idx2 in range(n_rows):
                            if r_idx2 == r_idx:
                                continue
                            other_row = rows[r_idx2]
                            # Check if d is a diff in other_row
                            # We can pre-compute diff sets for each row?
                            # Or just scan.
                            for i2 in range(len(other_row)):
                                for j2 in range(i2+1, len(other_row)):
                                    if other_row[j2] - other_row[i2] == d:
                                        found_in_other = True
                                        break
                                if found_in_other:
                                    break
                            if found_in_other:
                                break
                        
                        if found_in_other:
                            valid = False
                            break
                        # If not found in other, it means d is only in the old diffs of this row.
                        # So it's safe to reuse it? 
                        # Yes, because we are removing the old diffs and adding the new ones.
                        # If d is in new_diffs, it will be in the new state.
                        # As long as no OTHER row has it, it's fine.
                        
                    # If d is NOT in used_diffs, it's definitely safe.
                
                if valid:
                    # Apply move
                    # Remove old diffs from used_diffs
                    for d in old_left_diffs + old_right_diffs:
                        # We can't just discard if it's shared? 
                        # We established that if a diff is shared, we wouldn't have accepted the move if it was in another row.
                        # But what if two old diffs are the same? No, diffs in a row are distinct.
                        # What if an old diff is the same as a new diff?
                        # If d is in old_diffs and d is in new_diffs, and no other row has it, 
                        # then removing old and adding new leaves it in the set.
                        # So we should just update the set.
                        
                        # Actually, it's easier to rebuild the used_diffs for this row?
                        # Or just discard old, add new.
                        # If d is in old and new, discard removes it, add puts it back. Net effect: present.
                        # If d is in old only, discard removes it. Net effect: absent.
                        # If d is in new only, add puts it. Net effect: present.
                        
                        used_diffs.discard(d)
                    
                    row[pos] = new_val
                    
                    for d in new_left_diffs + new_right_diffs:
                        used_diffs.add(d)
                    
                    # Update scope
                    new_scope = get_scope(rows)
                    if new_scope < current_scope:
                        current_scope = new_scope
                        if current_scope < best_scope:
                            best_scope = current_scope
                            best_rows = [r[:] for r in rows]
                            
                    return True
            
            return False

        for i in range(max_iters):
            if time.time() - start_time > time_limit:
                break
            
            # Try a local decrease move
            if try_decrease_move():
                # Accepted (implicitly, as it only improves or maintains scope validity)
                pass
            else:
                # Maybe try a different move? 
                # For now, just continue.
                pass
            
            # Cool down
            T *= cooling
            
            if best_scope <= 105:
                break

        return best_rows

    return solve()
