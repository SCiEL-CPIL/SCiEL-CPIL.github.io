"""
Strategy: Local search over gap representations with global difference-conflict cost.

Core structural change from the seed:
- Seed program: random pool generation + DFS backtracking over row selections.
- This program: build an initial feasible (valid) configuration via greedy, then
  minimize the SCOPE (max element) using local search that moves individual
  elements (or swaps gaps) while tracking a global difference-conflict set.
  The objective is a composite: primary = scope, secondary = conflict count (must
  be 0 for validity). We use simulated-annealing-style moves with a hard validity
  gate: only accept a move if it does not introduce new global difference conflicts
  beyond what we are willing to temporarily carry, and always track the best
  conflict-free configuration.

Representation: each row is a list of k+1 increasing integers starting at 0.
Moves:
  1. Element move: change row[i][p] to a new value (keeping increasing order and >0 for p>=1).
  2. Gap swap: within a row, swap two adjacent gaps.
  3. Row element swap: swap an element from row A with a compatible element from row B.

We track:
  - global_diffs: dict from difference value -> set of (row_idx, i, j) producing it.
  - A move is "conflict-free" if after the move, no difference value appears more than once
    globally.

To make this tractable within 150s, we:
  1. Build a quick valid initial solution via greedy (scope ~150-200).
  2. Run local search with deadline guard, aiming to reduce scope.
  3. Return the best valid (conflict-free) solution found.
"""
import time
import random
import math

# EVOLVE-BLOCK-START

def _row_diffs(row):
    """Compute all positive differences for a single row."""
    n = len(row)
    diffs = []
    for i in range(n):
        for j in range(i + 1, n):
            diffs.append(row[j] - row[i])
    return diffs

def _scope_of(rows):
    """Max element across all rows."""
    return max(max(r) for r in rows) if rows else 0

def _is_valid(rows):
    """Check if all positive differences across all rows are globally distinct."""
    all_diffs = []
    for row in rows:
        all_diffs.extend(_row_diffs(row))
    return len(all_diffs) == len(set(all_diffs))

def _build_diff_map(rows):
    """Build a dict: diff_value -> list of (row_idx, i, j). Returns (diff_map, conflict_count)."""
    diff_map = {}
    conflict_count = 0
    for r_idx, row in enumerate(rows):
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                d = row[j] - row[i]
                if d not in diff_map:
                    diff_map[d] = [(r_idx, i, j)]
                else:
                    diff_map[d].append((r_idx, i, j))
                    if len(diff_map[d]) > 1:
                        conflict_count += 1
    return diff_map, conflict_count

def _greedy_construct(n_rows, k, rng, max_scope=250):
    """
    Greedily construct a valid set of n_rows rows.
    Each row has k+1 elements starting at 0, strictly increasing.
    Uses random sampling with retry.
    """
    num_elements = k + 1
    rows = []
    used_diffs = set()
    
    for _ in range(n_rows):
        found = False
        for attempt in range(5000):
            # Sample k distinct values from 1..max_scope
            if max_scope < k:
                break
            try:
                candidates = rng.sample(range(1, max_scope + 1), k)
            except ValueError:
                break
            candidates.sort()
            row = [0] + candidates
            diffs = _row_diffs(row)
            if len(diffs) != len(set(diffs)):
                continue
            if any(d in used_diffs for d in diffs):
                continue
            rows.append(row)
            used_diffs.update(diffs)
            found = True
            break
        if not found:
            # Increase scope and retry this row
            max_scope += 30
            # Retry
            for attempt in range(5000):
                try:
                    candidates = rng.sample(range(1, max_scope + 1), k)
                except ValueError:
                    break
                candidates.sort()
                row = [0] + candidates
                diffs = _row_diffs(row)
                if len(diffs) != len(set(diffs)):
                    continue
                if any(d in used_diffs for d in diffs):
                    continue
                rows.append(row)
                used_diffs.update(diffs)
                found = True
                break
            if not found:
                return None  # Failure
    
    return rows if len(rows) == n_rows else None

def _local_search(rows, n_rows, k, deadline, rng):
    """
    Local search to minimize scope while maintaining validity.
    
    Strategy:
    - Maintain the current solution and its diff_map.
    - At each step, pick a random row and a random element position.
    - Try to move that element to a smaller value (to reduce scope) or to a
      different value that reduces the max element of that row.
    - Accept the move if it preserves validity (no new global conflicts).
    - Also try "gap compression" moves: reduce a gap in a row if the resulting
      element doesn't create conflicts.
    
    We track best_scope and best_rows (the best valid configuration found).
    """
    num_elements = k + 1
    current_rows = [list(r) for r in rows]
    best_rows = [list(r) for r in rows]
    best_scope = _scope_of(best_rows)
    
    # Precompute initial diff map
    diff_map, conflict_count = _build_diff_map(current_rows)
    assert conflict_count == 0, "Initial solution must be valid"
    
    step = 0
    no_improve_steps = 0
    max_no_improve = 20000
    
    while time.time() < deadline - 5:
        step += 1
        if no_improve_steps > max_no_improve:
            break
        
        # Choose a move type
        move_type = rng.random()
        
        if move_type < 0.6:
            # Element move: pick a row and an element (position >= 1), try to change it
            r_idx = rng.randrange(n_rows)
            pos = rng.randrange(1, num_elements)  # positions 1..k
            row = current_rows[r_idx]
            old_val = row[pos]
            
            # Try to reduce this element
            # New value must satisfy: row[pos-1] < new_val < row[pos+1] (if pos < k)
            low_bound = row[pos - 1] + 1
            high_bound = row[pos + 1] - 1 if pos < num_elements - 1 else best_scope + 20
            
            if high_bound < low_bound:
                no_improve_steps += 1
                continue
            
            # Try a few candidate new values
            tried = 0
            for _ in range(15):
                if tried >= 15:
                    break
                tried += 1
                if rng.random() < 0.7:
                    # Try to reduce
                    if low_bound < old_val:
                        new_val = rng.randint(low_bound, old_val - 1)
                    else:
                        new_val = old_val  # can't reduce, try nearby
                else:
                    # Try a random nearby value
                    span = max(1, (high_bound - low_bound) // 4)
                    new_val = rng.randint(low_bound, high_bound)
                
                if new_val == old_val:
                    continue
                
                # Check if this move preserves validity
                # Remove old differences involving (r_idx, pos)
                old_diffs_to_remove = []
                for i in range(num_elements):
                    if i == pos:
                        continue
                    d_old = abs(row[pos] - row[i])
                    old_diffs_to_remove.append((i, d_old))
                
                # Compute new differences
                new_row = list(row)
                new_row[pos] = new_val
                new_diffs_to_add = []
                for i in range(num_elements):
                    if i == pos:
                        continue
                    d_new = abs(new_row[pos] - new_row[i])
                    new_diffs_to_add.append((i, d_new))
                
                # Check conflicts: for each new diff, see if it's already used by another pair
                valid_move = True
                for i, d_new in new_diffs_to_add:
                    if d_new in diff_map:
                        # Check if the existing usage is from this same pair
                        for (rr, ii, jj) in diff_map[d_new]:
                            if not ((rr == r_idx and ii == i and jj == pos) or
                                    (rr == r_idx and ii == pos and jj == i)):
                                valid_move = False
                                break
                    if not valid_move:
                        break
                if not valid_move:
                    continue
                
                # Also check internal row validity: new diffs within row must be distinct
                row_diffs_new = _row_diffs(new_row)
                if len(row_diffs_new) != len(set(row_diffs_new)):
                    continue
                
                # Accept the move
                # Update diff_map
                for i, d_old in old_diffs_to_remove:
                    if d_old in diff_map:
                        diff_map[d_old] = [x for x in diff_map[d_old]
                                           if not ((x[0] == r_idx and x[1] == i and x[2] == pos) or
                                                   (x[0] == r_idx and x[1] == pos and x[2] == i))]
                        if not diff_map[d_old]:
                            del diff_map[d_old]
                
                for i, d_new in new_diffs_to_add:
                    if d_new not in diff_map:
                        diff_map[d_new] = []
                    diff_map[d_new].append((r_idx, min(i, pos), max(i, pos)))
                
                current_rows[r_idx] = new_row
                row = new_row
                
                new_scope = _scope_of(current_rows)
                if new_scope < best_scope:
                    best_scope = new_scope
                    best_rows = [list(r) for r in current_rows]
                    no_improve_steps = 0
                else:
                    no_improve_steps += 1
                break  # Only one move per iteration
                
        elif move_type < 0.85:
            # Gap compression: try to reduce a gap in a row
            r_idx = rng.randrange(n_rows)
            row = current_rows[r_idx]
            if len(row) < 2:
                no_improve_steps += 1
                continue
            # Pick a gap position: between row[gap_idx] and row[gap_idx+1]
            gap_idx = rng.randrange(0, num_elements - 1)
            # Try to reduce the element at position gap_idx+1
            pos = gap_idx + 1
            old_val = row[pos]
            low_bound = row[pos - 1] + 1
            high_bound = row[pos + 1] - 1 if pos < num_elements - 1 else best_scope + 20
            
            if high_bound < low_bound or old_val <= low_bound:
                no_improve_steps += 1
                continue
            
            new_val = rng.randint(low_bound, old_val - 1)
            if new_val == old_val:
                no_improve_steps += 1
                continue
            
            # Check validity
            old_diffs_to_remove = []
            for i in range(num_elements):
                if i == pos:
                    continue
                d_old = abs(row[pos] - row[i])
                old_diffs_to_remove.append((i, d_old))
            
            new_row = list(row)
            new_row[pos] = new_val
            new_diffs_to_add = []
            for i in range(num_elements):
                if i == pos:
                    continue
                d_new = abs(new_row[pos] - new_row[i])
                new_diffs_to_add.append((i, d_new))
            
            valid_move = True
            for i, d_new in new_diffs_to_add:
                if d_new in diff_map:
                    for (rr, ii, jj) in diff_map[d_new]:
                        if not ((rr == r_idx and ii == i and jj == pos) or
                                (rr == r_idx and ii == pos and jj == i)):
                            valid_move = False
                            break
                if not valid_move:
                    break
            if not valid_move:
                no_improve_steps += 1
                continue
            
            row_diffs_new = _row_diffs(new_row)
            if len(row_diffs_new) != len(set(row_diffs_new)):
                no_improve_steps += 1
                continue
            
            # Accept
            for i, d_old in old_diffs_to_remove:
                if d_old in diff_map:
                    diff_map[d_old] = [x for x in diff_map[d_old]
                                       if not ((x[0] == r_idx and x[1] == i and x[2] == pos) or
                                               (x[0] == r_idx and x[1] == pos and x[2] == i))]
                    if not diff_map[d_old]:
                        del diff_map[d_old]
            
            for i, d_new in new_diffs_to_add:
                if d_new not in diff_map:
                    diff_map[d_new] = []
                diff_map[d_new].append((r_idx, min(i, pos), max(i, pos)))
            
            current_rows[r_idx] = new_row
            new_scope = _scope_of(current_rows)
            if new_scope < best_scope:
                best_scope = new_scope
                best_rows = [list(r) for r in current_rows]
                no_improve_steps = 0
            else:
                no_improve_steps += 1
                
        else:
            # Row-level shuffle: try swapping two elements in different rows
            # (only if they fit and don't create conflicts)
            if n_rows < 2:
                no_improve_steps += 1
                continue
            r1, r2 = rng.sample(range(n_rows), 2)
            row1 = current_rows[r1]
            row2 = current_rows[r2]
            p1 = rng.randrange(1, num_elements)
            p2 = rng.randrange(1, num_elements)
            
            val1 = row1[p1]
            val2 = row2[p2]
            if val1 == val2:
                no_improve_steps += 1
                continue
            
            # Check if swapping maintains increasing order
            new_row1 = list(row1)
            new_row2 = list(row2)
            new_row1[p1] = val2
            new_row2[p2] = val1
            
            # Check increasing order for new_row1
            if any(new_row1[i] >= new_row1[i+1] for i in range(num_elements - 1)):
                no_improve_steps += 1
                continue
            if any(new_row2[i] >= new_row2[i+1] for i in range(num_elements - 1)):
                no_improve_steps += 1
                continue
            
            # Check global validity
            # This is complex; skip for simplicity and rely on element moves
            no_improve_steps += 1
    
    return best_rows, best_scope

def _refine_with_backtrack(rows, n_rows, k, deadline, rng):
    """
    Additional refinement: try to reduce scope by systematically lowering
    the maximum element. For the row containing the max element, try to
    lower that element while maintaining validity.
    """
    best_rows = [list(r) for r in rows]
    best_scope = _scope_of(best_rows)
    
    while time.time() < deadline - 5:
        scope = _scope_of(best_rows)
        if scope <= 105:
            break
        
        # Find the row and position with the max element
        max_val = scope
        max_r = -1
        max_p = -1
        for r_idx, row in enumerate(best_rows):
            for p, v in enumerate(row):
                if v == max_val:
                    max_r = r_idx
                    max_p = p
                    break
            if max_r >= 0:
                break
        
        if max_r < 0:
            break
        
        row = best_rows[max_r]
        # Try to lower row[max_p]
        low_bound = row[max_p - 1] + 1 if max_p > 0 else 1
        tried = 0
        improved = False
        for _ in range(50):
            if tried >= 50 or time.time() > deadline - 5:
                break
            tried += 1
            new_val = rng.randint(low_bound, max_val - 1)
            if new_val >= max_val:
                continue
            
            new_row = list(row)
            new_row[max_p] = new_val
            
            # Check increasing
            if any(new_row[i] >= new_row[i+1] for i in range(len(new_row) - 1)):
                continue
            
            # Check global validity
            test_rows = [list(r) if r != max_r else new_row for r in best_rows]
            if _is_valid(test_rows):
                new_scope = _scope_of(test_rows)
                if new_scope < best_scope:
                    best_rows = test_rows
                    best_scope = new_scope
                    improved = True
                    break
        
        if not improved:
            # Try a different approach: random perturbation
            r_idx = rng.randrange(n_rows)
            pos = rng.randrange(1, k + 1)
            row = best_rows[r_idx]
            old_val = row[pos]
            new_val = rng.randint(1, best_scope + 10)
            # Ensure increasing
            low_b = row[pos-1] + 1 if pos > 0 else 1
            high_b = row[pos+1] - 1 if pos < k else best_scope + 10
            if low_b > high_b:
                continue
            new_val = max(low_b, min(new_val, high_b))
            if new_val == old_val:
                continue
            
            new_row = list(row)
            new_row[pos] = new_val
            if any(new_row[i] >= new_row[i+1] for i in range(len(new_row) - 1)):
                continue
            
            test_rows = [list(r) if r != r_idx else new_row for r in best_rows]
            if _is_valid(test_rows):
                new_scope = _scope_of(test_rows)
                if new_scope <= best_scope:
                    best_rows = test_rows
                    if new_scope < best_scope:
                        best_scope = new_scope
    
    return best_rows, best_scope

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0
    such that all positive differences within rows are distinct globally.
    Minimize the largest entry (scope).
    """
    start_time = time.time()
    DEADLINE = start_time + 140.0  # Safety margin before 150s kill
    
    rng = random.Random(12345)
    
    # Step 1: Greedy construction to get a valid initial solution
    initial_rows = _greedy_construct(n_rows, k, rng, max_scope=250)
    
    if initial_rows is None:
        # Fallback: even larger scope
        initial_rows = _greedy_construct(n_rows, k, rng, max_scope=400)
        if initial_rows is None:
            # Last resort: sequential construction
            rows = []
            used = set()
            for _ in range(n_rows):
                row = [0]
                x = 0
                while len(row) < k + 1:
                    x += 1
                    new_diffs = {x - a for a in row}
                    if not new_diffs & used:
                        row.append(x)
                        used |= new_diffs
                rows.append(row)
            return rows
    
    # Step 2: Local search to minimize scope
    best_rows, best_scope = _local_search(initial_rows, n_rows, k, DEADLINE, rng)
    
    if time.time() > DEADLINE - 10:
        return best_rows
    
    # Step 3: Additional refinement
    best_rows, best_scope = _refine_with_backtrack(best_rows, n_rows, k, DEADLINE, rng)
    
    if time.time() > DEADLINE - 5:
        return best_rows
    
    # Step 4: Try another round of local search from the refined solution
    if time.time() < DEADLINE - 15:
        best_rows2, best_scope2 = _local_search(best_rows, n_rows, k, DEADLINE, rng)
        if best_scope2 < best_scope:
            best_rows = best_rows2
            best_scope = best_scope2
    
    # Final verification
    if not _is_valid(best_rows):
        # Should not happen, but safety
        initial_rows = _greedy_construct(n_rows, k, rng, max_scope=300)
        if initial_rows is not None:
            return initial_rows
    
    return best_rows

# EVOLVE-BLOCK-END
