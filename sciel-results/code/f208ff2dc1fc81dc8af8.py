import time
import random
import math
from collections import Counter

_DEADLINE = 0.0

def _scope(rows):
    return max(a for r in rows for a in r[1:])

def _valid(rows, k):
    seen = set()
    for r in rows:
        if r[0] != 0 or len(r) != k + 1:
            return False
        for i in range(len(r)):
            for j in range(i + 1, len(r)):
                d = r[j] - r[i]
                if d <= 0 or d in seen:
                    return False
                seen.add(d)
    return True

def _build_diff_counts(rows):
    counts = Counter()
    for r in rows:
        for i in range(len(r)):
            for j in range(i + 1, len(r)):
                d = r[j] - r[i]
                counts[d] += 1
    return counts

def _remove_row_diffs(diff_counts, row):
    for i in range(len(row)):
        for j in range(i + 1, len(row)):
            d = row[j] - row[i]
            if d in diff_counts:
                diff_counts[d] -= 1
                if diff_counts[d] == 0:
                    del diff_counts[d]
            else:
                # Should not happen if state is consistent, but safe
                pass

def _add_row_diffs(diff_counts, row):
    for i in range(len(row)):
        for j in range(i + 1, len(row)):
            d = row[j] - row[i]
            diff_counts[d] = diff_counts.get(d, 0) + 1

def _get_cost(diff_counts, scope):
    collisions = sum(max(0, c - 1) for c in diff_counts.values())
    return collisions * 1000 + scope

def _greedy_row(k, used_diffs, rng, max_scope):
    """
    Greedily construct a row of k+1 elements starting with 0,
    avoiding differences in used_diffs.
    """
    row = [0]
    internal_diffs = set()
    for j in range(1, k + 1):
        found = False
        # Determine search range
        start = row[-1] + 1
        # Bias towards smaller numbers to keep scope low, but allow jumps
        # Use a random window
        max_step = 20
        for t in range(start, min(max_scope, start + max_step) + 1):
            ok = True
            new_diffs = []
            for a in row:
                d = t - a
                if d <= 0:
                    continue
                if d in internal_diffs or d in used_diffs:
                    ok = False
                    break
                new_diffs.append(d)
            
            if ok:
                row.append(t)
                internal_diffs.update(new_diffs)
                found = True
                break
        
        if not found:
            # If we can't find a small step, try larger steps
            # This fallback is rarely needed if max_scope is large enough
            for t in range(start, max_scope + 1):
                ok = True
                new_diffs = []
                for a in row:
                    d = t - a
                    if d <= 0:
                        continue
                    if d in internal_diffs or d in used_diffs:
                        ok = False
                        break
                    new_diffs.append(d)
                
                if ok:
                    row.append(t)
                    internal_diffs.update(new_diffs)
                    found = True
                    break
        
        if not found:
            return None
    return row

def _hill_climb(rows, k, diff_counts, rng, max_iters, deadline):
    """
    Local search to minimize cost.
    Returns improved rows and diff_counts.
    """
    current_rows = [list(r) for r in rows]
    current_diffs = dict(diff_counts)
    current_scope = _scope(current_rows)
    current_cost = _get_cost(current_diffs, current_scope)
    
    best_rows = [list(r) for r in current_rows]
    best_diffs = dict(current_diffs)
    best_cost = current_cost
    best_scope = current_scope
    
    no_improve_count = 0
    max_no_improve = 500
    
    for _ in range(max_iters):
        if time.time() > deadline:
            break
            
        if no_improve_count > max_no_improve:
            break
            
        # Pick a random row and a random element to move
        i = rng.randint(0, len(current_rows) - 1)
        j = rng.randint(1, k) # 1 to k, index of element
        
        old_val = current_rows[i][j]
        prev_val = current_rows[i][j-1]
        next_val = current_rows[i][j+1] if j < k else current_scope + 10
        
        lo = prev_val + 1
        hi = next_val - 1
        
        if lo > hi:
            continue
        
        # Try a few random new values
        improved = False
        for _ in range(5):
            new_val = rng.randint(lo, hi)
            if new_val == old_val:
                continue
                
            # Apply move
            _remove_row_diffs(current_diffs, current_rows[i])
            current_rows[i][j] = new_val
            _add_row_diffs(current_diffs, current_rows[i])
            
            new_scope = _scope(current_rows)
            new_cost = _get_cost(current_diffs, new_scope)
            
            if new_cost < current_cost:
                current_cost = new_cost
                improved = True
                no_improve_count = 0
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_rows = [list(r) for r in current_rows]
                    best_diffs = dict(current_diffs)
                    best_scope = new_scope
                break
            else:
                # Revert
                current_rows[i][j] = old_val
                _remove_row_diffs(current_diffs, current_rows[i])
                _add_row_diffs(current_diffs, current_rows[i])
        
        if not improved:
            no_improve_count += 1
            
    return best_rows, best_diffs, best_scope, best_cost

def _perturb(rows, k, diff_counts, rng, strength):
    """
    Create a perturbed version of the solution.
    strength: number of rows to rebuild or elements to kick.
    """
    new_rows = [list(r) for r in rows]
    new_diffs = dict(diff_counts)
    
    # Rebuild 'strength' rows
    rows_to_rebuild = rng.sample(range(len(new_rows)), min(strength, len(new_rows)))
    for i in rows_to_rebuild:
        old_row = new_rows[i]
        _remove_row_diffs(new_diffs, old_row)
        
        # Rebuild row
        new_row = _greedy_row(k, new_diffs, rng, _scope(new_rows) + 50)
        if new_row is None:
            # Fallback: keep old row
            _add_row_diffs(new_diffs, old_row)
        else:
            new_rows[i] = new_row
            _add_row_diffs(new_diffs, new_row)
            
    return new_rows, new_diffs

def construct(n_rows, k):
    global _DEADLINE
    _DEADLINE = time.time() + 115.0
    
    rng = random.Random(123)
    
    # 1. Initial Solution
    # Try to find a valid solution using greedy
    rows = []
    used = set()
    for _ in range(n_rows):
        row = _greedy_row(k, used, rng, 200)
        if row is None:
            # Fallback: simple arithmetic progression (might have collisions)
            row = [0, 1, 2, 3, 4, 5]
        rows.append(row)
        for i in range(len(row)):
            for j in range(i+1, len(row)):
                used.add(row[j]-row[i])
                
    # Ensure we have a valid starting point if possible
    # If greedy failed to produce valid, we start with collisions and let ILS fix it
    # But it's better to start valid if we can.
    # The greedy above adds to 'used' only if no conflict found in the loop?
    # No, _greedy_row checks against 'used_diffs' passed in.
    # So if it returns a row, it is valid w.r.t previous rows.
    # So 'rows' is a valid set.
    
    diff_counts = _build_diff_counts(rows)
    current_scope = _scope(rows)
    
    best_rows = [list(r) for r in rows]
    best_diffs = dict(diff_counts)
    best_scope = current_scope
    best_cost = _get_cost(best_diffs, best_scope)
    
    # 2. Iterated Local Search
    # We run ILS until deadline
    while time.time() < _DEADLINE - 5.0:
        # Hill Climb
        hc_rows, hc_diffs, hc_scope, hc_cost = _hill_climb(
            best_rows, k, best_diffs, rng, 
            max_iters=2000, 
            deadline=time.time() + 10.0 # Time limit per HC phase
        )
        
        if hc_cost < best_cost:
            best_rows = hc_rows
            best_diffs = hc_diffs
            best_scope = hc_scope
            best_cost = hc_cost
            
        # Perturb
        # Increase perturbation strength if we are stuck?
        # For now, constant strength 1 or 2
        strength = 1 if rng.random() < 0.5 else 2
        new_rows, new_diffs = _perturb(best_rows, k, best_diffs, rng, strength)
        
        # Check if perturbation made it invalid (it shouldn't if greedy works, 
        # but if greedy fails and we keep old row, it's valid.
        # If greedy succeeds, it's valid w.r.t other rows.
        # So new_rows should be valid.)
        
        # If perturbation resulted in a better cost (unlikely directly, but possible if scope dropped),
        # we could accept it immediately, but usually we Hill Climb from it.
        
        # To save time, we don't Hill Climb from every perturbation immediately if we are close to deadline.
        # But ILS structure is: Perturb -> Local Search.
        # So we should Local Search from new_rows.
        
        # However, to keep it simple and fast, let's just Hill Climb the best found so far in the next loop?
        # No, ILS requires searching from the perturbed state.
        
        # Let's do a short Hill Climb from the perturbed state
        if time.time() < _DEADLINE - 10.0:
            hc2_rows, hc2_diffs, hc2_scope, hc2_cost = _hill_climb(
                new_rows, k, new_diffs, rng,
                max_iters=1000,
                deadline=time.time() + 5.0
            )
            
            if hc2_cost < best_cost:
                best_rows = hc2_rows
                best_diffs = hc2_diffs
                best_scope = hc2_scope
                best_cost = hc2_cost
                
    # Final Polish: Run a long Hill Climb on the best solution found
    if time.time() < _DEADLINE - 2.0:
        final_rows, final_diffs, final_scope, final_cost = _hill_climb(
            best_rows, k, best_diffs, rng,
            max_iters=10000,
            deadline=_DEADLINE - 1.0
        )
        if final_cost < best_cost:
            best_rows = final_rows
            best_scope = final_scope
            
    return best_rows
