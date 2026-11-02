import time
import random
import math
import sys

sys.setrecursionlimit(10000)

def construct(n_rows, k):
    deadline = time.time() + 115  # Safety margin

    # 1. Initial Construction
    best_solution = _greedy_construct(n_rows, k)
    if not best_solution:
        return []
    
    best_score = max(max(r) for r in best_solution)
    current_solution = [r[:] for r in best_solution]
    current_score = best_score

    # 2. Simulated Annealing with Row Replacement
    T = 5.0
    cooling_rate = 0.995
    
    while time.time() < deadline - 10:
        # Pick a random row to replace
        idx = random.randint(0, n_rows - 1)
        
        # Calculate forbidden diffs from other rows
        forbidden_diffs = set()
        for i in range(n_rows):
            if i == idx:
                continue
            row = current_solution[i]
            n = len(row)
            for j in range(n):
                for l in range(j + 1, n):
                    forbidden_diffs.add(row[l] - row[j])
        
        # Try to generate a new compatible row
        # Bound the search to current_score + small margin to encourage improvement
        # or allow slight increase to escape local minima
        max_val_bound = current_score + 20 
        new_row = _generate_random_compatible_row(forbidden_diffs, k, max_val_bound)
        
        if new_row:
            new_solution = [r[:] for r in current_solution]
            new_solution[idx] = new_row
            new_score = max(max(r) for r in new_solution)
            
            delta = new_score - current_score
            
            # Acceptance criterion
            if delta < 0 or (T > 0.1 and random.random() < math.exp(-delta / T)):
                current_solution = new_solution
                current_score = new_score
                
                if current_score < best_score:
                    best_score = current_score
                    best_solution = [r[:] for r in current_solution]
        
        T *= cooling_rate
        if T < 0.1:
            T = 0.1 # Floor temperature to maintain some randomness

    # 3. Final Polish: Local Search to minimize scope
    if time.time() < deadline - 2:
        polished = _local_search_minimize(best_solution, n_rows, k, deadline)
        if polished:
            curr_score = max(max(r) for r in polished)
            if curr_score < best_score:
                best_score = curr_score
                best_solution = polished

    return best_solution

def _greedy_construct(n_rows, k):
    """Fast greedy baseline construction."""
    used_diffs = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        current_max = 0
        while len(row) < k + 1:
            found = False
            # Look ahead a bit
            for x in range(current_max + 1, current_max + 1000):
                new_diffs = []
                valid = True
                for a in row:
                    d = x - a
                    if d in used_diffs:
                        valid = False
                        break
                    new_diffs.append(d)

                if valid:
                    row.append(x)
                    used_diffs.update(new_diffs)
                    current_max = x
                    found = True
                    break
            if not found:
                return None
        rows.append(row)
    return rows

def _generate_random_compatible_row(forbidden_diffs, k, max_val_bound):
    """
    Generate a random row of length k+1 starting at 0.
    Must have distinct internal diffs and no diff in forbidden_diffs.
    Max entry <= max_val_bound.
    """
    # Backtracking search
    def backtrack(pos, current_row, used_new_diffs):
        if pos == k:
            return current_row[:]
        
        lower = current_row[-1] + 1
        # To ensure we don't exceed max_val_bound
        # We need to leave room for the remaining k-pos elements
        # Simple heuristic: upper bound is max_val_bound - (k - pos)
        upper = min(max_val_bound, lower + 30) 
        # If upper < lower, no solution
        if upper < lower:
            return None
            
        # Try random values in range
        # Shuffle a small window to avoid bias
        candidates = list(range(lower, upper + 1))
        random.shuffle(candidates)
        
        for val in candidates:
            valid = True
            new_diffs = set()
            for i in range(len(current_row)):
                d = val - current_row[i]
                if d in forbidden_diffs or d in used_new_diffs:
                    valid = False
                    break
                new_diffs.add(d)
            
            if valid:
                res = backtrack(pos + 1, current_row + [val], used_new_diffs | new_diffs)
                if res:
                    return res
                    
        return None

    return backtrack(1, [0], set())

def _local_search_minimize(rows, n_rows, k, deadline):
    """
    Greedy local search to minimize max entry.
    Applies single-point moves to compress rows.
    """
    best_rows = [r[:] for r in rows]
    used_diffs = set()
    for r in best_rows:
        n = len(r)
        for i in range(n):
            for j in range(i + 1, n):
                used_diffs.add(r[j] - r[i])

    while time.time() < deadline - 5:
        progress = False
        row_order = list(range(n_rows))
        random.shuffle(row_order)

        for r_idx in row_order:
            if time.time() > deadline - 5:
                break
            row = best_rows[r_idx]
            # Process positions right-to-left
            for pos in range(k, 0, -1):
                old_val = row[pos]
                
                # Compute forbidden diffs: all used diffs minus those involving old_val
                diffs_involving_old = set()
                for i in range(len(row)):
                    if i == pos: continue
                    if i < pos:
                        diffs_involving_old.add(old_val - row[i])
                    else:
                        diffs_involving_old.add(row[i] - old_val)
                
                forbidden_diffs = used_diffs - diffs_involving_old
                
                lower_bound = row[pos - 1] + 1
                upper_bound = old_val - 1
                if pos + 1 < len(row):
                    upper_bound = min(upper_bound, row[pos + 1] - 1)
                
                if upper_bound < lower_bound:
                    continue
                
                # Search for smallest valid value
                found = None
                for new_val in range(lower_bound, upper_bound + 1):
                    valid = True
                    new_diffs = set()
                    for i in range(len(row)):
                        if i == pos: continue
                        if i < pos:
                            d = new_val - row[i]
                        else:
                            d = row[i] - new_val
                        if d in forbidden_diffs or d in new_diffs:
                            valid = False
                            break
                        new_diffs.add(d)
                    
                    if valid:
                        found = new_val
                        break
                
                if found is not None and found < old_val:
                    # Update used_diffs
                    for d in diffs_involving_old:
                        used_diffs.discard(d)
                    for d in new_diffs:
                        used_diffs.add(d)
                    row[pos] = found
                    progress = True
        
        if not progress:
            break

    return best_rows
