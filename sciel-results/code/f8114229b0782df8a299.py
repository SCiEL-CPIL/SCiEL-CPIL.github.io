"""
Strategy: Randomized Backtracking with Forward Checking and Dynamic Ordering.
1. View the problem as assigning rows (sets of k+1 integers) such that their 
   pairwise differences are disjoint.
2. The "resource" is the set of used differences.
3. Use a backtracking search that constructs rows one by one.
4. Order rows dynamically: at each step, choose the next row to construct 
   based on the tightest constraints (e.g., fewest available candidates).
   For simplicity in this implementation, we will construct rows in a fixed 
   order but use a "tightest fit" heuristic for selecting values within a row,
   and restart with different random seeds/orders if stuck.
   Actually, a better structural change: 
   - Use a randomized greedy construction with multiple restarts.
   - Within each construction, when picking the next element for a row, 
     prefer values that minimize the "impact" on future rows (e.g., smaller 
     differences leave more room, or differences that are "rare").
   - To escape local minima, use a simulated annealing-like acceptance 
     criterion or random restarts with different random seeds.
   
   Let's refine: 
   The current program uses a deterministic greedy + repair. 
   The new strategy will use a **randomized constructive heuristic** with 
   **multiple restarts** and a **lookahead** for conflict minimization.
   
   Specifically:
   - For each row, generate candidate values.
   - Score candidates by how many future conflicts they might cause.
   - Use randomization to explore different branches.
   - Keep track of the best solution found across restarts.
"""
import time
import random
import sys

def construct(n_rows, k):
    deadline = time.time() + 140.0  # 10s safety margin
    
    best_rows = None
    best_score = float('inf')
    
    # Try multiple random restarts with different seeds
    for seed in range(1000):
        if time.time() > deadline - 2:
            break
            
        random.seed(seed)
        rows, score = _randomized_backtracking(n_rows, k, deadline)
        
        if rows is not None:
            if score < best_score:
                best_score = score
                best_rows = rows
                # If we hit the theoretical lower bound, we can stop
                # Lower bound is roughly n_rows * k * (k+1) / 2 / (k+1) ... 
                # Actually, the counting bound is given as 105 for (7,5).
                # We don't know the exact bound for other n_rows, k.
                # But if we find a very good solution, we can stop early.
                if best_score <= 105:
                    break

    if best_rows is None:
        # Fallback to deterministic greedy if backtracking fails
        best_rows = _greedy_construct(n_rows, k)
        
    return best_rows

def _randomized_backtracking(n_rows, k, deadline):
    """
    Attempts to construct n_rows rows of k+1 integers using randomized backtracking.
    Returns (rows, max_entry) or (None, inf) if no solution found within time.
    """
    used_diffs = set()
    rows = []
    
    def backtrack(idx):
        if time.time() > deadline - 1:
            return False
            
        if idx == n_rows:
            return True
            
        # Generate candidate rows
        # We want to find a row of k+1 increasing integers starting at 0
        # such that its differences are not in used_diffs.
        
        # Heuristic: Try to keep the row compact.
        # Start with a greedy construction, but with randomization.
        
        # For efficiency, we can pre-compute some things or use a smarter search.
        # Let's try a local search within the backtracking step.
        
        # Generate a random valid row that doesn't conflict with used_diffs.
        # We can do this by picking values one by one.
        
        row = [0]
        # We need to pick k more values.
        # Let's try a randomized greedy approach for this row.
        
        # To increase diversity, we can shuffle the order in which we try candidates.
        # But we must maintain increasing order.
        
        # Let's try to construct the row by picking the next value from a range.
        # Range: row[-1] + 1 to some upper bound.
        # Upper bound: We want to keep the max entry small.
        # Let's set a dynamic upper bound based on the current best score.
        
        current_best = best_score
        upper_bound = current_best if current_best < float('inf') else 200
        
        # Try to construct the row
        success = False
        for attempt in range(100):  # Limit attempts for this row
            if time.time() > deadline - 1:
                return False
                
            row = [0]
            valid = True
            row_diffs = set()
            
            for j in range(1, k + 1):
                # Pick next value
                # Minimum possible: row[-1] + 1
                # Maximum possible: upper_bound
                # We want to avoid conflicts with used_diffs and internal conflicts.
                
                # Generate candidates
                # To keep it compact, we should try smaller values first, but with randomization.
                
                # Let's try a range of values
                min_val = row[-1] + 1
                max_val = min(upper_bound, min_val + 20)  # Look ahead a bit
                
                candidates = list(range(min_val, max_val + 1))
                random.shuffle(candidates)
                
                found = False
                for cand in candidates:
                    # Check internal conflicts
                    new_diffs = {cand - a for a in row}
                    if len(new_diffs) != len(row):
                        continue  # Internal conflict
                    
                    # Check global conflicts
                    if new_diffs & used_diffs:
                        continue
                    
                    # Accept candidate
                    row.append(cand)
                    row_diffs |= new_diffs
                    found = True
                    break
                
                if not found:
                    valid = False
                    break
            
            if valid:
                # We found a valid row
                # Check if it improves the score
                max_entry = row[-1]
                if max_entry >= current_best:
                    # This row alone exceeds the best score, so it's not useful
                    # unless we are trying to find any solution
                    if current_best == float('inf'):
                        # Accept if it's the first solution
                        pass
                    else:
                        continue
                
                # Recurse
                used_diffs.update(row_diffs)
                rows.append(row)
                
                if backtrack(idx + 1):
                    return True
                
                # Backtrack
                rows.pop()
                used_diffs.difference_update(row_diffs)
                
        return False
        
    return None, float('inf')

def _greedy_construct(n_rows, k):
    used = set()
    rows = []
    for _ in range(n_rows):
        row, x = [0], 0
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if not new & used:
                row.append(x)
                used |= new
        rows.append(row)
    return rows

def _get_diffs(row):
    diffs = set()
    for i in range(len(row)):
        for j in range(i + 1, len(row)):
            diffs.add(row[j] - row[i])
    return diffs

def _max_entry(rows):
    return max(max(row) for row in rows)
