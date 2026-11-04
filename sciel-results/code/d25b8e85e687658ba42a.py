"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import time
import math
import numpy as np
from collections import deque

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    start_time = time.time()
    # Evaluator kills at 150s. We need to return before that.
    # Let's aim for 145s to be safe against overhead.
    deadline = start_time + 145.0
    
    # Total number of differences that must be distinct
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row
    
    # The maximum possible difference is M.
    # We need total_diffs distinct differences from 1..M.
    # So M must be at least total_diffs.
    # Lower bound: M >= total_diffs.
    # Also, the largest entry is M.
    
    # We will search for the smallest M such that a valid configuration exists.
    # We start from a greedy lower bound and go up.
    
    # Helper: Greedy construction to get an initial upper bound
    def greedy_construct(n_rows, k):
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            current_max = 0
            while len(row) < k + 1:
                # Try to add the smallest possible next number
                # We need to pick x > row[-1]
                # Such that x - a is not in used for all a in row
                found = False
                # Heuristic: try small increments first? Or jump?
                # Simple: scan upwards
                # Optimization: We know we need to fit in some range.
                # Let's just scan from row[-1] + 1
                # To make it faster, we can jump to avoid known conflicts if we track them, 
                # but for a simple baseline, scanning is okay if M is small.
                # However, if M is large, this is slow.
                # Better heuristic: 
                # The set of forbidden values for x is {a + d | a in row, d in used}.
                # We want the smallest x > row[-1] not in forbidden.
                
                # Actually, let's just do a simple random restart greedy to get a decent start.
                # Or better: use the backtracking solver directly with a generous M, 
                # then reduce M.
                
                # For now, a simple deterministic greedy:
                # Pick smallest x > row[-1] such that diffs are new.
                x = row[-1] + 1
                while True:
                    ok = True
                    for a in row:
                        if x - a in used:
                            ok = False
                            break
                    if ok:
                        break
                    x += 1
                
                row.append(x)
                for a in row[:-1]:
                    used.add(x - a)
                current_max = x
            rows.append(row)
        return rows, max(max(r) for r in rows)

    # Initial upper bound
    baseline_rows, initial_max = greedy_construct(n_rows, k)
    
    # We want to minimize M.
    # We know M >= total_diffs.
    # Let's try values of M starting from total_diffs upwards.
    # If we find a solution for M, we update best.
    # We can also use the result of greedy as a fallback.
    
    best_rows = baseline_rows
    best_max = initial_max
    
    # Bitset for used differences
    # Size: M + 1 (differences 1..M)
    # We will use a Python set for simplicity first, or a bitarray if available? 
    # Standard library doesn't have bitarray. 
    # We can use a list of ints or just a set. 
    # Given M <= ~200-300, a set of ints is fast enough.
    
    def solve_with_backtracking(M, time_limit):
        """
        Tries to construct n_rows of k+1 increasing integers starting at 0,
        max value <= M, such that all differences are distinct.
        Returns list of rows or None.
        """
        # State:
        # current_rows: list of lists
        # used_diffs: set of ints
        # 
        # Variable ordering: 
        # We fill rows one by one? Or interleave?
        # Filling row by row is simpler. 
        # Within a row, we fill positions 1..k.
        # 
        # Heuristic for value selection:
        # For the next position in the current row, we have a set of candidate values.
        # We prefer smaller values to keep the max low? 
        # Or we can randomize to escape local minima.
        # Since we are checking for existence for a fixed M, 
        # we can use DFS with random tie-breaking.
        
        rows = []
        used = set()
        
        # We will use a recursive backtracking function.
        # To avoid recursion depth issues (though k is small, n_rows is small),
        # we can implement it iteratively or just use recursion with sys.setrecursionlimit.
        # n_rows=4, k=8 -> depth 4*8=32. Safe.
        
        import sys
        sys.setrecursionlimit(10000)
        
        def backtrack(row_idx, pos_idx):
            # row_idx: which row we are filling (0..n_rows-1)
            # pos_idx: which position in the row (1..k, since 0 is fixed)
            
            if row_idx == n_rows:
                return True
            
            if pos_idx == 0:
                # Start new row
                rows.append([0])
                return backtrack(row_idx, 1)
            
            if pos_idx > k:
                # Row complete, move to next
                return backtrack(row_idx + 1, 0)
            
            current_row = rows[row_idx]
            last_val = current_row[-1]
            
            # Candidate values for current_row[pos_idx]
            # Must be > last_val
            # Must be <= M
            # Must not create duplicate differences with any a in current_row
            
            # Optimization: 
            # The forbidden values for x are {a + d | a in current_row, d in used}.
            # We iterate x from last_val + 1 to M.
            
            # To add randomness, we can shuffle the candidate range? 
            # Or just pick a random valid x?
            # Picking random valid x might skip good paths.
            # Better: iterate, but maybe with some randomization in order if M is large?
            # For now, deterministic ascending order is good for finding *a* solution quickly if it exists.
            # If we want to explore different solutions for the same M, we can randomize.
            
            # Let's try values in ascending order.
            # If we fail, we return False.
            
            # To improve performance, we can check if there are enough remaining differences?
            # Not easy to compute on the fly.
            
            for x in range(last_val + 1, M + 1):
                # Check conflicts
                valid = True
                for a in current_row:
                    d = x - a
                    if d in used:
                        valid = False
                        break
                    # Also d must be <= M (guaranteed since x<=M, a>=0)
                
                if not valid:
                    continue
                
                # Try x
                current_row.append(x)
                new_diffs = []
                for a in current_row[:-1]:
                    d = x - a
                    used.add(d)
                    new_diffs.append(d)
                
                if backtrack(row_idx, pos_idx + 1):
                    return True
                
                # Backtrack
                for d in new_diffs:
                    used.remove(d)
                current_row.pop()
            
            return False
        
        # Start
        rows = []
        used = set()
        
        # We need to handle the "start new row" logic in the initial call
        # Let's re-structure slightly for clarity
        # Actually, the logic above handles pos_idx=0 by appending [0] and calling pos_idx=1.
        # But it modifies 'rows' and 'used' which are in closure.
        
        # One issue: if we restart the search (randomized), we need fresh state.
        # This function creates fresh state.
        
        return backtrack(0, 0)

    # Strategy:
    # 1. Try M = total_diffs, total_diffs + 1, ... up to best_max.
    # 2. For each M, run the backtracking solver with a time limit.
    # 3. If found, update best_max and break (since we are looking for min M).
    # 4. If time runs out, return the best found so far (which might be the greedy one).
    
    # To increase the chance of finding a solution for a tight M, 
    # we can run the solver multiple times with different random seeds?
    # The current solver is deterministic (ascending order).
    # Deterministic backtracking is strong but can get stuck in bad branches.
    # Let's modify the solver to allow randomization of the value order.
    
    def solve_with_backtracking_random(M, time_limit, rng):
        rows = []
        used = set()
        
        def backtrack(row_idx, pos_idx):
            if row_idx == n_rows:
                return True
            
            if pos_idx == 0:
                rows.append([0])
                return backtrack(row_idx, 1)
            
            if pos_idx > k:
                return backtrack(row_idx + 1, 0)
            
            current_row = rows[row_idx]
            last_val = current_row[-1]
            
            # Generate candidates
            candidates = []
            for x in range(last_val + 1, M + 1):
                valid = True
                for a in current_row:
                    if x - a in used:
                        valid = False
                        break
                if valid:
                    candidates.append(x)
            
            if not candidates:
                return False
            
            # Randomize order to explore different branches
            rng.shuffle(candidates)
            
            for x in candidates:
                current_row.append(x)
                new_diffs = []
                for a in current_row[:-1]:
                    d = x - a
                    used.add(d)
                    new_diffs.append(d)
                
                if backtrack(row_idx, pos_idx + 1):
                    return True
                
                for d in new_diffs:
                    used.remove(d)
                current_row.pop()
            
            return False
        
        return backtrack(0, 0)

    # We will iterate M upwards.
    # Start from max(total_diffs, lower_bound)
    # We know best_max from greedy.
    # We try to beat it.
    
    start_m = total_diffs
    # If start_m > best_max, we can't improve.
    # But greedy might be bad.
    # Let's try from start_m to best_max - 1.
    
    # Time management:
    # We have ~145s.
    # We should allocate time to different M values.
    # Smaller M are harder (more constrained).
    # Let's give more time to smaller M? Or just iterate and stop if time is low.
    
    # Let's do a loop:
    # For M in range(start_m, best_max):
    #    if time_left < 10s: break
    #    Try solver for M with time_limit = min(5s, time_left/2)
    #    If success: best_max = M, break
    
    # To make it robust, we can also run the greedy with randomization to get a better initial upper bound.
    # But let's stick to the plan.
    
    # One optimization: 
    # If n_rows * k is small, the search space is manageable.
    # For (4, 8), total_diffs = 4 * 36 = 144.
    # M is around 144-160.
    # Search space is huge, but backtracking with forward checking is powerful.
    
    # Let's run the solver.
    # We need to capture the rows if successful.
    # The solver modifies 'rows' in the closure. We need to return it.
    
    def solve_and_get_rows(M, time_limit, rng):
        rows = []
        used = set()
        success = False
        
        def backtrack(row_idx, pos_idx):
            nonlocal success
            if time.time() > deadline - 5: # Safety break
                return False
                
            if row_idx == n_rows:
                success = True
                return True
            
            if pos_idx == 0:
                rows.append([0])
                return backtrack(row_idx, 1)
            
            if pos_idx > k:
                return backtrack(row_idx + 1, 0)
            
            current_row = rows[row_idx]
            last_val = current_row[-1]
            
            candidates = []
            for x in range(last_val + 1, M + 1):
                valid = True
                for a in current_row:
                    if x - a in used:
                        valid = False
                        break
                if valid:
                    candidates.append(x)
            
            if not candidates:
                return False
            
            rng.shuffle(candidates)
            
            for x in candidates:
                current_row.append(x)
                new_diffs = []
                for a in current_row[:-1]:
                    d = x - a
                    used.add(d)
                    new_diffs.append(d)
                
                if backtrack(row_idx, pos_idx + 1):
                    return True
                
                for d in new_diffs:
                    used.remove(d)
                current_row.pop()
            
            return False
        
        if backtrack(0, 0):
            return rows
        return None

    # Iterate M
    current_m = start_m
    # We might not want to go all the way to best_max if it's very large.
    # But we want the best possible.
    
    # To handle the time budget better, we can run multiple random restarts for a fixed M
    # before moving to the next M.
    
    # Let's say we have 100 seconds for search.
    # We try M = start_m. Give it 20 seconds. If fail, M = start_m + 1. Give it 20 seconds.
    # This might be too rigid.
    
    # Better: 
    # While time_left > 10s:
    #    if current_m >= best_max: break
    #    time_left_for_this_m = min(10.0, time_left / 2.0)
    #    rows = solve_and_get_rows(current_m, time_left_for_this_m, rng)
    #    if rows:
    #        best_max = current_m
    #        best_rows = rows
    #        break # Found smallest M, we are done? 
    #               # Actually, we want the smallest M. If we found one for current_m, 
    #               # we should check if we can find one for smaller M? 
    #               # No, we are iterating upwards from start_m. 
    #               # So the first success is the minimum M found.
    #               # However, it's possible that a solution exists for a smaller M 
    #               # that we missed due to time limits or search failure?
    #               # No, we start from start_m. If we succeed at start_m, that's the best.
    #               # If we fail at start_m, we try start_m+1.
    #               # So the first success is the optimal M (within the range we search).
    #    else:
    #        current_m += 1
    
    # Wait, if we fail at M due to time limit, we don't know if a solution exists.
    # So we should move to M+1.
    
    rng = random.Random(42)
    
    # We should also try to improve the greedy solution with SA if we have time?
    # The backtracking is a complete search (for a fixed M). 
    # If it finds a solution, it's valid.
    # The main risk is it taking too long for a specific M.
    
    # Let's refine the loop:
    # We will try each M from start_m to best_max - 1.
    # We allocate a fraction of the remaining time to each M.
    
    # Actually, if start_m is close to best_max, we have few M's to try.
    # If start_m is much smaller, we have many.
    
    # Let's just loop and check time.
    
    for m in range(start_m, best_max):
        if time.time() > deadline - 10:
            break
            
        # Give this M a chance. 
        # If we have lots of time, give it more.
        # If we are close to deadline, give it less.
        time_left = deadline - time.time()
        # Let's give 5 seconds per M attempt, or half the remaining time, whichever is smaller.
        # But if we have 100s left and 50 M's to try, 5s each is 250s. Too much.
        # So we should cap it.
        
        # Heuristic: 
        # If (best_max - start_m) is small, we can afford more time per M.
        # Let's say we want to finish before deadline - 5s.
        # num_m_to_try = best_max - m
        # time_per_m = (deadline - time.time() - 5) / num_m_to_try
        # Cap time_per_m to 20s?
        
        remaining_m = best_max - m
        time_per_m = (deadline - time.time() - 5.0) / max(1, remaining_m)
        time_per_m = min(20.0, time_per_m)
        time_per_m = max(1.0, time_per_m) # At least 1s
        
        # Run solver
        # We can run multiple random restarts within time_per_m
        # But the solver itself is a single DFS. 
        # To add diversity, we can run it multiple times with different seeds?
        # The internal shuffle uses rng. 
        # If we call it again, it uses the same rng state (continues).
        # To get a fresh start, we should create a new rng or reset it.
        
        # Let's just run it once with the current rng.
        # If it fails, we move to next M.
        
        rows = solve_and_get_rows(m, time_per_m, rng)
        
        if rows is not None:
            # Verify
            # (Solver ensures validity, but let's be safe)
            # Update best
            best_max = m
            best_rows = rows
            break # Found the smallest M in our search range.
            
    # If we didn't find anything better, return the greedy solution.
    # But wait, the greedy solution might have max > best_max if we updated best_max?
    # No, best_max was initialized to greedy's max.
    # If we break the loop without finding a solution, best_max is still the greedy one.
    # If we find a solution, best_max is updated to the smaller m.
    
    return best_rows
# EVOLVE-BLOCK-END
