import time
import random
import sys

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 145.0  # 5 seconds margin
    
    # Required differences
    req_diffs = set(range(1, 31))
    
    best_solution = None
    best_max_val = float('inf')
    
    # Lower bound for k: k(k-1)/2 >= 30 => k >= 9
    # We will try k from 9 to 32
    for k in range(9, 33):
        if time.time() > deadline - 5:
            break
            
        # Backtracking search for a set of size k
        # We want to find a set with minimal max element.
        # We will do a DFS. To find minimal max element, we can't just stop at the first solution
        # if we want the absolute minimum, but the first solution found with increasing x
        # tends to be small.
        # However, to be robust, we can try to find ANY solution first, then try to improve it?
        # Or just let the DFS run with a time limit per k.
        
        # We will use a recursive function
        def solve(current_set, used_diffs, remaining_req):
            nonlocal best_solution, best_max_val
            nonlocal deadline
            
            if time.time() > deadline - 2:
                return False
            
            if len(current_set) == k:
                if not remaining_req:
                    # Found a valid solution
                    max_val = max(current_set)
                    if max_val < best_max_val:
                        best_max_val = max_val
                        best_solution = list(current_set)
                    return True
                return False
            
            # Pruning: Can we cover remaining_req with remaining slots?
            slots_left = k - len(current_set)
            n_current = len(current_set)
            # Max new diffs: slots_left * n_current + slots_left * (slots_left - 1) // 2
            max_new_diffs = slots_left * n_current + slots_left * (slots_left - 1) // 2
            if len(remaining_req) > max_new_diffs:
                return False
            
            # Heuristic: Try to add elements that cover many required diffs
            # But we also want small max element.
            # Let's try candidates in increasing order.
            # To limit search space, we can set an upper bound for the next element.
            # A rough bound: if we have slots_left elements left, and we need to cover
            # diffs up to 30, the elements shouldn't be too far apart.
            # Let's just try x from max(current_set) + 1 up to some limit.
            # If k is small, the max element won't be huge.
            # Let's start with a limit of 1000 for the next element to keep it fast.
            # If we fail, we can increase the limit or rely on the fact that
            # small solutions are likely to have small elements.
            
            max_curr = max(current_set)
            
            # Optimization: Generate candidate x values that create at least one new required diff
            # Or just iterate.
            
            # To speed up, we can try to pick x such that x - s in remaining_req for some s in current_set
            # or x - x' in remaining_req for some x' in current_set (if we add multiple, but we add one at a time)
            # Actually, adding x creates diffs x - s for s in current_set.
            
            candidates = set()
            # Try to cover specific remaining diffs
            for d in remaining_req:
                for s in current_set:
                    x = s + d
                    if x > max_curr:
                        candidates.add(x)
            
            # Also try some small increments to maintain density
            # If candidates is empty, we might be stuck, but we should have added enough.
            # If no candidates cover a req diff, we might still need to add elements to satisfy Sidon
            # or to enable future diffs.
            # But if we just add x that doesn't cover any req diff, we might waste space.
            # Let's prioritize candidates that cover req diffs.
            
            # Sort candidates to try smaller ones first
            sorted_candidates = sorted(candidates)
            
            # If no candidates, try max_curr + 1, max_curr + 2, ...
            # This can be slow. Let's limit the range.
            # If len(candidates) == 0, it means no single step covers a req diff.
            # This is possible if remaining_req are large and current_set is small.
            # e.g. current_set={0}, remaining_req={30}. x=30 is a candidate.
            # So candidates should not be empty if remaining_req is not empty and we can form x=s+d.
            # Wait, if d=30, s=0, x=30. Yes.
            
            # However, we must also ensure Sidon property.
            # We check Sidon property inside the loop.
            
            # To avoid missing solutions where we need to add an element that doesn't immediately
            # cover a required diff but sets up for later?
            # In a Sidon set, every pair is unique.
            # If we don't cover a req diff now, we must cover it later.
            # It is generally safe to prioritize moves that cover req diffs.
            
            # Let's also add a few "neutral" moves if candidates are few?
            # No, let's stick to covering req diffs first.
            
            # If candidates is empty, we are in trouble.
            # But theoretically, if remaining_req is not empty, there should be candidates.
            # Unless all s+d are <= max_curr?
            # If s+d <= max_curr, it's not a new element (since we add x > max_curr).
            # So if all s+d <= max_curr, we can't cover d with a single new element x > max_curr?
            # Yes, because x - s = d => x = s + d. If x <= max_curr, it's already in the set or invalid.
            # If x is in current_set, diff is already used.
            # If x < max_curr, we can't add it (order).
            # So if no candidate x > max_curr covers any d in remaining_req, we cannot make progress
            # on covering those diffs with a single new element?
            # Wait, we could add x such that x - x2 = d where x2 is also new?
            # But we add one by one.
            # So if we can't cover any remaining diff with a new element x > max_curr,
            # we might be stuck.
            # However, we can still add elements that don't cover remaining diffs,
            # but that seems suboptimal for minimizing max element.
            # Let's assume we can always find a candidate.
            
            if not sorted_candidates:
                # Fallback: try max_curr + 1, ...
                # This might be necessary if the required diffs are "far"
                # but actually, if d is in remaining_req, s=0 is in current_set,
                # so x=d is a candidate. If d > max_curr, it's valid.
                # If d <= max_curr, it means we missed it?
                # No, if d is in remaining_req, it's not covered.
                # If d <= max_curr, and 0 is in set, then d-0=d should have been covered
                # if d was generated.
                # So if d is in remaining_req, it means no pair (a,b) in current_set has b-a=d.
                # So for any s in current_set, s+d is not in current_set (as a diff endpoint? No).
                # s+d is a value. If s+d is in current_set, then diff d exists.
                # So s+d is NOT in current_set.
                # If s+d > max_curr, it's a valid candidate.
                # If s+d <= max_curr, it's a value in the range, but not in the set.
                # We can only add x > max_curr.
                # So if all s+d <= max_curr, we cannot cover d with a single new element?
                # Correct. We would need to add an element x such that x - y = d where y is also new?
                # But we add one at a time.
                # So if we can't cover d now, we can't cover it at all with future single additions?
                # No, we can add x1, then x2. x2 - x1 = d.
                # But x1 is added first.
                # So if we can't cover d with x1 - s, we might cover it with x2 - x1.
                # This implies we need to add "stepping stone" elements.
                # This makes the search harder.
                # However, for small k and small diffs, direct coverage is likely.
                
                # Let's just try a few small values to allow stepping stones.
                for x in range(max_curr + 1, max_curr + 10):
                    if x in current_set: continue
                    # Check Sidon
                    valid = True
                    new_diffs = []
                    for s in current_set:
                        d_val = x - s
                        if d_val in used_diffs:
                            valid = False
                            break
                        new_diffs.append(d_val)
                    
                    if not valid:
                        continue
                    
                    # Check if any new diff is in remaining_req
                    # Even if not, we might need it.
                    
                    new_used = used_diffs | set(new_diffs)
                    new_remaining = remaining_req - set(new_diffs)
                    
                    if solve(current_set | {x}, new_used, new_remaining):
                        return True
            
            else:
                for x in sorted_candidates:
                    if x in current_set:
                        continue
                    # Check Sidon property
                    valid = True
                    new_diffs = []
                    for s in current_set:
                        d_val = x - s
                        if d_val in used_diffs:
                            valid = False
                            break
                        new_diffs.append(d_val)
                    
                    if not valid:
                        continue
                    
                    new_used = used_diffs | set(new_diffs)
                    new_remaining = remaining_req - set(new_diffs)
                    
                    if solve(current_set | {x}, new_used, new_remaining):
                        return True
            
            return False

        # Start search
        # We initialize with {0}
        # To increase chance of finding small max, we can try to restart with different heuristics?
        # For now, just one run per k.
        # We can shuffle the order of candidates slightly to avoid getting stuck in a local minimum
        # of "bad" small numbers?
        # Actually, trying smallest first is good for minimizing max element.
        
        # We will run the solver.
        # To avoid spending all time on k=9 if it's hard, we set a per-k time limit?
        # The global deadline is handled in solve().
        
        # Initial state
        initial_set = {0}
        initial_used = set()
        initial_remaining = req_diffs.copy()
        
        # We call solve. It returns True if found.
        # We might want to try multiple times with slight randomization if it fails quickly?
        # But deterministic search is reproducible.
        
        # Let's add a small randomization to candidate selection to escape bad paths?
        # If we just sort, we might always go down the same path.
        # Let's add a random tie-breaker or slight shuffle.
        
        # Actually, the current logic picks candidates based on covering diffs.
        # If multiple x cover the same diff, we pick smallest.
        
        # Let's just run it.
        solve(initial_set, initial_used, initial_remaining)
        
        if best_solution is not None:
            # We found a solution for this k.
            # Since we iterate k upwards, and for each k we find a solution,
            # the first k that yields a solution is the minimal k?
            # Not necessarily minimal max element.
            # A solution with k=10 might have smaller max element than k=9?
            # Unlikely, but possible.
            # We should probably check a few k values or just take the best found so far.
            # Since we want to minimize max element, and k is a constraint "at most 32",
            # we should check all k or at least a range.
            # But larger k allows more flexibility, potentially smaller diameter?
            # No, more elements means more differences, harder to keep Sidon in small range.
            # Usually smaller k is easier to pack?
            # Let's just keep the global best.
            
            # We can break if we are confident k=9 is optimal?
            # Let's run for a few k's.
            # If we find a solution with max_val < 100, it's likely good.
            
            # To be safe, we continue to next k only if we haven't found a "very good" solution?
            # Or just let the time limit handle it.
            
            # If we found a solution, we can try to improve it with local search?
            # For now, just record it.
            
            # If best_max_val is small, we can stop early.
            if best_max_val < 50:
                break

    if best_solution is None:
        # Fallback to the original greedy if search failed (should not happen)
        a = [0]
        def diffs(s):
            return [y - x for i, x in enumerate(s) for y in s[i + 1:]]
        for d in range(1, 31):
            if d in diffs(sorted(a)):
                continue
            for x in sorted(set(v + s for v in a for s in (d, -d) if v + s >= 0) - set(a)):
                trial = sorted(a + [x])
                if len(set(diffs(trial))) == len(diffs(trial)):
                    a = trial
                    break
            else:
                y = 2 * max(a) + 3 * d + 1
                while True:
                    trial = sorted(a + [y, y + d])
                    if len(set(diffs(trial))) == len(diffs(trial)):
                        a = trial
                        break
                    y += 1
        return a

    return best_solution

# EVOLVE-BLOCK-END
