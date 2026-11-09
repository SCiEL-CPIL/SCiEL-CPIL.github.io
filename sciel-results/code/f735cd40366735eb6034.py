import time
import sys

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = 115.0  # Safety margin before 120s kill
    
    # Target: differences 1..30
    target_diffs = set(range(1, 31))
    target_mask = 0
    for d in range(1, 31):
        target_mask |= (1 << d)
    
    best_solution = None
    best_max_elem = float('inf')
    
    # We will search for a Sidon set of size k that covers 1..30.
    # The minimum size for a Sidon set covering 1..30 is at least sqrt(30) ~ 6, but likely larger.
    # Actually, for a set of size n, the number of distinct differences is n(n-1)/2.
    # To cover 30 differences, we need n(n-1)/2 >= 30 => n ~ 8 or 9.
    # Let's try sizes 8, 9, 10, etc.
    
    # We'll use a backtracking search.
    # State: current set S, current used differences D, current mask M.
    # We build S in increasing order.
    
    def dfs(current_elements, current_mask, used_diffs, depth, max_depth):
        nonlocal best_solution, best_max_elem
        
        if time.time() > deadline:
            return
        
        # Pruning: if current max element is already >= best_max_elem, stop
        if current_elements and current_elements[-1] >= best_max_elem:
            return
            
        # Check if we have covered all target diffs
        if current_mask == target_mask:
            # We found a valid set.
            if current_elements[-1] < best_max_elem:
                best_max_elem = current_elements[-1]
                best_solution = list(current_elements)
            return
        
        # If we have reached max_depth, stop
        if depth >= max_depth:
            return
            
        # Heuristic: Try to add elements that cover many missing diffs.
        # But we need to ensure Sidon property.
        
        last_elem = current_elements[-1] if current_elements else -1
        
        # Upper bound for next element
        # If we have a best solution, we can prune if last_elem + 1 >= best_max_elem
        upper_bound = best_max_elem - 1 if best_solution else 100
        
        if upper_bound < last_elem + 1:
            return
            
        # Generate candidate x values
        # We can iterate x from last_elem + 1 to upper_bound
        # But we can be smarter: only consider x that covers at least one missing diff.
        # A missing diff d can be covered by x = a + d for some a in current_elements.
        # So x must be in {a + d | a in current_elements, d in missing_diffs}.
        
        missing_diffs = [d for d in range(1, 31) if not (current_mask & (1 << d))]
        
        candidates = set()
        for a in current_elements:
            for d in missing_diffs:
                x = a + d
                if x > last_elem:
                    candidates.add(x)
        
        # Also, we might need to add elements that don't cover any missing diff immediately
        # but help later? Unlikely for minimizing max element.
        # But let's also include last_elem + 1, last_elem + 2, ... up to some small range
        # to allow for "filler" elements if needed.
        # Actually, if we don't cover a missing diff, we might get stuck.
        # So restricting to candidates that cover at least one missing diff is a good heuristic.
        
        # Sort candidates by how many missing diffs they cover
        def score_candidate(x):
            count = 0
            for a in current_elements:
                d = x - a
                if 1 <= d <= 30 and not (current_mask & (1 << d)):
                    count += 1
            return count
        
        candidates = sorted(candidates, key=score_candidate, reverse=True)
        
        for x in candidates:
            if x > upper_bound:
                continue
                
            # Check Sidon property
            new_diffs = set()
            valid = True
            for a in current_elements:
                d = x - a
                if d in used_diffs or d in new_diffs:
                    valid = False
                    break
                new_diffs.add(d)
            
            if not valid:
                continue
                
            # Update mask
            new_mask = current_mask
            for d in new_diffs:
                if 1 <= d <= 30:
                    new_mask |= (1 << d)
            
            # Recurse
            dfs(current_elements + [x], new_mask, used_diffs.union(new_diffs), depth + 1, max_depth)
            
            if time.time() > deadline:
                return
                
            # Pruning: if we found a solution with max < x, we can break?
            # No, because a larger x might lead to a solution with smaller final max?
            # Unlikely, but possible.
            # However, if we found a solution, best_max_elem is updated.
            # The loop condition `upper_bound` uses best_max_elem, so it will shrink.
            
    # Start with [0]
    # Try different max depths
    for max_depth in range(8, 13):
        dfs([0], 0, set(), 0, max_depth)
        if best_solution:
            break
            
    if best_solution:
        return best_solution
    
    # Fallback: Greedy construction
    a = [0]
    def get_diffs(s):
        return [y - x for i, x in enumerate(s) for y in s[i + 1:]]
    for d in range(1, 31):
        if d in get_diffs(sorted(a)):
            continue
        found = False
        candidates = set()
        for v in a:
            if v + d not in a:
                candidates.add(v + d)
            if v - d >= 0 and v - d not in a:
                candidates.add(v - d)
        
        for x in sorted(candidates):
            trial = sorted(a + [x])
            d_list = get_diffs(trial)
            if len(set(d_list)) == len(d_list):
                a = trial
                found = True
                break
        if not found:
            y = 2 * max(a) + 3 * d + 1
            while True:
                trial = sorted(a + [y, y + d])
                d_list = get_diffs(trial)
                if len(set(d_list)) == len(d_list):
                    a = trial
                    break
                y += 1
    return a

# EVOLVE-BLOCK-END
