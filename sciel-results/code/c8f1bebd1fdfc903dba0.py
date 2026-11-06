import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    # We need a Sidon set containing 0, with all differences in 1..30 present,
    # at most 32 elements, minimizing the largest element.
    #
    # Strategy: DFS building the set element by element (sorted, starting with 0),
    # tracking which differences have been seen. Prune aggressively.
    
    target_diffs = set(range(1, 31))
    min_size = 9  # C(9,2)=36 >= 30, so we need at least 9 elements
    max_size = 32
    
    best_result = None
    best_max_elem = float('inf')
    
    def dfs(elements, seen_diffs, covered_diffs, max_elem_bound):
        nonlocal best_result, best_max_elem, deadline
        
        if time.time() > deadline:
            return
        
        # If we've covered all required differences, check if this is a valid complete set
        if covered_diffs == target_diffs:
            # This is a valid solution
            if len(elements) <= max_size:
                max_elem = elements[-1]
                if max_elem < best_max_elem:
                    best_max_elem = max_elem
                    best_result = list(elements)
            return
        
        # Pruning: if even with max_size elements we can't cover remaining diffs
        remaining_needed = target_diffs - covered_diffs
        if len(remaining_needed) == 0:
            return
        
        # If we already have too many elements
        if len(elements) >= max_size:
            return
        
        # Lower bound on max element: we need to be able to form all remaining diffs
        # The largest remaining diff must be formable, so max_elem must be at least that
        max_remaining_diff = max(remaining_needed) if remaining_needed else 0
        current_max = elements[-1]
        
        # We need at least one more element if we haven't covered all diffs
        if len(elements) < max_size:
            # The next element x must satisfy: for all existing e, x-e and e-x (if positive) 
            # must not be in seen_diffs, and we want to cover remaining diffs
            pass
        
        # Try adding a new element
        # The new element must be > current_max (since we build in sorted order)
        # It must not create a duplicate difference
        # It should help cover remaining differences
        
        # Upper bound for next element: if best_result exists, don't exceed it
        upper_bound = best_max_elem - 1 if best_result else 1000000
        
        # Also, the next element can't be so large that we can't fit enough more elements
        # to cover remaining diffs within max_size
        
        # Determine range of possible next elements
        # Minimum: current_max + 1
        # But we can be smarter: the next element x, when combined with existing elements,
        # produces differences x - e for e in elements. We want at least one of these to be in remaining_needed.
        
        # For efficiency, try candidate x values that would cover at least one remaining diff
        candidate_xs = set()
        for e in elements:
            for d in remaining_needed:
                x = e + d
                if x > current_max and x <= upper_bound:
                    candidate_xs.add(x)
        
        # Also try some values that don't immediately cover a diff but might be useful
        # Actually, for a tight packing, every new element should cover at least one new diff
        # Let's sort candidates and try them
        
        if not candidate_xs:
            return
        
        # Sort candidates - try smaller ones first to find tighter packings
        for x in sorted(candidate_xs):
            if time.time() > deadline:
                return
            
            # Check Sidon property: x - e for all e in elements must not be in seen_diffs
            valid = True
            new_diffs = set()
            for e in elements:
                d = x - e
                if d <= 0:
                    continue
                if d in seen_diffs:
                    valid = False
                    break
                new_diffs.add(d)
            
            if not valid:
                continue
            
            # Check that new_diffs don't overlap with each other (they can't since they're from different e)
            # Actually they're all distinct by construction (different e give different x-e)
            
            new_covered = covered_diffs | (new_diffs & target_diffs)
            
            # Pruning: check if remaining diffs can still be covered
            # With len(elements)+1 elements, we have (len(elements)+1)*(len(elements))/2 total diffs
            # But we've used len(seen_diffs) so far
            # Remaining capacity: total possible diffs - used diffs
            # This is a weak bound, let's use a stronger one
            
            # Stronger bound: the number of additional differences we can still create
            # with up to (max_size - len(elements) - 1) more elements
            # Each new element y adds (len(elements)+1) new differences (y - e for each e)
            # But some of those might already be in seen_diffs or > 30
            # This is complex, so let's just proceed with the search
            
            dfs(elements + [x], seen_diffs | new_diffs, new_covered, max_elem_bound)
    
    # Start with [0]
    dfs([0], set(), set(), 1000000)
    
    if best_result is not None:
        return best_result
    else:
        # Fallback: use the greedy approach from the seed
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
