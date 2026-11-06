import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Prescribed elements
    fixed_elements = [1, 2, 5, 11]
    fixed_set = set(fixed_elements)
    
    # Pre-compute all possible candidates (1..284)
    all_candidates = list(range(1, 285))
    
    # Initial state with fixed elements
    def get_sums(a_list):
        s = set()
        for i, x in enumerate(a_list):
            for y in a_list[i:]:
                s.add(x + y)
        return s
    
    # Try multiple random seeds for CSP backtracking
    best_size = len(fixed_elements)
    best_set = set(fixed_elements)
    
    # CSP Backtracking with Forward Checking and MRV
    def csp_backtrack(selected, used_sums, remaining, target_size):
        """
        Backtracking search to find a Sidon set of at least target_size elements.
        selected: set of currently selected elements
        used_sums: set of all pairwise sums a+b (a<=b) for elements in selected
        remaining: list of candidate elements not yet considered
        target_size: minimum size we want to achieve
        
        Returns: True if we found a valid set of size >= target_size, else False
        """
        if len(selected) >= target_size:
            return True
        
        if not remaining:
            return False
        
        # MRV heuristic: pick the candidate with fewest conflicts
        # For each candidate, count how many remaining elements it conflicts with
        # We'll use a simpler approach: try candidates in order of least conflicts
        
        # Build conflict count for each remaining candidate
        def count_conflicts(x):
            count = 0
            for y in remaining:
                if y == x:
                    continue
                # Check if x and y can both be in the set
                if (x + y) in used_sums:
                    count += 1
                # Also check if adding x would block y or vice versa
                # This is approximate; we'll refine in the actual check
            return count
        
        # Sort remaining by conflict count (ascending) - MRV
        # But this is expensive; use a heuristic instead
        # Try candidates that have fewer potential conflicts with already selected
        
        # Simple heuristic: try candidates that are "compatible" with most of selected
        # Actually, let's just try in a smart order
        
        # For efficiency, we'll iterate through remaining and try to add
        # Use a priority: prefer candidates that don't conflict with many others
        
        # To keep it simple and fast, we'll use a greedy-like ordering with backtracking
        # Try each candidate in remaining, but order by how "compatible" they are
        
        # Compute compatibility score: number of existing elements they can coexist with
        # Actually, they must coexist with ALL existing elements by definition of used_sums
        
        # Let's just try in a randomized order with some bias toward smaller numbers
        # (smaller numbers tend to have more room)
        
        # Create a list of (candidate, priority) and sort
        # Priority: lower is better. Use negative of "freedom"
        # Freedom = number of sums that would be used if we add this candidate
        # Lower freedom = more restrictive = should try later? Or earlier?
        # MRV says: pick the variable with fewest remaining values
        # Here, all candidates have the same "domain" (can be added or not)
        # So we use a different heuristic: pick the one that constrains the most
        # i.e., the one that eliminates the most other candidates
        
        # This is getting complex. Let's use a simpler approach:
        # Try candidates in increasing order, but with some randomization
        
        # For performance, we'll limit the branching factor
        # Try up to 5 candidates at each level
        
        # Get candidates that can actually be added
        addable = []
        for x in remaining:
            # Check if x can be added
            can_add = True
            for y in selected:
                if (x + y) in used_sums:
                    can_add = False
                    break
            if can_add and (2 * x) in used_sums:
                can_add = False
            if can_add:
                addable.append(x)
        
        if not addable:
            return False
        
        # MRV: pick the candidate that eliminates the most other candidates
        # Compute for each addable candidate, how many other addable candidates it would eliminate
        def elimination_count(x):
            count = 0
            for y in addable:
                if y == x:
                    continue
                # If we add x, can y still be added?
                # y can be added if (x+y) not in used_sums and (y+y) not in used_sums
                # But we're checking if adding x blocks y
                if (x + y) in used_sums:
                    count += 1
            return count
        
        # Sort by elimination count (descending) - most constraining first
        addable.sort(key=lambda x: elimination_count(x), reverse=True)
        
        # Try each addable candidate
        for x in addable:
            if time.time() > deadline:
                return False
            
            # Add x
            new_selected = set(selected)
            new_selected.add(x)
            
            # Compute new sums
            new_sums = set(used_sums)
            for y in selected:
                new_sums.add(x + y)
            new_sums.add(2 * x)
            
            # Update remaining: remove x and any that conflict with x
            new_remaining = []
            for y in remaining:
                if y == x:
                    continue
                # Check if y can still be added after adding x
                can_add_y = True
                for z in new_selected:
                    if (y + z) in new_sums:
                        can_add_y = False
                        break
                if can_add_y and (2 * y) in new_sums:
                    can_add_y = False
                if can_add_y:
                    new_remaining.append(y)
            
            # Recurse
            if csp_backtrack(new_selected, new_sums, new_remaining, target_size):
                # Update best if we found something
                if len(new_selected) > best_size:
                    best_size = len(new_selected)
                    best_set = set(new_selected)
                return True
        
        return False
    
    # Strategy: try to find sets of increasing size
    # Start from size 5 (fixed + 1) and go up to 20
    
    # First, do a quick greedy to get a good initial solution
    def greedy_construct():
        selected = set(fixed_elements)
        used_sums = get_sums(list(fixed_elements))
        candidates = [x for x in all_candidates if x not in selected]
        random.shuffle(candidates)
        
        for x in candidates:
            if time.time() > deadline:
                break
            can_add = True
            for y in selected:
                if (x + y) in used_sums:
                    can_add = False
                    break
            if can_add and (2 * x) in used_sums:
                can_add = False
            if can_add:
                selected.add(x)
                for y in selected:
                    used_sums.add(x + y)
        
        return selected
    
    # Get initial solution via greedy
    initial_set = greedy_construct()
    if len(initial_set) > best_size:
        best_size = len(initial_set)
        best_set = set(initial_set)
    
    # Now try CSP backtracking to improve
    # Try to find sets of size best_size + 1, best_size + 2, etc.
    
    # For CSP, we need to be careful about time
    # We'll try a few random restarts
    
    num_restarts = 5
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Randomize the order of candidates
        random_candidates = list(all_candidates)
        random.shuffle(random_candidates)
        
        # Try to find a set of size best_size + 1
        target = best_size + 1
        
        # Use CSP with the randomized candidate order
        # We need to modify csp_backtrack to accept a candidate order
        
        # For simplicity, we'll use the greedy result as a starting point
        # and try to swap elements
        
        # Alternative: use local search with kicks
        # This is what the original code did, but let's try a different approach
        
        # Try a different greedy with different random seed
        random.seed(restart * 12345 + 67890)
        selected = set(fixed_elements)
        used_sums = get_sums(list(fixed_elements))
        
        # Use a smarter greedy: at each step, pick the candidate that leaves the most options
        remaining = [x for x in all_candidates if x not in selected]
        
        for step in range(20):  # Try to add up to 20 elements
            if time.time() > deadline:
                break
            
            if not remaining:
                break
            
            # For each candidate, count how many other candidates it would eliminate
            # Pick the one that eliminates the fewest (most "compatible")
            
            best_candidate = None
            best_score = -1
            
            # Sample a subset of candidates for efficiency
            sample_size = min(len(remaining), 50)
            sample = random.sample(remaining, sample_size)
            
            for x in sample:
                # Check if x can be added
                can_add = True
                for y in selected:
                    if (x + y) in used_sums:
                        can_add = False
                        break
                if can_add and (2 * x) in used_sums:
                    can_add = False
                if not can_add:
                    continue
                
                # Count how many other candidates x would eliminate
                elimination = 0
                for y in remaining:
                    if y == x:
                        continue
                    # Check if y would be eliminated by adding x
                    # y is eliminated if (x+y) in used_sums or if adding x blocks y
                    # Simplified: check if (x+y) in used_sums
                    if (x + y) in used_sums:
                        elimination += 1
                
                score = -elimination  # Lower elimination is better
                if score > best_score:
                    best_score = score
                    best_candidate = x
            
            if best_candidate is None:
                break
            
            # Add best_candidate
            selected.add(best_candidate)
            for y in selected:
                used_sums.add(best_candidate + y)
            
            # Update remaining
            new_remaining = []
            for y in remaining:
                if y == best_candidate:
                    continue
                can_add = True
                for z in selected:
                    if (y + z) in used_sums:
                        can_add = False
                        break
                if can_add and (2 * y) in used_sums:
                    can_add = False
                if can_add:
                    new_remaining.append(y)
            remaining = new_remaining
        
        if len(selected) > best_size:
            best_size = len(selected)
            best_set = set(selected)
    
    return list(best_set)

if __name__ == "__main__":
    # Quick self-test
    res = construct()
    print(f"Found size: {len(res)}")
    # Verify
    s = set()
    valid = True
    for i in range(len(res)):
        for j in range(i, len(res)):
            sm = res[i] + res[j]
            if sm in s:
                valid = False
            s.add(sm)
    print(f"Valid: {valid}")
    print(f"Contains fixed: {set([1,2,5,11]).issubset(set(res))}")
