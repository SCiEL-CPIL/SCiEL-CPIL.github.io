import time
import random

def construct():
    # Target: maximize size of set S subset of 1..256
    # Constraint: all non-empty subset sums are distinct
    # Constraint: no two distinct subset sums divide each other
    
    # Greedy heuristic:
    # Start with empty set.
    # Try adding candidates.
    # To speed up, we can pre-filter candidates.
    
    # A set of integers is "sum-distinct" if all subset sums are distinct.
    # A set is "divisibility-incomparable" if no two distinct subset sums divide each other.
    
    # We will try to build the set greedily.
    # Order of candidates: Try random permutations or sorted by some metric.
    # Let's try sorted descending, as larger numbers tend to create larger sums,
    # making it harder for a sum to divide another (since b >= 2a).
    
    candidates = list(range(256, 0, -1))
    
    best_set = []
    best_size = 0
    
    # Time limit guard
    start_time = time.time()
    deadline = start_time + 50.0  # 50s limit, leaving 10s margin
    
    # Try multiple random restarts with different candidate orders
    for restart in range(100):
        if time.time() > deadline:
            break
            
        # Shuffle candidates for diversity in restarts
        # Or use a specific heuristic
        curr_candidates = candidates[:]
        random.shuffle(curr_candidates)
        
        current_set = []
        current_sums = set()
        
        for x in curr_candidates:
            if time.time() > deadline:
                break
                
            # Calculate new sums
            # new_sums = {s + x for s in current_sums} U {x} U current_sums
            # But we need to check distinctness first.
            
            # If x is already in current_set, skip (distinct integers required)
            # x is from candidates, so it's distinct from current_set if we manage it.
            
            # Check distinctness:
            # The new sums are current_sums and {s+x for s in current_sums} and {x}.
            # For sums to be distinct, {s+x for s in current_sums} must be disjoint from current_sums and {x}.
            # And x must not be in current_sums.
            
            if x in current_sums:
                continue
                
            new_sums_part = set()
            valid = True
            for s in current_sums:
                ns = s + x
                if ns in current_sums or ns == x:
                    valid = False
                    break
                new_sums_part.add(ns)
            
            if not valid:
                continue
            
            # Check distinctness of new_sums_part internally?
            # If s1 != s2, then s1+x != s2+x. So new_sums_part is distinct internally.
            # We just need to ensure new_sums_part doesn't intersect with current_sums or {x}.
            # We already checked ns in current_sums and ns == x.
            
            # Now check divisibility condition.
            # New set of sums: all_sums = current_sums U new_sums_part U {x}
            # We need to check if any a, b in all_sums (a != b) divide each other.
            
            # Optimization:
            # We only need to check pairs involving the new elements (new_sums_part and x)
            # against each other and against current_sums.
            # Pairs within current_sums are already valid.
            # Pairs within new_sums_part:
            #   If a, b in new_sums_part, a = s1+x, b = s2+x.
            #   If a | b, then s1+x | s2+x.
            #   This is possible. We must check.
            
            # To keep it fast, we can collect all new sums and check them against the full set.
            # But full set size grows.
            
            # Let's form the list of new sums to check:
            # new_elements = list(new_sums_part) + [x]
            
            # For each new element n in new_elements:
            #   Check if n divides any existing sum in current_sums.
            #   Check if any existing sum in current_sums divides n.
            #   Check if n divides any other new element m.
            #   Check if any other new element m divides n.
            
            new_elements = list(new_sums_part) + [x]
            
            # Check against current_sums
            for n in new_elements:
                for s in current_sums:
                    if n == s: continue # already checked distinctness
                    if n > s:
                        if n % s == 0:
                            valid = False
                            break
                    else:
                        if s % n == 0:
                            valid = False
                            break
                if not valid:
                    break
            
            if not valid:
                continue
            
            # Check within new_elements
            for i in range(len(new_elements)):
                for j in range(i+1, len(new_elements)):
                    a = new_elements[i]
                    b = new_elements[j]
                    if a > b:
                        if a % b == 0:
                            valid = False
                            break
                    else:
                        if b % a == 0:
                            valid = False
                            break
                if not valid:
                    break
            
            if not valid:
                continue
            
            # If we reach here, x is a valid addition
            current_set.append(x)
            current_sums.update(new_elements)
            
            if len(current_set) > best_size:
                best_size = len(current_set)
                best_set = current_set[:]
                
        # End candidate loop
        # End restart loop

    # Return best set found
    # Ensure types are Python int
    return [int(x) for x in best_set]
