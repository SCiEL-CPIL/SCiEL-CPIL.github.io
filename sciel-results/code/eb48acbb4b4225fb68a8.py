import random
import time

N = 256

def _is_sidon(sel):
    """Check if sel is a Sidon set."""
    sums = set()
    for i in range(len(sel)):
        a = sel[i]
        for j in range(i, len(sel)):
            s = a + sel[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _is_maximal(sel, N):
    """Check if sel is maximal in [1, N]."""
    sums = set()
    for i in range(len(sel)):
        a = sel[i]
        for j in range(i, len(sel)):
            sums.add(a + sel[j])
    
    sel_set = set(sel)
    for x in range(1, N + 1):
        if x in sel_set:
            continue
        # Check if x can be added
        if 2 * x in sums:
            continue
        can_add = True
        for a in sel:
            if x + a in sums:
                can_add = False
                break
        if can_add:
            return False
    return True

def _blocking_heuristic_greedy(deadline, randomize_tie_break=True):
    """
    Greedy construction using a blocking heuristic.
    At each step, pick the element that invalidates the most other candidates.
    """
    valid = set(range(1, N + 1))
    sums = set()
    sel = []
    
    while valid:
        if time.time() > deadline:
            break
        
        best_x = None
        best_score = -1
        best_ties = []
        
        # To speed up, we might not need to check ALL valid elements if the set is large,
        # but N=256 is small enough.
        # We iterate through valid candidates.
        # Optimization: Instead of iterating all, we can just pick one. 
        # But to maximize blocking, we need to score them.
        
        # Shuffling valid helps with randomization if we just pick the first best.
        # Or we collect top K.
        
        candidates = list(valid)
        if randomize_tie_break:
            random.shuffle(candidates)
            
        for x in candidates:
            # Calculate blocking score
            # New sums added: 2x, x+a for a in sel
            new_sums_to_check = {2 * x}
            for a in sel:
                new_sums_to_check.add(x + a)
                
            # Count how many y in valid (y != x) become invalid
            # y becomes invalid if 2y in new_sums_to_check U sums 
            # OR y+a in new_sums_to_check U sums for some a in sel U {x}
            
            # Since y is currently valid, 2y not in sums, and y+a not in sums for a in sel.
            # So y becomes invalid ONLY if:
            # 1. 2y is in new_sums_to_check
            # 2. y+a is in new_sums_to_check for some a in sel
            # 3. y+x is in new_sums_to_check (which is x+y, same as 2)
            # Note: new_sums_to_check contains 2x and x+a.
            # So we check if 2y == 2x (no) or 2y == x+a.
            # And if y+a == 2x or y+a == x+b.
            
            score = 0
            # Pre-compute sums of sel for faster lookup? 
            # sums is already a set.
            
            for y in valid:
                if y == x:
                    continue
                # Check if y becomes invalid
                # Case 1: 2y collides with new sums
                if 2 * y in new_sums_to_check:
                    score += 1
                    continue # Already blocked, no need to check other collisions for count? 
                             # We just want the count of blocked elements.
                # Case 2: y+a collides with new sums
                # y+a must be in {2x} U {x+b | b in sel}
                # So y+a == 2x  => y == 2x - a
                # OR y+a == x+b => y == x + b - a
                
                # Efficient check:
                # Is 2y in new_sums_to_check?
                # Is any y+a in new_sums_to_check?
                
                # Let's just check membership in the union of sums and new sums?
                # No, we only care if it *becomes* invalid.
                
                # y is valid now, so y+a not in sums.
                # So we only check against new_sums_to_check.
                
                blocked = False
                # Check 2y
                if 2 * y in new_sums_to_check:
                    blocked = True
                else:
                    # Check y+a for a in sel
                    # y+a in new_sums_to_check
                    # new_sums_to_check is small (size 1 + len(sel))
                    # We can iterate new_sums_to_check and see if y+a is in it?
                    # Or iterate sel and check if y+a in new_sums_to_check.
                    # len(sel) is small.
                    for a in sel:
                        if y + a in new_sums_to_check:
                            blocked = True
                            break
                    # Also need to check if y+x is in new_sums_to_check?
                    # y+x is x+y. 
                    # x+y is in new_sums_to_check if x+y == 2x (y=x, skipped) 
                    # or x+y == x+b (y=b, but b in sel, so y in sel, but y in valid, so y not in sel).
                    # So y+x is NOT in new_sums_to_check unless y=x.
                    # Wait, new_sums_to_check contains x+a for a in sel.
                    # If y is in valid, y is not in sel.
                    # So x+y is not in {x+a | a in sel} unless y=a which is false.
                    # So we don't need to check y+x specifically against new_sums_to_check 
                    # unless we consider that adding x makes x+y a sum.
                    # Does adding x make x+y a sum? Yes.
                    # Does that block y?
                    # y is blocked if x+y collides with an EXISTING sum or NEW sum?
                    # No, y is blocked if 2y or y+a (for a in S_new) is in Sums_new.
                    # Sums_new = Sums_old U {2x} U {x+a | a in S_old}.
                    # For y to be blocked, 2y or y+a' must be in Sums_new.
                    # We already checked 2y in {2x} U {x+a}.
                    # We checked y+a (a in S_old) in {2x} U {x+b}.
                    # What about y+x?
                    # y+x is a sum in S_new.
                    # Does y+x being in S_new block y?
                    # No, the condition for y to be addable is that 2y and y+a (a in S) are NOT in Sums.
                    # If we add x, S becomes S U {x}.
                    # So we check if 2y is in Sums_new.
                    # And if y+a is in Sums_new for all a in S U {x}.
                    # We checked a in S.
                    # We need to check a=x.
                    # Is y+x in Sums_new?
                    # Sums_new contains 2x and x+a (a in S).
                    # Does Sums_new contain y+x?
                    # Only if y+x == 2x (y=x) or y+x == x+a (y=a).
                    # Since y != x and y not in S, y+x is NOT in Sums_new.
                    # So y+x does NOT cause a collision with Sums_new.
                    # Thus, y+x being a sum in the new set does not block y, 
                    # UNLESS y+x collides with some OTHER sum in Sums_new?
                    # No, the check is: is the sum value in the set of sums?
                    # The sum value is y+x.
                    # Is y+x in Sums_new?
                    # We just said No.
                    # So y+x does not block y.
                    
                    # Wait, my logic above "y is blocked if y+a is in Sums_new" is correct.
                    # For a=x, sum is y+x.
                    # Is y+x in Sums_new?
                    # Sums_new = Sums_old U {2x} U {x+a | a in S_old}.
                    # If y+x is in Sums_old, y would have been blocked already.
                    # If y+x == 2x, y=x.
                    # If y+x == x+a, y=a.
                    # So y+x is NOT in Sums_new.
                    # So adding x does NOT block y via the sum y+x.
                    
                    # Therefore, the only way x blocks y is if:
                    # 1. 2y is in {2x} U {x+a}
                    # 2. y+a is in {2x} U {x+b} for some a,b in S_old.
                    
                    # My loop `for a in sel: if y+a in new_sums_to_check` covers case 2.
                    # My check `if 2*y in new_sums_to_check` covers case 1.
                    
                if blocked:
                    score += 1
            
            if score > best_score:
                best_score = score
                best_x = x
                best_ties = [x]
            elif score == best_score:
                best_ties.append(x)
                
        if best_x is None:
            break
            
        # Pick from ties randomly to add diversity
        chosen_x = random.choice(best_ties)
        
        # Update state
        sel.append(chosen_x)
        valid.remove(chosen_x)
        sums.add(2 * chosen_x)
        for a in sel[:-1]:
            sums.add(chosen_x + a)
            
        # Optional: Prune valid set
        # Remove any y that is now invalid
        new_valid = set()
        for y in valid:
            if 2 * y in sums:
                continue
            ok = True
            for a in sel:
                if y + a in sums:
                    ok = False
                    break
            if ok:
                new_valid.add(y)
        valid = new_valid
        
    return sel

def construct():
    start = time.time()
    deadline = start + 55  # Safety margin
    
    best_set = None
    best_size = float('inf')
    
    # Run multiple trials with the blocking heuristic
    # The heuristic is deterministic if we don't randomize ties, 
    # but randomizing ties gives different maximal sets.
    
    num_trials = 5000 # We can run many because it's fast
    
    for i in range(num_trials):
        if time.time() > deadline:
            break
            
        sel = _blocking_heuristic_greedy(deadline, randomize_tie_break=True)
        
        if not sel:
            continue
            
        # Verify it's maximal (should be by construction, but check just in case)
        # And Sidon
        if _is_sidon(sel) and _is_maximal(sel, N):
            if len(sel) < best_size:
                best_size = len(sel)
                best_set = sel
                # If we find a very small set, we might stop early or try to improve
                # But let's keep searching for a bit
        
        # If we have found a set of size, say, 15, and we've run many trials, 
        # we might want to do local search on the best one.
        
    # If we found a best set, try local search to reduce size further
    if best_set is not None and time.time() < deadline - 10:
        # Simple local search: Try removing elements and re-optimizing?
        # Or just run more greedy trials.
        # Given the budget, more greedy trials is safer than complex LS.
        
        # Let's do a few more targeted trials if we have time
        for i in range(2000):
            if time.time() > deadline:
                break
            sel = _blocking_heuristic_greedy(deadline, randomize_tie_break=True)
            if sel and _is_sidon(sel) and _is_maximal(sel, N):
                if len(sel) < best_size:
                    best_size = len(sel)
                    best_set = sel

    # Fallback
    if best_set is None:
        # Simple greedy smallest first
        sel = []
        sums = set()
        for x in range(1, N + 1):
            if 2 * x not in sums:
                ok = True
                for a in sel:
                    if x + a in sums:
                        ok = False
                        break
                if ok:
                    sel.append(x)
                    sums.add(2 * x)
                    for a in sel[:-1]:
                        sums.add(x + a)
        best_set = sel

    return best_set
