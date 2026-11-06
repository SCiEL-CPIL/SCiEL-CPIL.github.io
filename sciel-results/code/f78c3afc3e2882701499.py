def try_improve(chosen, counts, order):
    chosen_set = set(chosen)
    improved = False
    
    # To avoid O(N^2) in the inner loop of try_improve, 
    # we can just try the first conflict found.
    
    for c in order:
        if c in chosen_set:
            continue
        
        # Find a conflict
        conflict_x = None
        if counts.get(2 * c, 0) >= 2:
            # If 2c is full, we must remove an x that contributes to 2c?
            # No, x contributes to 2c only if x=c.
            # So if 2c is full, we cannot add c unless we remove elements that form sums = 2c.
            # This is harder. Let's skip 2c conflicts for now to keep it simple.
            # Or, we can just try to add c and see if it works? No, it's full.
            # So if 2c is full, we cannot add c by removing a single x (unless x is part of a pair summing to 2c).
            # Let's ignore 2c conflicts in this simple heuristic.
            pass
            
        for x in chosen:
            s = c + x
            if counts.get(s, 0) >= 2:
                conflict_x = x
                break
        
        if conflict_x is None:
            # No conflict found? Should be added in greedy.
            # Maybe it was rejected due to 2c?
            if counts.get(2 * c, 0) < 2:
                # Add it
                chosen_set.add(c)
                chosen.append(c)
                counts[2 * c] = counts.get(2 * c, 0) + 1
                for x in chosen[:-1]:
                    counts[c + x] = counts.get(c + x, 0) + 1
                improved = True
            continue

        # Try swapping conflict_x with c
        # 1. Remove conflict_x
        # 2. Check if c is valid
        # 3. If valid, commit, else revert
        
        # Collect sums to decrement
        sums_to_decrement = []
        for y in chosen:
            if y == conflict_x:
                continue
            s = conflict_x + y
            sums_to_decrement.append(s)
            counts[s] -= 1
        
        # Check validity of c
        valid = True
        if counts.get(2 * c, 0) >= 2:
            valid = False
        else:
            for y in chosen:
                if y == conflict_x:
                    continue
                s = c + y
                if counts.get(s, 0) >= 2:
                    valid = False
                    break
        
        if valid:
            # Commit
            chosen.remove(conflict_x)
            chosen_set.remove(conflict_x)
            chosen.append(c)
            chosen_set.add(c)
            
            counts[2 * c] = counts.get(2 * c, 0) + 1
            for y in chosen:
                if y == c:
                    continue
                s = c + y
                counts[s] = counts.get(s, 0) + 1
            improved = True
        else:
            # Revert
            for s in sums_to_decrement:
                counts[s] += 1
                
    return chosen, counts, improved
