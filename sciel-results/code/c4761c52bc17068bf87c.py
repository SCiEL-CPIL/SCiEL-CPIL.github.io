def construct():
        # 1. Greedy Construction
        S = set()
        for x in range(1, 257):
            if all(2*b - a != x for a in S for b in S if a < b):
                S.add(x)
        
        # 2. Local Search
        # We will try to remove each element and see if we can gain more by adding others.
        # To speed up, we might limit iterations or use a heuristic.
        
        improved = True
        while improved:
            improved = False
            current_sum = sum(1.0/x for x in S)
            # Sort elements to remove in some order? Smallest first?
            # Removing small elements has high cost, so maybe try removing larger ones first?
            # Or just iterate through all.
            for x in list(S):
                S_trial = S - {x}
                sum_trial = current_sum - 1.0/x
                
                # Try to add any valid y
                # Iterate y from 1 to 256
                # If y is not in S_trial and is valid, add it
                added_any = False
                # We should add in order of highest reciprocal (smallest y)
                for y in range(1, 257):
                    if y in S_trial:
                        continue
                    # Check if y forms AP with any pair in S_trial
                    valid = True
                    for a in S_trial:
                        # Check if (a+y)/2 is in S_trial
                        s = a + y
                        if s % 2 == 0:
                            b = s // 2
                            if b in S_trial:
                                valid = False
                                break
                    if valid:
                        S_trial.add(y)
                        sum_trial += 1.0/y
                        added_any = True
                
                if sum_trial > current_sum:
                    S = S_trial
                    current_sum = sum_trial
                    improved = True
                    # Break and restart the while loop? Or continue?
                    # If we continue, we might use the updated S in the same pass.
                    # But 'x' was from the old S.
                    # It's safer to restart the while loop to re-evaluate all x in the new S.
                    break
        
        return list(S)
