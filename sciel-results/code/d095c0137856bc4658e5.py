        # Get slice parameters
        if old_r == 0:
            start_old = p
        else:
            start_old = old_r
            
        if new_r == 0:
            start_new = p
        else:
            start_new = new_r
            
        # Update counts using slices
        counts[start_old:max_check+1:p] -= 1
        counts[start_new:max_check+1:p] += 1
