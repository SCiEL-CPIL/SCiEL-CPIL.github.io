def construct():
        N = 1008
        p = 1009
        inv = [0] * (p)
        for i in range(1, p):
            inv[i] = pow(i, p-2, p)
        
        left = 1
        max_len = 0
        best_start = 1
        last_occ = [0] * (p)
        
        for right in range(1, N + 1):
            x = right
            y = inv[x]
            
            if x == y:
                # Self-inverse. Cannot be in the interval.
                # The interval ending at 'right' is invalid.
                # We effectively exclude 'right'.
                # The next valid start is right + 1.
                left = right + 1
                # We do not update last_occ[x] because we are excluding it?
                # Or rather, since left > right, the window is empty.
                # If we set left = right + 1, then in the next step,
                # any check for last_occ[z] >= left will fail for z < left.
                # But we should update last_occ[x] to right?
                # If we update last_occ[x] = right, and left = right + 1,
                # then last_occ[x] < left, so it's effectively out of the window.
                last_occ[x] = right
            else:
                # Check if y is in the current window [left, right-1]
                if last_occ[y] >= left:
                    left = last_occ[y] + 1
                
                last_occ[x] = right
            
            current_len = right - left + 1
            if current_len > max_len:
                max_len = current_len
                best_start = left
        
        return [best_start, max_len]
