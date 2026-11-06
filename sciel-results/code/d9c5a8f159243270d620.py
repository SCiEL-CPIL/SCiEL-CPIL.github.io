import time

def construct():
    start_time = time.time()
    best_len = 0
    best_seq = []
    deadline = start_time + 55
    
    def check_extension_possible(x, P_new, prefixes, used_sums):
        """Quick check: is there any valid next element after x? Only check next ~20 values."""
        upper = min(x + 20, 256)
        for y in range(x + 1, upper + 1):
            P_y = P_new + y
            valid = True
            for p in prefixes:
                d = P_y - p
                if d in used_sums:
                    valid = False
                    break
            if valid:
                return True
        return False
    
    def backtrack(seq, used_sums, prefixes):
        nonlocal best_len, best_seq
        
        # Update best
        if len(seq) > best_len:
            best_len = len(seq)
            best_seq = seq[:]
            
        # Time guard
        if time.time() > deadline:
            return
            
        last_x = seq[-1] if seq else 0
        P_last = prefixes[-1]
        
        # Upper bound: can't beat best if not enough room
        max_possible = len(seq) + (256 - last_x)
        if max_possible <= best_len:
            return
        
        # Try candidates in ascending order
        for x in range(last_x + 1, 257):
            # Periodic time check
            if (x - last_x) % 32 == 0 and time.time() > deadline:
                return
            
            P_new = P_last + x
            
            # Check for collisions with existing sums
            valid = True
            new_diffs = []
            for p in prefixes:
                d = P_new - p
                if d in used_sums:
                    valid = False
                    break
                new_diffs.append(d)
                
            if not valid:
                continue
            
            # Forward checking at deeper levels to prune dead ends
            if len(seq) >= 3 and not check_extension_possible(x, P_new, prefixes, used_sums):
                continue
            
            # Commit
            seq.append(x)
            prefixes.append(P_new)
            used_sums.update(new_diffs)
            
            # Recurse
            backtrack(seq, used_sums, prefixes)
            
            # Undo
            seq.pop()
            prefixes.pop()
            for d in new_diffs:
                used_sums.discard(d)
                
    backtrack([], set(), [0])
    return best_seq
