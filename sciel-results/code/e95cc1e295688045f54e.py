import time
import random
import numpy as np

N = 256

def construct():
    start_time = time.time()
    deadline = start_time + 25.0  # 25s limit to be safe
    
    best_set = None
    best_size = N + 1
    
    # Helper to check if a set is a valid Sidon set
    def is_valid_sidon(S):
        sums = set()
        for i in range(len(S)):
            for j in range(i, len(S)):
                s = S[i] + S[j]
                if s in sums:
                    return False
                sums.add(s)
        return True

    # Helper to check if a set is maximal
    def is_maximal(S):
        S_set = set(S)
        sums = set()
        for i in range(len(S)):
            for j in range(i, len(S)):
                sums.add(S[i] + S[j])
                
        for x in range(1, N + 1):
            if x in S_set:
                continue
            # Check if x can be added
            valid = True
            new_sums = set()
            new_sums.add(2 * x)
            for y in S:
                new_sums.add(x + y)
            
            if len(new_sums) != len(list(new_sums)): # Duplicates within new sums? No, x is unique, so x+y are distinct if y are distinct. 2x is distinct from x+y if y!=x.
                 pass # Actually, if y1 != y2, x+y1 != x+y2. 2x != x+y if y != x. So new sums are always distinct internally.
            
            # Check against existing sums
            if sums.intersection(new_sums):
                continue # Conflict, so x cannot be added
            
            # If no conflict, x CAN be added, so S is NOT maximal
            return False
        
        # If no x could be added, it is maximal
        return True

    # Greedy construction with blocking count
    def greedy_blocking():
        candidates = set(range(1, N + 1))
        selected = []
        sums = set()
        
        while candidates:
            best_x = None
            best_block = -1
            valid_xs = []
            
            for x in candidates:
                # Check if x is valid
                new_sums = [2 * x] + [x + y for y in selected]
                if len(set(new_sums)) != len(new_sums):
                    continue
                if sums.intersection(set(new_sums)):
                    continue
                
                valid_xs.append(x)
                
                # Calculate blocking count
                # How many candidates y become invalid if x is added?
                block_count = 0
                for y in candidates:
                    if y == x:
                        continue
                    # Check if y is valid in S (it is, since y in candidates)
                    # Check if y is blocked by x
                    # y is blocked if (S U {x}) makes y invalid
                    # y invalid if:
                    # 1. 2y conflicts with sums U new_sums
                    # 2. y+z conflicts with sums U new_sums for z in S
                    # 3. y+z1 conflicts with y+z2 for z1,z2 in S (already checked for S)
                    # 4. y+z conflicts with 2x or x+w for w in S
                    
                    # Simplified: Check if y is valid in S U {x}
                    # If y was valid in S, it only becomes invalid if it conflicts with x or sums involving x.
                    
                    # Sums involving x: 2x, x+z for z in S.
                    # Sums involving y: 2y, y+z for z in S.
                    
                    # Conflict if:
                    # 2y in sums or 2y == 2x (no) or 2y == x+z
                    # y+z in sums (no, y valid in S) or y+z == 2x or y+z == x+w
                    
                    blocked = False
                    # Check 2y
                    if 2 * y in sums:
                        blocked = True # Shouldn't happen if y in candidates
                    elif 2 * y == 2 * x:
                        blocked = True
                    else:
                        # Check if 2y == x + z for any z in S
                        # Or 2y in new_sums? new_sums are 2x and x+z.
                        # So check if 2y - x in S and 2y != 2x
                        if (2 * y - x) in selected and 2 * y != 2 * x:
                            blocked = True
                    
                    if not blocked:
                        # Check y+z
                        for z in selected:
                            if y + z == 2 * x:
                                blocked = True
                                break
                            if y + z in [x + w for w in selected]:
                                # This means y+z = x+w => y-w = x-z
                                # This is a conflict between sums of S U {x}
                                blocked = True
                                break
                            # Also check if y+z is in new_sums?
                            # new_sums are 2x and x+w.
                            # y+z == 2x checked above.
                            # y+z == x+w checked above.
                    
                    if blocked:
                        block_count += 1
                
                if block_count > best_block:
                    best_block = block_count
                    best_x = x
            
            if best_x is None:
                break # Should not happen if candidates is non-empty and S is valid?
            
            # Add best_x
            selected.append(best_x)
            sums.update([2 * best_x] + [best_x + y for y in selected[:-1]])
            candidates.remove(best_x)
            
            # Remove newly blocked candidates
            # We can just let the next iteration filter them, or prune now.
            # Pruning is faster.
            to_remove = set()
            for y in candidates:
                # Check if y is still valid
                new_sums_y = [2 * y] + [y + z for z in selected]
                if len(set(new_sums_y)) != len(new_sums_y):
                    to_remove.add(y)
                    continue
                if sums.intersection(set(new_sums_y)):
                    to_remove.add(y)
            candidates -= to_remove
            
        return selected

    # Run multiple restarts
    restarts = 0
    while time.time() < deadline:
        random.seed() # Re-seed? No, just use global random.
        S = greedy_blocking()
        if len(S) < best_size:
            best_size = len(S)
            best_set = S[:]
            # Try local search on this best set
            improved = True
            while improved and time.time() < deadline:
                improved = False
                for x in list(best_set):
                    S_try = [y for y in best_set if y != x]
                    if is_maximal(S_try):
                        best_set = S_try
                        best_size = len(best_set)
                        improved = True
                        break # Restart loop as set changed
        
        restarts += 1
        if restarts > 200: # Safety cap
            break

    if best_set is None:
        # Fallback: simple ascending greedy
        selected, sums = [], set()
        for x in range(1, N + 1):
            added = [x + a for a in selected] + [2 * x]
            if len(set(added)) == len(added) and not sums.intersection(added):
                selected.append(x)
                sums.update(added)
        return selected

    return best_set
