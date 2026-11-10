import time
import random

N = 256

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    # 1. Greedy Construction
    def greedy_construct():
        selected = []
        used_sums = set()
        candidates = list(range(1, N + 1))
        
        while candidates:
            if time.time() > deadline:
                return selected if selected else [1]
            
            best_candidate = None
            max_blocks = -1
            
            # Find valid candidates
            valid_candidates = []
            for x in candidates:
                new_sums = [2 * x] + [x + s for s in selected]
                if not any(s in used_sums for s in new_sums):
                    valid_candidates.append(x)
            
            if not valid_candidates:
                break
            
            # Pick candidate that blocks the most other valid candidates
            for x in valid_candidates:
                added_sums = set([2 * x] + [x + s for s in selected])
                blocks = 0
                for y in valid_candidates:
                    if y == x:
                        continue
                    # Check if y is blocked by x
                    # y's sums: 2y, y+s (s in selected)
                    # Check collision with added_sums
                    if 2 * y in added_sums:
                        blocks += 1
                        continue
                    blocked = False
                    for s in selected:
                        if y + s in added_sums:
                            blocked = True
                            break
                    if not blocked:
                        # Check y+x against used_sums
                        if y + x in used_sums:
                            blocked = True
                    if blocked:
                        blocks += 1
                
                if blocks > max_blocks:
                    max_blocks = blocks
                    best_candidate = x
            
            if best_candidate is None:
                break
                
            selected.append(best_candidate)
            used_sums.update([2 * best_candidate] + [best_candidate + s for s in selected[:-1]])
            
            # Remove blocked candidates
            new_candidates = []
            for y in candidates:
                if y == best_candidate:
                    continue
                # Check if y is still valid with new selected
                new_sums_y = [2 * y] + [y + s for s in selected]
                if not any(s in used_sums for s in new_sums_y):
                    new_candidates.append(y)
            candidates = new_candidates
            
        return selected

    S = greedy_construct()
    if not S:
        S = [1]
        
    # Helper to check if z can be added to a set given its sums
    def can_add(z, selected, used_sums):
        if z in selected:
            return False
        new_sums = [2 * z] + [z + s for s in selected]
        for s in new_sums:
            if s in used_sums:
                return False
        return True

    # Local Search: 2-for-1 swaps
    improved = True
    while improved and time.time() < deadline:
        improved = False
        S_copy = S[:]
        S.sort()
        
        # Iterate over all pairs
        for i in range(len(S_copy)):
            if time.time() > deadline:
                break
            for j in range(i + 1, len(S_copy)):
                x1 = S_copy[i]
                x2 = S_copy[j]
                
                S_rem = [s for s in S_copy if s != x1 and s != x2]
                
                # Compute used_sums for S_rem
                used_sums_rem = set()
                for idx, a in enumerate(S_rem):
                    used_sums_rem.add(2 * a)
                    for b in S_rem[idx+1:]:
                        used_sums_rem.add(a + b)
                
                # Identify Z_check: elements that were blocked ONLY by x1 or x2 in original S
                # And x1, x2 themselves
                Z_check = set([x1, x2])
                
                # Original used_sums for S
                used_sums_orig = set()
                for idx, a in enumerate(S_copy):
                    used_sums_orig.add(2 * a)
                    for b in S_copy[idx+1:]:
                        used_sums_orig.add(a + b)
                        
                # An element z is in Z_check if:
                # 1. z not in S
                # 2. can_add(z, S_rem, used_sums_rem) is True (i.e., z is NOT blocked by S_rem)
                #    This means z was blocked by x1 or x2 in S.
                #    If z was blocked by S_rem, it remains blocked in S'.
                #    So we only care about z that are "free" w.r.t S_rem.
                # 3. But we must ensure z is blocked in S'.
                
                # We want to find y such that S' = S_rem U {y} is maximal.
                # This requires:
                # A. y is valid with S_rem.
                # B. For all z not in S', z is blocked by S'.
                #    The only z that might not be blocked are those in Z_free = { z not in S | can_add(z, S_rem, used_sums_rem) == True }
                #    Note: x1 and x2 are in Z_free because S is independent.
                #    So Z_free includes x1, x2 and any other z that was only blocked by x1/x2.
                #    We need y to block ALL z in Z_free.
                
                Z_free = set()
                # We can optimize this by only checking z that are "close" or just iterate all z not in S
                # Since N=256, iterating all z is fine.
                for z in range(1, N + 1):
                    if z in S_copy:
                        continue
                    if can_add(z, S_rem, used_sums_rem):
                        Z_free.add(z)
                        
                # Now find y
                # y must be in range(1, N+1), y not in S
                # y must be valid with S_rem
                # y must block all z in Z_free.
                # "y blocks z" means can_add(z, S_rem U {y}, used_sums_rem U sums(y)) is False.
                # Which is equivalent to: z and y are not compatible given S_rem.
                
                found_y = None
                for y in range(1, N + 1):
                    if y in S_copy:
                        continue
                    # Check if y is valid with S_rem
                    if not can_add(y, S_rem, used_sums_rem):
                        continue
                    
                    # Check if y blocks all z in Z_free
                    # If Z_free is empty, any valid y works?
                    # If Z_free is empty, it means S_rem is already maximal?
                    # If S_rem is maximal, then S' is maximal if y is valid.
                    # But S_rem is NOT maximal because x1, x2 are in Z_free.
                    # So Z_free is never empty.
                    
                    blocks_all = True
                    for z in Z_free:
                        # Check if z can be added to S_rem U {y}
                        # This is equivalent to checking if z conflicts with y or S_rem
                        # z doesn't conflict with S_rem (by definition of Z_free).
                        # So we just need to check if z conflicts with y.
                        # Conflict if:
                        # 2z == 2y (z==y, impossible if z!=y)
                        # 2z == z+y => z==y
                        # z+y == 2y => z==y
                        # z+y == 2z => y==z
                        # z+y == s (s in used_sums_rem)?
                        #   If z+y in used_sums_rem, then z would have been blocked by S_rem?
                        #   No, z is valid with S_rem. So z+y not in used_sums_rem.
                        #   Wait, can_add(z, S_rem) checks:
                        #   2z not in used_sums_rem
                        #   z+s not in used_sums_rem for all s in S_rem
                        #   It does NOT check z+y because y is not in S_rem.
                        #   So z+y CAN be in used_sums_rem?
                        #   If z+y = a+b for a,b in S_rem, then adding y would make z+y collide with a+b.
                        #   So yes, we must check this.
                        
                        # So, z is blocked by y in S' if:
                        # 1. 2z == 2y (No)
                        # 2. 2z == z+y (No)
                        # 3. z+y == 2y (No)
                        # 4. z+y in used_sums_rem
                        # 5. 2y in ...? No, we are checking if z can be added.
                        #    The sums involving z in S' U {z} are: 2z, z+s (s in S'), z+z.
                        #    S' = S_rem U {y}.
                        #    Sums: 2z, z+s (s in S_rem), z+y, 2y (from y), y+s (from y).
                        #    We know 2z, z+s (s in S_rem) are unique and not in used_sums_rem.
                        #    We know 2y, y+s (s in S_rem) are unique and not in used_sums_rem (y valid).
                        #    Collisions can occur between:
                        #    - 2z and 2y (z!=y)
                        #    - 2z and y+s (2z == y+s)
                        #    - 2z and z+y (2z == z+y => z==y)
                        #    - z+s1 and 2y (z+s1 == 2y)
                        #    - z+s1 and y+s2 (z+s1 == y+s2)
                        #    - z+s1 and z+y (z+s1 == z+y => s1==y, impossible)
                        #    - z+y and 2y (z==y)
                        #    - z+y and y+s2 (z==s2, impossible)
                        
                        # So we need to check:
                        # a. 2z == y + s for some s in S_rem
                        # b. z + s1 == 2y for some s1 in S_rem
                        # c. z + s1 == y + s2 for some s1, s2 in S_rem
                        
                        blocked = False
                        # Check a
                        for s in S_rem:
                            if 2 * z == y + s:
                                blocked = True
                                break
                        if not blocked:
                            # Check b
                            for s1 in S_rem:
                                if z + s1 == 2 * y:
                                    blocked = True
                                    break
                        if not blocked:
                            # Check c
                            # This is expensive.
                            # But we can precompute sums of S_rem?
                            # Or just iterate. |S_rem| is small.
                            for s1 in S_rem:
                                for s2 in S_rem:
                                    if z + s1 == y + s2:
                                        blocked = True
                                        break
                                if blocked:
                                    break
                                        
                        if not blocked:
                            blocks_all = False
                            break
                    
                    if blocks_all:
                        found_y = y
                        break
                
                if found_y is not None:
                    # Perform swap
                    S = S_rem + [found_y]
                    improved = True
                    break
            if improved:
                break
                
    return S
