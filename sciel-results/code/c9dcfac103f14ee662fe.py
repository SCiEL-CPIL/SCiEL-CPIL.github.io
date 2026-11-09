import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s margin
    
    n = 128
    
    # Precompute APs ending at each index i.
    # An AP is (a, a+d, a+2d, a+3d).
    # If we assign color to position i, it affects APs where i is the last element (a+3d),
    # or potentially earlier if we do forward propagation.
    # Let's use a backtracking search assigning colors 0..127 in order.
    # For each position i, we check constraints involving i and previous positions.
    
    # Constraints for position i:
    # 1. For every d such that i-3d >= 0, the AP (i-3d, i-2d, i-d, i) must have >= 3 colors.
    #    This involves colors at i-3d, i-2d, i-d, and i.
    # 2. For every d such that i-d >= 0, the AP (i-2d, i-d, i, i+d) is not fully determined yet.
    #    However, we can check if the assignment of i makes it *impossible* to satisfy later?
    #    Actually, standard backtracking just checks satisfied constraints involving assigned vars.
    #    Constraints involving unassigned vars are not checked yet.
    #    But we can prune if the current assignment creates a conflict with *already assigned* vars.
    
    # Let's stick to checking constraints where all 4 positions are assigned.
    # When assigning color to i, we check all APs (a, a+d, a+2d, a+3d) such that a+3d == i.
    # These are the "completed" APs.
    
    # We want to minimize k. Let's try k=3, then 4, etc.
    # But the problem says "Minimize the number of labels actually used".
    # The labels are 0..127. We can just use the first k labels 0..k-1.
    
    def solve(k):
        """
        Backtracking solver using k colors (0..k-1).
        Returns list of colors if found, else None.
        """
        colors = [-1] * n
        # Precompute for each i, the list of (d, a) such that a+3d = i
        # a = i - 3d
        # d ranges from 1 to i//3
        # a >= 0 => i - 3d >= 0 => d <= i//3
        # Also a, a+d, a+2d, a+3d are distinct indices, which is true for d>0.
        
        # To speed up, we can precompute the "past" APs for each i.
        past_aps = [[] for _ in range(n)]
        for i in range(n):
            max_d = i // 3
            for d in range(1, max_d + 1):
                a = i - 3 * d
                past_aps[i].append((a, a+d, a+2*d, i))
        
        def is_valid(i, c):
            """Check if assigning color c to position i is consistent with previous assignments."""
            for (a, b, c_idx, d_idx) in past_aps[i]:
                # Positions a, b, c_idx are already assigned
                ca = colors[a]
                cb = colors[b]
                cc = colors[c_idx]
                cd = c
                # Check if {ca, cb, cc, cd} has at least 3 distinct elements
                # This is equivalent to NOT (all 4 same) AND NOT (exactly 2 distinct)
                # Distinct count <= 2 means:
                # 1. All 4 same: ca==cb==cc==cd
                # 2. Two pairs: (ca==cb and cc==cd and ca!=cc) OR (ca==cc and cb==cd and ca!=cb) OR (ca==cd and cb==cc and ca!=cb)
                # 3. One pair and two singles? No, that's 3 distinct.
                # 4. One triple and one single? That's 2 distinct. (ca==cb==cc and ca!=cd) etc.
                
                # Efficient check:
                # Create a set of the 4 values.
                # If len(set) <= 2, invalid.
                # To avoid set creation overhead in tight loop, use bitwise logic if colors are small?
                # k is small (3, 4, 5).
                
                # Let's just use a simple check.
                if ca == cb:
                    if cc == cd:
                        if ca == cc:
                            return False # All 4 same
                        # ca==cb, cc==cd, ca!=cc -> 2 distinct. Invalid.
                        return False
                    # ca==cb, cc!=cd
                    if cc == ca or cd == ca:
                        # ca==cb==cc or ca==cb==cd. 
                        # If cc==ca, then {ca, ca, ca, cd}. Distinct: {ca, cd}. Size 2. Invalid.
                        # If cd==ca, then {ca, ca, cc, ca}. Distinct: {ca, cc}. Size 2. Invalid.
                        return False
                    # Else {ca, ca, cc, cd} with ca!=cc, ca!=cd, cc!=cd. 3 distinct. Valid.
                elif ca == cc:
                    if cb == cd:
                        # ca==cc, cb==cd.
                        if ca == cb:
                            return False # All 4 same
                        # Else 2 distinct. Invalid.
                        return False
                    # ca==cc, cb!=cd
                    if cb == ca or cd == ca:
                        return False # 2 distinct
                    # Else 3 distinct. Valid.
                elif ca == cd:
                    if cb == cc:
                        # ca==cd, cb==cc.
                        if ca == cb:
                            return False
                        return False # 2 distinct
                    # ca==cd, cb!=cc
                    if cb == ca or cc == ca:
                        return False # 2 distinct
                    # Else 3 distinct. Valid.
                elif cb == cc:
                    if ca == cd:
                        # Handled in ca==cd case? No, ca!=cd here.
                        # cb==cc, ca!=cd.
                        # If ca==cb? No, ca!=cb (from elif chain).
                        # If cd==cb?
                        if cd == cb:
                            return False # 2 distinct: {ca, cb}
                        # Else 3 distinct. Valid.
                    # cb==cc, ca!=cd, ca!=cb (implied), cd!=cb (implied if we didn't return).
                    # Wait, if ca==cd was false, and we are in cb==cc branch.
                    # We need to check if any other pair is equal.
                    # We checked ca==cb (false), ca==cc (false), ca==cd (false).
                    # So ca is distinct from cb, cc, cd.
                    # cb==cc.
                    # If cd==cb, then {ca, cb, cb, cb} -> 2 distinct.
                    # If cd!=cb, then {ca, cb, cb, cd} -> 3 distinct (ca, cb, cd).
                    if cd == cb:
                        return False
                    # Valid
                elif cb == cd:
                    # ca!=cb, ca!=cc, ca!=cd, cb!=cc, cb==cd.
                    # If cc==ca? No.
                    # If cc==cd? i.e. cc==cb. Handled in cb==cc.
                    # So cc is distinct from ca, cb, cd.
                    # cb==cd.
                    # {ca, cb, cc, cb}. Distinct: ca, cb, cc. 3 distinct. Valid.
                    pass
                elif cc == cd:
                    # ca!=cb, ca!=cc, ca!=cd, cb!=cc, cb!=cd, cc==cd.
                    # {ca, cb, cc, cc}. Distinct: ca, cb, cc. 3 distinct. Valid.
                    pass
                else:
                    # All distinct. 4 distinct. Valid.
                    pass
                
            return True
            
        def backtrack(i):
            if time.time() > deadline:
                return False
            if i == n:
                return True
                
            # Heuristic: Try colors in random order or specific order?
            # Random order might help escape symmetries.
            color_order = list(range(k))
            random.shuffle(color_order)
            
            for c in color_order:
                if is_valid(i, c):
                    colors[i] = c
                    if backtrack(i + 1):
                        return True
                    colors[i] = -1
                    
            return False

        if backtrack(0):
            return colors
        return None

    # Try k=3 first
    for k in [3, 4, 5, 6]:
        if time.time() > deadline - 5:
            break
        sol = solve(k)
        if sol is not None:
            return sol
            
    # Fallback if backtracking fails (should not happen for small k)
    # Use the greedy method from seed as a last resort
    colors = []
    for x in range(1, 129):
        for color in range(128):
            if (all(len({color} | {colors[x-1-j*d] for j in (1,2,3)}) >= 3 for d in range(1, (x-1)//3+1))
                and all(any(colors[x-1-j*d] != color for j in (1,2)) for d in range(1, (x-1)//2+1))):
                colors.append(color)
                break
    return colors
