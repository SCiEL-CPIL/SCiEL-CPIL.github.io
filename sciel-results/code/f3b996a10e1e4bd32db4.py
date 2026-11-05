"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 88.0  # Safety margin for 90s limit
    n = 128
    
    # The constraint: Every nonconstant four-term progression must use AT LEAST 3 distinct colors.
    # A 4-term AP is (a, a+d, a+2d, a+3d).
    # Let the colors be c0, c1, c2, c3.
    # We need |{c0, c1, c2, c3}| >= 3.
    # This fails if:
    # 1. All 4 are the same (|S|=1)
    # 2. Only 2 are used (|S|=2). This happens if the set of values is of size 2.
    
    # When adding c3 at index i (where i = a+3d), we look at c0=colors[i-3d], c1=colors[i-2d], c2=colors[i-d].
    # If {c0, c1, c2} already has 3 distinct colors, any c3 is valid.
    # If {c0, c1, c2} has 2 distinct colors, say {X, Y}:
    #   - If we pick c3=X or c3=Y, the total set is still {X, Y} (size 2). INVALID.
    #   - If we pick c3=Z (Z!=X, Z!=Y), the total set is {X, Y, Z} (size 3). VALID.
    # If {c0, c1, c2} has 1 distinct color, say {X}:
    #   - If we pick c3=X, total set {X} (size 1). INVALID.
    #   - If we pick c3=Z (Z!=X), total set {X, Z} (size 2). INVALID.
    #   Wait, if c0=c1=c2=X, and c3=Z!=X, the set is {X, Z}. Size 2. This is INVALID.
    #   So if c0=c1=c2, we must pick a c3 such that we get 3 distinct colors?
    #   No, we can only pick ONE color c3.
    #   If c0=c1=c2=X, any c3 results in set {X, c3}. Max size 2.
    #   Therefore, it is IMPOSSIBLE to satisfy the condition if the first 3 terms of an AP are monochromatic.
    #   This implies that in any valid coloring, NO 3-term AP can be monochromatic?
    #   Let's check: If c0=c1=c2=X, then for the 4-term AP ending at c3, we fail.
    #   So, a necessary condition is that there are no monochromatic 3-term APs?
    #   Actually, if c0=c1=c2, the 4-term AP (c0,c1,c2,c3) will have at most 2 colors.
    #   Thus, we must avoid monochromatic 3-term APs (where the 3 terms are part of a 4-term AP).
    #   Specifically, for any index i that can be the 3rd term of a 4-term AP (i.e., i >= 2d),
    #   if colors[i-2d] == colors[i-d] == colors[i], then we have a problem for the next term?
    #   No, the problem is when we place the 4th term.
    #   If we are at index i (which is the 4th term), and colors[i-3d], colors[i-2d], colors[i-d] are all X,
    #   then we CANNOT place any color at i that makes the set size >= 3.
    #   So, we must ensure that we never create a situation where 3 previous terms of an AP are identical.
    #   This means we must avoid monochromatic 3-term APs (a, a+d, a+2d) wherever a+3d < n.
    
    # Strategy:
    # 1. Use a small number of colors, K. Start with K=3.
    # 2. Greedily assign colors from 0 to K-1.
    # 3. At each step i, determine which colors are "forbidden".
    #    A color c is forbidden if choosing it violates the condition for ANY 4-term AP ending at i.
    #    For a specific AP ending at i with d, let c0, c1, c2 be the previous colors.
    #    - If c0==c1==c2: No color is valid? 
    #      If c0=c1=c2=X, then {X, X, X, c} has size 1 (if c=X) or 2 (if c!=X).
    #      Both are < 3. So if c0=c1=c2, we are in a dead end.
    #      Therefore, the greedy choice must prevent c0=c1=c2.
    #      This implies that when choosing c2 (the 3rd term), we must ensure c2 != c0 if c1==c0?
    #      Actually, if c0=c1, and we choose c2=c0, then c0=c1=c2. Dead end for i+3d.
    #      So, if c0=c1, we MUST choose c2 != c0.
    #    - If c0, c1, c2 have 2 distinct values {X, Y}:
    #      We need |{X, Y, c}| >= 3. So c must be NOT X and NOT Y.
    #    - If c0, c1, c2 have 3 distinct values:
    #      Any c is valid.
    
    # To make this efficient, we can just try colors 0..K-1 and check validity.
    # Since K is small (3 or 4), checking is fast.
    # The "dead end" (c0=c1=c2) can be avoided by a smarter heuristic:
    # If c0==c1, we MUST pick c2 != c0.
    
    def solve_with_K(K):
        colors = [-1] * n
        # We build the array index by index.
        # However, the constraint on index i depends on i-1, i-2, i-3, etc.
        # Specifically, it depends on previous indices.
        # We can do a simple forward greedy.
        
        for i in range(n):
            if time.time() > deadline:
                return None
                
            # Determine forbidden colors for index i
            # A color c is invalid if there exists a d such that i-3d >= 0 
            # and the set {colors[i-3d], colors[i-2d], colors[i-d], c} has size < 3.
            
            # Let's precompute the "state" of previous terms for each d.
            # Actually, we can just iterate d.
            forbidden = [False] * K
            
            max_d = i // 3
            for d in range(1, max_d + 1):
                c0 = colors[i - 3*d]
                c1 = colors[i - 2*d]
                c2 = colors[i - d]
                
                # Check if any color c in 0..K-1 would fail
                # We want to find which c are FORBIDDEN.
                # If c0, c1, c2 are all same (X):
                #   Then for any c, size is 1 or 2. So ALL colors are forbidden?
                #   Yes. If we reach this state, we have failed.
                #   But we can prevent this by ensuring we never pick c2==c0 if c1==c0.
                #   So in a valid path, c0,c1,c2 cannot be all same.
                #   Let's assume we maintain the invariant that no 3-term AP is monochromatic.
                #   If we do that, c0,c1,c2 will have at least 2 distinct colors.
                
                if c0 == c1 == c2:
                    # This should not happen if our heuristic is good.
                    # If it does, we are stuck.
                    return None
                
                distinct = {c0, c1, c2}
                if len(distinct) == 2:
                    # We have 2 colors. We need the 4th to be a 3rd distinct color.
                    # So c cannot be in distinct.
                    for val in distinct:
                        if val < K:
                            forbidden[val] = True
                elif len(distinct) == 3:
                    # Already 3 distinct. Any c is fine (adds 3rd or 4th, but we only have K colors).
                    # If K=3, and we have 3 distinct, the set is {0,1,2}. 
                    # Adding any c in {0,1,2} keeps size 3. Valid.
                    # No forbidden colors.
                    pass
                else:
                    # len(distinct) == 1 is handled above (return None)
                    pass
            
            # Find a valid color
            # Heuristic: Pick the color that appears least frequently so far? Or just the first valid one?
            # To minimize total colors used, we should stick to 0..K-1.
            # Let's try to pick the smallest valid color to keep the range tight?
            # Or pick randomly from valid ones to break symmetry?
            
            valid_colors = [c for c in range(K) if not forbidden[c]]
            if not valid_colors:
                return None
            
            # To avoid getting stuck in patterns, we can use a simple heuristic.
            # If we just pick the first valid color, we might create monochromatic 3-APs later?
            # No, the constraint on i depends on i-1, i-2, i-3.
            # By picking c_i, we affect future constraints.
            # Specifically, if we pick c_i = X, and we have c_{i-d} = X and c_{i-2d} = X, 
            # then we create a monochromatic 3-AP (i-2d, i-d, i).
            # This would make the NEXT term (i+d) impossible to satisfy if it forms a 4-AP.
            # So we should AVOID creating monochromatic 3-APs.
            
            # Refined Heuristic:
            # Among valid colors, prefer one that does NOT create a monochromatic 3-AP.
            # i.e., avoid c such that for any d, colors[i-d] == colors[i-2d] == c.
            
            best_color = None
            for c in valid_colors:
                is_bad = False
                max_d_check = i // 2
                for d in range(1, max_d_check + 1):
                    if colors[i-d] == colors[i-2*d] == c:
                        is_bad = True
                        break
                if not is_bad:
                    best_color = c
                    break
            
            if best_color is None:
                # All valid colors create a monochromatic 3-AP.
                # We have to pick one, hoping it doesn't lead to a dead end, 
                # or we restart.
                best_color = valid_colors[0]
            
            colors[i] = best_color
            
        return colors

    # Try K=3
    result = solve_with_K(3)
    if result is not None:
        return result
        
    # Try K=4
    result = solve_with_K(4)
    if result is not None:
        return result

    # Fallback: Randomized Greedy with more colors
    for attempt in range(100):
        if time.time() > deadline:
            break
        colors = [-1] * n
        success = True
        for i in range(n):
            forbidden = [False] * 128
            max_d = i // 3
            for d in range(1, max_d + 1):
                c0 = colors[i - 3*d]
                c1 = colors[i - 2*d]
                c2 = colors[i - d]
                distinct = {c0, c1, c2}
                if len(distinct) == 1:
                    val = next(iter(distinct))
                    forbidden[val] = True
                    # Note: If len==1, ALL other colors are valid?
                    # No, if c0=c1=c2=X, then {X,X,X,c} has size 1 or 2.
                    # So ALL colors are invalid.
                    # So if we encounter this, we fail.
                    success = False
                    break
                elif len(distinct) == 2:
                    for val in distinct:
                        forbidden[val] = True
                # else len==3, no forbidden
            if not success:
                break
            
            valid = [c for c in range(128) if not forbidden[c]]
            if not valid:
                success = False
                break
            # Pick random valid
            c = random.choice(valid)
            colors[i] = c
            
        if success and all(c != -1 for c in colors):
            # Verify
            valid = True
            for i in range(n):
                for d in range(1, i//3 + 1):
                    if len({colors[i], colors[i-d], colors[i-2*d], colors[i-3*d]}) < 3:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                return colors

    # Last resort
    return [0] * 128
# EVOLVE-BLOCK-END
