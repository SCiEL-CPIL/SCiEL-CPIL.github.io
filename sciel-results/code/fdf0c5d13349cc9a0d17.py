import time
import sys

def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin before 90s kill
    
    # Precompute all 4-term arithmetic progressions (a, b, c, d) 
    # where a < b < c < d and b-a = c-b = d-c = step > 0
    # For each position i, store which progressions it participates in
    
    n = 128
    # For each position i, list of (i-d, i-2d, i-3d) for progressions ending at i
    # and (i-2d, i-d, i+d) for progressions where i is the 2nd element
    # and (i-d, i, i+d, i+2d) where i is the 1st element - but we assign left to right
    
    # Actually, let's precompute for each position i:
    # - progressions where i is the last element: (i-3d, i-2d, i-d, i)
    # - progressions where i is the 3rd element: (i-2d, i-d, i, i+d) - future
    # - progressions where i is the 2nd element: (i-d, i, i+d, i+2d) - future
    # - progressions where i is the 1st element: (i, i+d, i+2d, i+3d) - future
    
    # For greedy left-to-right, when assigning position i, we need to check:
    # 1. All progressions ending at i (all 4 positions determined)
    # 2. Progressions where i is 2nd or 3rd and the other 3 positions are already determined
    #    - If i is 2nd: (i-d, i, i+d, i+2d) - i+d and i+2d are future, so only 2 positions known
    #    - If i is 3rd: (i-2d, i-d, i, i+d) - i+d is future, so only 3 positions known
    #    - Actually, for constraint satisfaction, we need to ensure no dead-end
    
    # Simpler approach: just check progressions where all 4 positions are <= i
    # and ensure the partial assignment doesn't force a violation
    
    # Let's precompute for each position i, the list of progressions that end at i
    progressions_ending_at = [[] for _ in range(n)]
    for d in range(1, n // 3 + 1):
        for start in range(n - 3 * d):
            a = start
            b = start + d
            c = start + 2 * d
            dd = start + 3 * d
            if dd < n:
                progressions_ending_at[dd].append((a, b, c))
    
    # Also precompute progressions where i is the 3rd element (i-2d, i-d, i, i+d)
    # When assigning i, if i-2d and i-d are already assigned, we need to ensure
    # that i+d can be assigned a valid color later
    # For now, let's just do the basic check and see if it works
    
    # Try greedy with increasing number of colors
    for max_colors in range(3, 129):
        if time.time() - start_time > deadline - 5.0:
            break
            
        trial = [0] * n
        valid = True
        
        for i in range(n):
            found = False
            for c in range(max_colors):
                ok = True
                
                # Check all progressions ending at i
                for (a, b, cc) in progressions_ending_at[i]:
                    if len({c, trial[a], trial[b], trial[cc]}) < 3:
                        ok = False
                        break
                
                if not ok:
                    continue
                
                # Check for dead-ends: progressions where i is the 3rd element
                # (i-2d, i-d, i, i+d) where i+d < n
                # If trial[i-2d] == trial[i-d] == c, then i+d must be different from c,
                # but that's always possible as long as max_colors >= 2
                # More importantly, if two of the three known are equal and the third is different,
                # the fourth just needs to not make it <= 2 colors
                # Actually, the constraint is >= 3 distinct colors in the full progression
                # If we have 3 positions known with colors {x, y, z}, the 4th can be anything
                # as long as the total distinct count is >= 3
                # If the 3 known already have >= 3 distinct, any 4th works
                # If the 3 known have exactly 2 distinct, the 4th must be the third color
                # If the 3 known have 1 distinct, the 4th must be different (2 choices)
                
                # For now, just do the basic check. The greedy should work with enough colors.
                
                trial[i] = c
                found = True
                break
            
            if not found:
                valid = False
                break
        
        if valid:
            # Verify the solution
            all_ok = True
            for d in range(1, n // 3 + 1):
                for start in range(n - 3 * d):
                    a = start
                    b = start + d
                    c = start + 2 * d
                    dd = start + 3 * d
                    if dd < n:
                        if len({trial[a], trial[b], trial[c], trial[dd]}) < 3:
                            all_ok = False
                            break
                if not all_ok:
                    break
            
            if all_ok:
                return trial
    
    # Fallback: use a valid coloring with many colors
    # Assign color = i % 128 (all distinct) - trivially valid
    return [i for i in range(n)]
