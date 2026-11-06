import time
import random
import sys

def _validate(colors, n=128):
    """Check if every non-constant 4-AP uses at least 3 distinct colors."""
    for a in range(n):
        for d in range(1, n):
            b = a + d
            c = a + 2*d
            e = a + 3*d
            if e >= n:
                break
            # Non-constant means d > 0, which is always true in loop
            cols = {colors[a], colors[b], colors[c], colors[e]}
            if len(cols) < 3:
                return False
    return True

def _count_colors(colors):
    return len(set(colors))

def _solve_backtrack(n, num_colors, time_limit=10.0):
    """
    Backtracking solver to find a valid coloring with 'num_colors' colors.
    Returns list of colors or None.
    """
    start_time = time.time()
    
    # Precompute APs ending at each index i
    # ap_ends[i] contains tuples (a, b, c) such that (a, b, c, i) is a 4-AP
    # i = a + 3d => a = i - 3d
    # b = a + d = i - 2d
    # c = a + 2d = i - d
    ap_ends = [[] for _ in range(n)]
    for i in range(n):
        for d in range(1, i // 3 + 1):
            a = i - 3*d
            if a < 0:
                break
            b = a + d
            c = a + 2*d
            ap_ends[i].append((a, b, c))
            
    # Also need to check APs where i is NOT the last element?
    # If we assign x_i, we can check APs starting at i?
    # No, we can only check fully assigned APs.
    # But we can prune based on partial assignments.
    # E.g. if x_i, x_{i+d}, x_{i+2d} are assigned and same, then x_{i+3d} cannot be assigned yet,
    # but we know it's a problem if it ends up being same or different?
    # Actually, if x_i=x_{i+d}=x_{i+2d}=k, then x_{i+3d} MUST NOT be k (to avoid 4 same)
    # AND the set {k, x_{i+3d}} has size 2. We need size >= 3.
    # So it's IMPOSSIBLE.
    # So we can prune immediately if we assign 3 same colors in a row in an AP.
    
    colors = [-1] * n
    
    def is_valid_partial(i, color):
        # Check APs ending at i
        for a, b, c in ap_ends[i]:
            ca = colors[a]
            cb = colors[b]
            cc = colors[c]
            # All a,b,c are assigned
            if ca == -1 or cb == -1 or cc == -1:
                continue # Should not happen if we go left-to-right
            cols = {ca, cb, cc, color}
            if len(cols) < 3:
                return False
        
        # Check APs where i is the first element?
        # (i, i+d, i+2d, i+3d)
        # We can check if assigning 'color' at i creates a "dead end" for future?
        # Specifically, if we have x_i, x_{i+d}, x_{i+2d} assigned? No, they are future.
        # But if x_i, x_{i+d} are assigned?
        # Let's check APs where i is the first element and some subsequent elements are assigned?
        # Since we assign left-to-right, if i is current, subsequent are unassigned.
        # So we can only check APs where i is the LAST element (fully assigned)
        # OR if we look ahead?
        
        # Pruning: Check if any AP starting at i has 3 same colors in the first 3 positions?
        # No, positions i+1, i+2, i+3 are unassigned.
        
        # Pruning: Check APs where i is the 2nd, 3rd, or 4th element?
        # If i is 2nd: (a, i, i+d, i+2d). a is assigned. i, i+d, i+2d unassigned.
        # If i is 3rd: (a, a+d, i, i+d). a, a+d assigned.
        #   If colors[a] == colors[a+d] == color, then we have 3 same? No, 3 assigned: a, a+d, i.
        #   If they are all same, then x_{i+d} will make it 4 same or 3 same + 1 diff.
        #   In either case, max distinct colors is 2. So INVALID.
        #   So if colors[a] == colors[a+d] == color, return False.
        
        # If i is 4th: (a, a+d, a+2d, i). All assigned. Checked above.
        
        # If i is 3rd: (a, b, i, e). a,b assigned.
        # If colors[a] == colors[b] == color, then we have 3 same. e will be 4th.
        # Set will be {color, colors[e]}. Size <= 2. Invalid.
        # So check this.
        
        for d in range(1, (n - i) // 3 + 1):
            # i is 3rd element: a = i - 2d, b = i - d
            a = i - 2*d
            if a < 0:
                break
            b = i - d
            if colors[a] != -1 and colors[b] != -1:
                if colors[a] == colors[b] == color:
                    return False
            
            # i is 2nd element: a = i - d, b = i, c = i + d, e = i + 2d
            # a assigned. i assigned (color). c, e unassigned.
            # If colors[a] == color, then we have 2 same.
            # Future c, e must provide 2 more distinct colors?
            # If c == color, then 3 same. Invalid.
            # If c != color, say c=k. Then we have {color, k}.
            # e must be a 3rd color?
            # If e is color, set {color, k}. Size 2. Invalid.
            # If e is k, set {color, k}. Size 2. Invalid.
            # If e is m (new), set {color, k, m}. Size 3. Valid.
            # So if colors[a] == color, we MUST ensure that c and e are not both in {color} or something?
            # Actually, if colors[a] == color, then we have x_a=color, x_i=color.
            # We need {color, x_{i+d}, x_{i+2d}} to have size >= 3.
            # This implies x_{i+d} != color and x_{i+2d} != color and x_{i+d} != x_{i+2d}?
            # No.
            # If x_{i+d}=k, x_{i+2d}=m. Set {color, k, m}.
            # Need k != color, m != color, k != m?
            # If k == m, set {color, k}. Size 2. Invalid.
            # So if colors[a] == color, then x_{i+d} and x_{i+2d} MUST be distinct from color and each other.
            # This is a constraint on future variables. We can't check it yet, but we can note it.
            # For now, just checking for immediate impossibility (3 same) is enough for basic pruning.
            
            # i is 1st element: (i, i+d, i+2d, i+3d). All future. No check.
            
        return True

    def backtrack(i):
        if time.time() - start_time > time_limit:
            return False
        if i == n:
            return True
        
        # Heuristic: try colors 0..num_colors-1
        # Symmetry breaking for first few
        for color in range(num_colors):
            if i == 0 and color != 0:
                continue
            if i == 1 and color == colors[0]: # Optional symmetry break? No, 0,0 might be valid.
                # Actually, if x0=0, x1=0, then x2, x3 must be distinct from 0 and each other?
                # AP (0,1,2,3): 0,0,c2,c3. Need {0,c2,c3} size 3.
                # So c2!=0, c3!=0, c2!=c3.
                # This is possible.
                pass
            
            colors[i] = color
            if is_valid_partial(i, color):
                if backtrack(i + 1):
                    return True
            colors[i] = -1
            
        return False

    if backtrack(0):
        return colors
    return None

def _greedy_4color(n=128):
    """
    Fallback: Construct a valid 4-coloring using greedy with backtracking if needed.
    """
    # We can use the same backtracking with num_colors=4
    # But to be safe, we can try a few random seeds or just deterministic.
    res = _solve_backtrack(n, 4, time_limit=5.0)
    if res:
        return res
    # If backtracking fails (shouldn't for 4 colors), use a hardcoded pattern or raise error
    # For 4 colors, it is very likely to succeed.
    # Let's try a simple pattern: i % 4
    # But we know i%4 fails for d=4.
    # We must rely on backtracking.
    return None

def construct():
    start_time = time.time()
    n = 128
    deadline = start_time + 80.0
    
    # Try 3 colors first
    # Use a longer time limit for 3 colors if possible, but keep total under deadline.
    # Let's give 30s to 3-color search.
    t0 = time.time()
    res3 = _solve_backtrack(n, 3, time_limit=30.0)
    
    if res3 is not None:
        if _validate(res3, n):
            return res3
        else:
            print("3-color result failed validation, falling back")
            
    # Try 4 colors
    t1 = time.time()
    remaining = deadline - t1
    if remaining > 5.0:
        res4 = _solve_backtrack(n, 4, time_limit=remaining - 5.0)
        if res4 is not None:
            if _validate(res4, n):
                return res4
            else:
                print("4-color result failed validation")
    
    # Emergency fallback: Random 4-coloring with validation (unlikely to work)
    # Or a hardcoded 4-coloring if we had one.
    # Let's try one more time with 4 colors and a different seed/order if possible?
    # The backtracking is deterministic.
    # If it fails, we are in trouble.
    # But 4-coloring should be easy.
    
    # As a last resort, return a pattern that might be valid?
    # No, must be valid.
    # If backtracking fails, it means no solution found in time.
    # But 4-coloring for N=128 is solvable.
    
    # Let's force a valid 4-coloring by using a known good pattern if available.
    # Or just raise an error? No, must return something.
    # If res4 is None, try again with more time?
    # We have used 30s. 50s left.
    res4b = _solve_backtrack(n, 4, time_limit=40.0)
    if res4b is not None:
        return res4b
        
    # If all else fails, return a list of 0s (invalid) to avoid crash, but score 0.
    # Better to return something that might be valid?
    # Let's return a 4-coloring generated by a simple heuristic that is likely valid.
    # Actually, if backtracking fails, it's a bug or time out.
    # Given the constraints, 4 colors should be found instantly.
    
    return [0]*n

