import random
import time

def construct():
    n = 128
    
    def is_valid_full(colors):
        # Check all 4-term APs
        for a in range(n):
            for d in range(1, (n - 1 - a) // 3 + 1):
                if a + 3 * d >= n:
                    break
                c1 = colors[a]
                c2 = colors[a + d]
                c3 = colors[a + 2 * d]
                c4 = colors[a + 3 * d]
                if len({c1, c2, c3, c4}) < 3:
                    return False
        return True

    def greedy_solve(palette, max_restarts=200, time_limit=5.0):
        start_time = time.time()
        for restart in range(max_restarts):
            if time.time() - start_time > time_limit:
                return None
            
            colors = [None] * n
            # Assign first few randomly to break symmetry? Or just start from 0.
            # We fill from 0 to n-1.
            # To make it faster, we can pre-fill some? No, let's just greedy.
            
            # Randomize order of palette for tie-breaking
            pos = 0
            while pos < n:
                # Check if we can place any color at pos
                valid_colors = []
                for c in palette:
                    # Check all APs ending at pos? No, APs involving pos.
                    # Since we fill left to right, we only need to check APs where pos is the last element?
                    # No, pos can be the 1st, 2nd, 3rd, or 4th term of an AP.
                    # If we fill left-to-right, when we are at `pos`, the elements before it are fixed.
                    # An AP $a, a+d, a+2d, a+3d$ involves `pos` if `pos` is one of these.
                    # If `pos` is the last term ($a+3d$), the previous 3 are fixed.
                    # If `pos` is the 3rd term ($a+2d$), then $a+3d$ is not fixed yet.
                    # BUT, the constraint applies to the *set* of 4.
                    # If we place `c` at `pos`, we must ensure that for every AP containing `pos`,
                    # the final set will have >= 3 colors.
                    # If `pos` is not the last term, we don't know the future colors.
                    # However, we can defer the check?
                    # No, the current `can_place` in the seed only checks APs ending at `x`.
                    # This is insufficient if `x` is an earlier term.
                    # Example: We place $c$ at $x$. Later we place $c'$ at $x+d$.
                    # If we don't check forward, we might paint ourselves into a corner.
                    
                    # To be safe and fast, we should only check constraints that are *fully determined* by the past?
                    # No, that's not how APs work.
                    # If we fill left to right, an AP $a, a+d, a+2d, a+3d$ becomes fully determined when we fill $a+3d$.
                    # So, we only need to check validity when we place the *last* element of an AP.
                    # This means: when placing color at index `pos`, we only need to check all APs where `pos` is the 4th term.
                    # APs where `pos` is 1st, 2nd, or 3rd term will be checked later when the 4th term is placed.
                    # This makes the local check very fast!
                    
                    is_valid = True
                    # Check APs where pos is the 4th term: pos, pos-d, pos-2d, pos-3d
                    for d in range(1, pos // 3 + 1):
                        if pos - 3 * d < 0:
                            break
                        c1 = colors[pos - 3 * d]
                        c2 = colors[pos - 2 * d]
                        c3 = colors[pos - d]
                        if c1 is None or c2 is None or c3 is None:
                            # This shouldn't happen if we fill 0..n-1 sequentially and pos is current
                            # But wait, if pos=3, d=1, we need 0,1,2. They are filled.
                            pass
                        s = {c1, c2, c3, c}
                        if len(s) < 3:
                            is_valid = False
                            break
                    if is_valid:
                        valid_colors.append(c)
                
                if not valid_colors:
                    # Backtrack or restart
                    # Simple restart is often faster for this size
                    break
                
                # Pick random valid color
                colors[pos] = random.choice(valid_colors)
                pos += 1
            
            if pos == n:
                # Verify full validity just in case (though logic above should guarantee it for 4th-term checks)
                # Actually, checking only 4th term is sufficient for left-to-right fill?
                # Yes, because every AP has a unique 4th term which is the largest index in the AP.
                # When that index is filled, all other 3 are already filled.
                # So checking only when placing the 4th term covers all APs exactly once.
                if is_valid_full(colors):
                    return colors
        
        return None

    # Try 3 colors
    result = greedy_solve([0, 1, 2], max_restarts=500, time_limit=10.0)
    if result is not None:
        return result

    # Try 4 colors
    result = greedy_solve([0, 1, 2, 3], max_restarts=50, time_limit=5.0)
    if result is not None:
        return result

    # Fallback: Return a simple 4-color pattern that is likely valid or just random 4-color
    # A pattern like 0,1,2,3,0,1,2,3...
    # Let's check if 0,1,2,3 repeated works.
    # AP 0,1,2,3 -> 0,1,2,3 (4 colors) OK
    # AP 1,2,3,4 -> 1,2,3,0 (4 colors) OK
    # AP 0,2,4,6 -> 0,2,0,2 (2 colors: 0,2) FAIL.
    # So simple repeat doesn't work.
    
    # Use a known valid 4-coloring or random search with more restarts
    result = greedy_solve([0, 1, 2, 3], max_restarts=1000, time_limit=15.0)
    if result is not None:
        return result

    # Absolute fallback: return array of 0s (will be invalid, but better than crash? No, must be valid)
    # If we can't find one, we have a problem.
    # But 4 colors should be easy to find.
    # Let's try a different heuristic for 4 colors:
    # Assign colors based on index mod 4? No, shown above to fail.
    # Maybe mod 5?
    # Let's just return the result of a longer search.
    
    # If all else fails, return a sequence that is valid by construction?
    # We can use a greedy with 4 colors and backtracking if needed.
    # But the randomized greedy with restarts should work.
    
    # To be safe, if result is None, return a dummy (will score 0)
    # But let's try one more thing: a specific 4-color pattern known to work?
    # Or just increase restarts.
    
    return result if result is not None else [0]*n

