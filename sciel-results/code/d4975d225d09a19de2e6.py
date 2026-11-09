import time
import random

def _all_4tp(n):
    """Return list of (a, d) for all nonconstant 4-term progressions in [1..n]."""
    res = []
    for a in range(1, n - 2):
        max_d = (n - a) // 3
        for d in range(1, max_d + 1):
            res.append((a, d))
    return res

def _check_validity(colors, n):
    """Check if every 4TP in [1..n] uses at least 3 distinct colors."""
    for a in range(1, n - 2):
        max_d = (n - a) // 3
        for d in range(1, max_d + 1):
            c0 = colors[a - 1]
            c1 = colors[a + d - 1]
            c2 = colors[a + 2 * d - 1]
            c3 = colors[a + 3 * d - 1]
            if len({c0, c1, c2, c3}) < 3:
                return False
    return True

def _count_violations(colors, n):
    """Count the number of 4TPs with fewer than 3 distinct colors."""
    count = 0
    for a in range(1, n - 2):
        max_d = (n - a) // 3
        for d in range(1, max_d + 1):
            c0 = colors[a - 1]
            c1 = colors[a + d - 1]
            c2 = colors[a + 2 * d - 1]
            c3 = colors[a + 3 * d - 1]
            if len({c0, c1, c2, c3}) < 3:
                count += 1
    return count

def _find_periodic_3color(n, max_period=16):
    """Search for a 3-coloring with a small period."""
    # Precompute 4TPs for the full range n
    # But for periodic check, we can just check the full range with the repeated pattern
    
    for p in range(4, max_period + 1):
        # Iterate through all 3^p patterns
        # Use a recursive generator or base-3 counting
        for mask in range(3 ** p):
            # Construct the pattern
            pattern = []
            val = mask
            for _ in range(p):
                pattern.append(val % 3)
                val //= 3
            
            # Expand to n
            colors = [pattern[i % p] for i in range(n)]
            
            # Check validity
            if _check_validity(colors, n):
                return colors
        
    return None

def _local_search_3color(n, deadline):
    """Use local search to find a 3-coloring."""
    best_colors = None
    best_violations = float('inf')
    
    # Try multiple random starts
    for restart in range(50):
        if time.time() > deadline - 5:
            break
            
        colors = [random.randint(0, 2) for _ in range(n)]
        violations = _count_violations(colors, n)
        
        if violations == 0:
            return colors
        
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            # Try flipping each position
            for i in range(n):
                if time.time() > deadline - 5:
                    break
                old_c = colors[i]
                for new_c in range(3):
                    if new_c == old_c:
                        continue
                    colors[i] = new_c
                    new_violations = _count_violations(colors, n)
                    if new_violations < violations:
                        violations = new_violations
                        improved = True
                        if violations == 0:
                            return colors
                        break # Restart the loop to try other positions? Or continue?
                    else:
                        colors[i] = old_c
            
        if violations < best_violations:
            best_violations = violations
            best_colors = colors[:]
            
    if best_violations == 0:
        return best_colors
    return None

def construct():
    deadline = time.time() + 85.0
    n = 128
    
    # 1. Try to find a 3-coloring
    # First, periodic patterns (fast for small p)
    sol = _find_periodic_3color(n, max_period=12)
    if sol is not None:
        return sol
        
    # Second, local search for 3-coloring
    sol = _local_search_3color(n, deadline)
    if sol is not None:
        return sol

    # 2. If 3 colors failed, try 4 colors
    # We can use a similar local search or greedy
    # Greedy with 4 colors is likely to succeed quickly
    # Let's use a simple greedy with backtracking for 4 colors if needed
    # But 3 should work. If not, 4 is a safe fallback.
    
    # Fallback: Greedy with 4 colors
    colors = [-1] * n
    colors[0] = 0
    success = True
    for x in range(1, n):
        if time.time() > deadline - 2:
            success = False
            break
        placed = False
        for c in range(4):
            valid = True
            # Check all 4TPs ending at x or containing x
            # Simple check: all 4TPs
            # This is O(N^2) per step, so O(N^3) total. 128^3 ~ 2M. Fast enough.
            for a in range(1, n - 2):
                max_d = (n - a) // 3
                for d in range(1, max_d + 1):
                    pos = [a - 1, a + d - 1, a + 2 * d - 1, a + 3 * d - 1]
                    if x not in pos:
                        continue
                    # Check if all others are assigned
                    all_assigned = True
                    distinct = set()
                    for p in pos:
                        if p == x:
                            distinct.add(c)
                        else:
                            if colors[p] == -1:
                                all_assigned = False
                                break
                            distinct.add(colors[p])
                    if all_assigned:
                        if len(distinct) < 3:
                            valid = False
                            break
                if not valid:
                    break
            if valid:
                colors[x] = c
                placed = True
                break
        if not placed:
            success = False
            break
            
    if success and _check_validity(colors, n):
        return colors

    # Ultimate fallback
    return list(range(n))
