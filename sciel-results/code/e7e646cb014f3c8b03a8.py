import time
import random
import sys

def _count_violations(colors, n=128):
    """Count the number of 4-APs that use fewer than 3 distinct colors."""
    violations = 0
    for a in range(n):
        for d in range(1, n):
            b = a + d
            c = a + 2*d
            e = a + 3*d
            if e >= n:
                break
            # Non-constant AP check (d>0 ensures non-constant)
            cols = {colors[a], colors[b], colors[c], colors[e]}
            if len(cols) < 3:
                violations += 1
    return violations

def _local_search_min_colors(n=128, max_colors=10, time_limit=80.0):
    """
    Find a valid coloring with the minimum number of colors.
    Uses local search to find a valid coloring for a fixed K, then tries to reduce K.
    """
    start_time = time.time()
    
    # Precompute 4-APs for faster violation checking during local search steps
    # Actually, recalculating all violations is O(N^2 * D) which is slow.
    # We need incremental updates or a faster check.
    # N=128. Number of 4-APs is roughly N^2/4 = 4096.
    # Checking all is ~4000 ops.
    # If we do 10000 steps, that's 40M ops. Doable in Python in < 1s?
    # Python is slow. 40M ops might take 2-5 seconds.
    # We need to be careful.
    
    def get_violations(colors):
        v = 0
        for a in range(n):
            for d in range(1, n):
                b = a + d
                c = a + 2*d
                e = a + 3*d
                if e >= n:
                    break
                if len({colors[a], colors[b], colors[c], colors[e]}) < 3:
                    v += 1
        return v

    def local_search_for_k(k, time_limit_k):
        """Find a valid k-coloring using local search."""
        rng = random.Random(int(time.time()))
        end_time = start_time + time_limit_k
        best_violations = float('inf')
        best_colors = None
        
        # Multiple restarts
        while time.time() < end_time:
            colors = [rng.randint(0, k-1) for _ in range(n)]
            violations = get_violations(colors)
            
            if violations == 0:
                return colors
            
            current_violations = violations
            current_colors = list(colors)
            
            # Simple Hill Climbing / Random Walk
            while time.time() < end_time:
                if current_violations == 0:
                    return current_colors
                
                # Pick a random position
                pos = rng.randint(0, n-1)
                old_color = current_colors[pos]
                
                # Try a different color
                new_color = rng.randint(0, k-1)
                if new_color == old_color:
                    continue
                    
                current_colors[pos] = new_color
                
                # Calculate new violations
                # Optimization: Only check APs involving 'pos'
                new_violations = current_violations
                # We need to undo the contribution of APs involving pos with old_color
                # and add contribution with new_color.
                # This requires storing which APs are violated.
                # For simplicity and robustness against time, let's just recalculate fully
                # if the number of steps is small, or use a faster incremental method.
                # Given N=128, full recalculation is ~4000 checks.
                # If we do 1000 steps, that's 4M checks. ~1-2 seconds in Python.
                # This is too slow for many restarts.
                
                # Let's use a faster violation check by precomputing APs.
                # But incremental update is better.
                # Let's stick to full recalculation but limit steps per restart.
                
                new_violations = get_violations(current_colors)
                
                if new_violations < current_violations:
                    current_violations = new_violations
                    if new_violations == 0:
                        return current_colors
                else:
                    # Revert
                    current_colors[pos] = old_color
                    # Random move to escape local minima
                    if rng.random() < 0.1:
                        current_colors[pos] = new_color
                        current_violations = new_violations
                        if new_violations == 0:
                            return current_colors

        return None

    # Try K=3, 4, 5...
    for k in range(3, 11):
        # Allocate remaining time proportionally
        elapsed = time.time() - start_time
        remaining = time_limit - elapsed
        if remaining < 2:
            break
        
        # Give more time to smaller K
        time_for_k = min(remaining, 10.0 if k == 3 else 5.0 if k == 4 else 2.0)
        
        solution = local_search_for_k(k, time_for_k)
        if solution:
            return solution
            
    # Fallback: Return a greedy coloring (might use many colors)
    colors = [0] * n
    for i in range(n):
        for c in range(128):
            colors[i] = c
            if _count_violations(colors) == 0: # This is O(N^2), too slow for loop
                break
    # Better fallback: Random coloring with many colors
    rng = random.Random(0)
    colors = [rng.randint(0, 127) for _ in range(n)]
    return colors

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    # Try local search for minimum colors
    result = _local_search_min_colors(n=128, time_limit=80.0)
    
    if result is not None:
        # Verify
        if _count_violations(result) == 0:
            return result
    
    # Fallback if local search fails
    # Use a simple greedy that is known to work, even if it uses more colors
    n = 128
    colors = []
    for x in range(n):
        for c in range(128):
            colors.append(c)
            # Quick check: Only check APs ending at x
            valid = True
            for d in range(1, x//3 + 1):
                a = x - 3*d
                b = x - 2*d
                e = x - d
                if a < 0: break
                if len({colors[a], colors[b], colors[e], colors[x]}) < 3:
                    valid = False
                    break
            if valid:
                break
        if len(colors) <= x:
            # Should not happen
            colors.append(0)
            
    return colors

# Optimization: The full recalculation in local_search_for_k is the bottleneck.
# Let's optimize _count_violations and the local search loop.

def _fast_count_violations(colors, n=128):
    v = 0
    for a in range(n):
        for d in range(1, n):
            b = a + d
            c = a + 2*d
            e = a + 3*d
            if e >= n: break
            if len({colors[a], colors[b], colors[c], colors[e]}) < 3:
                v += 1
    return v

def _local_search_optimized(n=128, time_limit=80.0):
    start_time = time.time()
    
    def get_violations(colors):
        v = 0
        for a in range(n):
            for d in range(1, n):
                b = a + d
                c = a + 2*d
                e = a + 3*d
                if e >= n: break
                if len({colors[a], colors[b], colors[c], colors[e]}) < 3:
                    v += 1
        return v

    for k in range(3, 11):
        elapsed = time.time() - start_time
        remaining = time_limit - elapsed
        if remaining < 1: break
        
        t_limit_k = min(remaining, 10.0 if k == 3 else 5.0)
        
        rng = random.Random(int(time.time()) ^ k)
        end_time = start_time + t_limit_k
        
        while time.time() < end_time:
            colors = [rng.randint(0, k-1) for _ in range(n)]
            viol = get_violations(colors)
            
            if viol == 0:
                return colors
            
            # Hill climbing
            for _ in range(200): # Max steps per restart
                if time.time() > end_time: break
                if viol == 0: return colors
                
                pos = rng.randint(0, n-1)
                old_c = colors[pos]
                new_c = rng.randint(0, k-1)
                if new_c == old_c: continue
                
                colors[pos] = new_c
                new_viol = get_violations(colors)
                
                if new_viol < viol:
                    viol = new_viol
                    if viol == 0: return colors
                else:
                    colors[pos] = old_c
                    if rng.random() < 0.05:
                        colors[pos] = new_c
                        viol = new_viol
                        if viol == 0: return colors
                        
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    result = _local_search_optimized(n=128, time_limit=80.0)
    if result and _fast_count_violations(result) == 0:
        return result
    
    # Fallback
    n = 128
    colors = []
    for x in range(n):
        for c in range(128):
            colors.append(c)
            valid = True
            for d in range(1, x//3 + 1):
                a = x - 3*d
                b = x - 2*d
                e = x - d
                if a < 0: break
                if len({colors[a], colors[b], colors[e], colors[x]}) < 3:
                    valid = False
                    break
            if valid:
                break
        if len(colors) <= x:
            colors.append(0)
    return colors
