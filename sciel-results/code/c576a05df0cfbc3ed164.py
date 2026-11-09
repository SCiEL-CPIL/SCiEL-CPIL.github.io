import time
import random
from itertools import product

def construct():
    deadline = time.time() + 80  # safety margin
    
    n = 128
    
    def count_distinct_colors(colors):
        return len(set(colors))
    
    def is_valid_4ap(colors, pos, d):
        """Check if 4-AP starting at pos with diff d has >= 3 distinct colors"""
        c0 = colors[pos-1]
        c1 = colors[pos+d-1]
        c2 = colors[pos+2*d-1]
        c3 = colors[pos+3*d-1]
        return len({c0, c1, c2, c3}) >= 3
    
    def check_full(colors):
        """Check all 4-APs in the coloring"""
        for start in range(1, n+1):
            for d in range(1, (n - start)//3 + 1):
                if start + 3*d > n:
                    break
                if not is_valid_4ap(colors, start, d):
                    return False
        return True
    
    def check_partial(colors, pos):
        """Check all 4-APs ending at or before pos"""
        for start in range(1, pos+1):
            for d in range(1, (pos - start)//3 + 1):
                if start + 3*d > pos:
                    break
                if not is_valid_4ap(colors, start, d):
                    return False
        return True
    
    # Try to find a valid 3-coloring using backtracking with heuristics
    # Use a greedy approach first, then local search if needed
    
    best_colors = None
    best_count = 4  # Start with upper bound
    
    # Try periodic patterns with 3 colors
    for period in range(1, 20):
        if time.time() > deadline:
            break
        # Try all possible periodic patterns with 3 colors
        # But that's 3^period which is too large for large period
        # For small periods, enumerate
        if period <= 5:
            for pattern in product(range(3), repeat=period):
                if time.time() > deadline:
                    break
                colors = [pattern[i % period] for i in range(n)]
                if check_full(colors):
                    cnt = count_distinct_colors(colors)
                    if cnt < best_count:
                        best_count = cnt
                        best_colors = colors[:]
    
    # If we found a 3-coloring, try to improve
    if best_colors and best_count == 3:
        # Try local search to see if we can do better (unlikely with 3 colors)
        # But verify it's correct
        if check_full(best_colors):
            return best_colors
    
    # Backtracking search for minimal colors
    # Try with 3 colors first using backtracking
    def backtrack_search(num_colors, max_time=60):
        """Backtracking search for valid coloring with num_colors colors"""
        start_time = time.time()
        colors = [-1] * n
        
        def is_valid_partial(pos):
            """Check if coloring up to pos is valid for all 4-APs fully within [1,pos]"""
            for start in range(1, pos+1):
                for d in range(1, (pos - start)//3 + 1):
                    if start + 3*d > pos:
                        break
                    c0 = colors[start-1]
                    c1 = colors[start+d-1]
                    c2 = colors[start+2*d-1]
                    c3 = colors[start+3*d-1]
                    if len({c0, c1, c2, c3}) < 3:
                        return False
            return True
        
        def backtrack(pos):
            if time.time() - start_time > max_time:
                return None
            if pos > n:
                return colors[:]
            
            for c in range(num_colors):
                colors[pos-1] = c
                if is_valid_partial(pos):
                    result = backtrack(pos + 1)
                    if result is not None:
                        return result
            colors[pos-1] = -1
            return None
        
        return backtrack(1)
    
    # Try 3-color backtracking with time limit
    if best_count > 3 and time.time() < deadline - 10:
        result = backtrack_search(3, max_time=min(30, deadline - time.time() - 5))
        if result and check_full(result):
            best_colors = result
            best_count = 3
    
    # If still no 3-coloring, try 4-color backtracking
    if best_colors is None and time.time() < deadline - 10:
        result = backtrack_search(4, max_time=min(20, deadline - time.time() - 5))
        if result and check_full(result):
            best_colors = result
            best_count = 4
    
    # Fallback: greedy construction
    if best_colors is None:
        colors = []
        for x in range(1, n+1):
            found = False
            for c in range(3):
                colors.append(c)
                if check_partial(colors, x):
                    found = True
                    break
                colors.pop()
            if not found:
                # Try 4th color
                colors.append(3)
                if not check_partial(colors, x):
                    # This shouldn't happen, but safety
                    colors[-1] = 0
        best_colors = colors
        best_count = count_distinct_colors(colors)
    
    return best_colors
