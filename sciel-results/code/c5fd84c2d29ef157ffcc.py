import time

def _check_validity(colors):
    """Check if the coloring is valid for all 4-APs."""
    n = len(colors)
    for x in range(1, n + 1):
        max_d = (x - 1) // 3
        for d in range(1, max_d + 1):
            i1 = x - 3 * d - 1
            i2 = x - 2 * d - 1
            i3 = x - d - 1
            i4 = x - 1
            c1, c2, c3, c4 = colors[i1], colors[i2], colors[i3], colors[i4]
            if len({c1, c2, c3, c4}) < 3:
                return False
    return True

def _solve_k_colors(k, n=128, time_limit=55):
    start_time = time.time()
    
    def is_valid_assignment(colors, i, c):
        max_d = i // 3
        for d in range(1, max_d + 1):
            i1 = i - 3 * d
            i2 = i - 2 * d
            i3 = i - d
            c1, c2, c3 = colors[i1], colors[i2], colors[i3]
            s = {c1, c2, c3}
            if len(s) == 1:
                return False
            elif len(s) == 2:
                if c in s:
                    return False
        return True

    def backtrack(colors, i):
        if time.time() - start_time > time_limit:
            return False
        if i == n:
            return True
        
        # Try colors 0 to k-1
        # Optional: Shuffle for random restarts
        for c in range(k):
            if is_valid_assignment(colors, i, c):
                colors[i] = c
                if backtrack(colors, i + 1):
                    return True
                colors[i] = -1
        return False

    colors = [-1] * n
    if backtrack(colors, 0):
        return colors
    return None

def construct():
    # 3 colors is impossible for N=128
    # Try 4 colors
    res = _solve_k_colors(4)
    if res is not None:
        if _check_validity(res):
            return res
    
    # Try 5 colors
    res = _solve_k_colors(5)
    if res is not None:
        if _check_validity(res):
            return res

    # Fallback
    return [0] * 128
