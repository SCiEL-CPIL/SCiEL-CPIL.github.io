import time
import random

def construct():
    N = 128
    start_time = time.time()
    time_limit = 45.0  # Leave 15s margin for 60s limit
    
    # Precompute APs ending at each position i (1-indexed)
    # aps[i] contains tuples (idx1, idx2, idx3) corresponding to i-3d, i-2d, i-d
    aps = [[] for _ in range(N + 1)]
    for i in range(1, N + 1):
        max_d = (i - 1) // 3
        for d in range(1, max_d + 1):
            a = i - 3 * d
            b = i - 2 * d
            c = i - d
            aps[i].append((a, b, c))

    def is_valid_color(i, color, colors):
        # Check if assigning 'color' to position i (1-indexed) is valid
        # given colors[1..i-1]
        for a, b, c in aps[i]:
            # colors is 0-indexed, so index is pos-1
            c1 = colors[a - 1]
            c2 = colors[b - 1]
            c3 = colors[c - 1]
            if len({c1, c2, c3, color}) < 3:
                return False
        return True

    def dfs(pos, k, colors):
        if pos > N:
            return True
        
        if time.time() - start_time > time_limit:
            return False

        # Try colors 0 to k-1
        # Symmetry breaking: if pos == 1, only try 0
        # if pos == 2, only try 0, 1 (if k>=2)
        # This reduces search space significantly
        
        if pos == 1:
            candidates = [0]
        elif pos == 2:
            candidates = [0, 1] if k >= 2 else [0]
        else:
            candidates = list(range(k))
            
        # Optional: Shuffle candidates for pos > 2 to avoid bias?
        # No, deterministic is safer.
        
        for c in candidates:
            if is_valid_color(pos, c, colors):
                colors[pos - 1] = c
                if dfs(pos + 1, k, colors):
                    return True
                colors[pos - 1] = -1
        return False

    # Try increasing number of colors
    for k in range(3, 6):  # Try 3, 4, 5 colors
        colors = [-1] * N
        # Symmetry breaking initialization
        if k >= 1:
            colors[0] = 0
        if k >= 2:
            colors[1] = 1
        else:
            colors[1] = 0
            
        if dfs(3, k, colors):
            return colors

    # Fallback: Greedy approach from seed
    # The seed program logic
    colors = []
    for x in range(1, 129):
        found = False
        for color in range(128):
            valid = True
            # Check all APs ending at x
            max_d = (x - 1) // 3
            for d in range(1, max_d + 1):
                a = x - 3 * d
                b = x - 2 * d
                c = x - d
                # colors is 0-indexed list, so index is pos-1
                c1 = colors[a - 1]
                c2 = colors[b - 1]
                c3 = colors[c - 1]
                if len({c1, c2, c3, color}) < 3:
                    valid = False
                    break
            if valid:
                colors.append(color)
                found = True
                break
        if not found:
            # Should not happen if k is large enough, but just in case
            colors.append(0)
            
    return colors
