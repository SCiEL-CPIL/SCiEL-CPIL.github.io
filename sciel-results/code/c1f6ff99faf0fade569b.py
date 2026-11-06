import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 85s deadline with 5s margin
    
    n = 128
    # Try to find coloring with minimum number of colors
    # Start with 3 colors (theoretical minimum since 2 colors can't work)
    
    for num_colors in range(3, 129):
        if time.time() > deadline:
            break
        result = _try_coloring(n, num_colors, deadline)
        if result is not None:
            return result
    
    # Fallback: greedy with 128 colors (always valid)
    return list(range(128))

def _try_coloring(n, num_colors, deadline):
    """Try to find a valid coloring using at most num_colors colors."""
    colors = [None] * n
    
    def is_valid_position(pos, color):
        """Check if assigning color to position pos (0-indexed) is valid."""
        # Check all 4-term progressions that include pos
        # pos can be at index 0,1,2,3 in the progression
        # progression: a, a+d, a+2d, a+3d where pos = a+kd for k in 0,1,2,3
        
        # Case 1: pos is first term (k=0)
        # a=pos, a+d, a+2d, a+3d must be < n
        for d in range(1, (n - pos - 1) // 3 + 1):
            if pos + 3*d >= n:
                break
            p1, p2, p3 = pos + d, pos + 2*d, pos + 3*d
            if colors[p1] is None or colors[p2] is None or colors[p3] is None:
                continue
            prog_colors = {color, colors[p1], colors[p2], colors[p3]}
            if len(prog_colors) < 3:
                return False
        
        # Case 2: pos is second term (k=1)
        # a=pos-d, pos, pos+d, pos+2d
        for d in range(1, pos // 1 + 1):
            if pos - d < 0:
                break
            if pos + 2*d >= n:
                break
            p0, p2, p3 = pos - d, pos + d, pos + 2*d
            if colors[p0] is None or colors[p2] is None or colors[p3] is None:
                continue
            prog_colors = {colors[p0], color, colors[p2], colors[p3]}
            if len(prog_colors) < 3:
                return False
        
        # Case 3: pos is third term (k=2)
        # a=pos-2d, a+d=pos-d, pos, pos+d
        for d in range(1, (pos // 2) + 1):
            if pos - 2*d < 0:
                break
            if pos + d >= n:
                break
            p0, p1, p3 = pos - 2*d, pos - d, pos + d
            if colors[p0] is None or colors[p1] is None or colors[p3] is None:
                continue
            prog_colors = {colors[p0], colors[p1], color, colors[p3]}
            if len(prog_colors) < 3:
                return False
        
        # Case 4: pos is fourth term (k=3)
        # a=pos-3d, a+d=pos-2d, a+2d=pos-d, pos
        for d in range(1, (pos // 3) + 1):
            if pos - 3*d < 0:
                break
            p0, p1, p2 = pos - 3*d, pos - 2*d, pos - d
            if colors[p0] is None or colors[p1] is None or colors[p2] is None:
                continue
            prog_colors = {colors[p0], colors[p1], colors[p2], color}
            if len(prog_colors) < 3:
                return False
        
        return True
    
    def backtrack(pos):
        if time.time() > deadline:
            return False
        if pos == n:
            return True
        
        # Try each color in order
        for c in range(num_colors):
            if is_valid_position(pos, c):
                colors[pos] = c
                if backtrack(pos + 1):
                    return True
                colors[pos] = None
        
        return False
    
    if backtrack(0):
        # Map to smallest available labels
        used = sorted(set(colors))
        mapping = {old: new for new, old in enumerate(used)}
        return [mapping[c] for c in colors]
    
    return None
