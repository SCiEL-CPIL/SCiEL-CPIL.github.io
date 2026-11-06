import random
import sys

def construct():
    n = 128
    
    def is_valid_placement(colors, pos, color):
        # Check if placing 'color' at 'pos' (0-indexed) is valid
        # Only check APs ending at 'pos'
        for d in range(1, pos//3 + 1):
            if pos - 3*d < 0:
                break
            c1 = colors[pos-3*d]
            c2 = colors[pos-2*d]
            c3 = colors[pos-d]
            # c4 is 'color'
            if len({c1, c2, c3, color}) < 3:
                return False
        return True

    def solve_backtrack(colors, pos, palette):
        if pos == n:
            return colors
        
        # Heuristic: Try colors that appear less frequently in the last few positions? 
        # Or just random.
        p = palette[:]
        random.shuffle(p)
        
        for color in p:
            if is_valid_placement(colors, pos, color):
                colors.append(color)
                result = solve_backtrack(colors, pos + 1, palette)
                if result is not None:
                    return result
                colors.pop()
        
        return None

    # Try 3 colors with multiple restarts
    for attempt in range(50):
        random.seed(attempt)
        res = solve_backtrack([], 0, [0, 1, 2])
        if res is not None:
            # Final verification
            valid = True
            for x in range(n):
                for d in range(1, x//3 + 1):
                    if x - 3*d >= 0:
                        if len({res[x-3*d], res[x-2*d], res[x-d], res[x]}) < 3:
                            valid = False
                            break
                if not valid:
                    break
            if valid:
                return res

    # Try 4 colors
    for attempt in range(50):
        random.seed(attempt + 100)
        res = solve_backtrack([], 0, [0, 1, 2, 3])
        if res is not None:
            valid = True
            for x in range(n):
                for d in range(1, x//3 + 1):
                    if x - 3*d >= 0:
                        if len({res[x-3*d], res[x-2*d], res[x-d], res[x]}) < 3:
                            valid = False
                            break
                if not valid:
                    break
            if valid:
                return res

    # Fallback: Greedy with 4 colors (no backtrack, but with lookahead? No, just greedy)
    # If backtrack fails, it's rare. Greedy might work.
    colors = []
    for x in range(n):
        found = False
        for c in [0, 1, 2, 3]:
            colors.append(c)
            ok = is_valid_placement(colors, x, c)
            if ok:
                found = True
                break
            colors.pop()
        if not found:
            # This shouldn't happen with 4 colors
            colors.append(0)
    return colors
