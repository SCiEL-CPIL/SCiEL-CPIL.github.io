"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import time
    start_time = time.time()
    
    # F8 mul with primitive polynomial x^3 + x + 1
    def mul(a, b):
        r = 0
        for i in range(3):
            if b & (1 << i):
                r ^= a << i
        if r & 8:
            r = (r ^ 11) & 7
        return r

    # Find a valid 3-coloring of F8*
    elems = [1, 2, 3, 4, 5, 6, 7]
    color_map = {}
    found = False
    for mask in range(3**7):
        c = []
        m = mask
        for _ in range(7):
            c.append(m % 3)
            m //= 3
        valid = True
        for col in range(3):
            s = [elems[i] for i in range(7) if c[i] == col]
            for i in range(len(s)):
                for j in range(i+1, len(s)):
                    for k in range(j+1, len(s)):
                        if (s[i] + s[j] + s[k]) & 7 == 0:
                            valid = False
                            break
                    if not valid: break
                if not valid: break
            if not valid: break
        if valid:
            color_map = {elems[i]: c[i] for i in range(7)}
            found = True
            break
    
    if not found:
        # Fallback to n=50 if something goes wrong (should not happen)
        # Implement n=50 logic here if needed, but n=62 is expected
        pass

    # Precompute color for each non-zero difference in F8^2
    diff_color = {}
    for dx in range(8):
        for dy in range(8):
            if dx == 0 and dy == 0:
                continue
            if dy == 0:
                diff_color[(dx, dy)] = color_map[dx]
            else:
                # Find t such that t * dy = dx
                t = 0
                for candidate in range(1, 8):
                    if mul(candidate, dy) == dx:
                        t = candidate
                        break
                diff_color[(dx, dy)] = color_map[t]
    
    # Construct 3-coloring of K62 (subset of K64)
    colors = []
    n = 62
    for i in range(n):
        x1 = i % 8
        y1 = i // 8
        for j in range(i + 1, n):
            x2 = j % 8
            y2 = j // 8
            dx = (x2 - x1) & 7
            dy = (y2 - y1) & 7
            colors.append(diff_color[(dx, dy)])
    
    return {'n': n, 'colors': colors}
# EVOLVE-BLOCK-END
