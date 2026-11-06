import random

def construct():
    n = 128
    
    # Use Legendre symbol modulo p=127 to assign colors
    # Legendre symbol (a/p) is 1 if a is quadratic residue mod p, -1 if non-residue, 0 if a=0
    # We'll map positions 1..128 to colors based on their Legendre symbol mod 127
    # Position 0 (if used) gets a special color
    
    p = 127
    
    # Compute Legendre symbols for 0..126 mod p
    def legendre_symbol(a, p):
        """Compute the Legendre symbol (a/p) for prime p."""
        if a % p == 0:
            return 0
        # Use Euler's criterion: (a/p) = a^((p-1)/2) mod p
        result = pow(a, (p - 1) // 2, p)
        if result == 1:
            return 1
        elif result == p - 1:
            return -1
        else:
            return 0  # shouldn't happen for prime p
    
    # Assign colors based on Legendre symbol
    # Color 0: Legendre symbol = 1 (quadratic residues)
    # Color 1: Legendre symbol = -1 (quadratic non-residues)
    # Color 2: position 0 or any position where a % p == 0
    
    colors = [0] * n
    for i in range(n):
        pos = i + 1  # positions are 1-indexed in the problem
        ls = legendre_symbol(pos, p)
        if ls == 1:
            colors[i] = 0
        elif ls == -1:
            colors[i] = 1
        else:
            colors[i] = 2
    
    # Verify the coloring: every nonconstant 4-term AP must have at least 3 distinct colors
    # A nonconstant 4-term AP is (a, a+d, a+2d, a+3d) with d > 0
    valid = True
    for a in range(n):
        for d in range(1, (n - 1 - a) // 3 + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            if e < n:
                distinct = len({colors[a], colors[b], colors[c], colors[e]})
                if distinct < 3:
                    valid = False
                    break
        if not valid:
            break
    
    # If the Legendre symbol approach doesn't work, fall back to a better heuristic
    if not valid:
        # Try a different modular approach: use position mod small numbers
        # Color based on (i mod 2, i mod 3) pattern
        # This gives up to 6 colors, but let's try to minimize
        
        # Actually, let's try a greedy approach with better ordering
        best_colors = None
        best_num_colors = n + 1
        
        for attempt in range(50):
            colors = [-1] * n
            success = True
            for i in range(n):
                candidates = list(range(4))  # Try at most 4 colors
                random.shuffle(candidates)
                placed = False
                for c in candidates:
                    # Check if assigning c to i is valid
                    valid_c = True
                    for d in range(1, (i // 3) + 1):
                        a = i - 3 * d
                        if a >= 0:
                            if colors[a] != -1 and colors[a+d] != -1 and colors[a+2*d] != -1:
                                if len({colors[a], colors[a+d], colors[a+2*d], c}) < 3:
                                    valid_c = False
                                    break
                        if not valid_c:
                            break
                    if not valid_c:
                        continue
                    for d in range(1, n):
                        a = i - 2 * d
                        b = i + d
                        if a >= 0 and b < n:
                            if colors[a] != -1 and colors[b] != -1:
                                if len({colors[a], colors[a+d], colors[b], c}) < 3:
                                    valid_c = False
                                    break
                        if not valid_c:
                            break
                    if not valid_c:
                        continue
                    for d in range(1, n):
                        a = i - d
                        b = i + d
                        c2 = i + 2 * d
                        if a >= 0 and b < n and c2 < n:
                            if colors[a] != -1 and colors[b] != -1 and colors[c2] != -1:
                                if len({colors[a], colors[b], colors[c2], c}) < 3:
                                    valid_c = False
                                    break
                        if not valid_c:
                            break
                    if not valid_c:
                        continue
                    for d in range(1, n):
                        b = i + d
                        c2 = i + 2 * d
                        d2 = i + 3 * d
                        if b < n and c2 < n and d2 < n:
                            if colors[b] != -1 and colors[c2] != -1 and colors[d2] != -1:
                                if len({colors[b], colors[c2], colors[d2], c}) < 3:
                                    valid_c = False
                                    break
                        if not valid_c:
                            break
                    if valid_c:
                        colors[i] = c
                        placed = True
                        break
                if not placed:
                    success = False
                    break
            if success:
                num_colors = len(set(colors))
                if num_colors < best_num_colors:
                    best_num_colors = num_colors
                    best_colors = colors[:]
                if best_num_colors == 3:
                    break
        
        if best_colors is not None:
            return best_colors
        else:
            # Last resort: use all distinct colors
            return list(range(n))
    
    return colors
