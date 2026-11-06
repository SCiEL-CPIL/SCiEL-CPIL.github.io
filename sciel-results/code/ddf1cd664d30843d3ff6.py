def construct():
    n = 256
    # Thue-Morse initial coloring
    color = [0] * n
    for i in range(n):
        # t(i) = popcount(i) % 2
        color[i] = bin(i).count('1') % 2
    
    # Count monochromatic 4-APs
    def count_mono_4ap(color):
        count = 0
        for d in range(1, n // 3 + 1):
            for a in range(1, n - 3 * d + 1):
                if color[a-1] == color[a+d-1] == color[a+2*d-1] == color[a+3*d-1]:
                    count += 1
        return count
    
    # Local search
    def delta_from_flip(color, p, n):
        """Compute change in monochromatic 4-AP count when flipping position p (0-indexed)."""
        delta = 0
        # p is at index p, value is color[p]
        # For each role in each 4-term AP containing p:
        for d in range(1, n // 3 + 1):
            # p as first element: (p, p+d, p+2d, p+3d) 0-indexed
            if p + 3*d < n:
                others = [color[p+d], color[p+2*d], color[p+3*d]]
                if all(o == color[p] for o in others):
                    delta -= 1  # was mono, now not
                elif all(o != color[p] for o in others):
                    delta += 1  # now mono
            # p as second: (p-d, p, p+d, p+2d)
            if p - d >= 0 and p + 2*d < n:
                others = [color[p-d], color[p+d], color[p+2*d]]
                if all(o == color[p] for o in others):
                    delta -= 1
                elif all(o != color[p] for o in others):
                    delta += 1
            # p as third: (p-2d, p-d, p, p+d)
            if p - 2*d >= 0 and p + d < n:
                others = [color[p-2*d], color[p-d], color[p+d]]
                if all(o == color[p] for o in others):
                    delta -= 1
                elif all(o != color[p] for o in others):
                    delta += 1
            # p as fourth: (p-3d, p-2d, p-d, p)
            if p - 3*d >= 0:
                others = [color[p-3*d], color[p-2*d], color[p-d]]
                if all(o == color[p] for o in others):
                    delta -= 1
                elif all(o != color[p] for o in others):
                    delta += 1
        return delta
    
    # Run local search
    improved = True
    while improved:
        improved = False
        for p in range(n):
            d = delta_from_flip(color, p, n)
            if d < 0:
                color[p] = 1 - color[p]
                improved = True
    
    # Return 1-indexed colors
    return [color[i] for i in range(n)]
