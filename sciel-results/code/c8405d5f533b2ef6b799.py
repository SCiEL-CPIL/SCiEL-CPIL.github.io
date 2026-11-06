"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    """
    Constructs a 4-coloring of K_n without monochromatic triangles.
    Uses Cyclic Symmetry Assumption: c(i, j) depends only on (j - i) % n.
    This reduces the search space significantly.
    """
    
    def solve_cyclic(n, timeout=5):
        """
        Try to find a cyclic 4-coloring for K_n.
        Returns list of colors for edges (i, j) in lexicographic order, or None.
        """
        if n < 3:
            return [0] * (n * (n - 1) // 2)
        
        half_n = n // 2
        f = [None] * (half_n + 1) # 1-indexed for distances 1..half_n
        start_time = time.time()
        
        def get_dist(a, b):
            d = (b - a) % n
            if d > half_n:
                d = n - d
            return d
        
        constraints = []
        for x in range(1, n):
            for y in range(1, n):
                z = (x + y) % n
                if z == 0:
                    continue
                d1 = get_dist(0, x)
                d2 = get_dist(x, x + y)
                d3 = get_dist(x + y, 0)
                constraints.append((d1, d2, d3))
        
        constraints = list(set(constraints))
        
        # Sort constraints to find conflicts earlier?
        # Constraints involving smaller indices first?
        
        def backtrack(idx):
            if time.time() - start_time > timeout:
                return False
            
            if idx > half_n:
                return True
            
            # Heuristic: Try colors that appear less often in constraints?
            # For now, just 0,1,2,3
            
            for color in [0, 1, 2, 3]:
                f[idx] = color
                
                valid = True
                for (a, b, c) in constraints:
                    if a == idx or b == idx or c == idx:
                        val_a = f[a]
                        val_b = f[b]
                        val_c = f[c]
                        
                        if val_a is not None and val_b is not None and val_c is not None:
                            if val_a == val_b == val_c:
                                valid = False
                                break
                            
                if valid:
                    if backtrack(idx + 1):
                        return True
                
                f[idx] = None
            
            return False
        
        if backtrack(1):
            colors = []
            for i in range(n):
                for j in range(i + 1, n):
                    d = get_dist(i, j)
                    colors.append(f[d])
            return colors
        
        return None

    best_result = None
    best_n = 0
    
    # Try high n first
    for n in range(50, 31, -1):
        res = solve_cyclic(n)
        if res is not None:
            best_result = res
            best_n = n
            break
    
    if best_result is None:
        # Fallback: Return the known n=32 construction
        def mul(a, b):
            r = 0
            for i in range(4):
                if b >> i & 1:
                    r ^= a << i
            for i in range(7, 3, -1):
                if r >> i & 1:
                    r ^= 0b10011 << (i - 4)
            return r
        log, x = {}, 1
        for e in range(15):
            log[x] = e
            x = mul(x, 2)
        colors = []
        for i in range(32):
            for j in range(i + 1, 32):
                colors.append(log[(i % 16) ^ (j % 16)] % 3 if i // 16 == j // 16 else 3)
        return {'n': 32, 'colors': colors}
        
    return {'n': best_n, 'colors': best_result}
# EVOLVE-BLOCK-END
