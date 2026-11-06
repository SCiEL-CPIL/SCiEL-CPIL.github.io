"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    deadline = time.time() + 140.0  # safety margin before 150s kill
    
    # Start with the known K16 3-coloring (Clebsch graph), doubled to K32
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
    
    # Build K16 3-coloring
    k16_colors = []
    for i in range(16):
        for j in range(i + 1, 16):
            k16_colors.append(log[i ^ j] % 3)
    
    # Double to K32
    def double_coloring(base_colors, m):
        """Double a triangle-free 3-coloring of K_m to a 4-coloring of K_{2m}."""
        n = 2 * m
        colors = []
        for i in range(n):
            for j in range(i + 1, n):
                if i // m == j // m:
                    # Same copy: use base coloring
                    colors.append(base_colors[i % m * (2*m - i % m - 1) + (j % m) - (i % m) - 1])
                else:
                    # Different copies: color 3
                    colors.append(3)
        return colors
    
    # Better: build the base coloring as a 2D list for easy lookup
    def base_to_2d(base_colors, m):
        mat = [[-1] * m for _ in range(m)]
        idx = 0
        for i in range(m):
            for j in range(i + 1, m):
                mat[i][j] = mat[j][i] = base_colors[idx]
                idx += 1
        return mat
    
    def double_coloring_from_mat(mat, m):
        n = 2 * m
        colors = []
        for i in range(n):
            for j in range(i + 1, n):
                if i // m == j // m:
                    colors.append(mat[i % m][j % m])
                else:
                    colors.append(3)
        return colors
    
    k16_mat = base_to_2d(k16_colors, 16)
    best_colors = double_coloring_from_mat(k16_mat, 16)
    best_n = 32
    
    # Try to extend: search for triangle-free 3-colorings of larger K_m
    # Strategy: use backtracking with constraint propagation
    # Try m values from 17 up to 25 (giving n up to 50)
    
    import random
    random.seed(42)
    
    for target_m in range(17, 26):
        if time.time() > deadline:
            break
        
        # Try to find a triangle-free 3-coloring of K_target_m
        # Use randomized local search / backtracking hybrid
        
        def try_find_3coloring(m, timeout=30.0):
            """Find a triangle-free 3-coloring of K_m or return None."""
            start = time.time()
            
            # Initialize with random colors
            colors = [[random.randint(0, 2) for _ in range(m)] for _ in range(m)]
            
            # Build adjacency: colors[i][j] is the color of edge (i,j)
            # A triangle (i,j,k) is monochromatic if colors[i][j] == colors[j][k] == colors[i][k]
            
            def count_triangles():
                count = 0
                for i in range(m):
                    for j in range(i + 1, m):
                        for k in range(j + 1, m):
                            if colors[i][j] == colors[j][k] == colors[i][k]:
                                count += 1
                return count
            
            # Local search: flip colors to reduce triangles
            best_count = count_triangles()
            if best_count == 0:
                return colors
            
            for iteration in range(100000):
                if time.time() - start > timeout:
                    return None
                
                # Pick a random vertex and try flipping its incident edge colors
                v = random.randint(0, m - 1)
                # Try changing one edge incident to v
                u = random.randint(0, m - 1)
                if u == v:
                    continue
                
                old_color = colors[v][u]
                new_color = (old_color + 1) % 3
                
                # Count triangles affected by changing edge (v,u)
                # Triangles containing edge (v,u): for each w != v,u
                delta = 0
                for w in range(m):
                    if w == v or w == u:
                        continue
                    # Before change: triangle (v,u,w) has edges (v,u)=old, (v,w), (u,w)
                    before_mono = 1 if (old_color == colors[v][w] == colors[u][w]) else 0
                    # After change: triangle (v,u,w) has edges (v,u)=new, (v,w), (u,w)
                    after_mono = 1 if (new_color == colors[v][w] == colors[u][w]) else 0
                    delta += after_mono - before_mono
                
                if delta <= 0:
                    colors[v][u] = colors[u][v] = new_color
                    best_count += delta
                    if best_count == 0:
                        return colors
                # else: don't accept (or accept with small probability for simulated annealing)
            
            return None if best_count > 0 else colors
        
        # Try multiple random restarts
        found = None
        max_restarts = 50
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
            result = try_find_3coloring(target_m, timeout=min(20.0, max(1.0, (deadline - time.time()) / (max_restarts - restart))))
            if result is not None:
                found = result
                break
        
        if found is not None:
            # Verify
            m = target_m
            valid = True
            for i in range(m):
                for j in range(i + 1, m):
                    for k in range(j + 1, m):
                        if found[i][j] == found[j][k] == found[i][k]:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            if valid:
                mat = [[0] * m for _ in range(m)]
                for i in range(m):
                    for j in range(i + 1, m):
                        mat[i][j] = mat[j][i] = found[i][j]
                best_colors = double_coloring_from_mat(mat, m)
                best_n = 2 * m
                # Continue searching for larger m
                continue
        else:
            # Couldn't find for this m, stop
            break
    
    return {'n': best_n, 'colors': best_colors}
# EVOLVE-BLOCK-END
