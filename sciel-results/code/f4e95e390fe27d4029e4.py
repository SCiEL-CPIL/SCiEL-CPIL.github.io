"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    """Find a 4-coloring of K_n with no monochromatic triangles, maximizing n."""
    deadline = time.time() + 110  # Safety margin before 120s limit
    
    # Start with the known 32-vertex Greenwood-Gleason coloring
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
    
    # Build adjacency matrix for n vertices (n x n), color[i][j] = color of edge (i,j)
    # Use a list of lists for efficiency
    n = 32
    color_matrix = [[-1] * n for _ in range(n)]
    
    for i in range(n):
        for j in range(i + 1, n):
            if i // 16 == j // 16:
                c = log[(i % 16) ^ (j % 16)] % 3
            else:
                c = 3
            color_matrix[i][j] = c
            color_matrix[j][i] = c
    
    def count_mono_triangles(cm, n):
        """Count monochromatic triangles in the coloring."""
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = cm[i][j]
                for k in range(j + 1, n):
                    if cm[i][k] == c and cm[j][k] == c:
                        count += 1
        return count
    
    def flip_edge(cm, n, i, j, new_color):
        """Flip edge (i,j) to new_color and return delta in mono triangle count."""
        old_color = cm[i][j]
        if old_color == new_color:
            return 0
        
        # Count triangles that become mono or stop being mono
        # For each k, check if (i,k) and (j,k) form a triangle with (i,j)
        delta = 0
        for k in range(n):
            if k == i or k == j:
                continue
            ci_k = cm[i][k]
            cj_k = cm[j][k]
            
            # Before flip: triangle (i,j,k) is mono if all three edges have same color
            was_mono = (ci_k == old_color and cj_k == old_color)
            # After flip: triangle (i,j,k) is mono if all three edges have new_color
            is_mono = (ci_k == new_color and cj_k == new_color)
            
            if is_mono and not was_mono:
                delta += 1
            elif was_mono and not is_mono:
                delta -= 1
        
        cm[i][j] = new_color
        cm[j][i] = new_color
        return delta
    
    def local_search(cm, n, max_iter=50000):
        """Local search to reduce mono triangles to zero."""
        best_delta = count_mono_triangles(cm, n)
        
        for iteration in range(max_iter):
            if best_delta == 0:
                return True
            
            if time.time() > deadline - 10:
                return best_delta == 0
            
            # Find an edge to flip that reduces mono triangles
            best_flip = None
            best_improvement = 0
            
            # Sample a subset of edges for speed
            num_samples = min(n * (n - 1) // 2, 200)
            edges = []
            for i in range(n):
                for j in range(i + 1, n):
                    edges.append((i, j))
            
            random.shuffle(edges)
            for (i, j) in edges[:num_samples]:
                old_c = cm[i][j]
                for new_c in range(4):
                    if new_c == old_c:
                        continue
                    # Compute delta without actually flipping
                    delta = 0
                    for k in range(n):
                        if k == i or k == j:
                            continue
                        ci_k = cm[i][k]
                        cj_k = cm[j][k]
                        was_mono = (ci_k == old_c and cj_k == old_c)
                        is_mono = (ci_k == new_c and cj_k == new_c)
                        if is_mono and not was_mono:
                            delta += 1
                        elif was_mono and not is_mono:
                            delta -= 1
                    
                    if delta < -best_improvement:
                        best_improvement = -delta
                        best_flip = (i, j, new_c)
            
            if best_flip is None:
                # No improving move found, try random restart or accept
                # Try a random flip to escape local minimum
                i, j = random.randint(0, n - 1), random.randint(0, n - 1)
                if i == j:
                    continue
                if i > j:
                    i, j = j, i
                new_c = random.randint(0, 3)
                old_c = cm[i][j]
                if new_c != old_c:
                    delta = 0
                    for k in range(n):
                        if k == i or k == j:
                            continue
                        ci_k = cm[i][k]
                        cj_k = cm[j][k]
                        was_mono = (ci_k == old_c and cj_k == old_c)
                        is_mono = (ci_k == new_c and cj_k == new_c)
                        if is_mono and not was_mono:
                            delta += 1
                        elif was_mono and not is_mono:
                            delta -= 1
                    cm[i][j] = new_c
                    cm[j][i] = new_c
                    best_delta += delta
            else:
                i, j, new_c = best_flip
                delta = flip_edge(cm, n, i, j, new_c)
                best_delta += delta
        
        return best_delta == 0
    
    def try_grow(cm, n):
        """Try to add a new vertex to the coloring."""
        if time.time() > deadline - 15:
            return False
        
        new_n = n + 1
        new_cm = [row[:] for row in cm]
        new_cm.append([-1] * new_n)
        for i in range(new_n):
            new_cm[i].append(-1)
        
        # Determine colors for edges from new vertex to old vertices
        # For each old vertex i, choose color c such that no mono triangle is created
        # A mono triangle would be formed if there exist j, k < new_n such that
        # cm[i][j] == cm[i][k] == c and cm[j][k] == c
        
        colors_for_new = []
        for i in range(n):
            valid_colors = set([0, 1, 2, 3])
            for j in range(n):
                if j == i:
                    continue
                # If we set color(i, new) = c, and there exists k with color(i,k)=c and color(j,k)=c,
                # then triangle (i, j, k) would be mono if color(i,j)=c too... 
                # Actually, triangle (i, new, j) is mono if color(i,new)=color(new,j)=color(i,j)
                # So for color(i,new) = c, we need that for all j, NOT(color(i,j)=c and color(new,j)=c)
                # But we're choosing colors sequentially, so this is complex.
                
                # Simpler: for each potential color c for edge (i, new), check if it creates
                # a mono triangle with any pair (j, k) where j, k < new
                # Triangle (i, j, new) is mono if color(i,j)=color(i,new)=color(j,new)
                # So if we set color(i,new)=c, then for any j with color(i,j)=c, we must NOT have
                # color(j,new)=c.
                pass
            
            # For now, just try all 4 colors and see which ones don't immediately create issues
            # This is a simplified approach
            colors_for_new.append(None)
        
        # This is getting complex. Let's use a different approach:
        # Try to extend by checking if we can assign colors to new vertex's edges
        # such that no mono triangle is formed.
        
        # For the new vertex (index n), we need to assign colors to edges (i, n) for i in range(n)
        # such that no triangle (i, j, n) is monochromatic.
        # Triangle (i, j, n) is mono if color(i,j) == color(i,n) == color(j,n).
        
        # This is a constraint satisfaction problem. Let's try a greedy approach.
        assignment = [None] * n
        
        # Order vertices by some heuristic (e.g., degree in each color class)
        order = list(range(n))
        
        for idx in order:
            # Try each color for edge (idx, n)
            found = False
            for c in [0, 1, 2, 3]:
                # Check if assigning color c to (idx, n) creates a mono triangle
                # with any already-assigned vertex
                ok = True
                for j in range(idx):
                    if assignment[j] is not None:
                        if cm[idx][j] == c and assignment[j] == c:
                            ok = False
                            break
                if ok:
                    assignment[idx] = c
                    found = True
                    break
            
            if not found:
                # Try with backtracking or just fail
                return False
        
        # Apply the assignment
        for i in range(n):
            new_cm[i][n] = assignment[i]
            new_cm[n][i] = assignment[i]
        
        # Verify no mono triangles
        if count_mono_triangles(new_cm, new_n) > 0:
            return False
        
        cm[:] = new_cm
        return True
    
    # Run local search on the initial 32-vertex coloring
    if local_search(color_matrix, n, max_iter=100000):
        # Try to grow
        while time.time() < deadline - 20:
            if not try_grow(color_matrix, n):
                break
            n += 1
    
    # Extract colors in lexicographic order
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            colors.append(color_matrix[i][j])
    
    return {'n': n, 'colors': colors}
# EVOLVE-BLOCK-END
