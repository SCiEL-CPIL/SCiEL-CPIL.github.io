"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    random.seed(42)
    
    def count_mono_triangles(color_arr, n):
        """Count monochromatic triangles in the coloring."""
        # Build adjacency matrix for each color
        # color_arr is in lexicographic order: (0,1), (0,2), ..., (0,n-1), (1,2), ...
        # Map edge (i,j) to index
        def edge_idx(i, j):
            if i > j:
                i, j = j, i
            # Number of edges from vertex 0 to i-1, plus position in row i
            # Row 0: n-1 edges, Row 1: n-2 edges, ..., Row i: n-1-i edges
            # Edges before row i: sum_{k=0}^{i-1} (n-1-k) = i*(n-1) - i*(i-1)//2
            # Position in row i: j - i - 1
            return i * (n - 1) - i * (i - 1) // 2 + (j - i - 1)
        
        # Build color matrix
        color_mat = np.zeros((n, n), dtype=np.int8)
        for i in range(n):
            for j in range(i + 1, n):
                c = color_arr[edge_idx(i, j)]
                color_mat[i, j] = c
                color_mat[j, i] = c
        
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = color_mat[i, j]
                for k in range(j + 1, n):
                    if color_mat[i, k] == c and color_mat[j, k] == c:
                        count += 1
        return count
    
    def try_find_coloring(n, max_iters=500000, seed_offset=0):
        """Try to find a valid 4-coloring of K_n with no monochromatic triangles."""
        rng = random.Random(seed_offset)
        m = n * (n - 1) // 2
        
        # Start with random coloring
        colors = [rng.randint(0, 3) for _ in range(m)]
        
        def edge_idx(i, j):
            if i > j:
                i, j = j, i
            return i * (n - 1) - i * (i - 1) // 2 + (j - i - 1)
        
        def count_mono_triangles_fast(colors):
            """Fast count using numpy."""
            color_mat = np.zeros((n, n), dtype=np.int8)
            for i in range(n):
                for j in range(i + 1, n):
                    c = colors[edge_idx(i, j)]
                    color_mat[i, j] = c
                    color_mat[j, i] = c
            
            count = 0
            for i in range(n):
                for j in range(i + 1, n):
                    c = color_mat[i, j]
                    # Find common neighbors with same color
                    for k in range(j + 1, n):
                        if color_mat[i, k] == c and color_mat[j, k] == c:
                            count += 1
            return count
        
        # Simulated annealing
        current_colors = colors[:]
        current_count = count_mono_triangles_fast(current_colors)
        
        best_colors = current_colors[:]
        best_count = current_count
        
        T = 1.0
        cooling = 0.99999
        min_T = 0.001
        
        for it in range(max_iters):
            if current_count == 0:
                return current_colors
            
            # Pick random edge
            ei = rng.randint(0, m - 1)
            old_c = current_colors[ei]
            new_c = rng.randint(0, 3)
            if new_c == old_c:
                continue
            
            current_colors[ei] = new_c
            new_count = count_mono_triangles_fast(current_colors)
            
            delta = new_count - current_count
            if delta < 0 or rng.random() < np.exp(-delta / max(T, 1e-10)):
                current_count = new_count
                if new_count < best_count:
                    best_count = new_count
                    best_colors = current_colors[:]
            else:
                current_colors[ei] = old_c
            
            T *= cooling
            if T < min_T:
                T = min_T
        
        if best_count == 0:
            return best_colors
        return None
    
    # Try to find colorings for decreasing n
    for n in range(50, 32, -1):
        # Multiple restarts
        for attempt in range(20):
            result = try_find_coloring(n, max_iters=300000, seed_offset=attempt * 1000 + n)
            if result is not None:
                return {'n': n, 'colors': result}
    
    # Fallback: return the known 32-vertex construction
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
# EVOLVE-BLOCK-END
