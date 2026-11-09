import time
import random
import numpy as np

def construct():
    start_time = time.time()
    time_limit = 140.0  # Safety margin

    # 1. Define the n=32 coloring (Greenwood-Gleason doubled)
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

    colors_32 = []
    for i in range(32):
        for j in range(i + 1, 32):
            colors_32.append(log[(i % 16) ^ (j % 16)] % 3 if i // 16 == j // 16 else 3)

    base_n = 32
    base_colors = colors_32

    # 2. Extend to n=50
    target_n = 50
    n = target_n

    def count_triangles_vectorized(cm):
        """Count monochromatic triangles using vectorized NumPy operations."""
        count = 0
        for c in range(4):
            # Create adjacency matrix for color c
            adj = (cm == c).astype(np.int32)
            np.fill_diagonal(adj, 0)
            # Number of triangles = trace(adj^3) / 6
            adj2 = adj @ adj
            count += np.trace(adj2 @ adj) // 6
        return count

    def compute_delta(cm, u, v, old_c, new_c):
        """Compute change in triangle count when changing edge (u,v) from old_c to new_c."""
        delta = 0
        cuw = cm[u]
        cvw = cm[v]
        # For all w != u, v:
        # Old: triangle mono if old_c == cuw[w] and old_c == cvw[w]
        # New: triangle mono if new_c == cuw[w] and new_c == cvw[w]
        old_mono = (cuw == old_c) & (cvw == old_c)
        new_mono = (cuw == new_c) & (cvw == new_c)
        # Exclude u and v themselves
        old_mono[u] = False
        old_mono[v] = False
        new_mono[u] = False
        new_mono[v] = False
        delta = new_mono.sum() - old_mono.sum()
        return int(delta)

    best_overall_tri_count = float('inf')
    best_overall_matrix = None

    # Multiple restarts
    max_restarts = 5
    for restart in range(max_restarts):
        if time.time() - start_time >= time_limit:
            break

        # Initialize
        color_matrix = np.zeros((n, n), dtype=np.uint8)

        # Fill base 32
        idx = 0
        for i in range(32):
            for j in range(i + 1, 32):
                color_matrix[i, j] = base_colors[idx]
                color_matrix[j, i] = base_colors[idx]
                idx += 1

        # Fill remaining edges randomly
        for i in range(32, n):
            for j in range(i + 1, n):
                c = random.randint(0, 3)
                color_matrix[i, j] = c
                color_matrix[j, i] = c
        # Edges between base and new
        for i in range(32):
            for j in range(32, n):
                c = random.randint(0, 3)
                color_matrix[i, j] = c
                color_matrix[j, i] = c

        # Initial count using vectorized method
        tri_count = count_triangles_vectorized(color_matrix)

        best_tri_count = tri_count
        best_color_matrix = color_matrix.copy()

        T = 1.0
        T_min = 0.01
        alpha = 0.9995

        while time.time() - start_time < time_limit:
            if T < T_min:
                break

            # Pick a random edge (u, v)
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v:
                continue
            if u > v:
                u, v = v, u

            old_c = color_matrix[u, v]
            new_c = random.randint(0, 3)
            if new_c == old_c:
                continue

            # Calculate delta in triangle count (vectorized)
            delta = compute_delta(color_matrix, u, v, old_c, new_c)

            new_tri_count = tri_count + delta

            # Accept if better or with probability exp(-delta/T) if delta > 0
            if delta < 0 or (delta > 0 and random.random() < np.exp(-delta / T)):
                color_matrix[u, v] = new_c
                color_matrix[v, u] = new_c
                tri_count = new_tri_count

                if tri_count < best_tri_count:
                    best_tri_count = tri_count
                    best_color_matrix = color_matrix.copy()
                    if best_tri_count == 0:
                        break

            T *= alpha

        # Update overall best
        if best_tri_count < best_overall_tri_count:
            best_overall_tri_count = best_tri_count
            best_overall_matrix = best_color_matrix.copy()
            if best_overall_tri_count == 0:
                break

    # 4. Return result
    if best_overall_tri_count == 0:
        # Flatten best_color_matrix
        colors = []
        for i in range(n):
            for j in range(i + 1, n):
                colors.append(int(best_overall_matrix[i, j]))
        return {'n': n, 'colors': colors}
    else:
        # Return base n=32
        return {'n': 32, 'colors': base_colors}
