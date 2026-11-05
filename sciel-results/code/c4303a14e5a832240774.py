import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # We aim for n=50
    N = 50
    
    # Function to count monochromatic triangles
    def count_triangles(colors):
        total = 0
        for c in range(4):
            # Create adjacency matrix for color c
            # colors is a flattened list or 2D array? 
            # Let's use a 2D numpy array for speed
            A = (colors == c).astype(np.int8)
            # Triangles = trace(A^3) / 6
            # A is symmetric, 0 diagonal
            # A^3 diagonal elements are 6 * (number of triangles in color c)
            A3 = A @ A @ A
            total += np.trace(A3) // 6
        return total

    # Initialize colors
    # Using a Latin Square based initial guess
    # V = {(i, j) | i in 0..4, j in 0..9}
    # colors[i, j] for i < j
    colors = np.zeros((N, N), dtype=np.int8)
    
    # Latin Square of order 5
    L = np.array([
        [0, 1, 2, 3, 4],
        [1, 2, 3, 4, 0],
        [2, 3, 4, 0, 1],
        [3, 4, 0, 1, 2],
        [4, 0, 1, 2, 3]
    ])
    
    # Simple 4-coloring for K10 (within parts)
    # We can just use (j + m) % 4 for now, or a known good one
    # For K10, (j+m)%4 might have triangles. 
    # Let's just use random for within parts initially, or a specific pattern.
    # Actually, let's just initialize with the formula:
    for i in range(N):
        for j in range(i + 1, N):
            ii = i // 10
            jj = j // 10
            ki = i % 10
            kj = j % 10
            if ii == jj:
                # Within part: use (ki + kj) % 4
                colors[i, j] = (ki + kj) % 4
            else:
                # Between parts: use L[ii][jj] + ki + kj
                colors[i, j] = (L[ii][jj] + ki + kj) % 4
            colors[j, i] = colors[i, j]

    current_count = count_triangles(colors)
    
    best_count = current_count
    best_colors = colors.copy()
    
    # Local Search
    # We use a greedy approach with random restarts/kicks if stuck
    stuck_counter = 0
    
    while time.time() < deadline:
        if current_count == 0:
            break
        
        # Pick a random edge
        i = random.randint(0, N - 1)
        j = random.randint(0, N - 1)
        if i == j:
            continue
        if i > j:
            i, j = j, i
            
        old_color = colors[i, j]
        
        # Try all 4 colors
        best_new_color = old_color
        best_new_count = current_count
        
        for c in range(4):
            if c == old_color:
                continue
            colors[i, j] = c
            colors[j, i] = c
            new_count = count_triangles(colors)
            if new_count < best_new_count:
                best_new_count = new_count
                best_new_color = c
            # Restore for next trial
            colors[i, j] = old_color
            colors[j, i] = old_color
            
        # Apply best color
        if best_new_color != old_color:
            colors[i, j] = best_new_color
            colors[j, i] = best_new_color
            current_count = best_new_count
            stuck_counter = 0
        else:
            stuck_counter += 1
            # If stuck for a while, do a random kick
            if stuck_counter > 100:
                # Pick a random edge and change it randomly
                k = random.randint(0, N - 1)
                m = random.randint(0, N - 1)
                if k != m:
                    if k > m: k, m = m, k
                    new_c = random.randint(0, 3)
                    colors[k, m] = new_c
                    colors[m, k] = new_c
                    current_count = count_triangles(colors)
                    stuck_counter = 0
        
        if current_count < best_count:
            best_count = current_count
            best_colors = colors.copy()
            
        # If we found a valid coloring, we can return immediately
        if current_count == 0:
            break

    # Use the best colors found
    final_colors = best_colors
    
    # Convert to list
    res_colors = []
    for i in range(N):
        for j in range(i + 1, N):
            res_colors.append(int(final_colors[i, j]))
            
    return {'n': N, 'colors': res_colors}
