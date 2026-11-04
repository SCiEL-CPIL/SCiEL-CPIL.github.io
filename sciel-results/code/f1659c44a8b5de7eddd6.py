"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130

    n, m, p = DIMENSIONS

    def add_strassen(U, V, W, i0, j0, k0, col):
        # M1
        U[i0 * m + j0][col] += 1
        U[(i0 + 1) * m + j0 + 1][col] += 1
        V[j0 * p + k0][col] += 1
        V[(j0 + 1) * p + k0 + 1][col] += 1
        W[k0 * n + i0][col] += 1
        W[(k0 + 1) * n + i0 + 1][col] += 1
        col += 1
        # M2
        U[(i0 + 1) * m + j0][col] += 1
        U[(i0 + 1) * m + j0 + 1][col] += 1
        V[j0 * p + k0][col] += 1
        W[k0 * n + i0 + 1][col] += 1
        W[(k0 + 1) * n + i0 + 1][col] -= 1
        col += 1
        # M3
        U[i0 * m + j0][col] += 1
        V[j0 * p + k0 + 1][col] += 1
        V[(j0 + 1) * p + k0 + 1][col] -= 1
        W[(k0 + 1) * n + i0][col] += 1
        W[(k0 + 1) * n + i0 + 1][col] += 1
        col += 1
        # M4
        U[(i0 + 1) * m + j0 + 1][col] += 1
        V[(j0 + 1) * p + k0][col] += 1
        V[j0 * p + k0][col] -= 1
        W[k0 * n + i0][col] += 1
        W[k0 * n + i0 + 1][col] += 1
        col += 1
        # M5
        U[i0 * m + j0][col] += 1
        U[i0 * m + j0 + 1][col] += 1
        V[(j0 + 1) * p + k0 + 1][col] += 1
        W[(k0 + 1) * n + i0][col] += 1
        W[k0 * n + i0][col] -= 1
        col += 1
        # M6
        U[(i0 + 1) * m + j0][col] += 1
        U[i0 * m + j0][col] -= 1
        V[j0 * p + k0][col] += 1
        V[j0 * p + k0 + 1][col] += 1
        W[(k0 + 1) * n + i0 + 1][col] += 1
        col += 1
        # M7
        U[i0 * m + j0 + 1][col] += 1
        U[(i0 + 1) * m + j0 + 1][col] -= 1
        V[(j0 + 1) * p + k0][col] += 1
        V[(j0 + 1) * p + k0 + 1][col] += 1
        W[k0 * n + i0][col] += 1
        col += 1
        return col

    def add_schoolbook(U, V, W, i_range, j_range, k_range, col):
        for i in i_range:
            for j in j_range:
                for k in k_range:
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return col

    # Greedy search for maximum number of disjoint 2x2x2 Strassen blocks
    # A block is identified by (i0, j0, k0) where i0 in [0, n-2], j0 in [0, m-2], k0 in [0, p-2]
    # Two blocks conflict if they share any row in U, any row in V, or any row in W
    
    # Precompute the "footprint" of each possible block
    # U footprint: rows {i0*m + j0, (i0+1)*m + j0, i0*m + j0+1, (i0+1)*m + j0+1}
    # V footprint: rows {j0*p + k0, j0*p + k0+1, (j0+1)*p + k0, (j0+1)*p + k0+1}
    # W footprint: rows {k0*n + i0, k0*n + i0+1, (k0+1)*n + i0, (k0+1)*n + i0+1}
    
    all_blocks = []
    for i0 in range(n - 1):
        for j0 in range(m - 1):
            for k0 in range(p - 1):
                U_rows = frozenset([i0 * m + j0, (i0 + 1) * m + j0, i0 * m + j0 + 1, (i0 + 1) * m + j0 + 1])
                V_rows = frozenset([j0 * p + k0, j0 * p + k0 + 1, (j0 + 1) * p + k0, (j0 + 1) * p + k0 + 1])
                W_rows = frozenset([k0 * n + i0, k0 * n + i0 + 1, (k0 + 1) * n + i0, (k0 + 1) * n + i0 + 1])
                all_blocks.append((i0, j0, k0, U_rows, V_rows, W_rows))
    
    # Greedy: repeatedly pick the block that conflicts with the fewest remaining blocks
    # For simplicity and speed, use a simple greedy: pick blocks in order, skip if conflicts
    # But we want to maximize count, so let's try a few different orderings
    
    best_blocks = []
    
    # Try multiple random orderings
    import random
    random.seed(42)
    
    for attempt in range(50):
        if time.time() > deadline - 10:
            break
        
        indices = list(range(len(all_blocks)))
        random.shuffle(indices)
        
        selected = []
        used_U = set()
        used_V = set()
        used_W = set()
        
        for idx in indices:
            i0, j0, k0, U_rows, V_rows, W_rows = all_blocks[idx]
            if U_rows & used_U or V_rows & used_V or W_rows & used_W:
                continue
            selected.append((i0, j0, k0))
            used_U |= U_rows
            used_V |= V_rows
            used_W |= W_rows
        
        if len(selected) > len(best_blocks):
            best_blocks = selected
            if len(best_blocks) >= 12:  # Good enough, stop early
                break
    
    # Now build the decomposition
    # First, add Strassen blocks for selected positions
    # Then, cover the remaining terms with schoolbook
    
    # We need to track which (i,j,k) terms are covered by Strassen blocks
    covered = set()
    for (i0, j0, k0) in best_blocks:
        for di in [0, 1]:
            for dj in [0, 1]:
                for dk in [0, 1]:
                    covered.add((i0 + di, j0 + dj, k0 + dk))
    
    rank = 105  # Start with schoolbook rank
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    
    # Add Strassen blocks
    for (i0, j0, k0) in best_blocks:
        if time.time() > deadline - 5:
            break
        col = add_strassen(U, V, W, i0, j0, k0, col)
    
    # Add schoolbook for uncovered terms
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in covered:
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
    
    # Trim unused columns
    U = [row[:col] for row in U]
    V = [row[:col] for row in V]
    W = [row[:col] for row in W]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
