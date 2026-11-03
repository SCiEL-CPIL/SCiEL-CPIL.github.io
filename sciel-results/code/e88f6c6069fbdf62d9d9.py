import time
import numpy as np

DIMENSIONS = [3, 5, 5]

def construct():
    start = time.time()
    deadline = start + 140.0
    n, m, p = DIMENSIONS  # 3, 5, 5

    # Find maximum set of non-overlapping 2x2x2 Strassen blocks
    # via greedy search with backtracking
    def find_max_blocks(n, m, p, time_limit=30.0):
        """Find maximum number of non-overlapping 2x2x2 blocks."""
        best_count = 0
        best_blocks = []
        
        # All possible block positions
        all_positions = []
        for i0 in range(n - 1):
            for j0 in range(m - 1):
                for k0 in range(p - 1):
                    all_positions.append((i0, j0, k0))
        
        # Greedy with local search
        used = [[[False]*p for _ in range(m)] for _ in range(n)]
        
        def is_compatible(pos, used):
            i0, j0, k0 = pos
            for di in range(2):
                for dj in range(2):
                    for dk in range(2):
                        if used[i0+di][j0+dj][k0+dk]:
                            return False
            return True
        
        def place_block(pos, used, sign=1):
            i0, j0, k0 = pos
            for di in range(2):
                for dj in range(2):
                    for dk in range(2):
                        used[i0+di][j0+dj][k0+dk] = sign
        
        # Greedy: sort by some heuristic and place
        # Try multiple orderings
        import random
        random.seed(42)
        
        best = 0
        best_set = []
        
        for attempt in range(50):
            if time.time() - start > time_limit + start:
                break
            perm = all_positions[:]
            random.shuffle(perm)
            used = [[[False]*p for _ in range(m)] for _ in range(n)]
            placed = []
            for pos in perm:
                if is_compatible(pos, used):
                    place_block(pos, used, 1)
                    placed.append(pos)
            if len(placed) > best:
                best = len(placed)
                best_set = placed[:]
        
        return best_set
    
    blocks = find_max_blocks(n, m, p, time_limit=20.0)
    n_blocks = len(blocks)
    n_strassen = n_blocks * 7
    n_school = n * m * p - n_blocks * 8
    total_rank = n_strassen + n_school

    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]

    col = 0

    def add_strassen_block(i0, j0, k0, col_offset):
        c = col_offset + 0
        U[i0*m + j0][c] += 1
        U[(i0+1)*m + (j0+1)][c] += 1
        V[j0*p + k0][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        W[k0*n + i0][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1

        c = col_offset + 1
        U[(i0+1)*m + j0][c] += 1
        U[(i0+1)*m + (j0+1)][c] += 1
        V[j0*p + k0][c] += 1
        W[k0*n + (i0+1)][c] += 1
        W[(k0+1)*n + (i0+1)][c] += -1

        c = col_offset + 2
        U[i0*m + j0][c] += 1
        V[j0*p + (k0+1)][c] += 1
        V[(j0+1)*p + (k0+1)][c] += -1
        W[(k0+1)*n + i0][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1

        c = col_offset + 3
        U[(i0+1)*m + (j0+1)][c] += 1
        V[(j0+1)*p + k0][c] += 1
        V[j0*p + k0][c] += -1
        W[k0*n + i0][c] += 1
        W[k0*n + (i0+1)][c] += 1

        c = col_offset + 4
        U[i0*m + j0][c] += 1
        U[i0*m + (j0+1)][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        W[k0*n + i0][c] += -1
        W[(k0+1)*n + i0][c] += 1

        c = col_offset + 5
        U[(i0+1)*m + j0][c] += 1
        U[i0*m + j0][c] += -1
        V[j0*p + k0][c] += 1
        V[j0*p + (k0+1)][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1

        c = col_offset + 6
        U[i0*m + (j0+1)][c] += 1
        U[(i0+1)*m + (j0+1)][c] += -1
        V[(j0+1)*p + k0][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        W[k0*n + i0][c] += 1

    for i0, j0, k0 in blocks:
        add_strassen_block(i0, j0, k0, col)
        col += 7

    # Mark covered triples
    covered = [[[False]*p for _ in range(m)] for _ in range(n)]
    for i0, j0, k0 in blocks:
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    covered[i0+di][j0+dj][k0+dk] = True

    # Schoolbook for remaining
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if not covered[i][j][k]:
                    U[i*m+j][col] += 1
                    V[j*p+k][col] += 1
                    W[k*n+i][col] += 1
                    col += 1

    # Verify decomposition
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    target = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target[i*m+j, j*p+k, k*n+i] = 1

    T_decomp = np.einsum('ar,br,cr->abc', U_np, V_np, W_np)
    if not np.allclose(T_decomp, target, atol=1e-10):
        # Fallback to pure schoolbook
        rank_s = n * m * p
        U_f = [[0]*rank_s for _ in range(n*m)]
        V_f = [[0]*rank_s for _ in range(m*p)]
        W_f = [[0]*rank_s for _ in range(p*n)]
        c2 = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_f[i*m+j][c2] = 1
                    V_f[j*p+k][c2] = 1
                    W_f[k*n+i][c2] = 1
                    c2 += 1
        return {"U": U_f, "V": V_f, "W": W_f}

    return {"U": U, "V": V, "W": W}
