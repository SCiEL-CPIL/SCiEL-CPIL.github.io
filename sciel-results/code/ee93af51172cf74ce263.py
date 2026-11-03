"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140
    
    def build_decomp(n, m, p):
        """Build U, V, W for n×m × m×p multiplication. Returns (U, V, W, rank)."""
        # Base case: 1×1×1
        if n == 1 and m == 1 and p == 1:
            return [[1]], [[1]], [[1]], 1
        
        # Degenerate cases
        if n == 1:
            rank = m * p
            U = [[0] * rank for _ in range(m)]
            V = [[0] * rank for _ in range(m * p)]
            W = [[0] * rank for _ in range(p)]
            col = 0
            for j in range(m):
                for k in range(p):
                    U[j][col] = 1
                    V[j * p + k][col] = 1
                    W[k][col] = 1
                    col += 1
            return U, V, W, rank
        
        if m == 1:
            rank = n * p
            U = [[0] * rank for _ in range(n)]
            V = [[0] * rank for _ in range(p)]
            W = [[0] * rank for _ in range(p * n)]
            col = 0
            for i in range(n):
                for k in range(p):
                    U[i][col] = 1
                    V[k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
            return U, V, W, rank
        
        if p == 1:
            rank = n * m
            U = [[0] * rank for _ in range(n * m)]
            V = [[0] * rank for _ in range(m)]
            W = [[0] * rank for _ in range(n)]
            col = 0
            for i in range(n):
                for j in range(m):
                    U[i * m + j][col] = 1
                    V[j][col] = 1
                    W[i][col] = 1
                    col += 1
            return U, V, W, rank
        
        # Strassen for 2×2×2
        if n == 2 and m == 2 and p == 2:
            rank = 7
            U = [[0] * rank for _ in range(4)]
            V = [[0] * rank for _ in range(4)]
            W = [[0] * rank for _ in range(4)]
            
            # M1 = (A11+A22)*(B11+B22)
            U[0][0] = 1; U[3][0] = 1
            V[0][0] = 1; V[3][0] = 1
            W[0][0] = 1; W[3][0] = 1  # C11, C22
            
            # M2 = (A21+A22)*B11
            U[2][1] = 1; U[3][1] = 1
            V[0][1] = 1
            W[1][1] = 1; W[3][1] = 1  # C21, C22
            
            # M3 = A11*(B12-B22)
            U[0][2] = 1
            V[1][2] = 1; V[3][2] = -1
            W[2][2] = 1; W[3][2] = -1  # C12, -C22
            
            # M4 = A22*(B21-B11)
            U[3][3] = 1
            V[2][3] = 1; V[0][3] = -1
            W[1][3] = 1; W[0][3] = -1  # C21, -C11
            
            # M5 = (A11+A12)*B22
            U[0][4] = 1; U[1][4] = 1
            V[3][4] = 1
            W[2][4] = 1; W[0][4] = 1  # C12, C11
            
            # M6 = (A21-A11)*(B11+B12)
            U[2][5] = 1; U[0][5] = -1
            V[0][5] = 1; V[1][5] = 1
            W[3][5] = 1  # C22
            
            # M7 = (A12-A22)*(B21+B22)
            U[1][6] = 1; U[3][6] = -1
            V[2][6] = 1; V[3][6] = 1
            W[2][6] = 1; W[0][6] = -1  # C12, -C11
            
            return U, V, W, rank
        
        # Recursive: try all splits
        best_rank = n * m * p
        best_result = None
        
        # Split n
        for n1 in range(1, n):
            n2 = n - n1
            U1, V1, W1, r1 = build_decomp(n1, m, p)
            U2, V2, W2, r2 = build_decomp(n2, m, p)
            r = r1 + r2
            if r < best_rank:
                best_rank = r
                U = U1 + [row[:] for row in U2]
                V = [row[:] for row in V1] + [row[:] for row in V2]
                W = [row[:] for row in W1] + [row[:] for row in W2]
                best_result = (U, V, W)
        
        # Split m
        for m1 in range(1, m):
            m2 = m - m1
            U1, V1, W1, r1 = build_decomp(n, m1, p)
            U2, V2, W2, r2 = build_decomp(n, m2, p)
            r = r1 + r2
            if r < best_rank:
                best_rank = r
                U = [row[:] for row in U1] + [row[:] for row in U2]
                V = [row[:] for row in V1] + [row[:] for row in V2]
                W = [row[:] for row in W1] + [row[:] for row in W2]
                best_result = (U, V, W)
        
        # Split p
        for p1 in range(1, p):
            p2 = p - p1
            U1, V1, W1, r1 = build_decomp(n, m, p1)
            U2, V2, W2, r2 = build_decomp(n, m, p2)
            r = r1 + r2
            if r < best_rank:
                best_rank = r
                U = [row[:] for row in U1] + [row[:] for row in U2]
                V = [row[:] for row in V1] + [row[:] for row in V2]
                W = [row[:] for row in W1] + [row[:] for row in W2]
                best_result = (U, V, W)
        
        if best_result is None:
            rank = n * m * p
            U = [[0] * rank for _ in range(n * m)]
            V = [[0] * rank for _ in range(m * p)]
            W = [[0] * rank for _ in range(p * n)]
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[i * m + j][col] = 1
                        V[j * p + k][col] = 1
                        W[k * n + i][col] = 1
                        col += 1
            return U, V, W, rank
        
        U, V, W = best_result
        return U, V, W, best_rank
    
    n, m, p = DIMENSIONS
    U, V, W, rank = build_decomp(n, m, p)
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
