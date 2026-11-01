"""Rank-32 decomposition for 2x2x9 matrix multiplication.

Uses Strassen (rank 7) for 2x2x2 blocks and schoolbook (rank 4) for 2x2x1 blocks.
Split: 4 Strassen blocks (cols 0-7) + 1 schoolbook block (col 8) = 4*7 + 4 = 32.

The index mapping follows the official trace convention W[k*n+i, r].
"""
import time
import numpy as np

DIMENSIONS = [2, 2, 9]

# EVOLVE-BLOCK-START

def _verify(U, V, W, n, m, p):
    """Verify decomposition: T[i,j,k] = sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1."""
    U3 = np.array(U, dtype=float).reshape(n, m, -1)
    V3 = np.array(V, dtype=float).reshape(m, p, -1)
    W3 = np.array(W, dtype=float).reshape(p, n, -1)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U3[i, j, :], V3[j, k, :] * W3[k, i, :])
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = DIMENSIONS  # 2, 2, 9
    
    # 4 Strassen blocks (each rank 7) for cols (0,1),(2,3),(4,5),(6,7)
    # 1 schoolbook block (rank 4) for col 8
    rank = 4 * 7 + 4  # 32
    
    U = [[0.0] * rank for _ in range(n * m)]  # (4, 32)
    V = [[0.0] * rank for _ in range(m * p)]  # (18, 32)
    W = [[0.0] * rank for _ in range(p * n)]  # (18, 32)
    
    col = 0
    
    def add_strassen(k0, k1):
        """Add Strassen 7-term decomposition for B columns k0,k1."""
        nonlocal col
        # M1: (a00+a11)(b_{0,k0}+b_{1,k1})
        #   U: rows 0,3 (a00,a11); V: rows 0*p+k0, 1*p+k1; W: C_{k0,0}, C_{k1,1}
        U[0][col] = 1.0; U[3][col] = 1.0
        V[0*p + k0][col] = 1.0; V[1*p + k1][col] = 1.0
        W[k0*n + 0][col] = 1.0; W[k1*n + 1][col] = 1.0
        col += 1
        
        # M2: (a10+a11) * b_{0,k0}
        #   U: rows 2,3 (a10,a11); V: row 0*p+k0; W: C_{k0,1}, C_{k1,1}
        U[2][col] = 1.0; U[3][col] = 1.0
        V[0*p + k0][col] = 1.0
        W[k0*n + 1][col] = 1.0; W[k1*n + 1][col] = -1.0
        col += 1
        
        # M3: a00 * (b_{0,k1} - b_{1,k1})
        #   U: row 0; V: rows 0*p+k1, 1*p+k1; W: C_{k1,0}, C_{k1,1}
        U[0][col] = 1.0
        V[0*p + k1][col] = 1.0; V[1*p + k1][col] = -1.0
        W[k1*n + 0][col] = 1.0; W[k1*n + 1][col] = 1.0
        col += 1
        
        # M4: a11 * (b_{1,k0} - b_{0,k0})
        #   U: row 3; V: rows 1*p+k0, 0*p+k0; W: C_{k0,0}, C_{k0,1}
        U[3][col] = 1.0
        V[1*p + k0][col] = 1.0; V[0*p + k0][col] = -1.0
        W[k0*n + 0][col] = 1.0; W[k0*n + 1][col] = 1.0
        col += 1
        
        # M5: (a00+a01) * b_{1,k1}
        #   U: rows 0,1; V: row 1*p+k1; W: C_{k0,0}, C_{k1,0}
        U[0][col] = 1.0; U[1][col] = 1.0
        V[1*p + k1][col] = 1.0
        W[k0*n + 0][col] = -1.0; W[k1*n + 0][col] = 1.0
        col += 1
        
        # M6: (a10-a00) * (b_{0,k0}+b_{0,k1})
        #   U: rows 2,0; V: rows 0*p+k0, 0*p+k1; W: C_{k1,1}
        U[2][col] = 1.0; U[0][col] = -1.0
        V[0*p + k0][col] = 1.0; V[0*p + k1][col] = 1.0
        W[k1*n + 1][col] = 1.0
        col += 1
        
        # M7: (a01-a11) * (b_{1,k0}+b_{1,k1})
        #   U: rows 1,3; V: rows 1*p+k0, 1*p+k1; W: C_{k0,0}
        U[1][col] = 1.0; U[3][col] = -1.0
        V[1*p + k0][col] = 1.0; V[1*p + k1][col] = 1.0
        W[k0*n + 0][col] = 1.0
        col += 1
    
    def add_schoolbook(k):
        """Add schoolbook 4-term decomposition for B column k."""
        nonlocal col
        # a00 * b_{0,k} -> C_{k,0}
        U[0][col] = 1.0
        V[0*p + k][col] = 1.0
        W[k*n + 0][col] = 1.0
        col += 1
        # a01 * b_{1,k} -> C_{k,0}
        U[1][col] = 1.0
        V[1*p + k][col] = 1.0
        W[k*n + 0][col] = 1.0
        col += 1
        # a10 * b_{0,k} -> C_{k,1}
        U[2][col] = 1.0
        V[0*p + k][col] = 1.0
        W[k*n + 1][col] = 1.0
        col += 1
        # a11 * b_{1,k} -> C_{k,1}
        U[3][col] = 1.0
        V[1*p + k][col] = 1.0
        W[k*n + 1][col] = 1.0
        col += 1
    
    # Build the decomposition
    add_strassen(0, 1)
    add_strassen(2, 3)
    add_strassen(4, 5)
    add_strassen(6, 7)
    add_schoolbook(8)
    
    # Verify
    if col != rank:
        raise ValueError(f"Column count {col} != expected rank {rank}")
    
    if not _verify(U, V, W, n, m, p):
        # Fallback: full schoolbook rank 36
        rank_fb = n * m * p
        U_fb = [[0.0] * rank_fb for _ in range(n * m)]
        V_fb = [[0.0] * rank_fb for _ in range(m * p)]
        W_fb = [[0.0] * rank_fb for _ in range(p * n)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i*m+j][c] = 1.0
                    V_fb[j*p+k][c] = 1.0
                    W_fb[k*n+i][c] = 1.0
                    c += 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
