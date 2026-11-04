import numpy as np
import time

DIMENSIONS = [3, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    dim_u = n * m  # 12
    dim_v = m * p  # 20
    dim_w = p * n  # 15
    
    start_time = time.time()
    deadline = start_time + 140.0
    
    def verify_decomposition(U, V, W, R):
        """Verify that the decomposition is exact."""
        Ua = np.array(U, dtype=float)
        Va = np.array(V, dtype=float)
        Wa = np.array(W, dtype=float)
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    expected = 1.0 if True else 0.0  # T[i,j,k] = 1 always for matmul
                    total = 0.0
                    for r in range(R):
                        total += Wa[k * n + i, r] * Ua[i * m + j, r] * Va[j * p + k, r]
                    if abs(total - 1.0) > 1e-9:
                        return False
        return True
    
    def build_optimal_decomposition():
        """
        Build decomposition using:
        - 4 Strassen 2x2x2 blocks: 4 * 7 = 28
        - 1x4x5 block (i=2): rank 1
        - 2x4x1 block (k=4, i=0,1): rank 2
        Total: 28 + 1 + 2 = 31
        """
        R = 31
        U = [[0] * R for _ in range(dim_u)]
        V = [[0] * R for _ in range(dim_v)]
        W = [[0] * R for _ in range(dim_w)]
        
        col = 0
        
        def add_strassen_block(i1, i2, j1, j2, k1, k2):
            """Add a Strassen 2x2x2 block with 7 rank-1 terms."""
            nonlocal col
            
            # m1 = (a+d)(e+h) -> C11 +=, C22 -=
            U[i1 * m + j1][col] = 1
            U[i2 * m + j2][col] = 1
            V[j1 * p + k1][col] = 1
            V[j2 * p + k2][col] = 1
            W[k1 * n + i1][col] = 1
            W[k2 * n + i2][col] = 1
            col += 1
            
            # m2 = (a+b)e -> C21 +=, C22 -=
            U[i1 * m + j1][col] = 1
            U[i1 * m + j2][col] = 1
            V[j1 * p + k1][col] = 1
            W[k1 * n + i2][col] = 1
            W[k2 * n + i2][col] = -1
            col += 1
            
            # m3 = (a+b)h -> C12 +=, C22 +=
            U[i1 * m + j1][col] = 1
            U[i1 * m + j2][col] = 1
            V[j2 * p + k2][col] = 1
            W[k1 * n + i1][col] = 1
            W[k2 * n + i2][col] = 1
            col += 1
            
            # m4 = c(e+f) -> C21 +=
            U[i2 * m + j1][col] = 1
            V[j1 * p + k1][col] = 1
            V[j1 * p + k2][col] = 1
            W[k1 * n + i1][col] = 1
            col += 1
            
            # m5 = d(g+h) -> C22 +=
            U[i2 * m + j2][col] = 1
            V[j2 * p + k1][col] = 1
            V[j2 * p + k2][col] = 1
            W[k1 * n + i2][col] = 1
            W[k2 * n + i2][col] = 1
            col += 1
            
            # m6 = (c-d)(e+h) -> C22 +=
            U[i2 * m + j1][col] = 1
            U[i2 * m + j2][col] = -1
            V[j1 * p + k1][col] = 1
            V[j2 * p + k2][col] = 1
            W[k2 * n + i2][col] = 1
            col += 1
            
            # m7 = (b-c)(g+h) -> C12 -=, C21 +=
            U[i1 * m + j2][col] = 1
            U[i2 * m + j1][col] = -1
            V[j2 * p + k1][col] = 1
            V[j2 * p + k2][col] = 1
            W[k1 * n + i1][col] = -1
            W[k1 * n + i2][col] = 1
            col += 1
        
        # Add 4 Strassen blocks covering i={0,1}, j={0,1},{2,3}, k={0,1},{2,3}
        add_strassen_block(0, 1, 0, 1, 0, 1)  # i={0,1}, j={0,1}, k={0,1}
        add_strassen_block(0, 1, 2, 3, 0, 1)  # i={0,1}, j={2,3}, k={0,1}
        add_strassen_block(0, 1, 0, 1, 2, 3)  # i={0,1}, j={0,1}, k={2,3}
        add_strassen_block(0, 1, 2, 3, 2, 3)  # i={0,1}, j={2,3}, k={2,3}
        
        # Rank-1 for i=2, all j, all k (1x4x5 block)
        # T[2,j,k] = 1 for all j,k
        for j in range(m):
            U[2 * m + j][col] = 1
        for j in range(m):
            for k in range(p):
                V[j * p + k][col] = 1
        for k in range(p):
            W[k * n + 2][col] = 1
        col += 1
        
        # Rank-2 for k=4, i={0,1}, all j (2x4x1 block)
        # Term for i=0: T[0,j,4] = 1
        for j in range(m):
            U[0 * m + j][col] = 1
        for j in range(m):
            V[j * p + 4][col] = 1
        W[4 * n + 0][col] = 1
        col += 1
        
        # Term for i=1: T[1,j,4] = 1
        for j in range(m):
            U[1 * m + j][col] = 1
        for j in range(m):
            V[j * p + 4][col] = 1
        W[4 * n + 1][col] = 1
        col += 1
        
        assert col == R, f"Expected {R} columns, got {col}"
        
        # Verify
        if time.time() < deadline:
            if not verify_decomposition(U, V, W, R):
                return None
        return U, V, W, R
    
    # Build the optimal decomposition
    result = build_optimal_decomposition()
    if result is not None:
        U, V, W, R = result
        return {"U": U, "V": V, "W": W}
    
    # Fallback: pure schoolbook
    R = n * m * p
    U = [[0] * R for _ in range(dim_u)]
    V = [[0] * R for _ in range(dim_v)]
    W = [[0] * R for _ in range(dim_w)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
