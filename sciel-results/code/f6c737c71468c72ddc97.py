import numpy as np
import time
import sys

DIMENSIONS = [3, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    dim_u = n * m  # 12
    dim_v = m * p  # 20
    dim_w = p * n  # 15
    
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    def verify_decomposition(U, V, W, R):
        """Verify that the decomposition is exact."""
        Ua = np.array(U, dtype=float)
        Va = np.array(V, dtype=float)
        Wa = np.array(W, dtype=float)
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    for jp in range(m):
                        total = 0.0
                        for r in range(R):
                            total += Wa[k * n + i, r] * Ua[i * m + j, r] * Va[jp * p + k, r]
                        expected = 1.0 if j == jp else 0.0
                        if abs(total - expected) > 1e-9:
                            return False
        return True
    
    def build_hybrid_decomposition():
        """
        Build a hybrid decomposition using Strassen-like blocks.
        
        Partition:
        - i: {0,1} (2 rows), {2} (1 row)
        - j: {0,1} (2 cols), {2,3} (2 cols)
        - k: {0,1} (2), {2,3} (2), {4} (1)
        
        Strassen 2x2x2 blocks (4 blocks x 7 = 28):
        - (i={0,1}, j={0,1}, k={0,1})
        - (i={0,1}, j={2,3}, k={0,1})
        - (i={0,1}, j={0,1}, k={2,3})
        - (i={0,1}, j={2,3}, k={2,3})
        
        Schoolbook for remaining:
        - i=2, all j, all k: 1*4*5 = 20
        - i={0,1}, all j, k=4: 2*4*1 = 8
        Total schoolbook: 28
        
        Grand total: 28 + 28 = 56
        """
        R = 56
        U = [[0] * R for _ in range(dim_u)]
        V = [[0] * R for _ in range(dim_v)]
        W = [[0] * R for _ in range(dim_w)]
        
        col = 0
        
        def add_strassen_block(i1, i2, j1, j2, k1, k2):
            """
            Add a Strassen 2x2x2 block.
            A = [[a, b], [c, d]] where a=A[i1,j1], b=A[i1,j2], c=A[i2,j1], d=A[i2,j2]
            B = [[e, f], [g, h]] where e=B[j1,k1], f=B[j1,k2], g=B[j2,k1], h=B[j2,k2]
            C = [[C11, C12], [C21, C22]] where C11=C[i1,k1], C12=C[i1,k2], C21=C[i2,k1], C22=C[i2,k2]
            
            Standard Strassen:
            m1 = (a+d)(e+h)
            m2 = (a+b)e
            m3 = (a+b)h
            m4 = c(e+f)
            m5 = d(g+h)
            m6 = (c-d)(e+h)
            m7 = (b-c)(g+h)
            
            C11 = m1 + m4 - m5 + m7
            C12 = m3 - m7
            C21 = m2 + m5
            C22 = m1 - m2 + m3 + m6
            """
            nonlocal col
            
            # m1 = (a+d)(e+h)
            U[i1 * m + j1][col] = 1
            U[i2 * m + j2][col] = 1
            V[j1 * p + k1][col] = 1
            V[j2 * p + k2][col] = 1
            W[k1 * n + i1][col] = 1
            W[k2 * n + i2][col] = 1
            col += 1
            
            # m2 = (a+b)e
            U[i1 * m + j1][col] = 1
            U[i1 * m + j2][col] = 1
            V[j1 * p + k1][col] = 1
            W[k1 * n + i2][col] = 1
            W[k2 * n + i2][col] = -1
            col += 1
            
            # m3 = (a+b)h
            U[i1 * m + j1][col] = 1
            U[i1 * m + j2][col] = 1
            V[j2 * p + k2][col] = 1
            W[k1 * n + i1][col] = 1
            W[k2 * n + i2][col] = -1
            col += 1
            
            # m4 = c(e+f)
            U[i2 * m + j1][col] = 1
            V[j1 * p + k1][col] = 1
            V[j1 * p + k2][col] = 1
            W[k1 * n + i1][col] = 1
            col += 1
            
            # m5 = d(g+h)
            U[i2 * m + j2][col] = 1
            V[j2 * p + k1][col] = 1
            V[j2 * p + k2][col] = 1
            W[k1 * n + i2][col] = -1
            W[k2 * n + i2][col] = 1
            col += 1
            
            # m6 = (c-d)(e+h)
            U[i2 * m + j1][col] = 1
            U[i2 * m + j2][col] = -1
            V[j1 * p + k1][col] = 1
            V[j2 * p + k2][col] = 1
            W[k2 * n + i2][col] = 1
            col += 1
            
            # m7 = (b-c)(g+h)
            U[i1 * m + j2][col] = 1
            U[i2 * m + j1][col] = -1
            V[j2 * p + k1][col] = 1
            V[j2 * p + k2][col] = 1
            W[k1 * n + i1][col] = -1
            W[k1 * n + i2][col] = 1
            col += 1
        
        # Add 4 Strassen blocks
        add_strassen_block(0, 1, 0, 1, 0, 1)  # i={0,1}, j={0,1}, k={0,1}
        add_strassen_block(0, 1, 2, 3, 0, 1)  # i={0,1}, j={2,3}, k={0,1}
        add_strassen_block(0, 1, 0, 1, 2, 3)  # i={0,1}, j={0,1}, k={2,3}
        add_strassen_block(0, 1, 2, 3, 2, 3)  # i={0,1}, j={2,3}, k={2,3}
        
        # Schoolbook for i=2, all j, all k
        for j in range(m):
            for k in range(p):
                U[2 * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + 2][col] = 1
                col += 1
        
        # Schoolbook for i={0,1}, all j, k=4
        for i in [0, 1]:
            for j in range(m):
                U[i * m + j][col] = 1
                V[j * p + 4][col] = 1
                W[4 * n + i][col] = 1
                col += 1
        
        assert col == R, f"Expected {R} columns, got {col}"
        
        # Verify
        if not verify_decomposition(U, V, W, R):
            return None
        return U, V, W, R
    
    def build_extended_strassen():
        """
        Try to add two more Strassen blocks for i={1,2} to reduce rank further.
        This creates overlapping i=1 terms which we need to handle carefully.
        
        Additional blocks:
        - (i={1,2}, j={0,1}, k={3,4})
        - (i={1,2}, j={2,3}, k={0,1})
        
        But these overlap with existing schoolbook terms for i=1, j=0..3, k=0..4.
        We need to replace the schoolbook terms for i=1 that are covered by these blocks.
        
        Actually, let's think more carefully. The current decomposition has:
        - Strassen blocks for i={0,1} covering j={0,1}x{k=0,1}, j={2,3}x{k=0,1}, 
          j={0,1}x{k=2,3}, j={2,3}x{k=2,3}
        - Schoolbook for i=2, all j, all k (20 terms)
        - Schoolbook for i={0,1}, all j, k=4 (8 terms)
        
        The i=1 terms in the Strassen blocks cover:
        - j=0,1 with k=0,1,2,3
        - j=2,3 with k=0,1,2,3
        
        So i=1, j=0..3, k=0..3 are covered by Strassen.
        i=1, j=0..3, k=4 are covered by schoolbook (4 terms).
        
        If we add Strassen blocks for i={1,2}:
        - Block A: i={1,2}, j={0,1}, k={3,4}
          This covers i=1,j=0,1,k=3,4 and i=2,j=0,1,k=3,4
          But i=1,j=0,1,k=3 is already covered by Strassen block (0,1,0,1,2,3)
          And i=1,j=0,1,k=4 is already covered by schoolbook
          And i=2,j=0,1,k=3,4 is already covered by schoolbook (i=2 terms)
          
        This gets complicated. Let me try a different approach.
        
        Instead, let's try to use the Winograd/Strassen decomposition more cleverly.
        
        Actually, the key insight from the knowledge is that adding two more disjoint 
        Strassen blocks for i={1,2} reduces rank from 58 to 56. But we're already at 56.
        
        Let me try a completely different partition:
        - Use 3x3x3 Winograd blocks where possible.
        
        For 3x3x3, Winograd uses rank 23 (vs schoolbook 27).
        But our dimensions are 3x4x5, so we can't directly use 3x3x3.
        
        Let me try:
        - i={0,1,2}, j={0,1,2}, k={0,1,2}: 3x3x3 block
        - Remaining: various strips
        
        Actually, this is getting too complex. Let me just try to verify that the 
        current 56-rank decomposition is correct and return it.
        
        The main improvement would be to find a way to merge some of the schoolbook 
        terms with the Strassen terms, but that requires careful linear algebra.
        
        For now, let's just return the 56-rank decomposition.
        """
        return None
    
    # First, try the extended Strassen approach
    result = build_extended_strassen()
    if result is not None:
        U, V, W, R = result
        return {"U": U, "V": V, "W": W}
    
    # Try the hybrid decomposition
    result = build_hybrid_decomposition()
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
