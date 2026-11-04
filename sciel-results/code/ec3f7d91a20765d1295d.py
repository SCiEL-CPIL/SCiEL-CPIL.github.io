import numpy as np
import time

DIMENSIONS = [3, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140

    def trivial_decomp():
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
        return {"U": U, "V": V, "W": W}

    def verify(U, V, W, n, m, p):
        rank = len(U[0]) if U else 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if abs(s - 1.0) > 1e-6:
                        return False
        return True

    # Rank 7 decomposition for 2x2x2
    # U: (4, 7), V: (4, 7), W: (4, 7)
    # Indices: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    def decomp_222():
        rank = 7
        U = [[0]*rank for _ in range(4)]
        V = [[0]*rank for _ in range(4)]
        W = [[0]*rank for _ in range(4)]
        
        # 1. A00 B00 -> C00
        U[0][0] = 1; V[0][0] = 1; W[0][0] = 1
        # 2. A01 B10 -> C00
        U[1][1] = 1; V[3][1] = 1; W[0][1] = 1
        # 3. A10 B01 -> C11
        U[2][2] = 1; V[1][2] = 1; W[3][2] = 1
        # 4. A11 B11 -> C11
        U[3][3] = 1; V[3][3] = 1; W[3][3] = 1
        # 5. (A00+A11)(B00+B11) -> C00, C11
        U[0][4] = 1; U[3][4] = 1
        V[0][4] = 1; V[3][4] = 1
        W[0][4] = 1; W[3][4] = 1
        # 6. (A01+A11)(B00+B10) -> C00, C10
        U[1][5] = 1; U[3][5] = 1
        V[0][5] = 1; V[2][5] = 1
        W[0][5] = 1; W[2][5] = 1
        # 7. (A00+A10)(B01+B11) -> C01, C11
        U[0][6] = 1; U[2][6] = 1
        V[1][6] = 1; V[3][6] = 1
        W[1][6] = 1; W[3][6] = 1
        
        # Check: C00 = 1 + 1 + 1 - 1 - 1 + 1 + 1? 
        # This specific set is NOT verified. 
        # I will use the trivial rank 8 for 2x2x2 to be safe.
        pass

    def decomp_222_trivial():
        rank = 8
        U = [[0]*rank for _ in range(4)]
        V = [[0]*rank for _ in range(4)]
        W = [[0]*rank for _ in range(4)]
        col = 0
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    U[i*2+j][col] = 1
                    V[j*2+k][col] = 1
                    W[k*2+i][col] = 1
                    col += 1
        return U, V, W

    # Use trivial for 2x2x2 to ensure correctness, then build up.
    # Rank 8 for 2x2x2.
    U222, V222, W222 = decomp_222_trivial()
    
    # Build 2x2x3 (rank 9)
    # Split p=3 into 2+1.
    # First 8 terms for 2x2x2 (k=0,1).
    # Last 1 term for 2x2x1 (k=2).
    def decomp_223():
        rank = 9
        U = [[0]*rank for _ in range(4)]
        V = [[0]*rank for _ in range(6)] # m*p = 2*3 = 6
        W = [[0]*rank for _ in range(6)] # p*n = 3*2 = 6
        
        # Copy 2x2x2 part
        for c in range(8):
            for i in range(4):
                U[i][c] = U222[i][c]
            for j in range(2):
                for k in range(2):
                    V[j*2+k][c] = V222[j*2+k][c]
            for k in range(2):
                for i in range(2):
                    W[k*2+i][c] = W222[k*2+i][c]
        
        # Add 2x2x1 part for k=2
        # T[i, j, 2] = 1
        # U[i*2+j, 8] = 1, V[j*2+2, 8] = 1, W[2*2+i, 8] = 1
        for i in range(2):
            for j in range(2):
                U[i*2+j][8] = 1
                V[j*2+2][8] = 1
                W[2*2+i][8] = 1
                
        return U, V, W

    U223, V223, W223 = decomp_223()
    
    # Build 3x2x2 (rank 9)
    # Isomorphic to 2x2x3.
    # T[i, j, k] for i in 0..2, j in 0..1, k in 0..1.
    # Map to T'[k, j, i] for k in 0..1, j in 0..1, i in 0..2.
    # U' (2x2, rank) -> U (3x2, rank)?
    # U'[k, j] corresponds to U[i, j] with i=k? No.
    # The tensor T[i, j, k] = sum U[i, j, c] V[j, k, c] W[k, i, c].
    # T'[k, j, i] = sum U'[k, j, c] V'[j, i, c] W'[i, k, c].
    # We want T[i, j, k] = T'[k, j, i].
    # So U[i, j, c] should be U'[k, j, c] with k=i? No.
    # The indices are permuted.
    # Let's just use trivial for 3x2x2 as well to be safe.
    
    def decomp_322_trivial():
        rank = 12
        U = [[0]*rank for _ in range(6)] # 3*2
        V = [[0]*rank for _ in range(4)] # 2*2
        W = [[0]*rank for _ in range(6)] # 2*3
        col = 0
        for i in range(3):
            for j in range(2):
                for k in range(2):
                    U[i*2+j][col] = 1
                    V[j*2+k][col] = 1
                    W[k*3+i][col] = 1
                    col += 1
        return U, V, W

    # Using trivial for 3x2x2 gives rank 12.
    # Then 3x2x3 rank 13.
    # Then 3x4x6 rank 52.
    # This is better than 72.
    
    U322, V322, W322 = decomp_322_trivial()
    
    # Build 3x2x3 (rank 13)
    def decomp_323():
        rank = 13
        U = [[0]*rank for _ in range(6)]
        V = [[0]*rank for _ in range(6)] # 2*3
        W = [[0]*rank for _ in range(9)] # 3*3
        
        for c in range(12):
            for i in range(6):
                U[i][c] = U322[i][c]
            for j in range(2):
                for k in range(2):
                    V[j*2+k][c] = V322[j*2+k][c]
            for k in range(2):
                for i in range(3):
                    W[k*3+i][c] = W322[k*3+i][c]
        
        # Add 3x2x1 for k=2
        for i in range(3):
            for j in range(2):
                U[i*2+j][12] = 1
                V[j*2+2][12] = 1
                W[2*3+i][12] = 1
                
        return U, V, W

    U323, V323, W323 = decomp_323()
    
    # Build 3x4x6 (rank 52)
    # Split m=4 into 2+2, p=6 into 3+3.
    # 4 blocks of 3x2x3.
    def decomp_346():
        rank = 52
        U = [[0]*rank for _ in range(12)] # 3*4
        V = [[0]*rank for _ in range(24)] # 4*6
        W = [[0]*rank for _ in range(18)] # 6*3
        
        block_rank = 13
        for a in range(2): # m split
            for b in range(2): # p split
                offset = (a*2 + b) * block_rank
                for c in range(block_rank):
                    for i in range(3):
                        for j in range(2):
                            U[i*4 + (a*2 + j)][offset + c] = U323[i*2 + j][c]
                    for j in range(2):
                        for k in range(3):
                            V[(a*2 + j)*6 + (b*3 + k)][offset + c] = V323[j*2 + k][c]
                    for k in range(3):
                        for i in range(3):
                            W[(b*3 + k)*3 + i][offset + c] = W323[k*3 + i][c]
                            
        return U, V, W

    U, V, W = decomp_346()
    
    if verify(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    else:
        return trivial_decomp()

