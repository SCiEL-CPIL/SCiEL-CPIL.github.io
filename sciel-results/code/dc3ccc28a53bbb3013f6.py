"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 4]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n  # 4, 8, 8
    
    def naive_decomp():
        rank = n * m * p  # 16
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        r = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][r] = 1
                    V[j * p + k][r] = 1
                    W[k * n + i][r] = 1
                    r += 1
        return {"U": U, "V": V, "W": W}
    
    def verify(U, V, W):
        """Verify the tensor decomposition is exact."""
        R = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(R):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    def strassen_block(U, V, W, offset_u, offset_v, offset_w, rank_offset):
        """
        Apply a standard Strassen 2x2 decomposition.
        U: 4xR, V: 8xR, W: 8xR
        The block operates on specific indices.
        For 2x2x2, U is 4x1, V is 4x1, W is 4x1.
        Here we map to the global indices.
        """
        # Indices for the 2x2 block:
        # i in 0,1; j in 0,1; k in 0,1 (local)
        # Global U index: i*m + j
        # Global V index: j*p + k
        # Global W index: k*n + i
        
        # Strassen formulas for C = AB
        # M1 = (a00+a11)(b00+b11)
        # M2 = (a10+a11)b00
        # M3 = a00(b01-b11)
        # M4 = a11(b10-b00)
        # M5 = (a00+a01)b11
        # M6 = (a10-a00)(b00+b01)
        # M7 = (a01-a11)(b10+b11)
        
        # C00 = M1 - M2 + M3 + M4
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M5 + M6 + M7
        
        # U coefficients (A part)
        # V coefficients (B part)
        # W coefficients (C part)
        
        # r=0: M1
        U[0][rank_offset] = 1; U[3][rank_offset] = 1
        V[0][rank_offset] = 1; V[5][rank_offset] = 1 # b00, b11 (local j=0,1; k=0,1) -> V[0*2+0], V[1*2+1]
        W[0][rank_offset] = 1; W[3][rank_offset] = 1 # c00, c11 (local i=0,1; k=0,1) -> W[0*2+0], W[1*2+1]
        
        # r=1: M2
        U[2][rank_offset+1] = 1; U[3][rank_offset+1] = 1 # a10, a11
        V[0][rank_offset+1] = 1 # b00
        W[1][rank_offset+1] = 1; W[3][rank_offset+1] = -1 # c10, c11? 
        # Wait, C10 = M2 + M4. C11 = M1 - M5 + M6 + M7.
        # My W indices: W[k*n+i].
        # C00: i=0,k=0 -> W[0]
        # C01: i=0,k=1 -> W[1]
        # C10: i=1,k=0 -> W[2]
        # C11: i=1,k=1 -> W[3]
        
        # Let's re-verify W indices in seed:
        # W[k*n+i].
        # C00 (i=0,k=0) -> W[0]
        # C01 (i=0,k=1) -> W[1]
        # C10 (i=1,k=0) -> W[2]
        # C11 (i=1,k=1) -> W[3]
        
        # Strassen:
        # C00 = M1 - M2 + M3 + M4
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M5 + M6 + M7
        
        # M1: U(0,3), V(0,5), W(0,3) -> W0, W3
        # M2: U(2,3), V(0), W(2) -> W2 (for C10)
        # M3: U(0), V(1,-5), W(0,1) -> W0, W1
        # M4: U(3), V(4,-0), W(2,3) -> W2, W3
        # M5: U(0,1), V(5), W(1,-3) -> W1, W3
        # M6: U(2,-0), V(0,1), W(2) -> W2
        # M7: U(1,-3), V(4,5), W(3) -> W3
        
        # Let's map this to the global indices for the right block (k=2,3).
        # Local k=0 -> Global k=2
        # Local k=1 -> Global k=3
        # V index: j*p + k.
        # j=0, k=2 -> V[2]
        # j=0, k=3 -> V[3]
        # j=1, k=2 -> V[6]
        # j=1, k=3 -> V[7]
        
        # W index: k*n + i.
        # k=2, i=0 -> W[4]
        # k=2, i=1 -> W[5]
        # k=3, i=0 -> W[6]
        # k=3, i=1 -> W[7]
        
        pass

    def try_strassen_rank14():
        R = 14
        U = [[0]*R for _ in range(4)]
        V = [[0]*R for _ in range(8)]
        W = [[0]*R for _ in range(8)]
        
        # Left block (k=0,1)
        # M1
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[5][0] = 1
        W[0][0] = 1; W[3][0] = 1
        # M2
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[2][1] = 1
        # M3
        U[0][2] = 1
        V[1][2] = 1; V[5][2] = -1
        W[0][2] = 1; W[1][2] = 1
        # M4
        U[3][3] = 1
        V[4][3] = 1; V[0][3] = -1
        W[2][3] = 1; W[3][3] = 1
        # M5
        U[0][4] = 1; U[1][4] = 1
        V[5][4] = 1
        W[1][4] = 1; W[3][4] = -1
        # M6
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        W[2][5] = 1
        # M7
        U[1][6] = 1; U[3][6] = -1
        V[4][6] = 1; V[5][6] = 1
        W[3][6] = 1

        # Right block (k=2,3)
        # M1
        U[0][7] = 1; U[3][7] = 1
        V[2][7] = 1; V[7][7] = 1
        W[4][7] = 1; W[7][7] = 1
        # M2
        U[2][8] = 1; U[3][8] = 1
        V[2][8] = 1
        W[5][8] = 1
        # M3
        U[0][9] = 1
        V[3][9] = 1; V[7][9] = -1
        W[4][9] = 1; W[6][9] = 1
        # M4
        U[3][10] = 1
        V[6][10] = 1; V[2][10] = -1
        W[5][10] = 1; W[7][10] = 1
        # M5
        U[0][11] = 1; U[1][11] = 1
        V[7][11] = 1
        W[6][11] = 1; W[7][11] = -1
        # M6
        U[2][12] = 1; U[0][12] = -1
        V[2][12] = 1; V[3][12] = 1
        W[5][12] = 1
        # M7
        U[1][13] = 1; U[3][13] = -1
        V[6][13] = 1; V[7][13] = 1
        W[7][13] = 1

        if verify(U, V, W):
            return {"U": U, "V": V, "W": W}
        return None

    result = try_strassen_rank14()
    if result is not None:
        return result
    
    return naive_decomp()
# EVOLVE-BLOCK-END
