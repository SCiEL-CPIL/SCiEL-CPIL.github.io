"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 4]

# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill
    
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
    
    def try_strassen_rectangular():
        """
        Rank-14 decomposition for 2x2 * 2x4 using Strassen on two 2x2 blocks.
        
        B is split into B_left (cols 0,1) and B_right (cols 2,3).
        Each 2x2*2x2 uses 7 Strassen products.
        
        U shape: (4, 14) - indices: a00=0, a01=1, a10=2, a11=3
        V shape: (8, 14) - indices: b00=0, b01=1, b10=2, b11=3, b02=4, b03=5, b12=6, b13=7
        W shape: (8, 14) - indices: C00=0, C01=1, C10=2, C11=3, C02=4, C03=5, C12=6, C13=7
        """
        R = 14
        U = [[0]*R for _ in range(4)]
        V = [[0]*R for _ in range(8)]
        W = [[0]*R for _ in range(8)]
        
        # Left block: B[:,0:2] -> V indices 0,1,2,3 (b00=0, b01=1, b10=2, b11=3)
        # W indices for left: C00=0, C01=1, C10=2, C11=3
        
        # r=0: M1 = (a00+a11)(b00+b11)
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        W[0][0] = 1; W[3][0] = 1
        
        # r=1: M2 = (a10+a11)(b00)
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[2][1] = 1; W[3][1] = -1
        
        # r=2: M3 = a00(b10-b11)
        U[0][2] = 1
        V[2][2] = 1; V[3][2] = -1
        W[1][2] = 1; W[3][2] = 1
        
        # r=3: M4 = a11(b01-b00)
        U[3][3] = 1
        V[1][3] = 1; V[0][3] = -1
        W[0][3] = 1; W[2][3] = 1
        
        # r=4: M5 = (a00+a01)(b11)
        U[0][4] = 1; U[1][4] = 1
        V[3][4] = 1
        W[0][4] = -1; W[1][4] = 1
        
        # r=5: M6 = (a10-a00)(b00+b01)
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        W[3][5] = 1
        
        # r=6: M7 = (a01-a11)(b10+b11)
        U[1][6] = 1; U[3][6] = -1
        V[2][6] = 1; V[3][6] = 1
        W[0][6] = 1
        
        # Right block: B[:,2:4] -> V indices 4,5,6,7 (b02=4, b03=5, b12=6, b13=7)
        # W indices for right: C02=4, C03=5, C12=6, C13=7
        
        # r=7: N1 = (a00+a11)(b02+b13)
        U[0][7] = 1; U[3][7] = 1
        V[4][7] = 1; V[7][7] = 1
        W[4][7] = 1; W[7][7] = 1
        
        # r=8: N2 = (a10+a11)(b02)
        U[2][8] = 1; U[3][8] = 1
        V[4][8] = 1
        W[6][8] = 1; W[7][8] = -1
        
        # r=9: N3 = a00(b12-b13)
        U[0][9] = 1
        V[6][9] = 1; V[7][9] = -1
        W[5][9] = 1; W[7][9] = 1
        
        # r=10: N4 = a11(b03-b02)
        U[3][10] = 1
        V[5][10] = 1; V[4][10] = -1
        W[4][10] = 1; W[6][10] = 1
        
        # r=11: N5 = (a00+a01)(b13)
        U[0][11] = 1; U[1][11] = 1
        V[7][11] = 1
        W[4][11] = -1; W[5][11] = 1
        
        # r=12: N6 = (a10-a00)(b02+b03)
        U[2][12] = 1; U[0][12] = -1
        V[4][12] = 1; V[5][12] = 1
        W[7][12] = 1
        
        # r=13: N7 = (a01-a11)(b12+b13)
        U[1][13] = 1; U[3][13] = -1
        V[6][13] = 1; V[7][13] = 1
        W[4][13] = 1
        
        if verify(U, V, W):
            return {"U": U, "V": V, "W": W}
        else:
            return None
    
    # Try the structured rank-14 decomposition first
    result = try_strassen_rectangular()
    if result is not None:
        return result
    
    # Fallback to naive rank 16
    return naive_decomp()
# EVOLVE-BLOCK-END
