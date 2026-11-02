import numpy as np
import time
from scipy.optimize import least_squares

DIMENSIONS = [2, 2, 4]

def construct():
    start_time = time.time()
    deadline = 140.0
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n  # 4, 8, 8

    def verify(U, V, W):
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

    def try_rank14():
        # Standard two Strassen blocks
        R = 14
        U = [[0]*R for _ in range(4)]
        V = [[0]*R for _ in range(8)]
        W = [[0]*R for _ in range(8)]
        
        # Left block
        U[0][0] = 1; U[3][0] = 1; V[0][0] = 1; V[5][0] = 1; W[0][0] = 1; W[3][0] = 1
        U[2][1] = 1; U[3][1] = 1; V[0][1] = 1; W[1][1] = 1; W[3][1] = -1
        U[0][2] = 1; V[1][2] = 1; V[5][2] = -1; W[2][2] = 1; W[3][2] = 1
        U[3][3] = 1; V[4][3] = 1; V[0][3] = -1; W[0][3] = 1; W[1][3] = 1
        U[0][4] = 1; U[1][4] = 1; V[5][4] = 1; W[0][4] = -1; W[2][4] = 1
        U[2][5] = 1; U[0][5] = -1; V[0][5] = 1; V[1][5] = 1; W[3][5] = 1
        U[1][6] = 1; U[3][6] = -1; V[4][6] = 1; V[5][6] = 1; W[0][6] = 1
        
        # Right block
        U[0][7] = 1; U[3][7] = 1; V[2][7] = 1; V[7][7] = 1; W[4][7] = 1; W[7][7] = 1
        U[2][8] = 1; U[3][8] = 1; V[2][8] = 1; W[5][8] = 1; W[7][8] = -1
        U[0][9] = 1; V[3][9] = 1; V[7][9] = -1; W[6][9] = 1; W[7][9] = 1
        U[3][10] = 1; V[6][10] = 1; V[2][10] = -1; W[4][10] = 1; W[5][10] = 1
        U[0][11] = 1; U[1][11] = 1; V[7][11] = 1; W[4][11] = -1; W[6][11] = 1
        U[2][12] = 1; U[0][12] = -1; V[2][12] = 1; V[3][12] = 1; W[7][12] = 1
        U[1][13] = 1; U[3][13] = -1; V[6][13] = 1; V[7][13] = 1; W[4][13] = 1
        
        if verify(U, V, W):
            return {"U": U, "V": V, "W": W}
        return None

    def try_rank12():
        if time.time() > deadline - 30:
            return None
            
        R = 12
        # We need to find U (4,12), V (8,12), W (8,12)
        # such that sum_r U[i,j,r] V[j,k,r] W[k,i,r] = delta(i,j,k)
        
        # Use the 7 Strassen terms for the first block as a starting point
        # and try to fit the second block with 5 more terms.
        
        U = np.zeros((4, R))
        V = np.zeros((8, R))
        W = np.zeros((8, R))
        
        # Fill first 7 columns with Strassen left block
        U[:, 0] = [1, 0, 0, 1]
        V[:, 0] = [1, 0, 0, 0, 0, 1, 0, 0]
        W[:, 0] = [1, 0, 0, -1, 0, 0, 0, 0]
        
        U[:, 1] = [0, 0, 1, 1]
        V[:, 1] = [1, 0, 0, 0, 0, 0, 0, 0]
        W[:, 1] = [0, 1, 0, -1, 0, 0, 0, 0]
        
        U[:, 2] = [1, 0, 0, 0]
        V[:, 2] = [0, 1, 0, 0, 0, -1, 0, 0]
        W[:, 2] = [0, 0, 1, 1, 0, 0, 0, 0]
        
        U[:, 3] = [0, 0, 0, 1]
        V[:, 3] = [1, 0, 0, 0, 1, -1, 0, 0]
        W[:, 3] = [1, 1, 0, 0, 0, 0, 0, 0]
        
        U[:, 4] = [1, 1, 0, 0]
        V[:, 4] = [0, 0, 0, 0, 0, 1, 0, 0]
        W[:, 4] = [-1, 0, 1, 0, 0, 0, 0, 0]
        
        U[:, 5] = [1, 0, -1, 0]
        V[:, 5] = [1, 1, 0, 0, 0, 0, 0, 0]
        W[:, 5] = [0, 0, 0, 1, 0, 0, 0, 0]
        
        U[:, 6] = [0, 1, 0, -1]
        V[:, 6] = [0, 0, 0, 0, 1, 1, 0, 0]
        W[:, 6] = [1, 0, 0, 0, 0, 0, 0, 0]
        
        # Now we need to find 5 more terms to satisfy the equations for the second block.
        # The equations for the second block involve V indices 2,3,6,7 and W indices 4,5,6,7.
        # We have 5 free terms (columns 7-11).
        # This is a system of 16 equations (for the 16 entries of the tensor, though 8 are already satisfied by the first 7 terms for the first block? No, the first 7 terms only contribute to the first block's W indices 0-3. They contribute 0 to W indices 4-7).
        # So the first 7 terms satisfy the 8 equations for the first block (C00, C01, C10, C11) and contribute 0 to the second block.
        # We need the remaining 5 terms to satisfy the 8 equations for the second block (C02, C03, C12, C13) AND contribute 0 to the first block?
        # No, the remaining 5 terms can contribute to the first block as long as the sum is correct.
        # But it's easier to force the remaining 5 terms to have 0 in the first block's V and W indices.
        # If V[:, 7:12] has 0 in indices 0,1,4,5 and W[:, 7:12] has 0 in indices 0,1,2,3, then they don't affect the first block.
        # Then we just need to solve for the second block using 5 terms.
        # But a 2x2x2 tensor has rank 7. We cannot decompose it with 5 terms.
        # So this approach fails. We must allow the 5 terms to interact with the first block.
        
        # This implies a more complex search.
        # Given the time, I will skip the numerical search for rank 12 and return rank 14.
        return None

    result = try_rank14()
    if result:
        return result
        
    # Fallback
    R = 16
    U = [[0]*R for _ in range(4)]
    V = [[0]*R for _ in range(8)]
    W = [[0]*R for _ in range(8)]
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][r] = 1
                V[j*p+k][r] = 1
                W[k*n+i][r] = 1
                r += 1
    return {"U": U, "V": V, "W": W}
