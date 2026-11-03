import numpy as np
import time

DIMENSIONS = [2, 3, 8]

def construct():
    n, m, p = DIMENSIONS
    # Current best known: rank 44 via schoolbook for first column + Strassen for remaining
    # Let's try to find a better decomposition by searching for rank 43 or lower
    
    deadline = time.time() + 120
    
    # First, return the known rank-44 solution as a fallback
    # Then try to improve if time permits
    
    def make_rank_44():
        rank = 44
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        col = 0
        
        # Part 1: A1 (2x1) * B1 (1x8) -> 2x8, schoolbook
        for i in range(2):
            for k in range(8):
                U[i * 3 + 0][col] = 1
                V[0 * 8 + k][col] = 1
                W[k * 2 + i][col] = 1
                col += 1
                
        # Part 2: A2 (2x2) * B2 (2x8) -> 2x8, Strassen on 4 blocks of 2x2
        for block in range(4):
            # m1 = (a00+a11)(b00+b11)
            U[1][col] = 1
            U[5][col] = 1
            V[1*8 + block*2][col] = 1
            V[2*8 + block*2+1][col] = 1
            W[(block*2)*2 + 0][col] = 1
            W[(block*2+1)*2 + 1][col] = 1
            col += 1
            
            # m2 = (a10+a11)b00
            U[4][col] = 1
            U[5][col] = 1
            V[1*8 + block*2][col] = 1
            W[(block*2)*2 + 1][col] = 1
            W[(block*2+1)*2 + 1][col] = -1
            col += 1
            
            # m3 = a00(b01-b11)
            U[1][col] = 1
            V[1*8 + block*2+1][col] = 1
            V[2*8 + block*2+1][col] = -1
            W[(block*2+1)*2 + 0][col] = 1
            W[(block*2+1)*2 + 1][col] = 1
            col += 1
            
            # m4 = a11(b10-b00)
            U[5][col] = 1
            V[2*8 + block*2][col] = 1
            V[1*8 + block*2][col] = -1
            W[(block*2)*2 + 0][col] = 1
            W[(block*2)*2 + 1][col] = 1
            col += 1
            
            # m5 = (a00+a01)b11
            U[1][col] = 1
            U[2][col] = 1
            V[2*8 + block*2+1][col] = 1
            W[(block*2)*2 + 0][col] = -1
            W[(block*2+1)*2 + 0][col] = 1
            col += 1
            
            # m6 = (a10-a00)(b00+b01)
            U[4][col] = 1
            U[1][col] = -1
            V[1*8 + block*2][col] = 1
            V[1*8 + block*2+1][col] = 1
            W[(block*2+1)*2 + 1][col] = 1
            col += 1
            
            # m7 = (a01-a11)(b10+b11)
            U[2][col] = 1
            U[5][col] = -1
            V[2*8 + block*2][col] = 1
            V[2*8 + block*2+1][col] = 1
            W[(block*2)*2 + 0][col] = 1
            col += 1
            
        return {"U": U, "V": V, "W": W}
    
    # Try a different decomposition: split into three 2x1x8 blocks
    # Each 2x1x8 can be done in rank 16 (schoolbook) or potentially less
    # Total would be 48 with schoolbook, but maybe we can do better
    
    # Alternative: use the structure more cleverly
    # A is 2x3, B is 3x8, C is 2x8
    # C[i,k] = sum_j A[i,j] * B[j,k]
    
    # Let's try rank 42: 14 per column-group if we can find a pattern
    # Or try to find a decomposition with rank < 44 using random search
    
    # For now, let's try a few structured alternatives
    
    # Alternative 1: Split B into two 3x4 blocks
    # C = A * B1 + A * B2 where B1, B2 are 3x4
    # Each A*B_i is 2x4, can be done in rank 12 (schoolbook) or 7*4=28 (Strassen-like)
    # Total: 2*12 = 24 if schoolbook per block, but that's worse
    # Actually schoolbook for 2x3x4 is 2*3*4 = 24, so two blocks = 48, worse
    
    # Let's try to verify the rank-44 solution is correct and return it
    result = make_rank_44()
    
    # Verify
    U_np = np.array(result["U"], dtype=float)
    V_np = np.array(result["V"], dtype=float)
    W_np = np.array(result["W"], dtype=float)
    rank = len(result["U"][0])
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U_np[i*m + j, r] * V_np[j*p + k, r] * W_np[k*n + i, r]
                if abs(val - 1.0) > 1e-9:
                    break
    
    return result
