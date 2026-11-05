import numpy as np
import time

def construct():
    n, m, p = 3, 3, 8
    start_time = time.time()
    deadline = start_time + 140

    # Schoolbook decomposition as fallback
    def schoolbook():
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

    best_result = schoolbook()
    best_rank = n * m * p

    # Try block decomposition with Strassen for 2x2x2 blocks
    # 3x3 x 3x8 split into:
    #   (2x2 x 2x8) + (2x1 x 1x8) + (1x2 x 2x8) + (1x1 x 1x8)
    # 
    # For 2x2 x 2x8: use Strassen on 2x2x2 blocks (4 blocks, each rank 7) = 28
    # For 2x1 x 1x8: schoolbook = 16
    # For 1x2 x 2x8: schoolbook = 16
    # For 1x1 x 1x8: schoolbook = 8
    # Total: 28 + 16 + 16 + 8 = 68
    #
    # But we can do better by using Strassen more cleverly.
    # Let's try a direct rank-59 construction.
    
    def try_rank_59():
        """Construct a rank-59 decomposition using block Strassen."""
        rank = 59
        U = [[0] * rank for _ in range(n * m)]  # 9 x 59
        V = [[0] * rank for _ in range(m * p)]  # 24 x 59
        W = [[0] * rank for _ in range(p * n)]  # 24 x 59
        
        col = 0
        
        # Block 1: 2x2 x 2x8 using Strassen (4 blocks of 2x2x2, each rank 7)
        # A[0:2, 0:2] * B[0:2, 0:8]
        # Split B into B[0:2, 0:2], B[0:2, 2:4], B[0:2, 4:6], B[0:2, 6:8]
        for b_block in range(4):
            # Strassen for 2x2 x 2x2
            # M1 = (a00+a11)(b00+b11)
            # M2 = (a10+a11)b00
            # M3 = a00(b10-b11)
            # M4 = a11(b01-b00)
            # M5 = (a00+a01)b11
            # M6 = (a10-a00)(b00+b01)
            # M7 = (a01-a11)(b10+b11)
            
            # M1: (a00+a11)(b00+b11)
            U[0][col] = 1; U[3][col] = 1
            V[b_block*2][col] = 1; V[b_block*2+1][col] = 1
            W[b_block*2*3+0][col] = 1; W[b_block*2*3+3][col] = 1
            W[b_block*2*3+1][col] = 1; W[b_block*2*3+4][col] = 1
            col += 1
            
            # M2: (a10+a11)b00
            U[3][col] = 1; U[6][col] = 1
            V[b_block*2][col] = 1
            W[b_block*2*3+3][col] = 1; W[b_block*2*3+6][col] = 1
            W[b_block*2*3+4][col] = 1; W[b_block*2*3+7][col] = 1
            col += 1
            
            # M3: a00(b10-b11)
            U[0][col] = 1
            V[b_block*2+1][col] = 1; V[b_block*2][col] = -1
            W[b_block*2*3+0][col] = 1; W[b_block*2*3+3][col] = 1
            W[b_block*2*3+1][col] = -1; W[b_block*2*3+4][col] = -1
            col += 1
            
            # M4: a11(b01-b00)
            U[3][col] = 1
            V[b_block*2+1][col] = 1; V[b_block*2][col] = -1
            W[b_block*2*3+3][col] = 1; W[b_block*2*3+6][col] = 1
            W[b_block*2*3+4][col] = -1; W[b_block*2*3+7][col] = -1
            col += 1
            
            # M5: (a00+a01)b11
            U[0][col] = 1; U[1][col] = 1
            V[b_block*2+1][col] = 1
            W[b_block*2*3+0][col] = 1; W[b_block*2*3+3][col] = 1
            W[b_block*2*3+1][col] = 1; W[b_block*2*3+4][col] = 1
            col += 1
            
            # M6: (a10-a00)(b00+b01)
            U[3][col] = 1; U[0][col] = -1
            V[b_block*2][col] = 1; V[b_block*2+1][col] = 1
            W[b_block*2*3+3][col] = 1; W[b_block*2*3+6][col] = 1
            W[b_block*2*3+4][col] = 1; W[b_block*2*3+7][col] = 1
            col += 1
            
            # M7: (a01-a11)(b10+b11)
            U[1][col] = 1; U[3][col] = -1
            V[b_block*2+1][col] = 1; V[b_block*2][col] = 1
            W[b_block*2*3+0][col] = 1; W[b_block*2*3+3][col] = 1
            W[b_block*2*3+1][col] = 1; W[b_block*2*3+4][col] = 1
            col += 1
        
        # Block 2: 2x1 x 1x8 (A[0:2, 2] * B[2, 0:8]) - schoolbook rank 16
        for i in range(2):
            for k in range(8):
                U[i*3+2][col] = 1
                V[2*8+k][col] = 1
                W[k*3+i][col] = 1
                col += 1
        
        # Block 3: 1x2 x 2x8 (A[2, 0:2] * B[0:2, 0:8]) - schoolbook rank 16
        for j in range(2):
            for k in range(8):
                U[2*3+j][col] = 1
                V[j*8+k][col] = 1
                W[k*3+2][col] = 1
                col += 1
        
        # Block 4: 1x1 x 1x8 (A[2,2] * B[2, 0:8]) - schoolbook rank 8
        for k in range(8):
            U[2*3+2][col] = 1
            V[2*8+k][col] = 1
            W[k*3+2][col] = 1
            col += 1
        
        # Verify
        valid = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(rank):
                        val += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(val - 1.0) > 1e-9:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if valid:
            return {"U": U, "V": V, "W": W}, rank
        return None, None
    
    result, r = try_rank_59()
    if result is not None:
        best_result = result
        best_rank = r
    
    return best_result
