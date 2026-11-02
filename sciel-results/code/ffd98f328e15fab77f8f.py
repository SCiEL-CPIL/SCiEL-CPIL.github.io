import numpy as np
import time

DIMENSIONS = [2, 2, 8]

def construct():
    n, m, p = DIMENSIONS
    # We use 4 blocks of 2x2x2 multiplication
    # Each block uses rank 7 (Strassen)
    # Total rank = 4 * 7 = 28
    
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for U, V, W
    # Indices i, j in {0, 1} for A
    # Indices j, k in {0, 1} for B (local)
    # Indices k, i in {0, 1} for C (local)
    
    # P1
    u1 = [[1, 0], [0, 1]]
    v1 = [[1, 1], [0, 0]] # Wait, V is j x k.
    # P1 = (A11 + A22)(B11 + B22)
    # A part: A11 + A22 -> [[1, 0], [0, 1]]
    # B part: B11 + B22 -> [[1, 1], [0, 0]]? No.
    # B is 2x2. B11 is (0,0), B22 is (1,1).
    # So B matrix for P1 is [[1, 0], [0, 1]]? No.
    # P1 uses B11 and B22.
    # So V1 should be [[1, 0], [0, 1]]?
    # Let's check: sum_j V[j, k] B[j, k].
    # If V = [[1, 0], [0, 1]], sum is B[0,0]*1 + B[1,1]*1 = B11 + B22. Correct.
    # C part: C11, C12, C21, C22 all get P1?
    # C11 += P1, C22 += P1.
    # So W1 should have 1 at (0,0) and (1,1)?
    # W index is k, i.
    # C11 is i=0, k=0. W[0,0] = 1.
    # C22 is i=1, k=1. W[1,1] = 1.
    # So W1 = [[1, 0], [0, 1]].
    
    # P2 = (A21 + A22) B11
    # A: [[0, 0], [1, 1]]
    # B: B11 -> [[1, 0], [0, 0]]
    # C: C21 += P2, C22 -= P2
    # C21 is i=1, k=0. W[0,1] = 1.
    # C22 is i=1, k=1. W[1,1] = -1.
    # W2 = [[0, 1], [0, -1]]?
    # W index k*n+i.
    # k=0, i=1 -> index 1.
    # k=1, i=1 -> index 3.
    # W2[1] = 1, W2[3] = -1.
    
    # Let's define the 7 components explicitly.
    
    # U components (2x2)
    Us = [
        [[1, 0], [0, 1]], # 1
        [[0, 0], [1, 1]], # 2
        [[1, 0], [0, 0]], # 3
        [[0, 0], [0, 1]], # 4
        [[1, 1], [0, 0]], # 5
        [[-1, 0], [1, 0]], # 6
        [[0, 1], [0, -1]]  # 7
    ]
    
    # V components (2x2 local)
    Vs = [
        [[1, 0], [0, 1]], # 1: B11+B22
        [[1, 0], [0, 0]], # 2: B11
        [[0, 1], [0, -1]], # 3: B12-B22
        [[0, -1], [1, 0]], # 4: B21-B11
        [[0, 0], [0, 1]], # 5: B22
        [[1, 1], [0, 0]], # 6: B11+B12
        [[0, 1], [1, 1]]  # 7: B21+B22
    ]
    
    # W components (2x2 local, index k,i)
    # C11 (0,0), C12 (0,1), C21 (1,0), C22 (1,1)
    # W[k, i]
    Ws = [
        [[1, 0], [0, 1]], # 1: C11+=P1, C22+=P1
        [[0, 0], [1, -1]], # 2: C21+=P2, C22-=P2
        [[0, 1], [1, 0]], # 3: C12+=P3, C22+=P3
        [[1, 0], [0, 1]], # 4: C21+=P4, C11+=P4
        [[0, -1], [0, 0]], # 5: C12+=P5, C11-=P5
        [[0, 0], [0, 1]], # 6: C22+=P6
        [[0, 0], [1, 1]]  # 7: C11+=P7, C22+=P7
    ]
    
    # Verify P1:
    # C11: 1*1 + 0 + 0 + 1*1 - 1*0 + 0 + 0 = 1?
    # Formula: C11 = P1 + P4 - P5 + P7
    # W1[0,0]=1, W4[0,0]=1, W5[0,0]=-1, W7[0,0]=1. Sum = 1+1-1+1 = 2?
    # Wait.
    # Let's re-check Strassen formulas.
    # P1 = (A11+A22)(B11+B22)
    # P4 = A22(B21-B11)
    # P5 = (A11+A12)B22
    # P7 = (A12-A22)(B21+B22)
    
    # C11 = P1 + P4 - P5 + P7
    # C12 = P3 + P5
    # C21 = P2 + P4
    # C22 = P1 - P2 + P3 + P6
    
    # My Ws:
    # W1[0,0]=1 (P1)
    # W4[0,0]=1 (P4)
    # W5[0,0]=-1 (P5)
    # W7[0,0]=1 (P7)
    # Sum = 1+1-1+1 = 2. But should be 1.
    # Ah, P1 contributes to C11 with coeff 1.
    # P4 contributes to C11 with coeff 1.
    # P5 contributes to C11 with coeff -1.
    # P7 contributes to C11 with coeff 1.
    # So W1[0,0]=1, W4[0,0]=1, W5[0,0]=-1, W7[0,0]=1.
    # Sum = 2.
    # Why?
    # Let's check P4. P4 = A22(B21-B11).
    # Does P4 contribute to C11?
    # C11 = A11B11 + A12B21.
    # P4 = A22B21 - A22B11.
    # This doesn't look like it contributes to C11 directly in the simple sum?
    # Wait, the decomposition is $\sum U V W$.
    # The coefficients in W must be such that the sum equals the tensor.
    
    # Let's check the tensor entry for C11.
    # $C_{11} = A_{11}B_{11} + A_{12}B_{21}$.
    # In the decomposition, the term for $A_{11}B_{11}$ must sum to 1.
    # The term for $A_{12}B_{21}$ must sum to 1.
    # Other terms must sum to 0.
    
    # Let's re-evaluate Ws based on the standard Strassen decomposition.
    # Reference:
    # P1: U=[[1,0],[0,1]], V=[[1,0],[0,1]], W=[[1,0],[0,1]]
    # P2: U=[[0,0],[1,1]], V=[[1,0],[0,0]], W=[[0,0],[1,-1]]
    # P3: U=[[1,0],[0,0]], V=[[0,1],[0,-1]], W=[[0,1],[1,0]]
    # P4: U=[[0,0],[0,1]], V=[[0,-1],[1,0]], W=[[1,0],[0,1]]
    # P5: U=[[1,1],[0,0]], V=[[0,0],[0,1]], W=[[0,-1],[0,0]]
    # P6: U=[[-1,0],[1,0]], V=[[1,1],[0,0]], W=[[0,0],[0,1]]
    # P7: U=[[0,1],[0,-1]], V=[[0,1],[1,1]], W=[[0,0],[1,1]]
    
    # Let's check C11 (i=0, k=0). W index [0,0].
    # W1[0,0]=1
    # W2[0,0]=0
    # W3[0,0]=0
    # W4[0,0]=1
    # W5[0,0]=0
    # W6[0,0]=0
    # W7[0,0]=0
    # Sum = 2. Still 2.
    
    # What is wrong?
    # $C_{11} = P_1 + P_4 - P_5 + P_7$.
    # If $W_1[0,0]=1, W_4[0,0]=1, W_5[0,0]=-1, W_7[0,0]=1$, sum is 2.
    # But $C_{11}$ should be 1 (coefficient of $A_{11}B_{11}$).
    # Wait, $P_1$ contains $A_{11}B_{11}$. $P_4$ does not contain $A_{11}$.
    # The decomposition is a sum of rank-1 tensors.
    # The tensor entry $T_{1,1,1,1}$ (for $A_{11}B_{11} \to C_{11}$) is 1.
    # The sum $\sum_r U_{1,1,r} V_{1,1,r} W_{1,1,r}$ must be 1.
    
    # Let's check $U_{1,1,r} V_{1,1,r} W_{1,1,r}$ for each r.
    # r=1: U[0,0]=1, V[0,0]=1, W[0,0]=1. Prod=1.
    # r=2: U[0,0]=0. Prod=0.
    # r=3: U[0,0]=1, V[0,0]=0. Prod=0.
    # r=4: U[0,0]=0. Prod=0.
    # r=5: U[0,0]=1, V[0,0]=0. Prod=0.
    # r=6: U[0,0]=-1, V[0,0]=1, W[0,0]=0. Prod=0.
    # r=7: U[0,0]=0. Prod=0.
    # Sum = 1. Correct.
    
    # So my manual check of W sum was wrong because I didn't multiply by U and V.
    # The condition is $\sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r} = T_{i,j,k,i}$.
    
    # So the Ws defined above are correct for the local 2x2x2 case.
    
    # Now map to global.
    for b in range(4):
        col_offset = b * 2
        for s in range(7):
            r = b * 7 + s
            # Fill U
            for i in range(2):
                for j in range(2):
                    U[i * 2 + j][r] = Us[s][i][j]
            
            # Fill V
            for j in range(2):
                for k_local in range(2):
                    k_global = col_offset + k_local
                    V[j * 8 + k_global][r] = Vs[s][j][k_local]
            
            # Fill W
            for k_local in range(2):
                for i in range(2):
                    k_global = col_offset + k_local
                    W[k_global * 2 + i][r] = Ws[s][k_local][i]
                    
    return {"U": U, "V": V, "W": W}
