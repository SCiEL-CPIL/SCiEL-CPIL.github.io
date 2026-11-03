import time
import numpy as np

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    # n=2, m=4, p=8
    # U: (n*m, rank) = (8, rank)
    # V: (m*p, rank) = (32, rank)
    # W: (p*n, rank) = (16, rank)
    
    # Use 4 independent 2x2x2 Strassen decompositions
    # A is 2x4, B is 4x8, C is 2x8
    # Partition B into 4 blocks of 4x2: B = [B^(0), B^(1), B^(2), B^(3)]
    # For each j, A*B^(j) is 2x2, use Strassen (rank 7)
    # Total rank = 4*7 = 28
    
    # Strassen components for 2x2 matrix multiplication
    # A_local is 2x2: A00, A01, A10, A11
    # B_local is 2x2: B00, B01, B10, B11
    # C_local is 2x2: C00, C01, C10, C11
    # Local indices: 0=A00/B00/C00, 1=A01/B01/C01, 2=A10/B10/C10, 3=A11/B11/C11
    
    U_local = np.zeros((4, 7), dtype=int)
    V_local = np.zeros((4, 7), dtype=int)
    W_local = np.zeros((4, 7), dtype=int)
    
    # r=0 (M1): M1 = (A00+A11)(B00+B11)
    U_local[0, 0] = 1; U_local[3, 0] = 1
    V_local[0, 0] = 1; V_local[3, 0] = 1
    W_local[0, 0] = 1
    W_local[3, 0] = 1
    
    # r=1 (M2): M2 = (A10+A11)B00
    U_local[2, 1] = 1; U_local[3, 1] = 1
    V_local[0, 1] = 1
    W_local[2, 1] = 1
    W_local[3, 1] = -1
    
    # r=2 (M3): M3 = A00(B10-B00)
    U_local[0, 2] = 1
    V_local[2, 2] = 1; V_local[0, 2] = -1
    W_local[1, 2] = 1
    W_local[3, 2] = 1
    
    # r=3 (M4): M4 = A11(B11-B01)
    U_local[3, 3] = 1
    V_local[3, 3] = 1; V_local[1, 3] = -1
    W_local[0, 3] = 1
    W_local[2, 3] = 1
    
    # r=4 (M5): M5 = (A00+A01)B11
    U_local[0, 4] = 1; U_local[1, 4] = 1
    V_local[3, 4] = 1
    W_local[0, 4] = -1
    W_local[1, 4] = 1
    
    # r=5 (M6): M6 = (A10-A00)(B00+B01)
    U_local[2, 5] = 1; U_local[0, 5] = -1
    V_local[0, 5] = 1; V_local[1, 5] = 1
    W_local[3, 5] = 1
    
    # r=6 (M7): M7 = (A01-A11)(B10+B11)
    U_local[1, 6] = 1; U_local[3, 6] = -1
    V_local[2, 6] = 1; V_local[3, 6] = 1
    W_local[0, 6] = 1
    
    total_rank = 28
    
    U = np.zeros((n * m, total_rank), dtype=int)
    V = np.zeros((m * p, total_rank), dtype=int)
    W = np.zeros((p * n, total_rank), dtype=int)
    
    # For each block j (0 to 3):
    # A is shared, so U indices are the same
    # B^(j) = B[:, j*2:(j+1)*2]
    # C^(j) = C[:, j*2:(j+1)*2]
    
    for j in range(4):
        base_r = j * 7
        
        for l in range(7):
            r = base_r + l
            
            # U mapping: local (i,j) -> global i*4+j
            # Local 0=A00=A[0,0:2], Local 1=A01=A[0,2:4]
            # Local 2=A10=A[1,0:2], Local 3=A11=A[1,2:4]
            # A[0,0]=U[0], A[0,1]=U[1], A[0,2]=U[2], A[0,3]=U[3]
            # A[1,0]=U[4], A[1,1]=U[5], A[1,2]=U[6], A[1,3]=U[7]
            # Local 0 (A00): U[0], U[1]
            # Local 1 (A01): U[2], U[3]
            # Local 2 (A10): U[4], U[5]
            # Local 3 (A11): U[6], U[7]
            for li in range(4):
                val = U_local[li, l]
                if val != 0:
                    if li == 0:
                        U[0, r] += val
                        U[1, r] += val
                    elif li == 1:
                        U[2, r] += val
                        U[3, r] += val
                    elif li == 2:
                        U[4, r] += val
                        U[5, r] += val
                    elif li == 3:
                        U[6, r] += val
                        U[7, r] += val
            
            # V mapping: local (i,j) -> global
            # B^(j) = B[:, j*2:(j+1)*2]
            # B00 = B[0:2, j*2:j*2+2] -> V[0*8+j*2], V[0*8+j*2+1], V[1*8+j*2], V[1*8+j*2+1]
            # B01 = B[0:2, j*2+2:j*2+4] -> V[0*8+j*2+2], V[0*8+j*2+3], V[1*8+j*2+2], V[1*8+j*2+3]
            # B10 = B[2:4, j*2:j*2+2] -> V[2*8+j*2], V[2*8+j*2+1], V[3*8+j*2], V[3*8+j*2+1]
            # B11 = B[2:4, j*2+2:j*2+4] -> V[2*8+j*2+2], V[2*8+j*2+3], V[3*8+j*2+2], V[3*8+j*2+3]
            # Local 0 (B00): V[0*8+j*2], V[0*8+j*2+1], V[1*8+j*2], V[1*8+j*2+1]
            # Local 1 (B01): V[0*8+j*2+2], V[0*8+j*2+3], V[1*8+j*2+2], V[1*8+j*2+3]
            # Local 2 (B10): V[2*8+j*2], V[2*8+j*2+1], V[3*8+j*2], V[3*8+j*2+1]
            # Local 3 (B11): V[2*8+j*2+2], V[2*8+j*2+3], V[3*8+j*2+2], V[3*8+j*2+3]
            for li in range(4):
                val = V_local[li, l]
                if val != 0:
                    if li == 0:
                        V[0*8+j*2, r] += val
                        V[0*8+j*2+1, r] += val
                        V[1*8+j*2, r] += val
                        V[1*8+j*2+1, r] += val
                    elif li == 1:
                        V[0*8+j*2+2, r] += val
                        V[0*8+j*2+3, r] += val
                        V[1*8+j*2+2, r] += val
                        V[1*8+j*2+3, r] += val
                    elif li == 2:
                        V[2*8+j*2, r] += val
                        V[2*8+j*2+1, r] += val
                        V[3*8+j*2, r] += val
                        V[3*8+j*2+1, r] += val
                    elif li == 3:
                        V[2*8+j*2+2, r] += val
                        V[2*8+j*2+3, r] += val
                        V[3*8+j*2+2, r] += val
                        V[3*8+j*2+3, r] += val
            
            # W mapping: local (i,j) -> global
            # C^(j) = C[:, j*2:(j+1)*2]
            # C00 = C[0, j*2:j*2+2] -> W[(j*2)*2+0]=W[4*j], W[(j*2+1)*2+0]=W[4*j+2]
            # C01 = C[0, j*2+2:j*2+4] -> W[(j*2+2)*2+0]=W[4*j+4], W[(j*2+3)*2+0]=W[4*j+6]
            # C10 = C[1, j*2:j*2+2] -> W[(j*2)*2+1]=W[4*j+1], W[(j*2+1)*2+1]=W[4*j+3]
            # C11 = C[1, j*2+2:j*2+4] -> W[(j*2+2)*2+1]=W[4*j+5], W[(j*2+3)*2+1]=W[4*j+7]
            # Local 0 (C00): W[4*j], W[4*j+2]
            # Local 1 (C01): W[4*j+4], W[4*j+6]
            # Local 2 (C10): W[4*j+1], W[4*j+3]
            # Local 3 (C11): W[4*j+5], W[4*j+7]
            for li in range(4):
                val = W_local[li, l]
                if val != 0:
                    if li == 0:
                        W[4*j, r] += val
                        W[4*j+2, r] += val
                    elif li == 1:
                        W[4*j+4, r] += val
                        W[4*j+6, r] += val
                    elif li == 2:
                        W[4*j+1, r] += val
                        W[4*j+3, r] += val
                    elif li == 3:
                        W[4*j+5, r] += val
                        W[4*j+7, r] += val
    
    elapsed = time.time() - start_time
    if elapsed > deadline:
        pass
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
