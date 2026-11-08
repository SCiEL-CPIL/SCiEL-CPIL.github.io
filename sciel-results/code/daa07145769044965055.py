import numpy as np

def construct():
    n, m, p = 4, 4, 6
    
    # M_{4,4,6} ≅ M_{2,2,2} ⊗ M_{2,2,3}
    # Rank = Rank(M_{2,2,2}) * Rank(M_{2,2,3}) = 7 * 6 = 42
    #
    # M_{2,2,2} via Strassen (rank 7)
    # M_{2,2,3} via rank-6 decomposition
    #
    # For M_{2,2,3}: A is 2x2, B is 2x3, C is 2x3
    # C[0,j] = a00*B[0,j] + a01*B[1,j]
    # C[1,j] = a10*B[0,j] + a11*B[1,j]
    #
    # Rank-6 decomposition:
    # Term 1: a00 * B_row0 -> C[0,:] (contributes to C[0,0], C[0,1], C[0,2])
    # Term 2: a01 * B_row1 -> C[0,:]
    # Term 3: a10 * B_row0 -> C[1,:]
    # Term 4: a11 * B_row1 -> C[1,:]
    # But wait, each term must be rank-1 in the tensor sense.
    # A rank-1 tensor for M_{2,2,3} is U[i,k]*V[k,j]*W[i,j]
    # where i in {0,1}, k in {0,1}, j in {0,1,2}
    #
    # For term 1: U[0,0]=1, all other U=0; V[0,j]=1 for all j, V[1,j]=0; W[0,j]=1 for all j, W[1,j]=0
    # This gives: sum_{i,k,j} U[i,k]V[k,j]W[i,j] A[i,k]B[k,j] = A[0,0]*B[0,0]*W[0,0] + A[0,0]*B[0,1]*W[0,1] + A[0,0]*B[0,2]*W[0,2]
    # = a00*(b00 + b01 + b02) if W[0,j]=1. But we want a00*b0j for each j separately.
    #
    # Actually, a rank-1 term contributes to ALL (i,j) entries simultaneously based on U, V, W.
    # For it to contribute a00*b0j to C[0,j] for j=0,1,2, we need:
    # U[0,0]=1, U[0,1]=0, U[1,0]=0, U[1,1]=0
    # V[0,0]=1, V[0,1]=1, V[0,2]=1, V[1,0]=0, V[1,1]=0, V[1,2]=0
    # W[0,0]=1, W[0,1]=1, W[0,2]=1, W[1,0]=0, W[1,1]=0, W[1,2]=0
    # This gives C[0,0] += a00*b00, C[0,1] += a00*b01, C[0,2] += a00*b02. Correct!
    #
    # Similarly for the other 3 terms. So rank 4 suffices? Let me check:
    # Term 1: a00 * B_row0 -> C[0,:]
    # Term 2: a01 * B_row1 -> C[0,:]
    # Term 3: a10 * B_row0 -> C[1,:]
    # Term 4: a11 * B_row1 -> C[1,:]
    # Total: 4 terms. But wait, is this correct?
    # C[0,0] = a00*b00 + a01*b10 ✓
    # C[0,1] = a00*b01 + a01*b11 ✓
    # C[0,2] = a00*b02 + a01*b12 ✓
    # C[1,0] = a10*b00 + a11*b10 ✓
    # C[1,1] = a10*b01 + a11*b11 ✓
    # C[1,2] = a10*b02 + a11*b12 ✓
    # So rank(M_{2,2,3}) = 4!
    
    # Actually wait, let me re-examine. Each "term" above is a single rank-1 tensor.
    # Term 1: U=[[1,0],[0,0]], V=[[1,1,1],[0,0,0]], W=[[1,1,1],[0,0,0]]
    # This is valid. So rank is at most 4.
    # Lower bound: rank >= max(2, 2*3/2, 2*3/2) = max(2,3,3) = 3. So rank is 3 or 4.
    # Let's assume rank 4 for now. Total = 7 * 4 = 28.
    
    R = 7 * 4  # 28
    U = np.zeros((16, R))
    V = np.zeros((24, R))
    W = np.zeros((24, R))
    
    # Strassen for M_{2,2,2} (block level)
    # A blocks: A00, A01, A10, A11 (each 2x2)
    # B blocks: B00, B01, B10, B11 (each 2x3)
    # C blocks: C00, C01, C10, C11 (each 2x3)
    
    # Strassen coefficients
    # M0 = (A00+A11)(B00+B11)
    # M1 = (A10+A11)B00
    # M2 = A00(B01-B11)
    # M3 = A11(B10-B00)
    # M4 = (A00+A01)B11
    # M5 = (A10-A00)(B00+B01)
    # M6 = (A01-A11)(B10+B11)
    # C00 = M0 + M3 - M5 - M6
    # C01 = M2 + M4
    # C10 = M1 + M5
    # C11 = M0 - M1 + M2 + M6
    
    alpha = [
        [1, 0, 0, 1],   # M0: A00 + A11
        [0, 0, 1, 1],   # M1: A10 + A11
        [1, 0, 0, 0],   # M2: A00
        [0, 0, 0, 1],   # M3: A11
        [1, 1, 0, 0],   # M4: A00 + A01
        [-1, 0, 1, 0],  # M5: A10 - A00
        [0, 1, 0, -1]   # M6: A01 - A11
    ]
    beta = [
        [1, 0, 0, 1],   # M0: B00 + B11
        [1, 0, 0, 0],   # M1: B00
        [0, 1, 0, -1],  # M2: B01 - B11
        [-1, 0, 1, 0],  # M3: B10 - B00
        [0, 0, 0, 1],   # M4: B11
        [1, 1, 0, 0],   # M5: B00 + B01
        [0, 0, 1, 1]    # M6: B10 + B11
    ]
    recon = [
        [(0, 0, 1), (1, 1, 1)],    # M0 -> C00, C11
        [(1, 0, 1), (1, 1, -1)],   # M1 -> C10, C11
        [(0, 1, 1), (1, 1, 1)],    # M2 -> C01, C11
        [(0, 0, 1), (1, 0, 1)],    # M3 -> C00, C10
        [(0, 0, -1), (0, 1, 1)],   # M4 -> C00, C01
        [(1, 0, 1), (0, 0, -1)],   # M5 -> C10, C00
        [(1, 1, 1), (0, 0, -1)]    # M6 -> C11, C00
    ]
    
    # Inner rank-4 decomposition for M_{2,2,3}
    # Term 0: a00 * B_row0 -> C[0,:]
    # Term 1: a01 * B_row1 -> C[0,:]
    # Term 2: a10 * B_row0 -> C[1,:]
    # Term 3: a11 * B_row1 -> C[1,:]
    
    # U_inner[i,k] for term t:
    # t=0: U[0,0]=1
    # t=1: U[0,1]=1
    # t=2: U[1,0]=1
    # t=3: U[1,1]=1
    U_inner = [
        [1, 0],
        [0, 1],
        [0, 0],
        [0, 0]
    ]
    U_inner[2][0] = 1
    U_inner[3][1] = 1
    
    # V_inner[k,j] for term t:
    # t=0: V[0,0]=1, V[0,1]=1, V[0,2]=1
    # t=1: V[1,0]=1, V[1,1]=1, V[1,2]=1
    # t=2: V[0,0]=1, V[0,1]=1, V[0,2]=1
    # t=3: V[1,0]=1, V[1,1]=1, V[1,2]=1
    V_inner = [
        [1, 1, 1, 0, 0, 0],
        [0, 0, 0, 1, 1, 1],
        [1, 1, 1, 0, 0, 0],
        [0, 0, 0, 1, 1, 1]
    ]
    
    # W_inner[i,j] for term t:
    # t=0: W[0,0]=1, W[0,1]=1, W[0,2]=1
    # t=1: W[0,0]=1, W[0,1]=1, W[0,2]=1
    # t=2: W[1,0]=1, W[1,1]=1, W[1,2]=1
    # t=3: W[1,0]=1, W[1,1]=1, W[1,2]=1
    W_inner = [
        [1, 1, 1, 0, 0, 0],
        [1, 1, 1, 0, 0, 0],
        [0, 0, 0, 1, 1, 1],
        [0, 0, 0, 1, 1, 1]
    ]
    
    col = 0
    for s in range(7):
        for t in range(4):
            # U: A index = (block_row * 2 + inner_row) * 4 + (block_col * 2 + inner_col)
            for ar in range(2):
                for br in range(2):
                    if alpha[s][ar * 2 + br] != 0:
                        for ur in range(2):
                            for uc in range(2):
                                if U_inner[t][ur * 2 + uc] != 0:
                                    u_idx = (ar * 2 + ur) * 4 + (br * 2 + uc)
                                    U[u_idx, col] += alpha[s][ar * 2 + br] * U_inner[t][ur * 2 + uc]
            
            # V: B index = (block_row * 2 + inner_row) * 6 + (block_col * 3 + inner_col)
            for ac in range(2):
                for bc in range(2):
                    if beta[s][ac * 2 + bc] != 0:
                        for vc in range(2):
                            for vj in range(3):
                                if V_inner[t][vc * 3 + vj] != 0:
                                    v_idx = (ac * 2 + vc) * 6 + (bc * 3 + vj)
                                    V[v_idx, col] += beta[s][ac * 2 + bc] * V_inner[t][vc * 3 + vj]
            
            # W: C index = (block_row * 3 + inner_col) * 4 + (block_col * 2 + inner_row)
            # Wait, the convention is W[k*n+i, r] where k is row of C, i is col of C? 
            # Let me re-read: "output factor follows the official trace convention W[k*n+i,r]"
            # C is n x p = 4 x 6. So W has shape (p*n, rank) = (24, rank).
            # W[k*n+i, r] where k in range(p), i in range(n)? Or k in range(n), i in range(p)?
            # Typically for matrix multiplication C = A*B, C[i,j] = sum_k A[i,k]B[k,j]
            # The tensor is T[i,k,j] = A[i,k]B[k,j] -> C[i,j]
            # U has shape (n*m, rank) = (16, rank), indexed by (i,k)
            # V has shape (m*p, rank) = (24, rank), indexed by (k,j)
            # W has shape (p*n, rank) = (24, rank), indexed by (j,i) per the convention W[k*n+i]
            # So W[j*n + i, r] corresponds to C[i,j]
            
            for cr in range(2):
                for cc in range(2):
                    found = False
                    for (a, b, gamma) in recon[s]:
                        if a == cr and b == cc:
                            found = True
                            for wr in range(2):
                                for wj in range(3):
                                    if W_inner[t][wr * 3 + wj] != 0:
                                        w_idx = wj * 4 + (wr * 2 + wr)  # This is wrong
                                        # W[j*n + i] where i is row of C, j is col of C
                                        # i = cr * 2 + wr, j = cc * 3 + wj
                                        w_idx = wj * 4 + (cr * 2 + wr)
                                        W[w_idx, col] += gamma * W_inner[t][wr * 3 + wj]
                            break
            col += 1
    
    def to_list(mat):
        return [[int(round(x)) if abs(x - round(x)) < 1e-9 else x for x in row] for row in mat]
        
    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
