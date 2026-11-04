"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 5, 5
    start_time = time.time()
    deadline = start_time + 110

    def schoolbook(nn, mm, pp):
        rank = nn * mm * pp
        U = np.zeros((nn * mm, rank))
        V = np.zeros((mm * pp, rank))
        W = np.zeros((pp * nn, rank))
        c = 0
        for i in range(nn):
            for j in range(mm):
                for k in range(pp):
                    U[i * mm + j, c] = 1.0
                    V[j * pp + k, c] = 1.0
                    W[k * nn + i, c] = 1.0
                    c += 1
        return U, V, W, rank

    def strassen_2x2x2():
        U = np.zeros((4, 7)); V = np.zeros((4, 7)); W = np.zeros((4, 7))
        U[0,0]=1;U[3,0]=-1; V[2,0]=1;V[3,0]=1; W[0,0]=1;W[3,0]=-1
        U[2,1]=1;U[3,1]=1; V[0,1]=1; W[2,1]=1;W[3,1]=-1
        U[0,2]=1; V[0,2]=1;V[3,2]=-1; W[1,2]=1;W[3,2]=1
        U[3,3]=1; V[3,3]=1;V[0,3]=-1; W[0,3]=1;W[2,3]=1
        U[0,4]=1;U[1,4]=1; V[3,4]=1; W[0,4]=-1;W[1,4]=1
        U[2,5]=1;U[0,5]=-1; V[0,5]=1;V[1,5]=1; W[3,5]=1
        U[1,6]=1;U[3,6]=-1; V[2,6]=1;V[3,6]=1; W[0,6]=1
        return U, V, W, 7

    def winograd_2x3x3():
        # A 2x3, B 3x3, C 2x3.  Rank 16 (Winograd F(2,3) style).
        # A = [[a0,a1,a2],[b0,b1,b2]], B cols b0,b1,b2 (each 3), C = A@B
        U = np.zeros((6, 16)); V = np.zeros((9, 16)); W = np.zeros((6, 16))
        A = np.array([
            [1,0,1,1,1,0, 1,0,0,0,0,0, 0,0,0,0],
            [1,0,1,0,0,1, 0,0,1,1,1,0, 0,0,0,0],
            [1,0,0,0,0,0, 0,0,1,0,0,0, 0,0,0,0],
            [0,1,0,1,0,0, 0,0,0,0,0,0, 0,0,0,0],
            [0,1,0,0,1,0, 0,0,0,0,0,0, 0,0,0,0],
            [0,1,1,0,0,1, 0,0,0,0,0,0, 0,0,0,0],
        ])
        B = np.array([
            [1,1,0,1,0,0, 0,0,1,1,0,0, 0,0,1,0],
            [1,1,0,0,1,0, 0,0,1,0,1,0, 0,0,1,0],
            [1,0,0,0,0,1, 0,0,0,0,0,1, 0,0,0,1],
            [0,1,0,1,0,0, 0,0,0,1,0,0, 0,0,0,1],
            [0,1,0,0,1,0, 0,0,0,0,1,0, 0,0,0,1],
            [0,0,1,0,0,1, 0,0,0,0,0,1, 0,0,0,1],
            [0,0,1,0,0,0, 0,0,0,0,0,0, 0,0,0,1],
            [0,0,0,0,0,0, 1,0,0,0,0,0, 0,0,0,0],
            [0,0,0,0,0,0, 0,1,0,0,0,0, 0,0,0,0],
        ])
        C = np.array([
            [1,0,0,1,0,0, 1,0,0,0,0,0, 0,0,0,0],
            [1,0,0,0,1,0, 0,1,0,0,0,0, 0,0,0,0],
            [1,0,0,0,0,1, 0,0,1,0,0,0, 0,0,0,0],
            [0,1,0,1,0,0, 0,0,0,1,0,0, 0,0,0,0],
            [0,1,0,0,1,0, 0,0,0,0,1,0, 0,0,0,0],
            [0,1,0,0,0,1, 0,0,0,0,0,1, 0,0,0,0],
        ])
        U = A.astype(float); V = B.astype(float); W = C.astype(float)
        return U, V, W, 16

    def rank_2x2x3():
        # A 2x2, B 2x3, C 2x3.  Rank 10.
        # A = [[a0,a1],[b0,b1]], B = [[x0,x1,x2],[y0,y1,y2]], C = A@B
        U = np.zeros((4, 10)); V = np.zeros((6, 10)); W = np.zeros((6, 10))
        U[0,0]=1;U[1,0]=-1; V[0,0]=1;V[3,0]=1; W[0,0]=1;W[3,0]=-1
        U[2,1]=1;U[3,1]=1; V[0,1]=1; W[2,1]=1;W[3,1]=-1
        U[0,2]=1; V[0,2]=1;V[3,2]=-1; W[1,2]=1;W[3,2]=1
        U[3,3]=1; V[3,3]=1;V[0,3]=-1; W[0,3]=1;W[2,3]=1
        U[0,4]=1;U[1,4]=1; V[3,4]=1; W[0,4]=-1;W[1,4]=1
        U[2,5]=1;U[0,5]=-1; V[0,5]=1;V[3,5]=1; W[3,5]=1
        U[1,6]=1;U[3,6]=-1; V[0,6]=1;V[3,6]=1; W[0,6]=1
        U[1,7]=1; V[1,7]=1; W[1,7]=1
        U[3,8]=1; V[1,8]=1; W[3,8]=1
        U[0,9]=1; V[2,9]=1; W[2,9]=1
        return U, V, W, 10

    # Sub-problems: Split 4x5x5 into 2x2x5 + 2x3x5 blocks
    # Block (0,0): 2x5x5 -> split into 2x3x3(16) + 2x2x3(10) + 2x3x2(12) + 2x2x2(7) = 45
    # Block (1,0): 2x5x5 -> same = 45
    # Total = 90
    
    # Try alternative: use 2x5x5 decomposition directly
    # 2x5x5: can use Winograd-style or other methods
    # For now, keep the current decomposition but optimize verification
    
    U1, V1, W1, r1 = winograd_2x3x3()      # 16
    U2, V2, W2, r2 = rank_2x2x3()          # 10
    U3, V3, W3, r3 = schoolbook(2, 3, 2)   # 12
    U4, V4, W4, r4 = strassen_2x2x2()      # 7
    U5, V5, W5, r5 = winograd_2x3x3()      # 16
    U6, V6, W6, r6 = rank_2x2x3()          # 10
    U7, V7, W7, r7 = schoolbook(2, 3, 2)   # 12
    U8, V8, W8, r8 = strassen_2x2x2()      # 7
    total_rank = r1+r2+r3+r4+r5+r6+r7+r8   # 90

    U_full = np.zeros((n*m, total_rank))
    V_full = np.zeros((m*p, total_rank))
    W_full = np.zeros((p*n, total_rank))
    col = [0]

    def place(U_sub, V_sub, W_sub, r, ar, ac, br, bc, cr, cc, nn, mm, pp):
        for il in range(nn):
            for jl in range(mm):
                U_full[(ar+il)*m+(ac+jl), col[0]:col[0]+r] = U_sub[il*mm+jl, :]
        for jl in range(mm):
            for kl in range(pp):
                V_full[(br+jl)*p+(bc+kl), col[0]:col[0]+r] = V_sub[jl*pp+kl, :]
        for kl in range(pp):
            for il in range(nn):
                W_full[(cc+kl)*n+(cr+il), col[0]:col[0]+r] = W_sub[kl*nn+il, :]
        col[0] += r

    place(U1,V1,W1,r1, 0,0, 0,0, 0,0, 2,3,3)
    place(U2,V2,W2,r2, 0,3, 3,0, 0,0, 2,2,3)
    place(U3,V3,W3,r3, 0,0, 0,3, 0,3, 2,3,2)
    place(U4,V4,W4,r4, 0,3, 3,3, 0,3, 2,2,2)
    place(U5,V5,W5,r5, 2,0, 0,0, 2,0, 2,3,3)
    place(U6,V6,W6,r6, 2,3, 3,0, 2,0, 2,2,3)
    place(U7,V7,W7,r7, 2,0, 0,3, 2,3, 2,3,2)
    place(U8,V8,W8,r8, 2,3, 3,3, 2,3, 2,2,2)

    # Verification using tensor contraction
    # C[i,k] = sum_j A[i,j]*B[j,k] = sum_j sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
    # Vectorized: for each r, compute contribution
    A_test = np.random.randn(n, m)
    B_test = np.random.randn(m, p)
    C_expected = A_test @ B_test
    
    # Vectorized verification
    C_rec = np.zeros((n, p))
    for r in range(total_rank):
        u_col = U_full[:, r].reshape(n, m)
        v_col = V_full[:, r].reshape(m, p)
        w_col = W_full[:, r].reshape(p, n)
        # M[i,j] = A[i,j]*u_col[i,j], N[j,k] = B[j,k]*v_col[j,k]
        # C_rec[i,k] += w_col[k,i] * (M @ N)[i,k]
        M = A_test * u_col
        N = B_test * v_col
        C_rec += w_col.T * (M @ N)
    
    if np.max(np.abs(C_rec - C_expected)) > 1e-8:
        U, V, W, rk = schoolbook(n, m, p)
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
# EVOLVE-BLOCK-END
