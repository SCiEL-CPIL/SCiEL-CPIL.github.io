import time
import numpy as np

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS  # n=2, m=4, p=8

    # Strassen 2x2 local factorization (rank 7).
    # Local A indices a00=0,a01=1,a10=2,a11=3 ; local B b00=0,b01=1,b10=2,b11=3
    # local C c00=0,c01=1,c10=2,c11=3
    U_loc = [[0]*7 for _ in range(4)]
    V_loc = [[0]*7 for _ in range(4)]
    W_loc = [[0]*7 for _ in range(4)]

    # r=0: M1=(a00+a11)(b00+b11) -> c00+1, c11+1
    U_loc[0][0]=1; U_loc[3][0]=1
    V_loc[0][0]=1; V_loc[3][0]=1
    W_loc[0][0]=1; W_loc[3][0]=1
    # r=1: M2=(a10+a11)b00 -> c10+1, c11-1
    U_loc[2][1]=1; U_loc[3][1]=1
    V_loc[0][1]=1
    W_loc[2][1]=1; W_loc[3][1]=-1
    # r=2: M3=a00(b01-b11) -> c01+1, c11+1
    U_loc[0][2]=1
    V_loc[1][2]=1; V_loc[3][2]=-1
    W_loc[1][2]=1; W_loc[3][2]=1
    # r=3: M4=a11(b10-b00) -> c00+1, c10+1
    U_loc[3][3]=1
    V_loc[2][3]=1; V_loc[0][3]=-1
    W_loc[0][3]=1; W_loc[2][3]=1
    # r=4: M5=(a00+a01)b11 -> c00-1, c01+1
    U_loc[0][4]=1; U_loc[1][4]=1
    V_loc[3][4]=1
    W_loc[0][4]=-1; W_loc[1][4]=1
    # r=5: M6=(a10-a00)(b00+b01) -> c11+1
    U_loc[2][5]=1; U_loc[0][5]=-1
    V_loc[0][5]=1; V_loc[1][5]=1
    W_loc[3][5]=1
    # r=6: M7=(a01-a11)(b10+b11) -> c00+1
    U_loc[1][6]=1; U_loc[3][6]=-1
    V_loc[2][6]=1; V_loc[3][6]=1
    W_loc[0][6]=1

    num_blocks = (p // 2) * 2  # 4 column-pair blocks * 2 A-halves = 8 groups
    total_rank = num_blocks * 7  # 56

    U = [[0]*total_rank for _ in range(n*m)]   # 8 x 56
    V = [[0]*total_rank for _ in range(m*p)]   # 32 x 56
    W = [[0]*total_rank for _ in range(p*n)]   # 16 x 56

    group = 0
    for jj in range(p // 2):        # column-pair of output
        for half in range(2):       # which A-half / B-half
            base_r = group * 7
            group += 1

            a_map = [
                0*m + (2*half + 0),
                0*m + (2*half + 1),
                1*m + (2*half + 0),
                1*m + (2*half + 1),
            ]
            b_map = [
                (2*half + 0)*p + (2*jj + 0),
                (2*half + 0)*p + (2*jj + 1),
                (2*half + 1)*p + (2*jj + 0),
                (2*half + 1)*p + (2*jj + 1),
            ]
            c_map = [
                (2*jj + 0)*n + 0,
                (2*jj + 1)*n + 0,
                (2*jj + 0)*n + 1,
                (2*jj + 1)*n + 1,
            ]

            for r in range(7):
                rr = base_r + r
                for la in range(4):
                    val = U_loc[la][r]
                    if val:
                        U[a_map[la]][rr] += val
                for lb in range(4):
                    val = V_loc[lb][r]
                    if val:
                        V[b_map[lb]][rr] += val
                for lc in range(4):
                    val = W_loc[lc][r]
                    if val:
                        W[c_map[lc]][rr] += val

    # Verify correctness with a random test
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Build the multiplication tensor T[i,j,k] = 1 if output C[i,k] depends on A[i,j]*B[j,k]
    # C = A @ B, so C[i,k] = sum_j A[i,j] * B[j,k]
    # Tensor T[i,j,k] = 1 for all valid i,j,k
    # Decomposition: T[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
    # where W index is k*n+i, so W[k*n+i, r]
    
    # Verify: for random A, B, check A@B == sum_r (U[:,r] outer V[:,r]) applied
    # Actually: C[i,k] = sum_j A[i,j] B[j,k] = sum_r (sum_j A[i,j] U[i,j? no...
    # U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
    # U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] summed over r should give 1 for each (i,j,k)
    
    # Let's verify with specific values
    np.random.seed(42)
    A_test = np.random.randn(n, m)
    B_test = np.random.randn(m, p)
    C_expected = A_test @ B_test
    
    # Compute C via decomposition: C[i,k] = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] * A_test[i,j]) * (sum_j V[j*p+k, r] * B_test[j,k])
    # Wait, that's not right. The tensor decomposition is:
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # And C[i,k] = sum_j A[i,j] * B[j,k] = sum_{i,j,k} T[i,j,k] A[i,j] B[j,k]... no
    # Actually the tensor is T[i,j,k] such that C[i,k] = sum_j T[i,j,k] A[i,j] B[j,k]
    # But T[i,j,k] = delta(i,k independent of j)? No, T[i,j,k] = 1 for all i,j,k
    # So C[i,k] = sum_j A[i,j] B[j,k]
    # The decomposition: sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] should equal 1 for all i,j,k
    
    # Check a few entries
    correct = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(total_rank):
                    s += U_arr[i*m+j, r] * V_arr[j*p+k, r] * W_arr[k*n+i, r]
                if abs(s - 1.0) > 1e-9:
                    correct = False
                    break
            if not correct:
                break
        if not correct:
            break
    
    # Safety deadline guard
    if time.time() - start_time > 115:
        pass

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
