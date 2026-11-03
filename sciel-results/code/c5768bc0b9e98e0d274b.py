"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-56 decomposition of 2x4 * 4x8 matrix multiplication.
    
    Strategy: Split the 2x4x8 multiplication into 8 independent 2x2x2
    multiplications. For each (i, j, k) block where i in {0,1}, j in {0,1},
    k in {0,1}, we multiply A[2i:2i+2, 2j:2j+2] (2x2) by B[2j:2j+2, 2k:2k+2] (2x2)
    to get a contribution to C[2i:2i+2, 2k:2k+2] (2x2).
    
    Each 2x2x2 multiplication uses Strassen's algorithm with rank 7.
    Total rank = 8 * 7 = 56.
    
    The output convention is W[k*n+i, r] where n=2, so W has shape (8*2, rank) = (16, rank).
    Wait, the contract says W has shape (p*n, rank) = (8*2, rank) = (16, rank).
    
    Let me re-read the convention:
    U: (n*m, rank) = (2*4, rank) = (8, rank)
    V: (m*p, rank) = (4*8, rank) = (32, rank)
    W: (p*n, rank) = (8*2, rank) = (16, rank)
    
    The tensor decomposition satisfies:
    C[i, k] = sum_r U[i*m + j, r] * V[j*p + k, r] ... no wait.
    
    Actually, the standard matrix multiplication tensor M_{n,m,p} satisfies:
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    The tensor T has indices (i, j, k) with T[i,j,k] = 1 if... no.
    
    For matrix multiplication, the tensor is T[i, j, k] = 1 for the "middle" index,
    meaning the bilinear form is: for each output element C[i,k], 
    C[i,k] = sum_j A[i,j] * B[j,k].
    
    In the factorization U, V, W:
    C[i,k] = sum_r (sum_j U[i*m+j, r]) * (sum_j V[j*p+k, r]) ... no, that's not right either.
    
    Let me think about this more carefully. The matrix multiplication tensor M of order (n, m, p)
    is defined on the vector space with basis e_{i,j,k} where:
    T(e_{i,j}, e_{j,k}) = e_{i,k} for each valid i, j, k.
    
    A tensor decomposition of rank r means:
    T = sum_{r'=1}^{r} u_{r'} ⊗ v_{r'} ⊗ w_{r'}
    
    where u_{r'} is in the A-space (n*m dimensional), v_{r'} is in the B-space (m*p dimensional),
    w_{r'} is in the C-space (n*p dimensional).
    
    The entries of U, V, W are:
    U[i*m + j, r'] = coefficient of e_{i,j} in u_{r'}
    V[j*p + k, r'] = coefficient of e_{j,k} in v_{r'}
    W[k*n + i, r'] = coefficient of e_{i,k} in w_{r'}  [using the W[k*n+i, r] convention]
    
    Wait, the convention says W[k*n+i, r]. So the index for C is (i, k) but stored as k*n + i.
    So W[k*n+i, r'] = coefficient of e_{i,k} in w_{r'}.
    
    For the schoolbook decomposition, for each (i, j, k):
    U[i*m+j, col] = 1
    V[j*p+k, col] = 1
    W[k*n+i, col] = 1
    
    This gives C[i,k] = sum_col U[i*m+j_col, col] * V[j_col*p+k, col] * ... 
    
    Actually, the tensor equation is:
    T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    Wait, no. The tensor is a 3-tensor. The factorization is:
    T = sum_r u_r ⊗ v_r ⊗ w_r
    
    where (u_r)_{i,j} = U[i*m+j, r], (v_r)_{j,k} = V[j*p+k, r], (w_r)_{i,k} = W[k*n+i, r]
    
    And T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    But T for matrix multiplication is: T[i,j,k] = 1 if we're multiplying A[i,j] * B[j,k] to get C[i,k].
    So T[i,j,k] should be 1 for all valid (i,j,k), and the "action" is that:
    C[i,k] = sum_j T[i,j,k] * A[i,j] * B[j,k]
    
    Hmm, I think the correct interpretation is:
    The matrix multiplication tensor M has M[i,j,k] = 1 (it's the identity in the middle index).
    A rank-r factorization means:
    M[i,j,k] = sum_{r=1}^{r_max} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    This must hold for ALL i, j, k.
    
    For the schoolbook: for each (i,j,k), there's a column r with U[i*m+j,r]=1, V[j*p+k,r]=1, W[k*n+i,r]=1.
    So M[i,j,k] = sum_r ... = 1 for the column corresponding to (i,j,k), and 0 for others. So M[i,j,k] = 1. ✓
    
    Now for Strassen on 2x2x2:
    The 2x2x2 multiplication tensor has indices (i,j,k) with i,j,k in {0,1}.
    M2[i,j,k] = 1 for all i,j,k in {0,1}.
    
    Strassen gives rank 7. The 7 products are:
    m1 = (A00+A11)(B00+B11)
    m2 = (A10+A11)B00
    m3 = A00(B10-B11)
    m4 = A11(B01-B10)
    m5 = (A00+A10)B11
    m6 = (A01-A00)(B00+B10)
    m7 = (A10-A11)(B01+B11)
    
    C00 = m1 + m4 - m5 + m7
    C01 = m3 + m1 - m2 + m6
    C10 = m2 + m5 - m1 + m3
    C11 = m1 - m4 + m2 - m6
    
    So for Strassen, the factor matrices for a single 2x2x2 are:
    
    For each of the 7 ranks r = 0..6:
    
    r=0: m1 = (A00+A11)(B00+B11)
    U: (1,1,1,1) for (0,0),(0,1),(1,0),(1,1) → U[0,0]=1, U[0,1]=0... 
    Wait, U has shape (2*2, 7) = (4, 7). Index i*2+j.
    For A00: i=0,j=0 → U[0, r]
    For A01: i=0,j=1 → U[1, r]
    For A10: i=1,j=0 → U[2, r]
    For A11: i=1,j=1 → U[3, r]
    
    m1: U[0]=1, U[1]=0, U[2]=0, U[3]=1
    B00: j=0,k=0 → V[0, r]
    B01: j=0,k=1 → V[1, r]
    B10: j=1,k=0 → V[2, r]
    B11: j=1,k=1 → V[3, r]
    m1: V[0]=1, V[1]=0, V[2]=0, V[3]=1
    C: W[k*2+i, r]
    C00: i=0,k=0 → W[0, r]
    C01: i=0,k=1 → W[2, r]
    C10: i=1,k=0 → W[1, r]
    C11: i=1,k=1 → W[3, r]
    m1 contributes to C00 with +1, C01 with -1, C10 with -1, C11 with +1
    So W[0]=1, W[2]=-1, W[1]=-1, W[3]=1
    
    r=1: m2 = (A10+A11)B00
    U[0]=0, U[1]=0, U[2]=1, U[3]=1
    V[0]=1, V[1]=0, V[2]=0, V[3]=0
    C00: -1, C01: 1, C10: 1, C11: -1
    W[0]=-1, W[2]=1, W[1]=1, W[3]=-1
    
    r=2: m3 = A00(B10-B11)
    U[0]=1, U[1]=0, U[2]=0, U[3]=0
    V[0]=0, V[1]=-1, V[2]=1, V[3]=0
    C00: -1, C01: 1, C10: 1, C11: 0
    W[0]=-1, W[2]=1, W[1]=1, W[3]=0
    
    r=3: m4 = A11(B01-B10)
    U[0]=0, U[1]=0, U[2]=0, U[3]=1
    V[0]=0, V[1]=1, V[2]=-1, V[3]=0
    C00: 1, C01: 0, C10: -1, C11: -1
    W[0]=1, W[2]=0, W[1]=-1, W[3]=-1
    
    r=4: m5 = (A00+A10)B11
    U[0]=1, U[1]=0, U[2]=1, U[3]=0
    V[0]=0, V[1]=0, V[2]=0, V[3]=1
    C00: -1, C01: 0, C10: 1, C11: 0
    W[0]=-1, W[2]=0, W[1]=1, W[3]=0
    
    r=5: m6 = (A01-A00)(B00+B10)
    U[0]=-1, U[1]=1, U[2]=0, U[3]=0
    V[0]=1, V[1]=0, V[2]=1, V[3]=0
    C00: 0, C01: 1, C10: 0, C11: -1
    W[0]=0, W[2]=1, W[1]=0, W[3]=-1
    
    r=6: m7 = (A10-A11)(B01+B11)
    U[0]=0, U[1]=0, U[2]=1, U[3]=-1
    V[0]=0, V[1]=1, V[2]=0, V[3]=1
    C00: 1, C01: 0, C10: 0, C11: -1
    W[0]=1, W[2]=0, W[1]=0, W[3]=-1
    
    Now for the full 2x4x8 problem:
    We have 8 blocks: for (ii, jj, kk) where ii in {0,1}, jj in {0,1}, kk in {0,1}.
    
    Block (ii, jj, kk):
    A sub-block: A[2ii:2ii+2, 2jj:2jj+2] (2x2)
    B sub-block: B[2jj:2jj+2, 2kk:2kk+2] (2x2)
    C sub-block: C[2ii:2ii+2, 2kk:2kk+2] (2x2)
    
    For this block, using Strassen rank 7:
    For local indices (i, j, k) in {0,1}^3:
    Global A index: i_global = 2ii + i, j_global = 2jj + j
    Global B index: j_global = 2jj + j, k_global = 2kk + k
    Global C index: i_global = 2ii + i, k_global = 2kk + k
    
    U row: i_global * 4 + j_global (since n*m = 2*4 = 8)
    V row: j_global * 8 + k_global (since m*p = 4*8 = 32)
    W row: k_global * 2 + i_global (since p*n = 8*2 = 16)
    
    Let me build this.
    """
    n, m, p = DIMENSIONS  # n=2, m=4, p=8
    
    # Strassen coefficients for 2x2x2
    # For each of 7 ranks, the U, V, W coefficients for local (i,j) and (j,k) and (i,k)
    # Local A indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    # Local B indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    # Local C indices: (0,0)=0, (0,1)=2, (1,0)=1, (1,1)=3  [using W[k*2+i]]
    
    # Strassen: U_coeffs[r][i*2+j], V_coeffs[r][j*2+k], W_coeffs[r][k*2+i]
    U_strassen = [
        [1, 0, 0, 1],  # r=0: A00+A11
        [0, 0, 1, 1],  # r=1: A10+A11
        [1, 0, 0, 0],  # r=2: A00
        [0, 0, 0, 1],  # r=3: A11
        [1, 0, 1, 0],  # r=4: A00+A10
        [-1, 1, 0, 0], # r=5: A01-A00
        [0, 0, 1, -1], # r=6: A10-A11
    ]
    
    V_strassen = [
        [1, 0, 0, 1],  # r=0: B00+B11
        [1, 0, 0, 0],  # r=1: B00
        [0, -1, 1, 0], # r=2: B10-B11
        [0, 1, -1, 0], # r=3: B01-B10
        [0, 0, 0, 1],  # r=4: B11
        [1, 0, 1, 0],  # r=5: B00+B10
        [0, 1, 0, 1],  # r=6: B01+B11
    ]
    
    W_strassen = [
        [1, -1, -1, 1],  # r=0: C00+1, C01-1, C10-1, C11+1 → W[0]=1, W[2]=-1, W[1]=-1, W[3]=1
        [-1, 1, 1, -1],  # r=1: C00-1, C01+1, C10+1, C11-1
        [-1, 1, 1, 0],   # r=2: C00-1, C01+1, C10+1, C11+0
        [1, 0, -1, -1],  # r=3: C00+1, C01+0, C10-1, C11-1
        [-1, 0, 1, 0],   # r=4: C00-1, C01+0, C10+1, C11+0
        [0, 1, 0, -1],   # r=5: C00+0, C01+1, C10+0, C11-1
        [1, 0, 0, -1],   # r=6: C00+1, C01+0, C10+0, C11-1
    ]
    
    rank = 56
    U = [[0] * rank for _ in range(n * m)]   # (8, 56)
    V = [[0] * rank for _ in range(m * p)]   # (32, 56)
    W = [[0] * rank for _ in range(p * n)]   # (16, 56)
    
    for ii in range(2):
        for jj in range(2):
            for kk in range(2):
                base_col = (ii * 4 + jj * 2 + kk) * 7
                for r in range(7):
                    col = base_col + r
                    for i in range(2):
                        for j in range(2):
                            u_idx = (2 * ii + i) * 4 + (2 * jj + j)
                            U[u_idx][col] = U_strassen[r][i * 2 + j]
                    for j in range(2):
                        for k in range(2):
                            v_idx = (2 * jj + j) * 8 + (2 * kk + k)
                            V[v_idx][col] = V_strassen[r][j * 2 + k]
                    for i in range(2):
                        for k in range(2):
                            w_idx = (2 * kk + k) * 2 + (2 * ii + i)
                            W[w_idx][col] = W_strassen[r][k * 2 + i]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
