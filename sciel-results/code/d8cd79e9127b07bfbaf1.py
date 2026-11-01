import time
import random

# EVOLVE-BLOCK-START

def _check_decomposition(U, V, W, n, m, p):
    """Check that the decomposition is exact for random inputs."""
    rank = len(U[0])
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # The decomposition says: for all A, B: C[i,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Wait, that's not right. The tensor T_{n,m,p} has indices (i, j, k) where:
    #   i in [0, n), j in [0, m), k in [0, p)
    # And T[i, j, k] = 1 if and only if j is the shared index... 
    # Actually, the matrix multiplication tensor is: T: R^{n*m} x R^{m*p} -> R^{n*p}
    # T(A, B)_{i,k} = sum_j A_{i,j} * B_{j,k}
    # In terms of the flat indices:
    #   A is indexed by (i, j) -> flat index i*m + j
    #   B is indexed by (j, k) -> flat index j*p + k
    #   C is indexed by (i, k) -> flat index k*n + i (following W[k*n+i, r] convention)
    # 
    # The rank-r decomposition: C_{i,k} = sum_r U[i*m + j, r] ... wait no.
    # Actually for matrix multiplication tensor, the decomposition is:
    #   C_{i,k} = sum_r U[i, r] * V[j, r] ... no that's not right either.
    # 
    # Let me think more carefully. The tensor T_{n,m,p} is in (R^{nm})* x (R^{mp})* x (R^{np})*
    # T(A, B) = C where C_{ik} = sum_j A_{ij} B_{jk}
    # As a trilinear form: T(A, B, C) = sum_{i,j,k} A_{ij} B_{jk} C_{ki}
    # The tensor has rank r if there exist U_{(ij),r}, V_{(jk),r}, W_{(ki),r} such that:
    #   T_{(ij),(jk),(ki)} = sum_r U_{(ij),r} V_{(jk),r} W_{(ki),r}
    # where T_{(ij),(jk),(ki)} = 1 if j == j' (the shared index matches) and i,i and k,k are free.
    # Wait, T_{(ij),(jk),(ki)} = delta_{j,j'} where (ij) is the first index, (j'k') is the second, (k''i'') is third.
    # Actually T_{(ij),(jk),(ki)} = 1 if j == j AND i == i AND k == k, which is always 1.
    # No wait: T is the matrix multiplication tensor, so T_{(ij),(jk),(ki)} = delta_{j,j} * ... 
    # 
    # Let me just use the standard formulation:
    # The matrix multiplication tensor M_{n,m,p} is defined by:
    #   M_{n,m,p} = sum_{i=0}^{n-1} sum_{j=0}^{m-1} sum_{k=0}^{p-1} e_{i,j} ⊗ e_{j,k} ⊗ e_{i,k}
    # where e_{i,j} is the basis vector for the (i,j) entry.
    # So M_{n,m,p}[(i,j), (j',k'), (i'',k'')] = 1 if j == j' AND i == i'' AND k == k'', else 0.
    # 
    # In the flat indexing used here:
    #   First factor index: a = i*m + j, for i in [0,n), j in [0,m)
    #   Second factor index: b = j'*p + k', for j' in [0,m), k' in [0,p)
    #   Third factor index: c = k''*n + i'', for k'' in [0,p), i'' in [0,n)
    # 
    # M[(i*m+j), (j'*p+k'), (k''*n+i'')] = 1 if j == j' AND i == i'' AND k == k''
    # 
    # The rank-r decomposition: M[a, b, c] = sum_r U[a, r] * V[b, r] * W[c, r]
    # 
    # So we need: for all (i,j), (j',k'), (k'',i''):
    #   delta_{j,j'} * delta_{i,i''} * delta_{k,k''} = sum_r U[i*m+j, r] * V[j'*p+k', r] * W[k''*n+i'', r]

    # Let's verify with random matrices
    for _ in range(5):
        A = [[random.randint(-3, 3) for _ in range(m)] for _ in range(n)]
        B = [[random.randint(-3, 3) for _ in range(p)] for _ in range(m)]
        # Compute AB directly
        AB = [[sum(A[i][j] * B[j][k] for j in range(m)) for k in range(p)] for i in range(n)]
        
        # Compute using decomposition
        # First compute U*A: for each r, sum_j U[i*m+j, r] * A[i][j]... 
        # Wait, U[a, r] where a = i*m + j. So for fixed i, we sum over j: U[i*m+j, r] * A[i][j]
        # Let's define: X[r][i] = sum_j U[i*m+j, r] * A[i][j]
        # Similarly: Y[r][k] = sum_j V[j*p+k, r] * B[j][k]
        # Then: Z[r][i][k] = X[r][i] * Y[r][k]
        # And: C[i][k] = sum_r W[k*n+i, r] * Z[r][i][k]
        
        X = [[0] * n for _ in range(rank)]
        for r in range(rank):
            for i in range(n):
                s = 0
                for j in range(m):
                    s += U[i * m + j][r] * A[i][j]
                X[r][i] = s
        
        Y = [[0] * p for _ in range(rank)]
        for r in range(rank):
            for k in range(p):
                s = 0
                for j in range(m):
                    s += V[j * p + k][r] * B[j][k]
                Y[r][k] = s
        
        C = [[0] * p for _ in range(n)]
        for i in range(n):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += W[k * n + i][r] * X[r][i] * Y[r][k]
                C[i][k] = s
        
        if C != AB:
            return False
    return True


def _schoolbook(n, m, p):
    """Standard schoolbook decomposition of rank n*m*p."""
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


def _winograd_fft(n, m, p):
    """
    Try Winograd-style / FFT-based decomposition.
    
    For 2x2x6, we can think of this as:
    - The m=2 dimension is the "convolution" dimension
    - We can use a length-2 DFT (which is just {1,1}, {1,-1}) to diagonalize the m-dimension
    - This gives 2 "frequency" components, each of which is a 2x6 outer product
    
    Actually, let me think about this differently. The tensor is:
    T[(i,j), (j',k), (k',i')] = delta_{j,j'} * delta_{i,i'} * delta_{k,k'}
    
    For fixed i and k, we need T[(i,j), (j,k), (k,i)] = delta_{j,j} = 1 for all j.
    
    So for each (i,k) pair, the "fiber" over j is the all-ones vector of length m=2.
    The all-ones vector of length 2 has rank 1 as a vector, but that doesn't help directly.
    
    Let me try a different approach: use the fact that 2x2 matrix multiplication has rank 7 (Strassen),
    and see if we can extend that to 2x2x6.
    
    Actually, for 2x2x6, we can view it as 3 independent 2x2x2 multiplications (one for each pair of columns),
    but that gives rank 3*7 = 21, which is worse than schoolbook (24).
    
    Wait, schoolbook is n*m*p = 2*2*6 = 24. So we need to beat 24.
    
    Let me try a Winograd approach. The key idea is to embed the 2x2x6 problem into a larger
    polynomial multiplication problem and use FFT to diagonalize.
    
    Consider the 2x2x2 case first. Strassen's algorithm uses 7 multiplications.
    The idea is to form linear combinations of entries, multiply, and then extract.
    
    For 2x2x6, we can group the 6 columns into 3 groups of 2, and apply Strassen to each group.
    But that gives 3*7 = 21.
    
    Can we do better? Maybe by sharing some of the linear combinations across groups?
    
    Let me try a more systematic approach using the Winograd Fourier Transform Algorithm.
    
    The WFTA for convolution of length N uses the DFT of order N. For N=4, the DFT matrix is:
    F = [[1, 1, 1, 1],
         [1, -i, -1, i],
         [1, -1, 1, -1],
         [1, i, -1, -i]]
    
    But we need real or rational entries. Let me use N=2 instead.
    
    For N=2, the DFT is:
    F = [[1, 1],
         [1, -1]]
    
    This is invertible with (1/2)F^{-1} = F.
    
    Now, for 2x2x6, let me try to use a tensor product structure.
    
    Actually, let me just try to find a decomposition of rank < 24 by search or by using
    known results.
    
    For 2x2xn matrix multiplication, the rank is known to be:
    - 2x2x2: 7
    - 2x2x3: 11 (or 12?)
    - 2x2x4: 15 (or 16?)
    
    Actually, I recall that the rank of 2x2xn is at most 3n+1 for n>=2, using a Winograd-like approach.
    For n=6, that would be 3*6+1 = 19.
    
    Let me try to construct such a decomposition.
    
    The Winograd algorithm for 2x2xn uses the following idea:
    - Pad the 2x2 matrix to a 2x2 matrix (no padding needed for 2x2)
    - Use a length-2 FFT in the j-dimension
    
    Actually, let me look at this from the perspective of the tensor structure.
    
    T[(i,j), (j',k), (k',i')] = delta_{j,j'} * delta_{i,i'} * delta_{k,k'}
    
    Let's fix i and k. Then the fiber is T[(i,j), (j,k), (k,i)] = delta_{j,j} for j=0,1.
    This is the 2x2 all-ones tensor, which has rank 1.
    
    But we need the full tensor, not just fibers.
    
    Let me try a different approach: use the fact that the tensor can be written as:
    T = sum_{i=0}^{1} sum_{k=0}^{5} e_i ⊗ (sum_{j=0}^{1} e_j ⊗ e_{j,k}) ⊗ e_{k,i}
    
    Hmm, this is getting complicated. Let me just try to use a computational approach.
    
    Actually, let me try the following construction:
    
    For the 2x2x6 case, I'll use a Winograd-like scheme with rank 16.
    
    The idea: use a 4-point DFT (with real parts) to diagonalize the convolution structure.
    
    Actually, let me just implement a known rank-16 decomposition if one exists, or use
    a rank-21 decomposition (3 Strassen blocks) which is better than schoolbook.
    
    For now, let me implement the 3-Strassen-block decomposition of rank 21.
    """
    n, m, p = 2, 2, 6
    
    # Strassen for 2x2x2 has rank 7.
    # We have 6 columns, so 3 blocks of 2 columns each.
    # Each block uses 7 multiplications, total 21.
    
    # Strassen's 7 products for 2x2x2:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # For 2x2x6, we split B into three 2x2 blocks: B = [B0 | B1 | B2]
    # where B0, B1, B2 are 2x2.
    
    # For each block Br, we apply Strassen with 7 products.
    # Total rank = 3 * 7 = 21.
    
    rank = 21
    U = [[0] * rank for _ in range(n * m)]  # 4 x 21
    V = [[0] * rank for _ in range(m * p)]  # 12 x 21
    W = [[0] * rank for _ in range(p * n)]  # 12 x 21
    
    # Strassen coefficients for 2x2x2
    # U indices: (i,j) for i in {0,1}, j in {0,1}
    # V indices: (j,k) for j in {0,1}, k in {0,1}
    # W indices: (k,i) for k in {0,1}, i in {0,1}
    
    # Strassen products:
    # M1: U: (0,0)->1, (1,1)->1; V: (0,0)->1, (1,1)->1; W: (0,0)->1, (0,1)->1, (1,0)->1, (1,1)->1
    #     Wait, W for M1 should give: C00 += M1, C01 += M1, C10 += M1, C11 += M1
    #     But that's not right. Let me be more careful.
    
    # Strassen:
    # C00 = M1 - M4 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M3 + M6
    
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # U coefficients for each M (row is i*m+j):
    # M1: U[0,0]=1, U[3,1]=1  (A00 and A11)
    # M2: U[2,0]=1, U[3,1]=1  (A10 and A11)
    # M3: U[0,0]=1            (A00)
    # M4: U[3,1]=1            (A11)
    # M5: U[0,0]=1, U[1,1]=1  (A00 and A01)
    # M6: U[2,0]=1, U[0,-1]=1 (A10 and -A00)
    # M7: U[1,1]=1, U[3,-1]=1 (A01 and -A11)
    
    # Let me index U as U[i*m + j, r]
    # i=0,j=0: index 0
    # i=0,j=1: index 1
    # i=1,j=0: index 2
    # i=1,j=1: index 3
    
    # V coefficients for each M (row is j*p+k for the 2x2 case, so j*2+k):
    # M1: V[0,0]=1, V[3,1]=1  (B00 and B11)
    # M2: V[0,0]=1            (B00)
    # M3: V[1,1]=1, V[3,-1]=1 (B01 and -B11)
    # M4: V[2,1]=1, V[0,-1]=1 (B10 and -B00)
    # M5: V[3,1]=1            (B11)
    # M6: V[0,0]=1, V[1,1]=1  (B00 and B01)
    # M7: V[2,1]=1, V[3,1]=1  (B10 and B11)
    
    # W coefficients for each M (row is k*n+i for the 2x2 case, so k*2+i):
    # C00 = M1 - M4 + M7: W[0,0] gets M1 coeff 1, M4 coeff -1, M7 coeff 1
    # C01 = M3 + M5:       W[1,0] gets M3 coeff 1, M5 coeff 1
    # C10 = M2 + M4:       W[2,0] gets M2 coeff 1, M4 coeff 1
    # C11 = M1 - M3 + M6:  W[3,0] gets M1 coeff 1, M3 coeff -1, M6 coeff 1
    
    # W[0,0] = (k=0,i=0): M1:1, M4:-1, M7:1
    # W[1,0] = (k=0,i=1): M3:1, M5:1
    # W[2,0] = (k=1,i=0): M2:1, M4:1
    # W[3,0] = (k=1,i=1): M1:1, M3:-1, M6:1
    
    # Let me define the Strassen coefficients:
    # U_strassen[r, (i,j)]:
    U_s = [
        [1, 0, 0, 1],   # M1: A00, A11
        [0, 0, 1, 1],   # M2: A10, A11
        [1, 0, 0, 0],   # M3: A00
        [0, 0, 0, 1],   # M4: A11
        [1, 1, 0, 0],   # M5: A00, A01
        [-1, 0, 1, 0],  # M6: -A00, A10
        [0, 1, 0, -1],  # M7: A01, -A11
    ]
    
    # V_strassen[r, (j,k)]:
    V_s = [
        [1, 0, 0, 1],   # M1: B00, B11
        [1, 0, 0, 0],   # M2: B00
        [0, 1, 0, -1],  # M3: B01, -B11
        [-1, 0, 1, 0],  # M4: -B00, B10
        [0, 0, 0, 1],   # M5: B11
        [1, 1, 0, 0],   # M6: B00, B01
        [0, 0, 1, 1],   # M7: B10, B11
    ]
    
    # W_strassen[r, (k,i)]:
    W_s = [
        [1, 0, 0, 1],   # M1: C00, C11
        [0, 0, 1, 0],   # M2: C10
        [0, 1, 0, -1],  # M3: C01, -C11
        [-1, 0, 1, 0],  # M4: -C00, C10
        [0, 1, 0, 0],   # M5: C01
        [0, 0, 0, 1],   # M6: C11
        [1, 0, 0, 0],   # M7: C00
    ]
    
    # Now, for 2x2x6, we have 3 blocks of 2 columns.
    # Block 0: columns 0,1 of B
    # Block 1: columns 2,3 of B
    # Block 2: columns 4,5 of B
    
    for block in range(3):
        for s in range(7):
            r = block * 7 + s
            # U: same for all blocks
            for ij in range(4):
                U[ij][r] = U_s[s][ij]
            # V: offset by block*2 in the k-dimension
            for j in range(2):
                for k in range(2):
                    v_idx = j * 6 + (block * 2 + k)
                    U_idx = j * 2 + k  # for the 2x2 block
                    V[v_idx][r] = V_s[s][U_idx]
            # W: offset by block*2 in the k-dimension
            for k in range(2):
                for i in range(2):
                    w_idx = (block * 2 + k) * 2 + i
                    W_idx = k * 2 + i
                    W[w_idx][r] = W_s[s][W_idx]
    
    return {"U": U, "V": V, "W": W}


def construct():
    n, m, p = 2, 2, 6
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Try Winograd/Strassen-based decomposition first
    result = _winograd_fft(n, m, p)
    if _check_decomposition(result["U"], result["V"], result["W"], n, m, p):
        rank = len(result["U"][0])
        return result
    
    # Fall back to schoolbook
    result = _schoolbook(n, m, p)
    return result

# EVOLVE-BLOCK-END
