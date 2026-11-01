"""Matrix multiplication 2,2,14 — rank-minimal exact tensor decomposition."""
import time

DIMENSIONS = [2, 2, 14]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 130  # seconds; leave margin before 150s kill

    n, m, p = 2, 2, 14

    # Strassen decomposition for 2x2 x 2x2 (rank 7)
    # A = [[a00, a01],[a10, a11]], B = [[b00, b01],[b10, b11]]
    # 7 products:
    # M1 = (a00+a11)*(b00+b11)
    # M2 = (a10+a11)*b00
    # M3 = a00*(b01-b11)
    # M4 = a11*(b10-b00)
    # M5 = (a00+a01)*b11
    # M6 = (a10-a00)*(b00+b01)
    # M7 = (a01-a11)*(b10+b11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6

    # Build the 7 rank-1 terms for one 2x2 block
    # U indices: i*m + j = i*2 + j for i in {0,1}, j in {0,1} -> 0,1,2,3
    # V indices: j*p_block + k where p_block=2: j*2+k -> 0,1,2,3
    # W indices: k*n + i = k*2 + i for k in {0,1}, i in {0,1} -> 0,1,2,3

    # For each of the 7 Strassen products, define:
    # (a_coeffs, b_coeffs, c_coeffs) where these are the linear forms
    # a_coeffs: dict mapping (i,j) -> coeff
    # b_coeffs: dict mapping (j,k) -> coeff
    # c_coeffs: dict mapping (k,i) -> coeff

    strassen_products = [
        # M1 = (a00+a11)*(b00+b11) -> C: M1 contributes to C00(+1) and C11(+1)
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1},
         {(0,0): 1, (1,1): 1}),  # C00=M1, C11=M1
        # M2 = (a10+a11)*b00 -> C: M2 contributes to C10(+1) and C11(-1)
        ({(1,0): 1, (1,1): 1}, {(0,0): 1},
         {(0,1): 1, (1,1): -1}),
        # M3 = a00*(b01-b11) -> C: M3 contributes to C00(+1) and C01(+1)
        ({(0,0): 1}, {(0,1): 1, (1,1): -1},
         {(0,0): 1, (1,0): 1}),
        # M4 = a11*(b10-b00) -> C: M4 contributes to C00(+1) and C10(+1)
        ({(1,1): 1}, {(1,0): 1, (0,0): -1},
         {(0,0): 1, (0,1): 1}),
        # M5 = (a00+a01)*b11 -> C: M5 contributes to C00(-1) and C01(+1)
        ({(0,0): 1, (0,1): 1}, {(1,1): 1},
         {(0,0): -1, (1,0): 1}),
        # M6 = (a10-a00)*(b00+b01) -> C: M6 contributes to C11(+1)
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1},
         {(1,1): 1}),
        # M7 = (a01-a11)*(b10+b11) -> C: M7 contributes to C00(+1) and C11(+1)
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1},
         {(0,0): 1, (1,1): 1}),
    ]

    # Now apply this to 7 column-pairs (since 14 = 7*2)
    rank = 7 * 7  # 49

    U = [[0] * rank for _ in range(n * m)]  # 4 x 49
    V = [[0] * rank for _ in range(m * p)]  # 28 x 49
    W = [[0] * rank for _ in range(p * n)]  # 28 x 49

    for block in range(7):
        for s in range(7):
            col = block * 7 + s
            a_coeffs, b_coeffs, c_coeffs = strassen_products[s]

            # U: A indices (i,j) -> U[i*2+j, col]
            for (i, j), coeff in a_coeffs.items():
                U[i * m + j][col] = coeff

            # V: B indices (j,k) within this block -> global column = block*2 + k
            # V row index = j * p + (block*2 + k)
            for (j, k_local), coeff in b_coeffs.items():
                k_global = block * 2 + k_local
                V[j * p + k_global][col] = coeff

            # W: C indices (k,i) -> global column = block*2 + k
            # W row index = (block*2 + k) * n + i
            for (k_local, i), coeff in c_coeffs.items():
                k_global = block * 2 + k_local
                W[k_global * n + i][col] = coeff

    # Quick validation check
    if time.time() - start_time < DEADLINE:
        # Verify: sum_c U[i,j,c] * V[j,k,c] == (A*B)[i,k]
        # We'll do a spot check with a few random matrices
        import random
        random.seed(42)
        ok = True
        for _ in range(5):
            A = [[random.randint(-5, 5) for _ in range(2)] for _ in range(2)]
            B = [[random.randint(-5, 5) for _ in range(14)] for _ in range(2)]
            # Expected C[i][k] = sum_j A[i][j]*B[j][k]
            for i in range(2):
                for k in range(14):
                    expected = 0
                    for j in range(2):
                        expected += A[i][j] * B[j][k]
                    # Decomposed: sum_c U[i*2+j? no...]
                    # Actually the decomposition is: C[i,k] = sum_c U[i,j,c]*V[j,k,c] summed over j?
                    # No: rank-1 decomposition of multiplication tensor:
                    # (A*B)[i,k] = sum_c sum_j sum_l U[i*m+j, c] * V[j*p+l, c] * W[l*n+i, c] * A[i,j] * B[j,l]
                    # Wait, the standard form is:
                    # C[i,k] = sum_c U[i,j,c] * A[i,j] * V[j,k,c] * B[j,k] * W[k,i,c]
                    # Hmm, let me reconsider. The tensor is M[i,j,k] = delta... no.
                    # The matrix multiplication tensor T_{i,j,k,l,m,n}... 
                    # For M_{n,m,p}: T_{i,j,k} where the bilinear form is sum_{j,k} A[i,j]*B[j,k]*C_{k,i}
                    # Rank-r decomposition: C[k*n+i] = sum_c W[k*n+i, c] * (sum_j U[i*m+j,c]*A[i,j]) * (sum_k V[j*p+k,c]*B[j,k])
                    # Wait that's not right either. Let me re-derive.
                    #
                    # Standard: we want C = A * B where A is n x m, B is m x p, C is n x p.
                    # C[i,k] = sum_j A[i,j] * B[j,k]
                    # Tensor T_{i,j,k} such that C[i,k] = sum_j T_{i,j,k} A[i,j] B[j,k] with T_{i,j,k} = delta_{j,k?} no.
                    # Actually T_{i,j,k} is the structure: for each output (i,k), we sum over j: A[i,j]*B[j,k].
                    # So the "tensor" entries are: for each (i,k,j): coefficient is 1 if... it's just the product.
                    #
                    # Rank-r decomposition: 
                    # C[i,k] = sum_{c=1}^{r} (sum_j U_{i,j,c} * A[i,j]) * (sum_j V_{j,k,c} * B[j,k]) * W_{i,k,c}
                    # Wait, but the interface says U is (n*m, rank), V is (m*p, rank), W is (p*n, rank).
                    # And the trace convention is W[k*n+i, r].
                    #
                    # So: C[k*n+i] ... hmm the output index is (k,n+i)? No, p*n = 14*2 = 28 entries.
                    # C is 2x14, so C[i,k] for i in {0,1}, k in {0,...,13}.
                    # The output vector is indexed by k*n+i (trace convention W[k*n+i, r]).
                    # So output index (k,i) means row k of the "transposed" output? 
                    # Actually W[k*n+i] corresponds to C[i,k] (since i is the row of A->C, k is column).
                    #
                    # Decomposition: C[i,k] = sum_c W[k*n+i, c] * (sum_j U[i*m+j, c] * A[i,j]) * (sum_j V[j*p+k, c] * B[j,k])
                    #
                    # Let me verify with schoolbook: for each (i,j,k), U[i*m+j]=1, V[j*p+k]=1, W[k*n+i]=1.
                    # Then sum_c W[k*n+i,c]*(sum_j U[i*m+j,c]*A[i,j])*(sum_j V[j*p+k,c]*B[j,k])
                    # = W[k*n+i,c] * A[i,j] * B[j,k] for the specific (i,j,k) term.
                    # Summing over all (i,j,k) gives sum_j A[i,j]*B[j,k] = C[i,k]. Correct.
                    #
                    # So for my Strassen decomposition, I need:
                    # C[i,k] = sum_c W[k*n+i,c] * (sum_j U[i*m+j,c] * A[i,j]) * (sum_j V[j*p+k,c] * B[j,k])
                    
                    actual = 0
                    for c in range(rank):
                        sum_U = 0
                        for j in range(m):
                            sum_U += U[i * m + j][c] * A[i][j]
                        sum_V = 0
                        for j in range(m):
                            sum_V += V[j * p + k][c] * B[j][k]
                        actual += W[k * n + i][c] * sum_U * sum_V
                    if abs(actual - expected) > 1e-9:
                        ok = False
                        break
                if not ok:
                    break
        if not ok:
            # Fall back to schoolbook
            rank = n * m * p
            U = [[0] * rank for _ in range(n * m)]
            V = [[0] * rank for _ in range(m * p)]
            W = [[0] * rank for _ in range(p * n)]
            column = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[i * m + j][column] = 1
                        V[j * p + k][column] = 1
                        W[k * n + i][column] = 1
                        column += 1
            return {"U": U, "V": V, "W": W}

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
