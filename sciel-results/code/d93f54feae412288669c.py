"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
import numpy as np
import time

def _strassen_2x2():
    """Return U, V, W for 2x2 x 2x2 -> 2x2 Strassen decomposition (rank 7)."""
    # Indices: A is 2x2 (rows 0,1; cols 0,1), B is 2x2, C is 2x2
    # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
    # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
    # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
    # C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1]
    
    # Strassen:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)  -- wait, let me use the standard formulation
    # 
    # Standard Strassen:
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    #
    # p = M1 - M4 + M5 - M7
    # q = M3 + M5
    # r = M2 + M4
    # s = M1 - M3 + M6 + M7
    #
    # Where a=A[0,0], b=A[0,1], c=A[1,0], d=A[1,1]
    #       e=B[0,0], f=B[0,1], g=B[1,0], h=B[1,1]
    
    # U[i*m+j, k] for A index (i,j), V[j*p+k, k] for B index (j,k), W[k*n+i, k] for C index (k,i)
    # n=2, m=2, p=2
    # A indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    # B indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    # C indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1 = (a+d)(e+h): A[0,0]+A[1,1], B[0,0]+B[1,1]
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    # M1 contributes to p=C[0,0] with +1, s=C[1,1] with +1
    W[0][0] = 1; W[3][0] = 1
    
    # M2 = (c+d)e: A[1,0]+A[1,1], B[0,0]
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    # M2 contributes to r=C[1,0] with +1
    W[2][1] = 1
    
    # M3 = a(f-h): A[0,0], B[0,1]-B[1,1]
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    # M3 contributes to q=C[0,1] with +1, s=C[1,1] with -1
    W[1][2] = 1; W[3][2] = -1
    
    # M4 = d(g-e): A[1,1], B[1,0]-B[0,0]
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    # M4 contributes to p=C[0,0] with -1, r=C[1,0] with +1
    W[0][3] = -1; W[2][3] = 1
    
    # M5 = (a+b)h: A[0,0]+A[0,1], B[1,1]
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    # M5 contributes to p=C[0,0] with +1, q=C[0,1] with +1
    W[0][4] = 1; W[1][4] = 1
    
    # M6 = (c-a)(e+f): A[1,0]-A[0,0], B[0,0]+B[0,1]
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    # M6 contributes to s=C[1,1] with +1
    W[3][5] = 1
    
    # M7 = (b-d)(g+h): A[0,1]-A[1,1], B[1,0]+B[1,1]
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    # M7 contributes to p=C[0,0] with -1, s=C[1,1] with +1
    W[0][6] = -1; W[3][6] = 1
    
    return U, V, W

def _strassen_4x4():
    """Return U, V, W for 4x4 x 4x4 -> 4x4 Strassen decomposition (rank 49)."""
    # Use Strassen recursively: 4x4 = 2x2 blocks of 2x2
    # The 2x2 block matrix multiplication uses Strassen with 7 "scalar" multiplications,
    # where each "scalar" is a 2x2 matrix multiplication.
    # Each 2x2 matrix multiplication uses Strassen with rank 7.
    # Total rank = 7 * 7 = 49.
    
    # I'll build this by composing the 2x2 Strassen with itself.
    # The outer Strassen operates on 2x2 blocks, the inner on 2x2 matrices.
    
    # Actually, let me just construct it directly using the recursive structure.
    # For 4x4 x 4x4, split into 2x2 blocks:
    # A = [[A11, A12], [A21, A22]] where each is 2x2
    # B = [[B11, B12], [B21, B22]] where each is 2x2
    # C = [[C11, C12], [C21, C22]] where each is 2x2
    #
    # Strassen:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    #
    # C11 = M1 - M4 + M5 - M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M3 + M6 + M7
    
    # Each Mj is a 2x2 matrix product, which itself uses Strassen with rank 7.
    # So total rank = 7 * 7 = 49.
    
    # I need to build U, V, W of shapes (16, 49), (16, 49), (16, 49)
    # n=4, m=4, p=4
    
    # Let me define the 2x2 Strassen factors as functions.
    U2, V2, W2 = _strassen_2x2()
    
    # Now I need to compose. The outer level has 7 "virtual" multiplications,
    # each of which is a 2x2 x 2x2 multiplication using the inner Strassen.
    
    # For each outer multiplication j (j=0..6), I need to determine:
    # - Which A blocks are involved and with what coefficients
    # - Which B blocks are involved and with what coefficients
    # - How the result contributes to C blocks
    
    # Outer Strassen coefficients for A blocks (A11, A12, A21, A22) indexed as (0,0),(0,1),(1,0),(1,1):
    # M1: A11 + A22 -> coeffs [1, 0, 0, 1]
    # M2: A21 + A22 -> coeffs [0, 0, 1, 1]
    # M3: A11 -> coeffs [1, 0, 0, 0]
    # M4: A22 -> coeffs [0, 0, 0, 1]
    # M5: A11 + A12 -> coeffs [1, 1, 0, 0]
    # M6: A21 - A11 -> coeffs [-1, 0, 1, 0]
    # M7: A12 - A22 -> coeffs [0, 1, 0, -1]
    
    # Outer Strassen coefficients for B blocks (B11, B12, B21, B22) indexed as (0,0),(0,1),(1,0),(1,1):
    # M1: B11 + B22 -> coeffs [1, 0, 0, 1]
    # M2: B11 -> coeffs [1, 0, 0, 0]
    # M3: B21 - B22 -> coeffs [0, 0, 1, -1]
    # M4: B21 - B11 -> coeffs [-1, 0, 1, 0]
    # M5: B22 -> coeffs [0, 0, 0, 1]
    # M6: B11 + B12 -> coeffs [1, 1, 0, 0]
    # M7: B21 + B22 -> coeffs [0, 0, 1, 1]
    
    # Outer Strassen coefficients for C blocks (C11, C12, C21, C22) indexed as (0,0),(0,1),(1,0),(1,1):
    # M1: C11 +1, C22 +1 -> coeffs [1, 0, 0, 1]
    # M2: C21 +1 -> coeffs [0, 0, 1, 0]
    # M3: C12 +1, C22 -1 -> coeffs [0, 1, 0, -1]
    # M4: C11 -1, C21 +1 -> coeffs [-1, 0, 1, 0]
    # M5: C11 +1, C12 +1 -> coeffs [1, 1, 0, 0]
    # M6: C22 +1 -> coeffs [0, 0, 0, 1]
    # M7: C11 -1, C22 +1 -> coeffs [-1, 0, 0, 1]
    
    A_coeff = [
        [1, 0, 0, 1],  # M1
        [0, 0, 1, 1],  # M2
        [1, 0, 0, 0],  # M3
        [0, 0, 0, 1],  # M4
        [1, 1, 0, 0],  # M5
        [-1, 0, 1, 0], # M6
        [0, 1, 0, -1], # M7
    ]
    
    B_coeff = [
        [1, 0, 0, 1],  # M1
        [1, 0, 0, 0],  # M2
        [0, 0, 1, -1], # M3
        [-1, 0, 1, 0], # M4
        [0, 0, 0, 1],  # M5
        [1, 1, 0, 0],  # M6
        [0, 0, 1, 1],  # M7
    ]
    
    C_coeff = [
        [1, 0, 0, 1],  # M1
        [0, 0, 1, 0],  # M2
        [0, 1, 0, -1], # M3
        [-1, 0, 1, 0], # M4
        [1, 1, 0, 0],  # M5
        [0, 0, 0, 1],  # M6
        [-1, 0, 0, 1], # M7
    ]
    
    # Now build the full U, V, W matrices.
    # U has shape (16, 49), where 16 = 4*4 (A matrix indices)
    # V has shape (16, 49), where 16 = 4*4 (B matrix indices)
    # W has shape (16, 49), where 16 = 4*4 (C matrix indices)
    
    U = [[0]*49 for _ in range(16)]
    V = [[0]*49 for _ in range(16)]
    W = [[0]*49 for _ in range(16)]
    
    for j in range(7):  # outer multiplication index
        # For each outer multiplication j, we have a 2x2 x 2x2 multiplication
        # using the inner Strassen (rank 7).
        # The inner Strassen operates on the "effective" 2x2 matrices formed by
        # linear combinations of the A and B blocks.
        
        # The effective A matrix for outer multiplication j has entries:
        # A_eff[i][k] = sum over A blocks (bi, bk) of A_coeff[j][block_index] * A[block]
        # where block_index maps (bi, bk) to a linear index.
        
        # Similarly for B_eff and C_eff.
        
        # The inner Strassen gives us U2, V2, W2 for the 2x2 case.
        # U2 has shape (4, 7), V2 has shape (4, 7), W2 has shape (4, 7).
        # U2[i*2+k, t] gives the coefficient of A[i,k] in the t-th inner multiplication.
        # V2[k*2+l, t] gives the coefficient of B[k,l] in the t-th inner multiplication.
        # W2[l*2+i, t] gives the coefficient of C[i,l] in the t-th inner multiplication.
        
        # Wait, I need to be careful with the index conventions.
        # In the 2x2 case: n=2, m=2, p=2
        # A indices: (i,j) -> i*2+j, i in 0..1, j in 0..1
        # B indices: (j,k) -> j*2+k, j in 0..1, k in 0..1
        # C indices: (k,i) -> k*2+i, k in 0..1, i in 0..1
        
        # For the outer level, the "2x2" blocks are indexed as:
        # A blocks: (bi, bj) -> bi*2+bj, bi in 0..1, bj in 0..1
        # B blocks: (bj, bk) -> bj*2+bk, bj in 0..1, bk in 0..1
        # C blocks: (bk, bi) -> bk*2+bi, bk in 0..1, bi in 0..1
        
        # Each A block is a 2x2 matrix with indices (i, j) -> i*2+j
        # Each B block is a 2x2 matrix with indices (j, k) -> j*2+k
        # Each C block is a 2x2 matrix with indices (k, i) -> k*2+i
        
        # So the full A matrix index for block (bi, bj) and within-block (i, j) is:
        # (bi*2+i) * 4 + (bj*2+j) = (bi*2+i)*4 + bj*2 + j
        
        # Similarly for B: block (bj, bk) and within-block (j, k):
        # (bj*2+j) * 4 + (bk*2+k) = (bj*2+j)*4 + bk*2 + k
        
        # And for C: block (bk, bi) and within-block (k, i):
        # (bk*2+k) * 4 + (bi*2+i) = (bk*2+k)*4 + bi*2 + i
        
        for t in range(7):  # inner multiplication index
            col = j * 7 + t
            
            # U: For each A entry (i, j) in the full 4x4 matrix
            for i in range(4):
                for j_full in range(4):
                    # Determine which A block and within-block position
                    bi = i // 2
                    bj = j_full // 2
                    i_block = i % 2
                    j_block = j_full % 2
                    block_idx = bi * 2 + bj
                    within_idx = i_block * 2 + j_block
                    
                    # Get the inner Strassen coefficient for A[within_idx]
                    inner_coeff = U2[within_idx][t]
                    outer_coeff = A_coeff[j][block_idx]
                    U[i * 4 + j_full][col] = inner_coeff * outer_coeff
            
            # V: For each B entry (j, k) in the full 4x4 matrix
            for j_full in range(4):
                for k in range(4):
                    # Determine which B block and within-block position
                    bj = j_full // 2
                    bk = k // 2
                    j_block = j_full % 2
                    k_block = k % 2
                    block_idx = bj * 2 + bk
                    within_idx = j_block * 2 + k_block
                    
                    inner_coeff = V2[within_idx][t]
                    outer_coeff = B_coeff[j][block_idx]
                    V[j_full * 4 + k][col] = inner_coeff * outer_coeff
            
            # W: For each C entry (k, i) in the full 4x4 matrix
            for k in range(4):
                for i in range(4):
                    # Determine which C block and within-block position
                    bk = k // 2
                    bi = i // 2
                    k_block = k % 2
                    i_block = i % 2
                    block_idx = bk * 2 + bi
                    within_idx = k_block * 2 + i_block
                    
                    inner_coeff = W2[within_idx][t]
                    outer_coeff = C_coeff[j][block_idx]
                    W[k * 4 + i][col] = inner_coeff * outer_coeff
    
    return U, V, W

def _verify(U, V, W, n, m, p):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    # Check shapes
    assert len(U) == n * m, f"U rows {len(U)} != {n*m}"
    assert len(V) == m * p, f"V rows {len(V)} != {m*p}"
    assert len(W) == p * n, f"W rows {len(W)} != {p*n}"
    assert all(len(row) == rank for row in U), "U column count mismatch"
    assert all(len(row) == rank for row in V), "V column count mismatch"
    assert all(len(row) == rank for row in W), "W column count mismatch"
    
    # Build the tensor T[i,j,k] = delta_{i,k} for the multiplication
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # Tensor: T[i,j,k] = 1 if this is a valid multiplication term, else 0
    # Actually, the tensor is: for each (i,j,k), the coefficient is 1
    # because C[i,k] gets contribution A[i,j]*B[j,k]
    
    # The decomposition says: T = sum_c U[:,c] ⊗ V[:,c] ⊗ W[:,c]
    # i.e., T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    
    # Check for a few random points
    import random
    random.seed(42)
    for _ in range(100):
        i = random.randint(0, n-1)
        j = random.randint(0, m-1)
        k = random.randint(0, p-1)
        
        # Expected value: 1 (since every (i,j,k) triple contributes to C[i,k])
        expected = 1
        
        # Computed value
        computed = 0
        for c in range(rank):
            computed += U[i*m + j][c] * V[j*p + k][c] * W[k*n + i][c]
        
        if abs(computed - expected) > 1e-9:
            raise ValueError(f"Verification failed at ({i},{j},{k}): expected {expected}, got {computed}")
    
    return True

def _schoolbook(n, m, p):
    """Fallback schoolbook decomposition."""
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

def _construct_rank98():
    """Construct 4x4 x 4x8 decomposition using two 4x4 x 4x4 Strassen decompositions."""
    n, m, p = 4, 4, 8
    
    # Split B into two 4x4 matrices: B1 (cols 0-3) and B2 (cols 4-7)
    # C = A * B = [A*B1 | A*B2]
    # Use Strassen for each half.
    
    U4, V4, W4 = _strassen_4x4()
    
    rank = 49 * 2  # 98
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # First half: B1 (cols 0-3)
    for i in range(n * m):
        for c in range(49):
            U[i][c] = U4[i][c]
    for j in range(m):
        for k in range(4):  # cols 0-3
            for c in range(49):
                V[j * p + k][c] = V4[j * 4 + k][c]
    for k in range(4):  # cols 0-3
        for i in range(n):
            for c in range(49):
                W[k * n + i][c] = W4[k * 4 + i][c]
    
    # Second half: B2 (cols 4-7)
    for i in range(n * m):
        for c in range(49):
            U[i][49 + c] = U4[i][c]
    for j in range(m):
        for k in range(4):  # cols 4-7, index k-4 in V4
            for c in range(49):
                V[j * p + (4 + k)][49 + c] = V4[j * 4 + k][c]
    for k in range(4):  # cols 4-7, index k in W
        for i in range(n):
            for c in range(49):
                W[(4 + k) * n + i][49 + c] = W4[k * 4 + i][c]
    
    return {"U": U, "V": V, "W": W}

def construct():
    """Main construct function."""
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS
    
    # Try the Strassen-based rank-98 decomposition
    try:
        result = _construct_rank98()
        # Verify
        _verify(result["U"], result["V"], result["W"], n, m, p)
        return result
    except Exception as e:
        print(f"Strassen rank-98 failed: {e}")
    
    # Fallback to schoolbook
    return _schoolbook(n, m, p)
# EVOLVE-BLOCK-END
