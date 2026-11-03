import time

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a decomposition for 4x5x6 matrix multiplication.
    Strategy:
    1. Split j dimension into 4+1.
    2. For j in 0..3: split i into 2+2, k into 2+2+2.
       Each 2x2x2 block uses Strassen (rank 7).
       Total for j in 0..3: 4 * 2 * 3 * 7 = 168? No, that's wrong.
       
       Let me reconsider.
       i: 4 = 2 + 2
       j: 4 = 2 + 2 (if we split j=0..3 into two blocks of size 2)
       k: 6 = 2 + 2 + 2
       
       So for the j=0..3 part:
       Blocks: (2,2,2) for each combination.
       Number of blocks: (4/2) * (4/2) * (6/2) = 2 * 2 * 3 = 12 blocks.
       Each block has rank 7 (Strassen).
       Total rank for j=0..3: 12 * 7 = 84.
       
       For j=4: 
       i: 4, j: 1, k: 6.
       This is rank 4*1*6 = 24 for schoolbook.
       
       Total rank: 84 + 24 = 108.
       
       But wait, I can do better on j=4.
       j=4: A[i,4] * B[4,k] -> C[i,k].
       This is an outer product of vector A[:,4] and vector B[4,:].
       C = A[:,4] * B[4,:]
       C[i,k] = A[i,4] * B[4,k]
       
       This can be done with rank 1? No.
       C = u * v^T where u = A[:,4] (4x1) and v = B[4,:] (1x6).
       C[i,k] = u[i] * v[k]
       
       The tensor T_{i,k} = A_{i,4} * B_{4,k}
       This is a rank-1 tensor in the (i,k) indices for fixed j=4.
       
       Actually, let's look at the structure:
       C[i,k] = sum_j A[i,j] B[j,k]
       
       For j=4: C[i,k] += A[i,4] * B[4,k]
       
       This term is A[i,4] * B[4,k].
       U[i*m+4, r] = A coefficient
       V[4*p+k, r] = B coefficient
       W[k*n+i, r] = C coefficient
       
       We need: sum_r U[i*5+4, r] * V[4*6+k, r] * W[k*4+i, r] = A[i,4] * B[4,k]
       
       This is satisfied if we set:
       For each i in 0..3:
         For each k in 0..5:
           Use a separate rank-1 term? No, that's 24 terms.
           
       But notice: A[i,4] only depends on i, and B[4,k] only depends on k.
       So C[i,k] = A[i,4] * B[4,k] = (A[i,4]) * (B[4,k])
       
       This is a rank-1 matrix in the (i,k) plane!
       So we can use rank 1 for the entire j=4 strip?
       
       Let's check:
       Term r: 
         U[i*5+4, r] = A[i,4]
         V[4*6+k, r] = B[4,k]
         W[k*4+i, r] = 1
         
       Then sum_r U[i*5+4, r] V[4*6+k, r] W[k*4+i, r] = A[i,4] * B[4,k] * 1
       
       But we only have ONE rank index r for this. So we need:
       U[i*5+4, r] must equal A[i,4] for all i
       V[4*6+k, r] must equal B[4,k] for all k
       
       But U and V are just coefficient matrices, they don't depend on the input A and B!
       They are fixed numbers.
       
       Ah, I see my confusion. The decomposition is:
       C[i,k] = sum_j sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] * A[i,j] * B[j,k]
       
       No, that's not right either. The tensor decomposition is:
       T_{i,j,k} = sum_r U_{i,j}^r * V_{j,k}^r * W_{k,i}^r
       
       Where T_{i,j,k} is the matrix multiplication tensor.
       For matrix multiplication: T_{i,j,k} = delta_{i,j} * delta_{j,k} ? No.
       
       Let me re-read the task.
       "factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)"
       "U, V, W ... exactly decompose the matrix multiplication tensor"
       
       The matrix multiplication tensor T has indices (i, j, k) where:
       (A*B)[i,k] = sum_j A[i,j] * B[j,k]
       
       So T_{i,j,k} = 1 if we think of it as the "1" in A[i,j]*B[j,k]*C[i,k]
       Actually, the tensor is: T[i,j,k] = 1 for all valid i,j,k? No.
       
       The standard form: 
       C[i,k] = sum_j A[i,j] B[j,k]
       
       This can be written as:
       C[i,k] = sum_j sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] * ... 
       
       No, the decomposition is:
       The tensor T_{i,j,k} (where the multiplication is implicit) is decomposed as:
       T_{i,j,k} = sum_r U_{i,j}^{(r)} * V_{j,k}^{(r)} * W_{k,i}^{(r)}
       
       Where U has indices (i,j), V has indices (j,k), W has indices (k,i).
       
       For matrix multiplication, T_{i,j,k} = 1 (it's the identity in the sense that each term A[i,j]*B[j,k] contributes to C[i,k]).
       
       So we need: 1 = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for all valid i,j,k.
       
       Wait, that can't be right because the left side is 1 and the right side sums over r.
       
       Let me look at the seed code. It sets U[i*m+j][col] = 1, V[j*p+k][col] = 1, W[k*n+i][col] = 1 for schoolbook.
       So for each (i,j,k) triple, there's exactly one r (col) where all three are 1.
       Then sum_r U*V*W = 1 for that specific (i,j,k), and 0 for others.
       
       So the decomposition is: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
       And for matrix multiplication, T[i,j,k] = 1 for all valid i,j,k.
       
       So the task is to find rank-minimal U, V, W such that:
       For all i in 0..n-1, j in 0..m-1, k in 0..p-1:
         sum_{r=0}^{rank-1} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
       
       This is the standard matrix multiplication tensor rank problem.
       
       So my Strassen approach should work:
       For 2x2x2 blocks, use Strassen rank 7.
       
       Let me implement it correctly.
       
       Dimensions: n=4, m=5, p=6.
       
       Split:
       i: 0..3 -> two blocks of size 2: i0 in {0, 2}
       j: 0..4 -> 
          For j in 0..3: two blocks of size 2: j0 in {0, 2}
          For j=4: one block of size 1
       k: 0..5 -> three blocks of size 2: k0 in {0, 2, 4}
       
       Part 1: j in 0..3, i in 0..3, k in 0..5
       Blocks: (2,2,2) at offsets (i0, j0, k0) where i0 in {0,2}, j0 in {0,2}, k0 in {0,2,4}
       Number of blocks: 2 * 2 * 3 = 12
       Each block: rank 7
       Total: 84
       
       Part 2: j=4, i in 0..3, k in 0..5
       This is 4x1x6.
       We need T[i,4,k] = 1 for all i,k.
       sum_r U[i*5+4, r] * V[4*6+k, r] * W[k*4+i, r] = 1
       
       This is a rank-1 problem? 
       We need: U[i*5+4, r] * V[4*6+k, r] * W[k*4+i, r] = 1 for all i,k for some r?
       No, we sum over r.
       
       Actually, this is equivalent to: the 4x6 matrix of 1s (for fixed j=4) needs to be decomposed.
       But it's not a matrix multiplication, it's just setting T[i,4,k] = 1.
       
       We can use rank 1: 
       Set U[i*5+4, r0] = 1 for all i
       Set V[4*6+k, r0] = 1 for all k  
       Set W[k*4+i, r0] = 1 for all i,k
       
       Then sum_r U*V*W = 1*1*1 = 1. ✓
       
       So Part 2: rank 1.
       
       Total rank: 84 + 1 = 85.
       
       Let me implement this.
       """
    n, m, p = DIMENSIONS
    
    # Rank calculation
    # Part 1: 12 blocks of 2x2x2, each rank 7 -> 84
    # Part 2: 1 block of 4x1x6, rank 1 -> 1
    total_rank = 84 + 1
    
    U = [[0] * total_rank for _ in range(n * m)]  # (20, rank)
    V = [[0] * total_rank for _ in range(m * p)]  # (30, rank)
    W = [[0] * total_rank for _ in range(p * n)]  # (24, rank)
    
    col = 0
    
    # Strassen 2x2x2 factors
    # U2: (4, 7), V2: (4, 7), W2: (4, 7)
    # Indices: U2[i*2+j, c], V2[j*2+k, c], W2[k*2+i, c]
    
    U2 = [[0]*7 for _ in range(4)]
    V2 = [[0]*7 for _ in range(4)]
    W2 = [[0]*7 for _ in range(4)]
    
    # Strassen 7 terms
    # M1 = (A11+A22)(B11+B22)
    U2[0][0] = 1; U2[3][0] = 1
    V2[0][0] = 1; V2[3][0] = 1
    W2[0][0] = 1; W2[3][0] = 1
    
    # M2 = (A21+A22)B11
    U2[2][1] = 1; U2[3][1] = 1
    V2[0][1] = 1
    W2[1][1] = 1
    
    # M3 = A11(B12-B22)
    U2[0][2] = 1
    V2[1][2] = 1; V2[3][2] = -1
    W2[2][2] = 1
    
    # M4 = A22(B21-B11)
    U2[3][3] = 1
    V2[2][3] = 1; V2[0][3] = -1
    W2[0][3] = 1; W2[1][3] = 1
    
    # M5 = (A11+A12)B22
    U2[0][4] = 1; U2[1][4] = 1
    V2[3][4] = 1
    W2[0][4] = -1; W2[2][4] = 1
    
    # M6 = (A21-A11)(B11+B12)
    U2[2][5] = 1; U2[0][5] = -1
    V2[0][5] = 1; V2[1][5] = 1
    W2[3][5] = 1
    
    # M7 = (A12-A22)(B21+B22)
    U2[1][6] = 1; U2[3][6] = -1
    V2[2][6] = 1; V2[3][6] = 1
    W2[0][6] = 1
    
    def add_strassen_block(i0, j0, k0):
        """Add Strassen decomposition for 2x2x2 block at offset (i0, j0, k0)."""
        nonlocal col
        for c in range(7):
            # U part: for all (i,j) in the 2x2 block
            for di in range(2):
                for dj in range(2):
                    i = i0 + di
                    j = j0 + dj
                    u_idx = i * m + j
                    coeff = U2[di * 2 + dj][c]
                    if coeff != 0:
                        U[u_idx][col] += coeff
                        V[j * p + (k0 + 0)][col]  # placeholder, will do V and W below
                        # Actually, let me do all three together
                        pass
            
            # Let me do it more carefully
            # U[i*m+j, col] += U2[di*2+dj][c] for di,dj in 0..1
            # V[j*p+k, col] += V2[dj*2+dk][c] for dj,dk in 0..1
            # W[k*n+i, col] += W2[dk*2+di][c] for dk,di in 0..1
            
            for di in range(2):
                for dj in range(2):
                    i = i0 + di
                    j = j0 + dj
                    u_idx = i * m + j
                    coeff_u = U2[di * 2 + dj][c]
                    if coeff_u != 0:
                        U[u_idx][col] += coeff_u
            
            for dj in range(2):
                for dk in range(2):
                    j = j0 + dj
                    k = k0 + dk
                    v_idx = j * p + k
                    coeff_v = V2[dj * 2 + dk][c]
                    if coeff_v != 0:
                        V[v_idx][col] += coeff_v
            
            for dk in range(2):
                for di in range(2):
                    k = k0 + dk
                    i = i0 + di
                    w_idx = k * n + i
                    coeff_w = W2[dk * 2 + di][c]
                    if coeff_w != 0:
                        W[w_idx][col] += coeff_w
            
            col += 1
    
    # Part 1: j in 0..3
    for i0 in [0, 2]:
        for j0 in [0, 2]:
            for k0 in [0, 2, 4]:
                add_strassen_block(i0, j0, k0)
    
    # Part 2: j=4, rank 1
    # U[i*5+4, col] = 1 for all i
    # V[4*6+k, col] = 1 for all k
    # W[k*4+i, col] = 1 for all i,k
    for i in range(n):
        U[i * m + 4][col] += 1
    for k in range(p):
        V[4 * p + k][col] += 1
    for i in range(n):
        for k in range(p):
            W[k * n + i][col] += 1
    col += 1
    
    assert col == total_rank, f"Rank mismatch: {col} vs {total_rank}"
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
