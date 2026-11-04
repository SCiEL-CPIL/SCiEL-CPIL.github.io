"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
import numpy as np
import time

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition exactly matches the matrix multiplication tensor.
    U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank).
    The tensor entry T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] should equal delta[i,k] (1 if i==k else 0).
    """
    rank = U.shape[1]
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
    U_r = U.reshape(n, m, rank)
    V_r = V.reshape(m, p, rank)
    W_r = W.reshape(p, n, rank)
    # Compute tensor T[i,j,k] = sum_c U_r[i,j,c] * V_r[j,k,c] * W_r[k,i,c]
    # For each j, compute the (n, p) matrix M_j where M_j[i,k] = sum_c U_r[i,j,c] * W_r[k,i,c]
    # Wait, the contraction is over c, and we have U[i,j,c], V[j,k,c], W[k,i,c].
    # So for fixed j, the tensor slice is sum_c U[:, j, c] * W[:, j, c]^T * V[j, k, c]? No.
    # Let's do it step by step.
    # T[i,j,k] = sum_c U_r[i,j,c] * V_r[j,k,c] * W_r[k,i,c]
    # For a fixed j, let's compute M[i,k] = sum_c U_r[i,j,c] * W_r[k,i,c]
    # Then T[i,j,k] = sum_c (U_r[i,j,c] * W_r[k,i,c]) * V_r[j,k,c]
    # This is not a simple matrix product because W_r[k,i,c] depends on i and k.
    # Let's just loop over i, j, k.
    # Since n, m, p are small (4, 5, 5), we can afford to loop.
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # T[i,j,k] = sum_c U_r[i,j,c] * V_r[j,k,c] * W_r[k,i,c]
                # Vectorize over c
                val = np.sum(U_r[i, j, :] * V_r[j, k, :] * W_r[k, i, :])
                expected = 1.0 if i == k else 0.0
                err = abs(val - expected)
                if err > max_err:
                    max_err = err
    return max_err < 1e-6

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook rank
    rank_school = n * m * p
    
    # We will try to find a lower rank decomposition.
    # Since we have time constraints, we will first return schoolbook if we can't find better quickly.
    # But let's try a simple local search or random restart to see if we can find a lower rank.
    # Actually, for 4x5x5, the known rank is likely close to schoolbook or slightly lower.
    # Let's just return schoolbook to be safe, unless we have a specific algorithm to find lower rank.
    # The directive says "Make a STRUCTURAL change". Random search for tensor rank is a structural change.
    # But it's very unlikely to find a lower rank for 4x5x5 in 150s with random search.
    # Let's check if there's a known lower rank.
    # For 4x4x4, Strassen gives 49 (from 64).
    # For 4x5x5, we can embed the 4x4x4 part and add extra terms for the 5th dimension.
    # Let's try to construct a decomposition based on the 4x4x4 Strassen decomposition.
    
    # 1. Extract the 4x4x4 subproblem.
    # We want to compute C = A * B where A is 4x5 and B is 5x5.
    # Let A = [A_4 | A_5] where A_4 is 4x4 and A_5 is 4x1.
    # Let B = [B_1 B_2 B_3 B_4 B_5] where B_j is 5x1.
    # C = A_4 * B_14 + A_5 * B_5
    # Here B_14 is 4x4 (first 4 columns of B).
    # The term A_4 * B_14 can be computed using Strassen 4x4x4 (rank 49).
    # The term A_5 * B_5 is a 4x1 vector times a 5x1 vector? No, A_5 is 4x1, B_5 is 5x1.
    # Wait, B is 5x5. So B_5 is the 5th column, which is 5x1.
    # A_5 is the 5th column of A, which is 4x1.
    # So A_5 * B_5 is not a matrix product. It's an outer product? No, it's part of the sum.
    # C = A_4 * B_14 + A_5 * B_5 is incorrect.
    # C = sum_j A[:, j] * B[j, :]
    # C = A_4 * B_14 + A_5 * B[4, :]
    # Here A_5 is 4x1, B[4, :] is 1x5. So A_5 * B[4, :] is a 4x5 matrix.
    # This outer product has rank 1.
    # So we need 49 (for A_4 * B_14) + 1 (for A_5 * B[4, :]) = 50 terms?
    # But the decomposition must be linear. The Strassen decomposition is linear.
    # The outer product term is also linear.
    # So we can combine them.
    # Total rank = 49 + 1 = 50.
    # This is much better than 100.
    
    # Let's implement this.
    # First, we need the Strassen 4x4x4 decomposition.
    # Strassen 2x2x2 has rank 7.
    # Strassen 4x4x4 can be done by applying Strassen to 2x2 blocks of 2x2 matrices.
    # This gives rank 7 * 7 = 49.
    # We need the explicit U, V, W matrices for the 4x4x4 Strassen decomposition.
    
    # Let's generate the 4x4x4 Strassen decomposition.
    # The 2x2x2 Strassen decomposition has vectors u, v, w of length 7.
    # u = [[1,0,0,0], [0,1,0,0], [1,1,0,0], [0,0,1,0], [0,0,1,1], [0,0,0,1], [1,0,0,-1]]
    # v = [[1,0,0,0], [1,0,1,0], [0,1,0,0], [0,1,0,1], [0,0,1,0], [0,0,1,1], [1,0,-1,0]]
    # w = [[1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,1,1], [1,0,1,0], [1,0,1,1], [0,1,1,0]]
    # Wait, the standard Strassen vectors are:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    
    # Let's define the 2x2x2 vectors.
    # Indexing: A is 2x2, B is 2x2, C is 2x2.
    # U vectors for A:
    # u1 = [1, 0, 0, 1] (A11+A22) -> vec([1,0;0,1]) = [1,0,0,1]
    # u2 = [0, 0, 1, 1] (A21+A22) -> vec([0,0;1,1]) = [0,0,1,1]
    # u3 = [1, 0, 0, 0] (A11) -> vec([1,0;0,0]) = [1,0,0,0]
    # u4 = [0, 0, 0, 1] (A22) -> vec([0,0;0,1]) = [0,0,0,1]
    # u5 = [1, 1, 0, 0] (A11+A12) -> vec([1,1;0,0]) = [1,1,0,0]
    # u6 = [-1, 0, 1, 0] (A21-A11) -> vec([-1,0;1,0]) = [-1,0,1,0]
    # u7 = [0, 1, 0, -1] (A12-A22) -> vec([0,1;0,-1]) = [0,1,0,-1]
    
    # V vectors for B:
    # v1 = [1, 0, 0, 1] (B11+B22)
    # v2 = [1, 0, 0, 0] (B11)
    # v3 = [0, -1, 1, 0] (B21-B22) -> vec([0,-1;1,0]) = [0,-1,1,0]
    # v4 = [-1, 1, 0, 0] (B12-B11) -> vec([-1,1;0,0]) = [-1,1,0,0]
    # v5 = [0, 0, 0, 1] (B22)
    # v6 = [1, 1, 0, 0] (B11+B12)
    # v7 = [0, 1, 1, 1] (B21+B22)
    
    # W vectors for C:
    # w1 = [1, 0, 0, -1] (M1+M4-M5+M7) -> C11 gets M1, M4, -M5, M7
    #    C11 is at index 0. C12 at 1, C21 at 2, C22 at 3.
    #    w1 = [1, 0, 0, 0] for M1? No, w is the vector for C.
    #    The contribution of M1 to C is:
    #    C11 += M1, C22 += M1. So w1 = [1, 0, 0, 1]
    #    M4: C11 += M4. w4 = [1, 0, 0, 0]
    #    M5: C11 -= M5. w5 = [-1, 0, 0, 0]
    #    M7: C11 += M7. w7 = [1, 0, 0, 0]
    #    Wait, the W vector for each term is the vector that defines how the product M_l contributes to C.
    #    M1 = (A11+A22)(B11+B22) -> C11 += M1, C22 += M1. So w1 = [1, 0, 0, 1]
    #    M2 = (A21+A22)B11 -> C21 += M2. w2 = [0, 0, 1, 0]
    #    M3 = A11(B21-B22) -> C12 += M3, C22 += M3. w3 = [0, 1, 0, 1]
    #    M4 = A22(B12-B11) -> C11 += M4. w4 = [1, 0, 0, 0]
    #    M5 = (A11+A12)B22 -> C11 -= M5, C12 += M5. w5 = [-1, 1, 0, 0]
    #    M6 = (A21-A11)(B11+B12) -> C22 += M6. w6 = [0, 0, 0, 1]
    #    M7 = (A12-A22)(B21+B22) -> C11 += M7. w7 = [1, 0, 0, 0]
    
    # Let's list them:
    # u = [
    #   [1, 0, 0, 1],
    #   [0, 0, 1, 1],
    #   [1, 0, 0, 0],
    #   [0, 0, 0, 1],
    #   [1, 1, 0, 0],
    #   [-1, 0, 1, 0],
    #   [0, 1, 0, -1]
    # ]
    # v = [
    #   [1, 0, 0, 1],
    #   [1, 0, 0, 0],
    #   [0, -1, 1, 0],
    #   [-1, 1, 0, 0],
    #   [0, 0, 0, 1],
    #   [1, 1, 0, 0],
    #   [0, 1, 1, 1]
    # ]
    # w = [
    #   [1, 0, 0, 1],
    #   [0, 0, 1, 0],
    #   [0, 1, 0, 1],
    #   [1, 0, 0, 0],
    #   [-1, 1, 0, 0],
    #   [0, 0, 0, 1],
    #   [1, 0, 0, 0]
    # ]
    
    # Now, for 4x4x4, we treat the 4x4 matrix as a 2x2 matrix of 2x2 blocks.
    # A = [[A11, A12], [A21, A22]] where each Ai is 2x2.
    # B = [[B11, B12], [B21, B22]]
    # C = [[C11, C12], [C21, C22]]
    # The Strassen formulas are the same, but the "entries" are 2x2 matrices.
    # So for each Strassen term l (1..7), we compute M_l = L_A(l) * L_B(l), where L_A(l) and L_B(l) are 2x2 matrices of 2x2 blocks.
    # This is equivalent to a 4x4 matrix product.
    # We can decompose this 4x4 product using the base 2x2x2 Strassen decomposition on the block level.
    # So for each l, we have 7 sub-terms.
    # Total 49 terms.
    
    # The U vector for term (l, m) is formed by combining the block-level u_l with the base-case u_m.
    # Specifically, L_A(l) = sum_a alpha[l, a] * A_block[a], where a=0..3 (A11, A12, A21, A22).
    # And the base-case term m uses u_m to form a 2x2 matrix from the 2x2 blocks of L_A(l).
    # This is getting complicated. Let's just use the explicit indices.
    
    # Let's define the 4x4x4 Strassen decomposition explicitly.
    # The 4x4 matrix A has indices (i, j) where i, j in 0..3.
    # We can map (i, j) to (block_i, inner_i, block_j, inner_j) where block_i = i // 2, inner_i = i % 2, etc.
    # The Strassen decomposition at the block level uses u_l (2x2 vector) to combine the 4 blocks.
    # The base-case Strassen decomposition uses u_m (2x2 vector) to combine the 2x2 entries of the block.
    # So the total U vector for term (l, m) is:
    # U[(block_i * 2 + inner_i) * 4 + (block_j * 2 + inner_j)] = u_l[block_i * 2 + block_j] * u_m[inner_i * 2 + inner_j]
    # Wait, the indices for u_l are (block_i, block_j) and for u_m are (inner_i, inner_j).
    # So U[4*block_i + 2*inner_i + 2*block_j + inner_j] = u_l[2*block_i + block_j] * u_m[2*inner_i + inner_j]
    
    # Let's implement this.
    
    u_base = np.array([
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ])
    v_base = np.array([
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, -1, 1, 0],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 1, 1, 1]
    ])
    w_base = np.array([
        [1, 0, 0, 1],
        [0, 0, 1, 0],
        [0, 1, 0, 1],
        [1, 0, 0, 0],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ])
    
    # 4x4x4 Strassen decomposition
    rank_444 = 49
    U_444 = np.zeros((16, rank_444))
    V_444 = np.zeros((16, rank_444))
    W_444 = np.zeros((16, rank_444))
    
    for l in range(7):
        for m in range(7):
            col = l * 7 + m
            # U_444
            for block_i in range(2):
                for inner_i in range(2):
                    for block_j in range(2):
                        for inner_j in range(2):
                            i = block_i * 2 + inner_i
                            j = block_j * 2 + inner_j
                            idx = i * 4 + j
                            val = u_base[l, block_i * 2 + block_j] * u_base[m, inner_i * 2 + inner_j]
                            U_444[idx, col] = val
            # V_444
            for block_i in range(2):
                for inner_i in range(2):
                    for block_j in range(2):
                        for inner_j in range(2):
                            i = block_i * 2 + inner_i
                            j = block_j * 2 + inner_j
                            idx = i * 4 + j
                            val = v_base[l, block_i * 2 + block_j] * v_base[m, inner_i * 2 + inner_j]
                            V_444[idx, col] = val
            # W_444
            for block_i in range(2):
                for inner_i in range(2):
                    for block_j in range(2):
                        for inner_j in range(2):
                            i = block_i * 2 + inner_i
                            j = block_j * 2 + inner_j
                            idx = i * 4 + j
                            val = w_base[l, block_i * 2 + block_j] * w_base[m, inner_i * 2 + inner_j]
                            W_444[idx, col] = val
    
    # Now, we need to extend this to 4x5x5.
    # A is 4x5, B is 5x5.
    # C = A * B
    # We split A = [A_4 | A_5] where A_4 is 4x4, A_5 is 4x1.
    # We split B = [B_14 | B_5] where B_14 is 4x4, B_5 is 1x5 (wait, B is 5x5, so B_5 is the 5th column, 5x1).
    # C = A_4 * B_14 + A_5 * B[4, :]
    # The first term A_4 * B_14 is a 4x4 matrix product, which we can compute using the 4x4x4 Strassen decomposition.
    # The second term A_5 * B[4, :] is an outer product of a 4x1 vector and a 1x5 vector, which has rank 1.
    # So we need 49 + 1 = 50 terms.
    
    # Let's construct the U, V, W matrices for the 4x5x5 problem.
    # U has shape (4*5, 50) = (20, 50)
    # V has shape (5*5, 50) = (25, 50)
    # W has shape (5*4, 50) = (20, 50)
    
    U = np.zeros((20, 50))
    V = np.zeros((25, 50))
    W = np.zeros((20, 50))
    
    # First 49 terms: A_4 * B_14
    # A_4 is 4x4, indexed by (i, j) where i in 0..3, j in 0..3.
    # In the full A (4x5), A_4 is at indices (i, j) for j in 0..3.
    # So U[i*5 + j, col] = U_444[i*4 + j, col] for j in 0..3, and 0 for j=4.
    # B_14 is 4x4, indexed by (j, k) where j in 0..3, k in 0..3.
    # In the full B (5x5), B_14 is at indices (j, k) for j in 0..3, k in 0..3.
    # So V[j*5 + k, col] = V_444[j*4 + k, col] for k in 0..3, and 0 for k=4.
    # The result C_44 is 4x4, indexed by (i, k) where i in 0..3, k in 0..3.
    # In the full C (4x5), C_44 is at indices (i, k) for k in 0..3.
    # So W[k*4 + i, col] = W_444[k*4 + i, col] for k in 0..3.
    # Wait, the W index is k*n + i = k*4 + i.
    # The 4x4x4 W_444 has indices (k, i) where k, i in 0..3.
    # So W[k*4 + i, col] = W_444[k*4 + i, col].
    
    for col in range(49):
        for i in range(4):
            for j in range(4):
                U[i*5 + j, col] = U_444[i*4 + j, col]
        for j in range(4):
            for k in range(4):
                V[j*5 + k, col] = V_444[j*4 + k, col]
        for k in range(4):
            for i in range(4):
                W[k*4 + i, col] = W_444[k*4 + i, col]
    
    # 50th term: A_5 * B[4, :]
    # A_5 is the 5th column of A, indexed by (i, 4) where i in 0..3.
    # B[4, :] is the 5th row of B, indexed by (4, k) where k in 0..4.
    # The outer product is A_5[i] * B[4, k], which contributes to C[i, k].
    # So U[i*5 + 4, 49] = 1 for i in 0..3.
    # V[4*5 + k, 49] = 1 for k in 0..4.
    # W[k*4 + i, 49] = 1 for i in 0..3, k in 0..4.
    
    for i in range(4):
        U[i*5 + 4, 49] = 1
    for k in range(5):
        V[4*5 + k, 49] = 1
    for i in range(4):
        for k in range(5):
            W[k*4 + i, 49] = 1
    
    # Verify the decomposition
    if _verify(U, V, W, 4, 5, 5):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fallback to schoolbook
        rank_school = 4 * 5 * 5
        U_school = np.zeros((20, rank_school))
        V_school = np.zeros((25, rank_school))
        W_school = np.zeros((20, rank_school))
        col = 0
        for i in range(4):
            for j in range(5):
                for k in range(5):
                    U_school[i*5 + j, col] = 1
                    V_school[j*5 + k, col] = 1
                    W_school[k*4 + i, col] = 1
                    col += 1
        return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
# EVOLVE-BLOCK-END
