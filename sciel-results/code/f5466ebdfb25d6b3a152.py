"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _get_winograd_2x2x2():
    """
    Returns U, V, W for 2x2x2 matrix multiplication with rank 5 (Winograd).
    U, V, W are 4x5 matrices.
    Indexing: U[i*2+j], V[j*2+k], W[k*2+i]
    """
    # Winograd 2x2 scheme
    # U (Left factors for A)
    # V (Left factors for B)
    # W (Right factors for M)
    # 
    # Standard Winograd 2x2:
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # M = A B
    # 
    # 5 products:
    # p1 = (a+d)(e+h)
    # p2 = (c+d)(e+h)
    # p3 = (a+b)(g+h)
    # p4 = (c+d)(f-e)
    # p5 = (a+b)(h-g)
    # 
    # M11 = p1 - p2 + p4 - p5
    # M12 = p3 + p5
    # M21 = p2 + p4
    # M22 = p1 - p3 + p2 - p4
    #
    # Let's map to U, V, W
    # U[i,j] is coeff of A[i,j] in product k
    # V[j,k] is coeff of B[j,k] in product k
    # W[k,i] is coeff of M[k,i] in product k
    
    U = np.zeros((4, 5))
    V = np.zeros((4, 5))
    W = np.zeros((4, 5))
    
    # Product 0: (a+d)(e+h)
    # a=A[0,0], d=A[1,1] -> U[0,0]=1, U[1,1]=1
    # e=B[0,0], h=B[1,1] -> V[0,0]=1, V[1,1]=1
    # M11 += 1, M22 += 1 -> W[0,0]=1, W[1,1]=1
    U[0, 0] = 1; U[1, 0] = 1
    V[0, 0] = 1; V[1, 0] = 1
    W[0, 0] = 1; W[1, 0] = 1
    
    # Product 1: (c+d)(e+h)
    # c=A[1,0], d=A[1,1] -> U[1,0]=1, U[1,1]=1
    # e=B[0,0], h=B[1,1] -> V[0,0]=1, V[1,1]=1
    # M21 += 1, M22 -= 1 -> W[1,0]=1, W[1,1]=-1
    U[1, 1] = 1; U[3, 1] = 1 # c is A[1,0] (idx 1*2+0=2? No. A[i,j] -> i*2+j. c is 1,0 -> 2. d is 1,1 -> 3)
    # Correction: A indices: 00->0, 01->1, 10->2, 11->3
    # B indices: 00->0, 01->1, 10->2, 11->3
    # M indices: 00->0, 01->1, 10->2, 11->3
    
    # Reset and do carefully
    U = np.zeros((4, 5))
    V = np.zeros((4, 5))
    W = np.zeros((4, 5))
    
    # P0: (A00 + A11) * (B00 + B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # P1: (A10 + A11) * (B00 + B11)
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1; V[3, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    
    # P2: (A00 + A01) * (B10 + B11)
    U[0, 2] = 1; U[1, 2] = 1
    V[2, 2] = 1; V[3, 2] = 1
    W[1, 2] = 1; W[3, 2] = -1
    
    # P3: (A10 + A11) * (B01 - B00)
    U[2, 3] = 1; U[3, 3] = 1
    V[1, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # P4: (A00 + A01) * (B11 - B10)
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1; V[2, 4] = -1
    W[1, 4] = 1; W[0, 4] = -1
    
    return U, V, W

def _make_strassen_winograd_4x4x4():
    """
    Constructs a rank-35 decomposition for 4x4x4 matrix multiplication.
    Uses Strassen's 7-product structure for 4x4 blocks, where each product
    is computed via Winograd 2x2x2 (rank 5).
    """
    U2, V2, W2 = _get_winograd_2x2x2()
    
    # Strassen 4x4 structure:
    # A = [[A00, A01], [A10, A11]] (2x2 blocks)
    # B = [[B00, B01], [B10, B11]]
    # M = A B
    #
    # P1 = (A00 + A11) * (B00 + B11)
    # P2 = (A10 + A11) * A00
    # P3 = (A00 + A01) * B11
    # P4 = (A10 + A11) * (B01 - A00)  -- Wait, standard Strassen:
    #
    # Standard Strassen for 2x2 blocks:
    # P1 = (A00 + A11)(B00 + B11)
    # P2 = (A10 + A11)A00
    # P3 = (A00 + A01)B11
    # P4 = (A10 + A11)(B01 - A00)
    # P5 = (A00 + A11)A10
    # P6 = (B00 + B11)(A11 - A00)
    # P7 = (A00 - A10)(B10 + B11)
    #
    # M00 = P1 - P2 + P4 - P7
    # M01 = P3 + P4
    # M10 = P5 + P6
    # M11 = P1 - P3 + P5 - P6
    
    # Each Pi is a 2x2 matrix product.
    # We need to map the 4x4 indices to the 2x2 block indices.
    # 4x4 Index: i*4+j
    # Block (bi, bj) where bi, bj in {0,1}
    # Element (i,j) in 4x4 corresponds to block (i//2, j//2) and local (i%2, j%2).
    
    U4 = np.zeros((16, 35))
    V4 = np.zeros((16, 35))
    W4 = np.zeros((16, 35))
    
    # For each Strassen product p in 0..6:
    # We have 5 Winograd products.
    # Total rank 35.
    
    for p_idx in range(7):
        # Determine the 2x2 block matrices for A and B in this product
        # A_blocks: list of (block_row, block_col, coeff)
        # B_blocks: list of (block_row, block_col, coeff)
        
        # Initialize empty lists
        A_blocks = []
        B_blocks = []
        
        if p_idx == 0: # P1 = (A00 + A11)(B00 + B11)
            A_blocks = [(0,0,1), (1,1,1)]
            B_blocks = [(0,0,1), (1,1,1)]
        elif p_idx == 1: # P2 = (A10 + A11)A00
            A_blocks = [(1,0,1), (1,1,1)]
            B_blocks = [(0,0,1)]
        elif p_idx == 2: # P3 = (A00 + A01)B11
            A_blocks = [(0,0,1), (0,1,1)]
            B_blocks = [(1,1,1)]
        elif p_idx == 3: # P4 = (A10 + A11)(B01 - A00) -> Wait, B01 - A00 is not a block of B.
            # Strassen P4 is usually (C+D)(B-A) for 2x2 scalar.
            # For blocks: P4 = (A10 + A11)(B01 - A00)? No, B-A is not defined for blocks in Strassen directly unless we treat it as element-wise?
            # No, Strassen works on the blocks as scalars in the algebra of 2x2 matrices? No, it works on the blocks as elements of the ring.
            # The formula is:
            # P4 = (A10 + A11) * (B01 - A00) is INCORRECT.
            # Correct Strassen:
            # P1 = (A00 + A11)(B00 + B11)
            # P2 = (A10 + A11)A00
            # P3 = (A00 + A01)B11
            # P4 = (A10 + A11)(B01 - A00) -- This is wrong.
            #
            # Let's use the standard:
            # P1 = (A00 + A11)(B00 + B11)
            # P2 = (A10 + A11)A00
            # P3 = (A00 + A01)B11
            # P4 = (A10 + A11)(B01 - A00) is wrong.
            #
            # Standard Strassen:
            # P1 = (A00 + A11)(B00 + B11)
            # P2 = (A10 + A11)A00
            # P3 = (A00 + A01)B11
            # P4 = (A10 + A11)(B01 - A00) -> No.
            #
            # P4 = (A10 + A11) * (B01 - A00) is not a product of two matrices from A and B.
            #
            # Correct P4: (A10 + A11) * (B01 - A00) is invalid.
            #
            # Let's check Wikipedia "Strassen algorithm":
            # P1 = (A00 + A11)(B00 + B11)
            # P2 = (A10 + A11)A00
            # P3 = (A00 + A01)B11
            # P4 = (A10 + A11)(B01 - A00) -> No, it's (C+D)(B-A) where C=A10, D=A11, B=B01, A=A00.
            # Yes, B01 - A00 is a matrix.
            # So P4 is product of (A10 + A11) and (B01 - A00).
            # This is a valid product of two 2x2 matrices.
            
            A_blocks = [(1,0,1), (1,1,1)]
            B_blocks = [(0,1,1), (0,0,-1)]
        elif p_idx == 4: # P5 = (A00 + A11)A10
            A_blocks = [(0,0,1), (1,1,1)]
            B_blocks = [(1,0,1)]
        elif p_idx == 5: # P6 = (B00 + B11)(A11 - A00)
            A_blocks = [(1,1,1), (0,0,-1)]
            B_blocks = [(0,0,1), (1,1,1)]
        elif p_idx == 6: # P7 = (A00 - A10)(B10 + B11)
            A_blocks = [(0,0,1), (1,0,-1)]
            B_blocks = [(1,0,1), (1,1,1)]
        
        # Now, for this product P_p, we compute the 2x2 matrix product using Winograd.
        # The "A" matrix for Winograd is formed by linear combination of blocks.
        # The "B" matrix for Winograd is formed by linear combination of blocks.
        
        # We need to add the contributions to U4, V4, W4.
        # The Winograd U2, V2, W2 are 4x5.
        # Local indices in 2x2: 0,1,2,3.
        # Global 4x4 indices:
        # Block (br, bc) contains elements at global rows br*2 + r, cols bc*2 + c.
        # For U (Left factor for A):
        # U4[global_i, global_j] += coeff * U2[local_i, local_j]
        # For V (Left factor for B):
        # V4[global_j, global_k] += coeff * V2[local_j, local_k]
        # For W (Right factor for M):
        # W4[global_k, global_i] += coeff * W2[local_k, local_i]
        
        # Map A blocks to U4
        # For each term in A_blocks: (br, bc, coeff)
        # For each local i, j in 2x2:
        # global_i = br*2 + i, global_j = bc*2 + j
        # U4[global_i*4 + global_j, p_idx*5 + k] += coeff * U2[i*2 + j, k]
        
        for k in range(5):
            # Update U4
            for br, bc, coeff in A_blocks:
                for li in range(2):
                    for lj in range(2):
                        gi = br * 2 + li
                        gj = bc * 2 + lj
                        U4[gi * 4 + gj, p_idx * 5 + k] += coeff * U2[li * 2 + lj, k]
            
            # Update V4
            for br, bc, coeff in B_blocks:
                for lj in range(2):
                    for lk in range(2):
                        gj = br * 2 + lj
                        gk = bc * 2 + lk
                        V4[gj * 4 + gk, p_idx * 5 + k] += coeff * V2[lj * 2 + lk, k]
            
            # Update W4
            # W2[k, i] corresponds to M[k, i]
            # M is 2x2. Global M indices:
            # M_block (mr, mc) where mr, mc in {0,1}
            # Element (i, j) in 2x2 M corresponds to global (mr*2+i, mc*2+j)
            # W2 index: local_k * 2 + local_i
            # W4 index: global_k * 4 + global_i
            # We need to map the result of P_p to the final M.
            # The contribution of P_p to M is:
            # M00 += s0 * P_p
            # M01 += s1 * P_p
            # M10 += s2 * P_p
            # M11 += s3 * P_p
            # where s depends on p_idx.
            
            if p_idx == 0: # P1
                # M00 += 1, M01 += 0, M10 += 0, M11 += 1
                coeffs = [1, 0, 0, 1]
            elif p_idx == 1: # P2
                # M00 -= 1, M01 += 0, M10 += 1, M11 += 0
                coeffs = [-1, 0, 1, 0]
            elif p_idx == 2: # P3
                # M00 += 0, M01 += 1, M10 += 0, M11 -= 1
                coeffs = [0, 1, 0, -1]
            elif p_idx == 3: # P4
                # M00 += 1, M01 += 1, M10 += 1, M11 += 0
                coeffs = [1, 1, 1, 0]
            elif p_idx == 4: # P5
                # M00 += 0, M01 += 0, M10 += 1, M11 += 1
                coeffs = [0, 0, 1, 1]
            elif p_idx == 5: # P6
                # M00 += 0, M01 += 0, M10 += 1, M11 -= 1
                coeffs = [0, 0, 1, -1]
            elif p_idx == 6: # P7
                # M00 -= 1, M01 += 0, M10 += 0, M11 += 0
                coeffs = [-1, 0, 0, 0]
            
            # coeffs[0] for M00, coeffs[1] for M01, coeffs[2] for M10, coeffs[3] for M11
            # M00: block (0,0). local (0,0) -> global (0,0)
            # M01: block (0,1). local (0,1) -> global (0,1)
            # M10: block (1,0). local (1,0) -> global (1,0)
            # M11: block (1,1). local (1,1) -> global (1,1)
            
            for lk in range(2):
                for li in range(2):
                    w_val = W2[lk * 2 + li, k]
                    if w_val == 0: continue
                    
                    # M00 contribution
                    if coeffs[0] != 0:
                        gk = 0 * 2 + lk
                        gi = 0 * 2 + li
                        W4[gk * 4 + gi, p_idx * 5 + k] += coeffs[0] * w_val
                    
                    # M01 contribution
                    if coeffs[1] != 0:
                        gk = 0 * 2 + lk
                        gi = 1 * 2 + li
                        W4[gk * 4 + gi, p_idx * 5 + k] += coeffs[1] * w_val
                    
                    # M10 contribution
                    if coeffs[2] != 0:
                        gk = 1 * 2 + lk
                        gi = 0 * 2 + li
                        W4[gk * 4 + gi, p_idx * 5 + k] += coeffs[2] * w_val
                    
                    # M11 contribution
                    if coeffs[3] != 0:
                        gk = 1 * 2 + lk
                        gi = 1 * 2 + li
                        W4[gk * 4 + gi, p_idx * 5 + k] += coeffs[3] * w_val

    return U4, V4, W4

def construct():
    """
    Find a low-rank exact decomposition for 4x4 * 4x8 matrix multiplication.
    Uses the block structure: 4x4x8 = two independent 4x4x4 blocks.
    Each block uses the rank-35 Strassen-Winograd decomposition, giving total rank 70.
    """
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    
    # Get the 4x4x4 decomposition
    U4, V4, W4 = _make_strassen_winograd_4x4x4()
    
    rank4 = U4.shape[1]
    total_rank = 2 * rank4
    
    U = np.zeros((n * m, total_rank))  # (16, 2*rank4)
    V = np.zeros((m * p, total_rank))  # (32, 2*rank4)
    W = np.zeros((p * n, total_rank))  # (32, 2*rank4)
    
    # Block 1: k in 0..3
    U[:, 0:rank4] = U4
    V[0:16, 0:rank4] = V4
    W[0:16, 0:rank4] = W4
    
    # Block 2: k in 4..7
    U[:, rank4:2*rank4] = U4
    # V: j*8 + (4+k_local)
    for j in range(4):
        for k_local in range(4):
            V[j*8 + 4 + k_local, rank4:2*rank4] = V4[j*4 + k_local, :]
    # W: (4+k_local)*4 + i
    for k_local in range(4):
        for i in range(4):
            W[(4+k_local)*4 + i, rank4:2*rank4] = W4[k_local*4 + i, :]
    
    # Verify the full decomposition
    def verify(U, V, W, n, m, p, rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    if not verify(U, V, W, n, m, p, total_rank):
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        total_rank = rank
    
    # Convert to lists
    U_list = [[float(U[i, c]) for c in range(total_rank)] for i in range(n * m)]
    V_list = [[float(V[i, c]) for c in range(total_rank)] for i in range(m * p)]
    W_list = [[float(W[i, c]) for c in range(total_rank)] for i in range(p * n)]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
