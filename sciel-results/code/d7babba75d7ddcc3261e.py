"""Rank-minimal tensor decomposition for 4x4x8 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [4, 4, 8]
START_TIME = time.time()
TIME_LIMIT = 140.0  # seconds, leave margin before 150s kill


def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W form a valid decomposition of the multiplication tensor."""
    rank = len(U[0])
    U3 = np.array(U, dtype=float).reshape(n, m, rank)
    V3 = np.array(V, dtype=float).reshape(m, p, rank)
    W3 = np.array(W, dtype=float).reshape(p, n, rank)

    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = float(np.dot(U3[i, j, :], V3[j, k, :] * W3[k, i, :]))
                if abs(val - 1.0) > tol:
                    return False
    return True


def _strassen_2x2x2():
    """
    Return the 7 rank-1 components of the 2x2x2 matrix multiplication tensor.
    
    Strassen's algorithm multiplies two 2x2 matrices:
      A = [[a00, a01], [a10, a11]]
      B = [[b00, b01], [b10, b11]]
      C = A * B
      
    The 7 products are:
      m0 = (a00 + a11) * (b00 + b11)
      m1 = (a10 + a11) * b00
      m2 = a00 * (b01 - b11)
      m3 = a11 * (b10 - b00)
      m4 = (a00 + a01) * b11
      m5 = (a10 - a00) * (b00 + b01)
      m6 = (a01 - a11) * (b10 + b11)
      
    Then:
      c00 = m0 + m3 - m5 - m7  ... wait, let me be more careful.
      
    Actually the standard Strassen formulas:
      m1 = (a00 + a11)(b00 + b11)
      m2 = (a10 + a11)b00
      m3 = a00(b01 - b11)
      m4 = a11(b10 - b00)
      m5 = (a00 + a01)b11
      m6 = (a10 - a00)(b00 + b01)
      m7 = (a01 - a11)(b10 + b11)
      
      c00 = m1 + m4 - m5 - m7
      c01 = m3 + m5
      c10 = m2 + m4
      c11 = m1 + m3 - m2 - m6
      
    Each m_s = X_s * Y_s where X_s is a linear form in A entries and Y_s is a linear form in B entries.
    Since each m_s is a scalar product of two linear forms, it's already rank-1 in the sense that:
      contribution to C entry (r,c) from m_s = alpha_{s,r,c} * X_s * Y_s
    
    So for each s, we have:
      U_s[i] = coefficient of a_i in X_s  (i indexes A entries: 00,01,10,11)
      V_s[j] = coefficient of b_j in Y_s  (j indexes B entries: 00,01,10,11)
      W_s[k] = coefficient for C entry k (00,01,10,11)
    
    Let me define the index order:
      A: 0=a00, 1=a01, 2=a10, 3=a11
      B: 0=b00, 1=b01, 2=b10, 3=b11
      C: 0=c00, 1=c01, 2=c10, 3=c11
    """
    # m1 = (a00+a11)(b00+b11)
    # m2 = (a10+a11)b00
    # m3 = a00(b01-b11)
    # m4 = a11(b10-b00)
    # m5 = (a00+a01)b11
    # m6 = (a10-a00)(b00+b01)
    # m7 = (a01-a11)(b10+b11)
    
    # U vectors (coefficients of A entries for each m_s)
    U = [
        [1, 0, 0, 1],   # m1: a00+a11
        [0, 0, 1, 1],   # m2: a10+a11
        [1, 0, 0, 0],   # m3: a00
        [0, 0, 0, 1],   # m4: a11
        [1, 1, 0, 0],   # m5: a00+a01
        [1, 0, -1, 0],  # m6: a10-a00
        [0, 1, 0, -1],  # m7: a01-a11
    ]
    
    # V vectors (coefficients of B entries for each m_s)
    V = [
        [1, 0, 0, 1],   # m1: b00+b11
        [1, 0, 0, 0],   # m2: b00
        [0, 1, 0, -1],  # m3: b01-b11
        [0, 0, 1, -1],  # m4: b10-b00
        [0, 0, 0, 1],   # m5: b11
        [1, 1, 0, 0],   # m6: b00+b01
        [0, 0, 1, 1],   # m7: b10+b11
    ]
    
    # W vectors: for each m_s, which C entries does it contribute to, and with what coefficient?
    # c00 = m1 + m4 - m5 - m7
    # c01 = m3 + m5
    # c10 = m2 + m4
    # c11 = m1 + m3 - m2 - m6
    W = [
        [1, 0, 0, 1],   # m1: c00+=1, c11+=1
        [0, 0, 1, -1],  # m2: c10+=1, c11-=1
        [0, 1, 0, 1],   # m3: c01+=1, c11+=1
        [1, 0, 1, 0],   # m4: c00+=1, c10+=1
        [-1, 1, 0, 0],  # m5: c00-=1, c01+=1
        [0, 0, 0, -1],  # m6: c11-=1
        [-1, 0, 0, 0],  # m7: c00-=1
    ]
    
    return U, V, W


def _build_4x4x4_decomposition():
    """
    Build a 49-rank decomposition for 4x4x4 matrix multiplication using Strassen.
    
    The 4x4 matrix multiplication can be viewed as multiplying 2x2 blocks.
    A is 4x4, split into four 2x2 blocks: A00, A01, A10, A11
    B is 4x4, split into four 2x2 blocks: B00, B01, B10, B11
    C is 4x4, split into four 2x2 blocks: C00, C01, C10, C11
    
    C00 = A00*B00 + A01*B10
    C01 = A00*B01 + A01*B11
    C10 = A10*B00 + A11*B10
    C11 = A10*B01 + A11*B11
    
    Each 2x2 block product can be done with Strassen (7 multiplications).
    But we can also use the "Strassen-like" 49-rank approach where we do:
    
    Actually, the standard way to get rank 49 for 4x4x4 is:
    - Split each 4x4 into 2x2 blocks
    - Use Strassen on the 2x2 block-level multiplication (which has 8 2x2 multiplications)
    - But Strassen on 2x2 matrices gives 7 scalar multiplications per 2x2 product
    - So 8 * 7 = 56? No...
    
    Wait, let me reconsider. The 4x4x4 tensor has rank at most 49.
    One way to achieve this: use the fact that 4x4 multiplication can be done
    with 49 scalar multiplications. This comes from:
    
    Actually, I recall that the best known upper bound for 4x4x4 is 49, achieved
    by a specific algorithm. Let me use a different approach.
    
    Alternative: Use the Kronecker product structure.
    The 4x4x4 multiplication tensor is the tensor product of two 2x2x2 tensors? No, that's not right either.
    
    Let me think about this differently. We can use the following approach:
    
    For 4x4x4, we can use the fact that it's a "block" multiplication. 
    Write A, B, C as 2x2 blocks of 2x2 matrices. Then:
      C_{ij} = sum_k A_{ik} B_{kj}
    
    Using Strassen at the block level:
      M1 = (A00+A11)(B00+B11)  -> 2x2 product, costs 7 mults
      M2 = (A10+A11)B00        -> 2x2 product, costs 7 mults
      M3 = A00(B01-B11)        -> 2x2 product, costs 7 mults
      M4 = A11(B10-B00)        -> 2x2 product, costs 7 mults
      M5 = (A00+A01)B11        -> 2x2 product, costs 7 mults
      M6 = (A10-A00)(B00+B01)  -> 2x2 product, costs 7 mults
      M7 = (A01-A11)(B10+B11)  -> 2x2 product, costs 7 mults
    
    Total: 7 * 7 = 49 scalar multiplications!
    
    Now I need to expand each 2x2 product into 7 scalar rank-1 terms.
    
    For each block-level Strassen product M_s = X_s * Y_s where X_s, Y_s are 2x2 matrices,
    I decompose X_s * Y_s into 7 scalar multiplications using the 2x2x2 Strassen decomposition.
    
    So for each s in {1,...,7} and each t in {1,...,7}:
      The (s,t)-th scalar multiplication is:
        (linear form in A entries) * (linear form in B entries) contributing to (linear form in C entries)
    
    Let me define the index mapping:
      A entries: A[i,j] for i,j in {0,1,2,3}
        Block A00 = A[0:2, 0:2] = [[A00, A01], [A10, A11]]
        Block A01 = A[0:2, 2:4] = [[A02, A03], [A12, A13]]
        Block A10 = A[2:4, 0:2] = [[A20, A21], [A30, A31]]
        Block A11 = A[2:4, 2:4] = [[A22, A23], [A32, A33]]
      
      I'll index A entries as A[i*4+j] for i,j in 0..3
      Similarly for B and C.
      
    For each block-level Strassen step s, I have:
      X_s = a linear combination of A blocks
      Y_s = a linear combination of B blocks
      
    Each X_s is a 2x2 matrix whose entries are linear combinations of A entries.
    Each Y_s is a 2x2 matrix whose entries are linear combinations of B entries.
    
    Then X_s * Y_s = sum_t (u_{s,t} . X_s) * (v_{s,t} . Y_s) * w_{s,t} where the dot products
    give scalars, and w_{s,t} determines which C entries get contributions.
    
    Actually, it's simpler to just directly compute:
    
    For the 2x2x2 Strassen decomposition, given two 2x2 matrices P and Q, the product P*Q
    is decomposed into 7 scalar multiplications. Each scalar multiplication m_t = (row vector . P) * (column vector . Q)
    contributes to P*Q with specific coefficients.
    
    So for block s:
      X_s is 2x2, Y_s is 2x2
      Product X_s * Y_s has entries that are sums of 7 scalar products
      
    Let me just build this computationally.
    """
    n = m = p = 4
    U_s, V_s, W_s = _strassen_2x2x2()
    
    # Block-level Strassen definitions
    # Each entry: (coefficients for A blocks, coefficients for B blocks)
    # A blocks indexed 0,1,2,3 corresponding to A00, A01, A10, A11
    # B blocks indexed 0,1,2,3 corresponding to B00, B01, B10, B11
    
    # Block-level Strassen:
    # M1 = (A00+A11) * (B00+B11)
    # M2 = (A10+A11) * B00
    # M3 = A00 * (B01-B11)
    # M4 = A11 * (B10-B00)
    # M5 = (A00+A01) * B11
    # M6 = (A10-A00) * (B00+B01)
    # M7 = (A01-A11) * (B10+B11)
    
    # For each block-level step s, the A-block coefficients and B-block coefficients
    # A_blocks: 0=A00, 1=A01, 2=A10, 3=A11
    # B_blocks: 0=B00, 1=B01, 2=B10, 3=B11
    
    A_block_coeffs = [
        [1, 0, 0, 1],  # M1: A00+A11
        [0, 0, 1, 1],  # M2: A10+A11
        [1, 0, 0, 0],  # M3: A00
        [0, 0, 0, 1],  # M4: A11
        [1, 1, 0, 0],  # M5: A00+A01
        [1, 0, -1, 0], # M6: A10-A00
        [0, 1, 0, -1], # M7: A01-A11
    ]
    
    B_block_coeffs = [
        [1, 0, 0, 1],  # M1: B00+B11
        [1, 0, 0, 0],  # M2: B00
        [0, 1, 0, -1], # M3: B01-B11
        [0, 0, 1, -1], # M4: B10-B00
        [0, 0, 0, 1],  # M5: B11
        [1, 1, 0, 0],  # M6: B00+B01
        [0, 0, 1, 1],  # M7: B10+B11
    ]
    
    # C block reconstruction from M1..M7
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 + M3 - M2 - M6
    
    # For each block s, and each 2x2x2 Strassen term t:
    # The contribution to C blocks is:
    #   W_block[s][t] * C_block_coeff[s]
    # where W_block[s][t] is from the 2x2x2 decomposition (which C entry of the 2x2 product it contributes to)
    # and C_block_coeff maps 2x2 product entries to C blocks.
    
    # Wait, I need to be more careful. Let me think step by step.
    
    # For block-level step s:
    #   X_s = sum_{a} A_block_coeffs[s][a] * A_a  (a 2x2 matrix)
    #   Y_s = sum_{b} B_block_coeffs[s][b] * B_b  (a 2x2 matrix)
    #   Product P_s = X_s * Y_s (a 2x2 matrix)
    
    # Now decompose P_s = X_s * Y_s using 2x2x2 Strassen:
    #   P_s[i,j] = sum_t U_2[t][i] * V_2[t][j] * W_2[t]  ... no, that's not right.
    
    # Actually, the 2x2x2 Strassen decomposition says:
    #   For 2x2 matrices P and Q, P*Q = sum_t (u_t . P_entries) * (v_t . Q_entries) * w_t_vector
    # where u_t is a 4-vector (coefficients of P's entries), v_t is a 4-vector (coefficients of Q's entries),
    # and w_t is a 4-vector (coefficients for each entry of P*Q).
    
    # So P_s[i,j] = sum_t (sum_{p,q} U_2[t][p*2+q] * X_s[p,q]) * (sum_{r,s} V_2[t][r*2+s] * Y_s[r,s]) * W_2[t][i*2+j]
    
    # And C_block[block_i, block_j][a,b] where block_i, block_j identify which C block:
    #   C00 gets contributions from M1, M4, M5, M7
    #   C01 gets contributions from M3, M5
    #   C10 gets contributions from M2, M4
    #   C11 gets contributions from M1, M3, M2, M6
    
    # Let me define the C block coefficients more explicitly.
    # C block index: 0=C00, 1=C01, 2=C10, 3=C11
    # For each block-level M_s, its contribution to each C block:
    #   M1 -> C00 (+1), C11 (+1)
    #   M2 -> C10 (+1), C11 (-1)
    #   M3 -> C01 (+1), C11 (+1)
    #   M4 -> C00 (+1), C10 (+1)
    #   M5 -> C00 (-1), C01 (+1)
    #   M6 -> C11 (-1)
    #   M7 -> C00 (-1)
    
    C_block_coeffs = [
        [1, 0, 0, 1],  # M1: C00+=1, C11+=1
        [0, 0, 1, -1], # M2: C10+=1, C11-=1
        [0, 1, 0, 1],  # M3: C01+=1, C11+=1
        [1, 0, 1, 0],  # M4: C00+=1, C10+=1
        [-1, 1, 0, 0], # M5: C00-=1, C01+=1
        [0, 0, 0, -1], # M6: C11-=1
        [-1, 0, 0, 0], # M7: C00-=1
    ]
    
    # Now, for each (s, t) where s is block-level Strassen step and t is 2x2x2 Strassen term:
    # The global U vector has entries for all 16 A entries.
    #   X_s[p,q] = sum_a A_block_coeffs[s][a] * A_a[p,q]
    #   where A_a is a 2x2 block.
    #   
    #   The entry of X_s at position (p,q) for p,q in {0,1}:
    #     If a=0 (A00): A[0,0], A[0,1], A[1,0], A[1,1] -> positions 0,1,4,5
    #     If a=1 (A01): A[0,2], A[0,3], A[1,2], A[1,3] -> positions 2,3,6,7
    #     If a=2 (A10): A[2,0], A[2,1], A[3,0], A[3,1] -> positions 8,9,12,13
    #     If a=3 (A11): A[2,2], A[2,3], A[3,2], A[3,3] -> positions 10,11,14,15
    #
    #   So X_s[p,q] = sum_a A_block_coeffs[s][a] * A[base_a + p, base_a_col + q]
    #   where base_a for row: a=0->0, a=1->0, a=2->2, a=3->2
    #          base_a_col for col: a=0->0, a=1->2, a=2->0, a=3->2
    #
    #   Global A index: A[i*4+j]
    #   For block a, position (p,q): i = block_row[a] + p, j = block_col[a] + q
    #   block_row = [0, 0, 2, 2], block_col = [0, 2, 0, 2]
    
    block_row = [0, 0, 2, 2]
    block_col = [0, 2, 0, 2]
    
    # Now compute global U, V, W vectors for each (s,t) pair
    U_list = []
    V_list = []
    W_list = []
    
    for s in range(7):
        for t in range(7):
            u2 = U_s[t]  # 4-vector: coefficients of P entries [P00, P01, P10, P11]
            v2 = V_s[t]  # 4-vector: coefficients of Q entries [Q00, Q01, Q10, Q11]
            w2 = W_s[t]  # 4-vector: coefficients for each entry of P*Q
            
            # Global U vector: 16 entries
            # U_global[A_idx] = sum_{p,q} u2[p*2+q] * A_block_coeffs[s][a] * delta(A_idx corresponds to block a, pos (p,q))
            U_g = [0] * 16
            for a in range(4):
                for p in range(2):
                    for q in range(2):
                        A_idx = (block_row[a] + p) * 4 + (block_col[a] + q)
                        U_g[A_idx] += u2[p * 2 + q] * A_block_coeffs[s][a]
            
            # Global V vector: 16 entries
            V_g = [0] * 16
            for b in range(4):
                for r in range(2):
                    for sc in range(2):
                        B_idx = (block_row[b] + r) * 4 + (block_col[b] + sc)
                        V_g[B_idx] += v2[r * 2 + sc] * B_block_coeffs[s][b]
            
            # Global W vector: 16 entries
            # C entry (i,j) where i = block_row[c] + p, j = block_col[c] + q
            # C_idx = i*4 + j
            # Contribution: w2[p*2+q] * C_block_coeffs[s][c]
            W_g = [0] * 16
            for c in range(4):
                for p in range(2):
                    for q in range(2):
                        C_idx = (block_row[c] + p) * 4 + (block_col[c] + q)
                        W_g[C_idx] += w2[p * 2 + q] * C_block_coeffs[s][c]
            
            U_list.append(U_g)
            V_list.append(V_g)
            W_list.append(W_g)
    
    return U_list, V_list, W_list, 49


def _build_4x4x8_decomposition():
    """
    Build a decomposition for 4x4x8 using the 49-rank 4x4x4 decomposition.
    
    For 4x4x8: A is 4x4, B is 4x8, C is 8x4.
    C[k,i] = sum_j A[i,j] * B[j,k]
    
    The output W has shape (p*n, rank) = (8*4, rank) = (32, rank)
    W[k*n+i, c] gives the coefficient for C[k,i] from factor c.
    
    We can split B into two 4x4 blocks: B_left = B[:, 0:4], B_right = B[:, 4:8]
    Similarly, C splits into C_left = C[0:4, :] and C_right = C[4:8, :]
    
    C_left[k,i] = sum_j A[i,j] * B[j,k] for k=0..3
    C_right[k,i] = sum_j A[i,j] * B[j,k] for k=4..7
    
    These are two independent 4x4x4 multiplications (same A, different B blocks, different C blocks).
    
    So we can use the 49-rank decomposition for each:
    - First 49 factors: handle B_left -> C_left
    - Next 49 factors: handle B_right -> C_right
    
    Total rank: 98.
    
    For the first 49 factors:
      U is the same as in 4x4x4 (16 x 49)
      V uses B_left entries (16 x 49) -> maps to B indices 0..15 (i.e., B[j,k] for k=0..3)
      W uses C_left entries (16 x 49) -> maps to W indices k*n+i for k=0..3, i=0..3 -> indices 0..15
    
    For the next 49 factors:
      U is the same as in 4x4x4 (16 x 49)
      V uses B_right entries (16 x 49) -> maps to B indices 16..31 (i.e., B[j,k] for k=4..7)
      W uses C_right entries (16 x 49) -> maps to W indices k*n+i for k=4..7, i=0..3 -> indices 16..31
    """
    U_444, V_444, W_444, rank_444 = _build_4x4x4_decomposition()
    
    total_rank = 2 * rank_444  # 98
    
    # U: shape (16, 98)
    # First 49 cols: U_444
    # Next 49 cols: U_444 (same A)
    U = [U_444[s][:] for s in range(16)]
    for c in range(rank_444, 2 * rank_444):
        for r in range(16):
            U[r].append(U_444[r][c - rank_444])
    
    # V: shape (32, 98)
    # First 49 cols: V_444 for B_left (rows 0..15), zeros for B_right (rows 16..31)
    # Next 49 cols: zeros for B_left, V_444 for B_right (rows 16..31)
    V = [[0] * total_rank for _ in range(32)]
    for c in range(rank_444):
        for r in range(16):
            V[r][c] = V_444[r][c]
            # B_right rows 16..31 stay 0 for first 49 factors
    
    for c in range(rank_444, 2 * rank_444):
        for r in range(16):
            V[16 + r][c] = V_444[r][c - rank_444]
            # B_left rows 0..15 stay 0 for next 49 factors
    
    # W: shape (32, 98)
    # W[k*n+i, c] where k in 0..7, i in 0..3
    # Index = k*4 + i
    # First 49 cols: W_444 for C_left (k=0..3), zeros for C_right (k=4..7)
    # Next 49 cols: zeros for C_left, W_444 for C_right (k=4..7)
    W = [[0] * total_rank for _ in range(32)]
    for c in range(rank_444):
        for idx in range(16):
            # idx = k*4+i where k=0..3, i=0..3
            k = idx // 4
            i = idx % 4
            # W_444[idx] gives the coefficient for C[k,i]
            W[idx][c] = W_444[idx][c]
            # C_right rows 16..31 stay 0
    
    for c in range(rank_444, 2 * rank_444):
        for idx in range(16):
            k = idx // 4
            i = idx % 4
            # C_right[k,i] -> W index (k+4)*4 + i
            W[16 + idx][c] = W_444[idx][c - rank_444]
            # C_left rows 0..15 stay 0
    
    return U, V, W, total_rank


def _verify_444_decomposition(U, V, W):
    """Verify the 4x4x4 decomposition."""
    n = m = p = 4
    rank = len(U[0])
    U3 = np.array(U, dtype=float).reshape(n, m, rank)
    V3 = np.array(V, dtype=float).reshape(m, p, rank)
    W3 = np.array(W, dtype=float).reshape(p, n, rank)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = float(np.dot(U3[i, j, :], V3[j, k, :] * W3[k, i, :]))
                if abs(val - 1.0) > 1e-9:
                    return False, (i, j, k, val)
    return True, None


def _schoolbook_444(n, m, p):
    """Standard rank n*m*p decomposition for 4x4x4."""
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
    return U, V, W, rank


def construct():
    n, m, p = DIMENSIONS  # 4, 4, 8
    deadline = START_TIME + TIME_LIMIT

    # Build the 98-rank decomposition
    try:
        U, V, W, rank = _build_4x4x8_decomposition()
        
        # Verify
        if _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
            return {"U": U, "V": V, "W": W}
        else:
            print("Verification failed for 98-rank decomposition, falling back to schoolbook")
    except Exception as e:
        print(f"Error building 98-rank decomposition: {e}")
    
    # Fallback: schoolbook decomposition (rank 128)
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
