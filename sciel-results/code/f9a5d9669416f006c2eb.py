"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-44 decomposition for 3x4x4 matrix multiplication.
    Uses Strassen's algorithm (rank 7) for the four 2x2x2 sub-blocks
    involving the first two output rows, and trivial multiplication (rank 4)
    for the bottom output row.
    
    U: shape (12, 44)  -- n*m = 3*4 = 12
    V: shape (16, 44)  -- m*p = 4*4 = 16
    W: shape (12, 44)  -- p*n = 4*3 = 12
    """
    n, m, p = DIMENSIONS
    # 4 Strassen blocks (2x2x2) each of rank 7 = 28
    # 1 trivial block (1x4x4) of rank 4 = 4
    # Total rank = 32? No, 4*7 + 4 = 32. Wait, let me re-check the math.
    # Actually, the previous analysis said 44. Let's see.
    # The 3x4x4 matrix can be split into:
    # Rows 0,1 of A (2x4) times B (4x4) -> 2x4x4.
    # This can be split into four 2x2x2 blocks.
    # Row 2 of A (1x4) times B (4x4) -> 1x4x4.
    # 
    # For the 2x4x4 part:
    # Split B into 4 columns, so we have 4 blocks of 2x2x2? No.
    # A is 2x4, B is 4x4.
    # We can view this as 4 independent 2x2x2 multiplications if we split B's columns into pairs?
    # No, B is 4x4. We can split A into two 2x2 blocks? No, A is 2x4.
    # We can split the 2x4x4 tensor into four 2x2x2 tensors by splitting the middle dimension (j) into two parts?
    # A (2x4), B (4x4).
    # Let's split j into {0,1} and {2,3}.
    # Block 1: A[:, 0:2] (2x2) * B[0:2, :] (2x4). This is 2x2x4.
    # Block 2: A[:, 2:4] (2x2) * B[2:4, :] (2x4). This is 2x2x4.
    # Each 2x2x4 can be split into two 2x2x2 blocks.
    # So total 4 blocks of 2x2x2.
    # Each 2x2x2 block uses Strassen (rank 7).
    # So 4 * 7 = 28.
    # 
    # For the 1x4x4 part:
    # A[2, :] (1x4) * B (4x4).
    # This is just a vector-matrix multiplication.
    # It can be done with rank 4 (trivial).
    # 
    # Total rank = 28 + 4 = 32.
    
    rank = 32
    
    U = [[0] * rank for _ in range(n * m)]  # 12 x 32
    V = [[0] * rank for _ in range(m * p)]  # 16 x 32
    W = [[0] * rank for _ in range(p * n)]  # 12 x 32
    
    col = 0
    
    # Strassen coefficients for 2x2x2
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B10 - B11)
    # M4 = A11(B01 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # C00 = M1 - M4 + M5 + M7
    # C01 = M3 + M4
    # C10 = M2 + M5
    # C11 = M1 - M2 + M3 + M6
    
    # We define the Strassen terms as (U_coeff, V_coeff, W_coeff) for each of the 7 terms.
    # U indices: i*2 + j (local)
    # V indices: j*2 + k (local)
    # W indices: k*2 + i (local)
    
    # Term 1: (A00+A11)(B00+B11)
    # U: (0,0)=1, (1,1)=1
    # V: (0,0)=1, (1,1)=1
    # W: contributes to C00 (+1), C11 (+1)
    
    # Let's define the 7 terms explicitly.
    # Each term is a list of (u_idx, u_val), (v_idx, v_val), (w_idx, w_val)
    
    def add_term(u_coeffs, v_coeffs, w_coeffs):
        nonlocal col
        for u_idx, u_val in u_coeffs:
            U[u_idx][col] += u_val
        for v_idx, v_val in v_coeffs:
            V[v_idx][col] += v_val
        for w_idx, w_val in w_coeffs:
            W[w_idx][col] += w_val
        col += 1
    
    # Define the 4 blocks
    # Block (i_off, j_off, k_off)
    # i_off: 0 or 1 (row offset in A)
    # j_off: 0 or 2 (column offset in A / row offset in B)
    # k_off: 0 or 2 (column offset in B)
    
    blocks = [
        (0, 0, 0),
        (0, 2, 0),
        (0, 0, 2),
        (0, 2, 2),
    ]
    
    for i_off, j_off, k_off in blocks:
        # Local indices 0,1 map to global:
        # i = i_off + i'
        # j = j_off + j'
        # k = k_off + k'
        
        # U global index: (i_off + i') * m + (j_off + j')
        # V global index: (j_off + j') * p + (k_off + k')
        # W global index: (k_off + k') * n + (i_off + i')
        
        # Term 1: (A00+A11)(B00+B11)
        u1 = [( (i_off+0)*m + (j_off+0), 1 ), ( (i_off+1)*m + (j_off+1), 1 )]
        v1 = [( (j_off+0)*p + (k_off+0), 1 ), ( (j_off+1)*p + (k_off+1), 1 )]
        w1 = [( (k_off+0)*n + (i_off+0), 1 ), ( (k_off+1)*n + (i_off+1), 1 )]
        add_term(u1, v1, w1)
        
        # Term 2: (A10+A11)B00
        u2 = [( (i_off+1)*m + (j_off+0), 1 ), ( (i_off+1)*m + (j_off+1), 1 )]
        v2 = [( (j_off+0)*p + (k_off+0), 1 )]
        w2 = [( (k_off+0)*n + (i_off+1), 1 ), ( (k_off+1)*n + (i_off+1), 1 )]
        add_term(u2, v2, w2)
        
        # Term 3: A00(B10-B11)
        u3 = [( (i_off+0)*m + (j_off+0), 1 )]
        v3 = [( (j_off+1)*p + (k_off+0), 1 ), ( (j_off+1)*p + (k_off+1), -1 )]
        w3 = [( (k_off+0)*n + (i_off+0), 1 ), ( (k_off+1)*n + (i_off+0), 1 )]
        add_term(u3, v3, w3)
        
        # Term 4: A11(B01-B00)
        u4 = [( (i_off+1)*m + (j_off+1), 1 )]
        v4 = [( (j_off+0)*p + (k_off+1), 1 ), ( (j_off+0)*p + (k_off+0), -1 )]
        w4 = [( (k_off+0)*n + (i_off+0), -1 ), ( (k_off+1)*n + (i_off+0), 1 ),
              ( (k_off+0)*n + (i_off+1), 1 ), ( (k_off+1)*n + (i_off+1), 1 )]
        add_term(u4, v4, w4)
        
        # Term 5: (A00+A01)B11
        u5 = [( (i_off+0)*m + (j_off+0), 1 ), ( (i_off+0)*m + (j_off+1), 1 )]
        v5 = [( (j_off+1)*p + (k_off+1), 1 )]
        w5 = [( (k_off+0)*n + (i_off+0), 1 ), ( (k_off+1)*n + (i_off+0), 1 ),
              ( (k_off+0)*n + (i_off+1), 1 ), ( (k_off+1)*n + (i_off+1), 1 )]
        add_term(u5, v5, w5)
        
        # Term 6: (A10-A00)(B00+B01)
        u6 = [( (i_off+1)*m + (j_off+0), 1 ), ( (i_off+0)*m + (j_off+0), -1 )]
        v6 = [( (j_off+0)*p + (k_off+0), 1 ), ( (j_off+0)*p + (k_off+1), 1 )]
        w6 = [( (k_off+0)*n + (i_off+1), 1 ), ( (k_off+1)*n + (i_off+1), 1 )]
        add_term(u6, v6, w6)
        
        # Term 7: (A01-A11)(B10+B11)
        u7 = [( (i_off+0)*m + (j_off+1), 1 ), ( (i_off+1)*m + (j_off+1), -1 )]
        v7 = [( (j_off+1)*p + (k_off+0), 1 ), ( (j_off+1)*p + (k_off+1), 1 )]
        w7 = [( (k_off+0)*n + (i_off+0), 1 ), ( (k_off+1)*n + (i_off+0), 1 ),
              ( (k_off+0)*n + (i_off+1), -1 ), ( (k_off+1)*n + (i_off+1), -1 )]
        add_term(u7, v7, w7)
        
    # Now the bottom row (i=2)
    # A[2, j] * B[j, k]
    # Trivial multiplication, rank 4
    for j in range(4):
        u_idx = 2 * m + j
        v_coeffs = []
        for k in range(4):
            v_idx = j * p + k
            w_idx = k * n + 2
            # We need to add a term for each k? No, Strassen terms are rank 1.
            # Trivial multiplication for 1x4x4 is rank 4.
            # Each term is A[2,j] * B[j,k].
            # So for a fixed j, we have 4 terms? No.
            # The tensor is sum_j A[2,j] B[j,k].
            # This is a sum of 4 rank-1 tensors? No.
            # A[2,j] is a scalar. B[j,k] is a vector.
            # The term is A[2,j] * B[j,:].
            # This is not a rank-1 tensor in the (i,j,k) space unless we fix j.
            # Wait, the tensor is T_{ijk} = A_{ij} B_{jk}.
            # For i=2, T_{2jk} = A_{2j} B_{jk}.
            # This is a sum over j of A_{2j} B_{jk}.
            # Each term A_{2j} B_{jk} is a product of a vector in j (only one non-zero at j) and a vector in k.
            # So yes, it is a sum of 4 rank-1 tensors.
            # Term j: U[2*m+j] = 1, V[j*p+k] = 1 for all k? No.
            # Rank 1 tensor is u_i v_j w_k.
            # Here u_2j = 1, v_jk = 1? No.
            # The term is A_{2j} * B_{jk}.
            # This corresponds to u_{2j}=1, v_{jk}=1? No.
            # The tensor entry is u_i v_j w_k.
            # We want u_{2j} v_{jk} w_{k2} = A_{2j} B_{jk}.
            # Let u_{2j} = 1.
            # Let v_{jk} = 1 if j'=j, 0 otherwise? No, v is a vector of length m*p.
            # v_{j*p+k} = 1 for the specific j.
            # Let w_{k*n+2} = 1.
            # Then u_{2j} v_{j*p+k} w_{k*n+2} = 1 * 1 * 1 = 1.
            # But we need A_{2j} B_{jk}.
            # This decomposition works if A and B have entries 0 or 1.
            # But they are arbitrary.
            # The decomposition is exact if the coefficients are correct.
            # The rank-1 term is u * v * w.
            # We need sum_{col} u_i col v_j col w_k col = A_{ij} B_{jk}.
            # For the trivial algorithm, we have one term for each (i,j,k).
            # So for i=2, we have 4*4 = 16 terms?
            # No, the schoolbook algorithm has n*m*p terms.
            # For i=2, j in 0..3, k in 0..3, we have 16 terms.
            # Each term is u_{2j} v_{jk} w_{k2} = 1.
            # So rank is 16 for the bottom row?
            # Wait, the schoolbook rank is n*m*p = 48.
            # For i=0,1, we used Strassen.
            # For i=2, we use schoolbook.
            # Schoolbook for i=2 has 4*4 = 16 terms.
            # So total rank = 4*7 + 16 = 28 + 16 = 44.
            
            pass # This loop is not used yet
            
    # Let's redo the bottom row with schoolbook.
    # We already used col 0..27 for Strassen.
    # Now we add 16 terms for i=2.
    
    for j in range(4):
        for k in range(4):
            u_idx = 2 * m + j
            v_idx = j * p + k
            w_idx = k * n + 2
            U[u_idx][col] = 1
            V[v_idx][col] = 1
            W[w_idx][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
