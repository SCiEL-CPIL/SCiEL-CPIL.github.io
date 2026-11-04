import time
import sys

# EVOLVE-BLOCK-START

def _trivial_decomposition(n, m, p):
    """
    Return the trivial (rank n*m*p) exact tensor decomposition for n x m x p matrix multiplication.
    
    U: shape (n*m, rank)  -- entries for A[i,j] indexed as i*m+j
    V: shape (m*p, rank)  -- entries for B[j,k] indexed as j*p+k
    W: shape (p*n, rank)  -- entries for C[i,k] indexed as k*n+i (trace convention)
    
    The decomposition satisfies:
    C[i,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]  for all i,j,k
    """
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    t = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                U[u_idx][t] = 1
                V[v_idx][t] = 1
                W[w_idx][t] = 1
                t += 1
    
    return U, V, W

def _strassen_2x2_2x2_UVW():
    """
    Return the Strassen decomposition for 2x2 * 2x2 -> 2x2 multiplication.
    Rank = 7.
    
    U: shape (4, 7) for A[i,j] indexed as i*2+j
    V: shape (4, 7) for B[j,k] indexed as j*2+k  
    W: shape (4, 7) for C[i,k] indexed as k*2+i (trace convention)
    """
    # Strassen's 7 products for 2x2 matrix multiplication
    # A = [[a11, a12], [a21, a22]], B = [[b11, b12], [b21, b22]]
    # C = [[c11, c12], [c21, c22]]
    
    # The 7 intermediate products:
    # p1 = (a11+a22)(b11+b22)
    # p2 = (a21+a22)b11
    # p3 = a11(b12-b22)
    # p4 = a22(b21-b11)
    # p5 = (a11+a12)b22
    # p6 = (a21-a11)(b11+b12)
    # p7 = (a12-a22)(b21+b22)
    
    # c11 = p1 + p4 - p5 + p7
    # c12 = p3 + p5
    # c21 = p2 + p4
    # c22 = p1 - p2 + p3 + p6
    
    U = [
        # A[0,0]=a11: p1, p3, p5
        [1, 0, 1, 0, 1, 0, 0],
        # A[0,1]=a12: p5, p7
        [0, 0, 0, 0, 1, 0, 1],
        # A[1,0]=a21: p2, p6
        [0, 1, 0, 0, 0, -1, 0],
        # A[1,1]=a22: p1, p2, p4, p6
        [1, 1, 0, 1, 0, -1, 0],
    ]
    
    V = [
        # B[0,0]=b11: p1, p2, p6
        [1, 1, 0, 0, 0, 1, 0],
        # B[0,1]=b12: p3, p6
        [0, 0, 1, 0, 0, 1, 0],
        # B[1,0]=b21: p4, p7
        [0, 0, 0, 1, 0, 0, 1],
        # B[1,1]=b22: p1, p3, p5, p7
        [1, 0, -1, 0, 1, 0, 1],
    ]
    
    W = [
        # C[0,0]=c11: p1, p4, p5, p7 -> k=0,i=0 -> idx 0
        [1, 0, 0, 1, -1, 0, 1],
        # C[0,1]=c12: p3, p5 -> k=1,i=0 -> idx 1
        [0, 0, 1, 0, 1, 0, 0],
        # C[1,0]=c21: p2, p4 -> k=0,i=1 -> idx 2
        [0, 1, 0, 1, 0, 0, 0],
        # C[1,1]=c22: p1, p2, p3, p6 -> k=1,i=1 -> idx 3
        [1, -1, 1, 0, 0, 1, 0],
    ]
    
    return U, V, W

def _verify_decomposition(n, m, p, U, V, W):
    """
    Verify that the decomposition (U, V, W) correctly computes matrix multiplication.
    Returns True if valid, False otherwise.
    """
    try:
        import numpy as np
        np.random.seed(12345)
        
        # Test with several random matrices
        for trial in range(3):
            A = np.random.randint(-10, 10, (n, m), dtype=float)
            B = np.random.randint(-10, 10, (m, p), dtype=float)
            C_expected = A @ B
            
            rank = len(U[0])
            C_computed = np.zeros((n, p), dtype=float)
            
            # Efficient verification using einsum-like approach
            # C[i,k] = sum_{j,r} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            for i in range(n):
                for k in range(p):
                    s = 0.0
                    for j in range(m):
                        u_row = U[i * m + j]
                        v_row = V[j * p + k]
                        w_val = W[k * n + i]
                        for r in range(rank):
                            s += u_row[r] * v_row[r] * w_val[r]
                    C_computed[i, k] = s
            
            if not np.allclose(C_computed, C_expected, atol=1e-10):
                return False
        
        return True
    except Exception:
        return False

def _block_2x2x7_UVW():
    """
    Compute 2x2 * 2x7 -> 2x7 multiplication using Strassen for the 2x2 part
    combined with trivial for the 7 columns.
    
    This gives rank = 7 * 7 = 49 (Strassen rank 7 times 7 independent column blocks).
    Actually, we can do better by noting the structure.
    
    A is 2x2, B is 2x7, C is 2x7.
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    Using Strassen: we have 7 products that are linear combinations of A entries
    multiplied by linear combinations of B entries.
    
    For each of the 7 Strassen products, we need to multiply a 2x2 "effective" 
    matrix product by the 7 columns. But actually, the Strassen decomposition
    gives us 7 scalar products. For each scalar product p_r, we have:
    p_r = (linear combo of A) * (linear combo of B)
    
    But B is 2x7, so "linear combo of B" is a 1x7 vector.
    
    So for each Strassen term r, we get a 1x7 vector contribution to C.
    
    Total rank = 7 (Strassen) * 7 (columns) = 49? No, that's not right either.
    
    Let me think more carefully. The tensor for 2x2 * 2x7 -> 2x7 has:
    - U dimension: 2*2 = 4 (A entries)
    - V dimension: 2*7 = 14 (B entries)  
    - W dimension: 7*2 = 14 (C entries, trace convention k*2+i)
    
    The trivial rank is 4*7 = 28 (for each i,k we need sum over j, so 2*7=14 terms... 
    no, rank = n*m*p = 2*2*7 = 28).
    
    Using Strassen: The 2x2 part has rank 7. For each of the 7 Strassen products,
    we multiply a scalar (from A combination) by a 1x7 vector (from B combination).
    This gives 7 * 7 = 49 rank terms? That's worse than trivial.
    
    Actually, let's think of it differently. We can use the Strassen decomposition
    where we treat the 7 columns as 7 separate 2x1 multiplications, but that gives
    7*7 = 49 which is worse.
    
    Better approach: Use the fact that the multiplication is linear in the 7 columns.
    We can decompose B into 7 column vectors, and for each column, use Strassen.
    But that's 7 * 7 = 49.
    
    Alternatively, we can find a direct low-rank decomposition.
    
    Let me just use the trivial decomposition for 2x2x7: rank = 2*2*7 = 28.
    """
    n, m, p = 2, 2, 7
    U, V, W = _trivial_decomposition(n, m, p)
    return U, V, W

def _block_4x4x7_hierarchical():
    """
    Attempt a hierarchical/tensor-train style decomposition for 4x4 * 4x7 -> 4x7.
    
    Strategy: Split the 4x4 matrix into 2x2 blocks.
    A = [[A11, A12], [A21, A22]] where each Aij is 2x2
    B = [[B1], [B2]] where B1, B2 are 2x7
    C = [[C1], [C2]] where C1, C2 are 2x7
    
    C1 = A11*B1 + A12*B2
    C2 = A21*B1 + A22*B2
    
    Each C1, C2 is 2x7 = sum of two 2x2 * 2x7 products.
    
    For each 2x2 * 2x7 product, we can use a rank-? decomposition.
    
    The trivial rank for 2x2 * 2x7 is 2*2*7 = 28.
    If we could use Strassen-like for the 2x2 part with rank 7, and handle the 7 columns,
    we'd need 7*7 = 49 per 2x2*2x7 block, which is worse.
    
    So let's try a different approach: use the known fact that 
    M(4,4,7) <= M(4,4,4) + M(4,4,3) or similar subadditivity.
    
    Actually, let's try to use a block Strassen approach:
    Treat the 4x4 multiplication as a 2x2 block multiplication where each block
    is 2x2. This uses Strassen with rank 7, but each "element" multiplication is
    2x2 * 2x7 -> 2x7.
    
    For the block Strassen:
    p1 = (A11+A22)(B1+B2)  -- 2x2 * 2x7 -> 2x7
    p2 = (A21+A22)B1       -- 2x2 * 2x7 -> 2x7
    p3 = A11(B2-B1)        -- 2x2 * 2x7 -> 2x7
    p4 = A22(B1-B2)        -- 2x2 * 2x7 -> 2x7
    p5 = (A11+A12)B2       -- 2x2 * 2x7 -> 2x7
    p6 = (A21-A11)(B1+B2)  -- 2x2 * 2x7 -> 2x7
    p7 = (A12-A22)(B1+B2)  -- 2x2 * 2x7 -> 2x7
    
    Wait, this isn't quite right because B is 4x7, not 2x2 blocks of scalars.
    
    Let me reconsider. B is 4x7. Split B into B1 (2x7) and B2 (2x7).
    The "block Strassen" for 2x2 blocks:
    
    p1 = (A11+A22) * (B1+B2)  -- product of 2x2 and 2x7 -> 2x7
    p2 = (A21+A22) * B1       -- 2x2 * 2x7 -> 2x7
    p3 = A11 * (B2-B1)        -- 2x2 * 2x7 -> 2x7
    p4 = A22 * (B1-B2)        -- 2x2 * 2x7 -> 2x7
    p5 = (A11+A12) * B2       -- 2x2 * 2x7 -> 2x7
    p6 = (A21-A11) * (B1+B2)  -- 2x2 * 2x7 -> 2x7
    p7 = (A12-A22) * (B1+B2)  -- 2x2 * 2x7 -> 2x7
    
    Then:
    C11 = p1 + p4 - p5 + p7  (2x7)
    C12 = p3 + p5            (2x7)
    C21 = p2 + p4            (2x7)
    C22 = p1 - p2 + p3 + p6  (2x7)
    
    So C = [C11 C12; C21 C22] = 4x7.
    
    Each pr is a 2x2 * 2x7 -> 2x7 product. If we use trivial rank 28 for each,
    total rank = 7 * 28 = 196, which is worse than 112.
    
    This approach doesn't help directly. Let me try a different strategy.
    
    Alternative: Use the fact that M(4,4,7) <= 2*M(4,4,4) + M(4,4,3) or similar.
    But I don't have good decompositions for those either.
    
    Let me try a direct search approach with random initialization and gradient-free optimization
    to find a low-rank exact decomposition.
    """
    return None  # Placeholder

def _strassen_4x4_4x4_UVW():
    """
    Return the Strassen-based decomposition for 4x4 * 4x4 -> 4x4.
    Uses block Strassen: treat 4x4 as 2x2 blocks of 2x2 matrices.
    Each block multiplication uses Strassen (rank 7).
    Total rank = 7 * 7 = 49 (7 Strassen products, each involving 2x2*2x2).
    
    Actually, the block Strassen gives rank 7 where each "scalar" product is
    a 2x2 * 2x2 -> 2x2 multiplication. If each such multiplication uses
    Strassen with rank 7, total rank = 7 * 7 = 49.
    
    U: shape (16, 49) for A[i,j] indexed as i*4+j
    V: shape (16, 49) for B[j,k] indexed as j*4+k
    W: shape (16, 49) for C[i,k] indexed as k*4+i (trace convention)
    """
    # First get the 2x2 Strassen decomposition
    U2, V2, W2 = _strassen_2x2_2x2_UVW()
    
    # Block Strassen for 4x4:
    # A = [[A11, A12], [A21, A22]] (each 2x2)
    # B = [[B11, B12], [B21, B22]] (each 2x2)
    # C = [[C11, C12], [C21, C22]] (each 2x2)
    #
    # p1 = (A11+A22)(B11+B22)
    # p2 = (A21+A22)B11
    # p3 = A11(B12-B22)
    # p4 = A22(B21-B11)
    # p5 = (A11+A12)B22
    # p6 = (A21-A11)(B11+B12)
    # p7 = (A12-A22)(B21+B22)
    #
    # C11 = p1 + p4 - p5 + p7
    # C12 = p3 + p5
    # C21 = p2 + p4
    # C22 = p1 - p2 + p3 + p6
    
    # For each block Strassen product pr, we need to compute a 2x2 * 2x2 -> 2x2
    # multiplication using the 2x2 Strassen decomposition (rank 7).
    
    # The overall decomposition has rank 7 (block) * 7 (inner) = 49.
    
    # Let's construct it:
    # For each block Strassen term r_block (0..6), and each inner Strassen term r_inner (0..6),
    # we have a combined term.
    
    # Block Strassen coefficients for A:
    # A11 appears in: p1, p3, p5 -> coeffs: [1, 1, 0, 0, 1, 0, 0]
    # A12 appears in: p5, p7 -> coeffs: [0, 0, 0, 0, 1, 0, 1]
    # A21 appears in: p2, p6 -> coeffs: [0, 1, 0, 0, 0, -1, 0]
    # A22 appears in: p1, p2, p4, p6 -> coeffs: [1, 1, 0, 1, 0, -1, 0]
    
    # Block Strassen coefficients for B:
    # B11 appears in: p1, p2, p6 -> coeffs: [1, 1, 0, 0, 0, 1, 0]
    # B12 appears in: p3, p6 -> coeffs: [0, 0, 1, 0, 0, 1, 0]
    # B21 appears in: p4, p7 -> coeffs: [0, 0, 0, 1, 0, 0, 1]
    # B22 appears in: p1, p3, p5, p7 -> coeffs: [1, -1, 0, 0, 1, 0, 1]
    
    # Block Strassen coefficients for C:
    # C11 appears in: p1, p4, p5, p7 -> coeffs: [1, 0, 0, 1, -1, 0, 1]
    # C12 appears in: p3, p5 -> coeffs: [0, 0, 1, 0, 1, 0, 0]
    # C21 appears in: p2, p4 -> coeffs: [0, 1, 0, 1, 0, 0, 0]
    # C22 appears in: p1, p2, p3, p6 -> coeffs: [1, -1, 1, 0, 0, 1, 0]
    
    # Now, for each block term, we use the 2x2 Strassen decomposition.
    # The 2x2 Strassen has:
    # U2: 4x7 matrix for 2x2 A entries
    # V2: 4x7 matrix for 2x2 B entries
    # W2: 4x7 matrix for 2x2 C entries
    
    # Combined decomposition:
    # For block term rb and inner term ri:
    # U_entry[i,j, rb*7+ri] = block_A_coeff[block_A_index(i,j), rb] * U2[inner_A_index(i,j), ri]
    # where block_A_index maps (i,j) to which 2x2 block it's in
    # and inner_A_index maps to position within that 2x2 block
    
    # A[i,j] for i,j in 0..3:
    # Block index: bi = i//2, bj = j//2 -> block (bi,bj)
    # Inner index: ii = i%2, jj = j%2 -> position in 2x2 block
    # inner_A_index = ii*2 + jj
    
    U = [[0] * 49 for _ in range(16)]
    V = [[0] * 49 for _ in range(16)]
    W = [[0] * 49 for _ in range(16)]
    
    # Block A coefficients: for each 2x2 block position, which block Strassen terms it participates in
    # Block (0,0) = A11: [1, 0, 1, 0, 1, 0, 0]
    # Block (0,1) = A12: [0, 0, 0, 0, 1, 0, 1]
    # Block (1,0) = A21: [0, 1, 0, 0, 0, -1, 0]
    # Block (1,1) = A22: [1, 1, 0, 1, 0, -1, 0]
    block_A_coeffs = {
        (0,0): [1, 0, 1, 0, 1, 0, 0],
        (0,1): [0, 0, 0, 0, 1, 0, 1],
        (1,0): [0, 1, 0, 0, 0, -1, 0],
        (1,1): [1, 1, 0, 1, 0, -1, 0],
    }
    
    # Block B coefficients
    block_B_coeffs = {
        (0,0): [1, 1, 0, 0, 0, 1, 0],
        (0,1): [0, 0, 1, 0, 0, 1, 0],
        (1,0): [0, 0, 0, 1, 0, 0, 1],
        (1,1): [1, -1, 0, 0, 1, 0, 1],
    }
    
    # Block C coefficients
    block_C_coeffs = {
        (0,0): [1, 0, 0, 1, -1, 0, 1],
        (0,1): [0, 0, 1, 0, 1, 0, 0],
        (1,0): [0, 1, 0, 1, 0, 0, 0],
        (1,1): [1, -1, 1, 0, 0, 1, 0],
    }
    
    # Fill in U
    for i in range(4):
        for j in range(4):
            bi, bj = i // 2, j // 2
            ii, jj = i % 2, j % 2
            inner_idx = ii * 2 + jj
            for rb in range(7):
                for ri in range(7):
                    combined_r = rb * 7 + ri
                    U[i*4+j][combined_r] = block_A_coeffs[(bi,bj)][rb] * U2[inner_idx][ri]
    
    # Fill in V
    for j in range(4):
        for k in range(4):
            bi, bj = j // 2, k // 2
            ii, jj = j % 2, k % 2
            inner_idx = ii * 2 + jj
            for rb in range(7):
                for ri in range(7):
                    combined_r = rb * 7 + ri
                    V[j*4+k][combined_r] = block_B_coeffs[(bi,bj)][rb] * V2[inner_idx][ri]
    
    # Fill in W (trace convention: k*4+i)
    for i in range(4):
        for k in range(4):
            bi, bj = i // 2, k // 2
            ii, jj = i % 2, k % 2
            inner_idx = ii * 2 + jj
            for rb in range(7):
                for ri in range(7):
                    combined_r = rb * 7 + ri
                    W[k*4+i][combined_r] = block_C_coeffs[(bi,bj)][rb] * W2[inner_idx][ri]
    
    return U, V, W

def _extend_4x4_4x4_to_4x4_7():
    """
    Extend the 4x4*4x4 -> 4x4 Strassen decomposition (rank 49) to 4x4*4x7 -> 4x7.
    
    The 4x7 matrix B can be viewed as 4x4 + 4x3, but that's not directly helpful.
    
    Alternative: Use the fact that we can pad B to 4x8 and use a decomposition for 4x4*4x8.
    But we don't have that.
    
    Better approach: Use the 4x4*4x4 Strassen (rank 49) for the first 4 columns,
    and a trivial decomposition (rank 4*4*3=48) for the last 3 columns.
    Total rank = 49 + 48 = 97.
    
    This is better than 112!
    """
    # Get the 4x4*4x4 Strassen decomposition
    U4, V4, W4 = _strassen_4x4_4x4_UVW()
    
    # Get the trivial decomposition for 4x4*4x3
    U3, V3, W3 = _trivial_decomposition(4, 4, 3)
    
    # Combine them
    # U: first 49 columns from U4, next 48 from U3
    # V: first 49 columns from V4, next 48 from V3
    # W: first 49 columns from W4, next 48 from W3
    
    # But we need to handle the indexing carefully.
    # For the 4x4 part: B columns 0-3, C columns 0-3
    # For the 4x3 part: B columns 4-6, C columns 4-6
    
    # U4 is 16x49, U3 is 16x48
    # V4 is 16x49, V3 is 16x48
    # W4 is 16x49 (for 4x4 C), W3 is 12x48 (for 4x3 C, trace convention k*4+i for k in 0..2, i in 0..3)
    
    # We need to combine W4 (16x49) and W3 (12x48) into a 28x97 matrix.
    # W4 corresponds to C entries with k in 0..3 (rows 0..15 in trace convention)
    # W3 corresponds to C entries with k in 0..2 (rows 0..11 in trace convention for 4x3)
    # But in the 4x7 case, C has k in 0..6, so W should be 28x97.
    
    # W4: 16x49, entries for k in 0..3, i in 0..3 -> row index k*4+i
    # W3: 12x48, entries for k in 0..2, i in 0..3 -> row index k*4+i (but this is for 4x3)
    # In the 4x7 case, W3's rows should be mapped to k in 4..6, i in 0..3 -> row index (4+k3)*4+i where k3 is 0..2
    
    U = [row4 + row3 for row4, row3 in zip(U4, U3)]
    V = [row4 + row3 for row4, row3 in zip(V4, V3)]
    
    # W combination is trickier
    W = [[0] * 97 for _ in range(28)]
    
    # Copy W4 into W (rows 0-15)
    for i in range(16):
        for r in range(49):
            W[i][r] = W4[i][r]
    
    # Copy W3 into W (rows 16-27, corresponding to k=4,5,6)
    for i3 in range(12):
        k3, i_val = i3 // 4, i3 % 4
        row_in_W = (4 + k3) * 4 + i_val
        for r in range(48):
            W[row_in_W][49 + r] = W3[i3][r]
    
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    n, m, p = 4, 4, 7
    
    # Start with the trivial decomposition (rank = 4*4*7 = 112)
    # This is always correct.
    U, V, W = _trivial_decomposition(n, m, p)
    
    # Verify the trivial decomposition works
    if not _verify_decomposition(n, m, p, U, V, W):
        # Should never happen, but return something
        pass
    
    # Try the extended Strassen decomposition (rank 97)
    try:
        U_strassen, V_strassen, W_strassen = _extend_4x4_4x4_to_4x4_7()
        
        # Verify
        if _verify_decomposition(n, m, p, U_strassen, V_strassen, W_strassen):
            U, V, W = U_strassen, V_strassen, W_strassen
    except Exception:
        pass
    
    # Verify final result
    if not _verify_decomposition(n, m, p, U, V, W):
        # Fallback to trivial
        U, V, W = _trivial_decomposition(n, m, p)
    
    # Return the best decomposition found
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
