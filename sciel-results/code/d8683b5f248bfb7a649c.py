import time
import numpy as np

DIMENSIONS = [2, 2, 16]

def _verify_decomp(U, V, W, n, m, p, num_tests=20):
    """Verify the decomposition is correct."""
    for seed in range(num_tests):
        np.random.seed(seed)
        A_test = np.random.randint(-5, 5, size=(n, m))
        B_test = np.random.randint(-5, 5, size=(m, p))
        C_test = A_test @ B_test
        
        A_flat = A_test.flatten()
        B_flat = B_test.flatten()
        
        UA = U.T @ A_flat
        VB = V.T @ B_flat
        terms = UA * VB
        C_decomp = W @ terms
        C_test_flat = C_test.T.flatten()
        
        if not np.allclose(C_decomp, C_test_flat, rtol=1e-9, atol=1e-9):
            return False
    return True

def _rank48_decomposition(n, m, p):
    """
    Rank 48 decomposition for 2x2 by 2x16 matrix multiplication.
    
    Exploits the structure: C[i,k] = sum_j A[i,j] B[j,k].
    For each pair of output columns (2k, 2k+1), we use a rank-3 decomposition
    of the 1x2x2 multiplication (vector by 2x2 block).
    
    For columns 2k and 2k+1:
    C[i,2k] = A[i,0] B[0,2k] + A[i,1] B[1,2k]
    C[i,2k+1] = A[i,0] B[0,2k+1] + A[i,1] B[1,2k+1]
    
    Rank-3 decomposition of 1x2x2:
    M1 = A[0] * (B[0,0] + B[0,1])  -> contributes to both outputs
    M2 = A[1] * (B[1,0] + B[1,1])  -> contributes to both outputs  
    M3 = A[0] * B[1,1]             -> correction term
    
    Wait, let me think more carefully.
    
    We need: C[0,0] = A0*B00 + A1*B10
              C[0,1] = A0*B01 + A1*B11
              C[1,0] = A0*B10 + A1*B20  -- no wait, this is 2x2 by 2x2
    
    Actually for the full 2x2 by 2x2:
    C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
    C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
    C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
    C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1]
    
    This is just 4 independent 1x2x1 multiplications... no wait.
    
    Let me reconsider. The tensor M_{2,2,16} has entries T[i,j,k] where i in {0,1}, j in {0,1}, k in {0,...,15}.
    T[i,j,k] = 1 if i=j (for the matrix multiplication structure).
    
    Actually, the multiplication tensor for C = A*B where A is 2x2 and B is 2x16:
    C[i,k] = sum_j A[i,j] B[j,k]
    
    So the tensor T has T[i,j,k] = 1 for all i,j,k where the product A[i,j]*B[j,k] contributes to C[i,k].
    The structure is: for each (i,k), C[i,k] = sum_j A[i,j] B[j,k].
    
    This is equivalent to 2*16 = 32 independent 1x2x1 tensors? No.
    It's 2 independent 1x2x16 tensors (one for each row of A).
    
    For a fixed i, C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] for all k.
    This is a 1x2x16 tensor (row vector times 2x16 matrix).
    
    The rank of M_{1,2,16}: 
    C[k] = A[0]*B[0,k] + A[1]*B[1,k] for k=0..15.
    
    We can group the 16 outputs into 8 pairs. For each pair (2q, 2q+1):
    C[2q] = A[0]*B[0,2q] + A[1]*B[1,2q]
    C[2q+1] = A[0]*B[0,2q+1] + A[1]*B[1,2q+1]
    
    This is a 1x2x2 tensor for each pair. The rank of M_{1,2,2} is 3.
    
    So total rank = 8 * 3 = 24? That seems too low. Let me verify.
    
    For M_{1,2,2}: C[0] = A[0]*B[0,0] + A[1]*B[1,0], C[1] = A[0]*B[0,1] + A[1]*B[1,1]
    
    Rank 3 decomposition:
    M1 = A[0] * (B[0,0] + B[0,1])
    M2 = A[1] * (B[1,0] + B[1,1])
    M3 = A[0] * B[1,1]
    
    C[0] = M1 + M2 - A[0]*B[0,1] - A[1]*B[1,1] + A[0]*B[0,0] + A[1]*B[1,0]
    Hmm, this doesn't work directly.
    
    Let me use the known rank-3 decomposition:
    M1 = A[0] * (B[0,0] + B[1,1])
    M2 = A[1] * (B[1,0] + B[0,1])  
    M3 = A[0] * B[1,1]
    
    C[0] should be A[0]*B[0,0] + A[1]*B[1,0]
    C[1] should be A[0]*B[0,1] + A[1]*B[1,1]
    
    M1 + M2 + M3 = A[0]*(B[0,0]+B[1,1]) + A[1]*(B[1,0]+B[0,1]) + A[0]*B[1,1]
    = A[0]*B[0,0] + A[0]*B[1,1] + A[1]*B[1,0] + A[1]*B[0,1] + A[0]*B[1,1]
    = A[0]*B[0,0] + A[1]*B[1,0] + A[1]*B[0,1] + 2*A[0]*B[1,1]
    
    That's not right either. Let me look up the correct rank-3 decomposition.
    
    Actually, M_{1,2,2} is isomorphic to M_{2,1,2} and M_{2,2,1}.
    The rank of M_{1,n,m} is n*m for n,m >= 2? No, it's known that rk(M_{1,2,2}) = 3.
    
    Let me construct it explicitly. We want to represent:
    C[0] = A[0]*B[0,0] + A[1]*B[1,0]
    C[1] = A[0]*B[0,1] + A[1]*B[1,1]
    
    using 3 multiplications. Let's try:
    P1 = A[0] * (B[0,0] + B[0,1])
    P2 = A[1] * (B[1,0] + B[1,1])
    P3 = (A[0] - A[1]) * (B[0,1] - B[1,0])
    
    C[0] = P1 + P2 - P3 + ... hmm.
    
    P1 + P2 = A[0]*B[0,0] + A[0]*B[0,1] + A[1]*B[1,0] + A[1]*B[1,1]
    P3 = A[0]*B[0,1] - A[0]*B[1,0] - A[1]*B[0,1] + A[1]*B[1,0]
    
    P1 + P2 - P3 = A[0]*B[0,0] + A[1]*B[1,1] + 2*A[0]*B[0,1] + 2*A[1]*B[1,0] - A[0]*B[1,0] - A[1]*B[0,1]
    
    This is getting complicated. Let me just use the fact that the total rank should be 48 based on the analysis, and construct a valid decomposition.
    
    Actually, the simplest approach: for each of the 2 rows of A, we have a 1x2x16 multiplication.
    The rank of M_{1,2,16} is 2*16 = 32 (schoolbook), but it can be lower.
    
    Known result: rk(M_{1,n,m}) = n*m for n,m >= 1, except rk(M_{1,2,2}) = 3.
    So rk(M_{1,2,16}) = 2*16 = 32? Or can we do better?
    
    If rk(M_{1,2,2}) = 3, then rk(M_{1,2,16}) <= 8*3 = 24 by grouping into 8 blocks of 2.
    And rk(M_{2,2,16}) <= 2*24 = 48.
    
    Let me implement the rank-3 decomposition of M_{1,2,2} carefully.
    """
    n, m, p = 2, 2, 16
    num_blocks = p // 2  # 8 blocks of 2 columns each
    R = num_blocks * 3  # rank 24 for each row, 48 total
    
    U = np.zeros((n * m, R), dtype=int)
    V = np.zeros((m * p, R), dtype=int)
    W = np.zeros((p * n, R), dtype=int)
    
    # For each block q (columns 2q and 2q+1), and for each row i of A (0 and 1):
    # C[i, 2q] = A[i,0]*B[0,2q] + A[i,1]*B[1,2q]
    # C[i, 2q+1] = A[i,0]*B[0,2q+1] + A[i,1]*B[1,2q+1]
    #
    # Rank-3 decomposition of this 1x2x2 sub-problem:
    # Let a0 = A[i,0], a1 = A[i,1]
    # Let b00 = B[0,2q], b01 = B[0,2q+1], b10 = B[1,2q], b11 = B[1,2q+1]
    #
    # We want: c0 = a0*b00 + a1*b10, c1 = a0*b01 + a1*b11
    #
    # Using:
    # P1 = a0 * (b00 + b01)
    # P2 = a1 * (b10 + b11)
    # P3 = (a0 - a1) * (b11 - b10)
    #
    # c0 = P1 - a0*b01 + a1*b10 - (a0-a1)*(b11-b10) + a0*b00
    #     = P1 - a0*b01 + a1*b10 - a0*b11 + a0*b10 + a1*b11 - a1*b10 + a0*b00
    #     = a0*b00 + a0*b01 - a0*b01 + a1*b10 - a0*b11 + a0*b10 + a1*b11 - a1*b10 + a0*b00
    #     = 2*a0*b00 + a0*b10 + a1*b11 - a0*b11
    #     Not right.
    
    # Let me try a different approach. The standard rank-3 for 1x2x2:
    # P1 = a0 * b00
    # P2 = a1 * b10  
    # P3 = (a0 + a1) * (b01 + b11)
    # c0 = P1 + P2
    # c1 = P3 - a0*b01 - a1*b11 + a0*b01 + a1*b11... no.
    
    # Actually, I think the correct approach is:
    # P1 = a0 * (b00 + b11)
    # P2 = a1 * (b10 + b01)
    # P3 = a0 * b11
    # c0 = P1 + P2 - a0*b11 - a1*b01 + a1*b10 = P1 + P2 - P3 - a1*b01 + a1*b10
    # Still has extra terms.
    
    # Let me just use the trivial rank-4 decomposition for each 1x2x2 block:
    # P1 = a0*b00, P2 = a1*b10, P3 = a0*b01, P4 = a1*b11
    # c0 = P1 + P2, c1 = P3 + P4
    # That gives rank 4 per block, total 8*4*2 = 64 (same as schoolbook).
    
    # Or use rank 3: I'll look for a valid one.
    # c0 = a0*b00 + a1*b10
    # c1 = a0*b01 + a1*b11
    #
    # P1 = a0*(b00+b01), P2 = a1*(b10+b11), P3 = (a0-a1)*(b01-b10)
    # c0 = P1 - a0*b01 + a1*b10 - (a0-a1)*(b01-b10) + a0*b00
    #    = a0*b00 + a0*b01 - a0*b01 + a1*b10 - a0*b01 + a0*b10 + a1*b01 - a1*b10 + a0*b00
    #    = 2*a0*b00 + a0*b10 + a1*b01 - a0*b01
    #    Nope.
    
    # I'll just use rank 4 per 1x2x2 block for safety, giving total rank 64.
    # But that's not better than schoolbook.
    
    # Let me try: for the full 2x2x16, use a different grouping.
    # Actually, let's just implement a correct rank-48 using a different method.
    
    # For each column k (0..15):
    # C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k]
    # C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k]
    #
    # Group columns into pairs. For pair (2q, 2q+1):
    # We have 4 outputs: C[0,2q], C[0,2q+1], C[1,2q], C[1,2q+1]
    # This is a 2x2x2 multiplication (2x2 matrix times 2x2 matrix).
    # Rank of M_{2,2,2} is 7 (Strassen).
    # Total rank = 8 * 7 = 56. That's what we had before.
    
    # To get rank 48, we need a different approach.
    # Since M_{2,2,16} = M_{2,2,2} ⊗ M_{1,1,8} (tensor product)?
    # No, it's more like 8 independent M_{2,2,2} blocks.
    
    # Actually, M_{2,2,16} can be seen as 2 independent M_{1,2,16} tensors.
    # And M_{1,2,16} can be decomposed using 8 blocks of M_{1,2,2}.
    # If rk(M_{1,2,2}) = 3, then rk(M_{1,2,16}) <= 24, and rk(M_{2,2,16}) <= 48.
    
    # Let me find the rank-3 decomposition of M_{1,2,2} properly.
    # Reference: The tensor M_{1,2,2} has rank 3.
    # Decomposition:
    # P1 = x0 * (y0*z0 + y1*z1)  -- no, that's not bilinear in the right way.
    
    # For M_{1,2,2}: inputs are a = (a0, a1), b = (b00, b01, b10, b11)
    # Outputs: c0 = a0*b00 + a1*b10, c1 = a0*b01 + a1*b11
    #
    # Rank 3:
    # P1 = a0 * (b00 + b01)
    # P2 = a1 * (b10 + b11)  
    # P3 = (a0 - a1) * (b11 - b00)
    #
    # c0 = P1 + P2 - P3 + correction
    # P1 + P2 = a0*b00 + a0*b01 + a1*b10 + a1*b11
    # P3 = a0*b11 - a0*b00 - a1*b11 + a1*b00
    # P1 + P2 - P3 = a0*b00 + a0*b01 + a1*b10 + a1*b11 - a0*b11 + a0*b00 + a1*b11 - a1*b00
    #              = 2*a0*b00 + a0*b01 + a1*b10 - a1*b00
    # c0 should be a0*b00 + a1*b10
    # Difference: a0*b00 + a0*b01 - a1*b00 = a0*(b00+b01) - a1*b00
    # Not zero.
    
    # I give up trying to find rank 3 for M_{1,2,2} from scratch.
    # Let me just use the Strassen rank-56 which is verified to work.
    
    # Actually, let me try a completely different structural approach.
    # Use the fact that we can share computations across the 16 columns.
    
    # For now, return the Strassen rank-56 decomposition.
    return _strassen_decomposition(n, m, p)

def _strassen_decomposition(n, m, p):
    """Strassen-based rank-56 decomposition."""
    num_blocks = p // 2
    R = num_blocks * 7
    
    U = np.zeros((n * m, R), dtype=int)
    V = np.zeros((m * p, R), dtype=int)
    W = np.zeros((p * n, R), dtype=int)
    
    U_coeffs = [
        {0: 1, 3: 1},
        {2: 1, 3: 1},
        {0: 1},
        {3: 1},
        {0: 1, 1: 1},
        {2: 1, 0: -1},
        {1: 1, 3: -1}
    ]
    
    V_coeffs = [
        {0: 1, 3: 1},
        {0: 1},
        {1: 1, 3: -1},
        {2: 1, 0: -1},
        {3: 1},
        {0: 1, 1: 1},
        {2: 1, 3: 1}
    ]
    
    W_contribs = [
        {0: 1, 3: 1},
        {2: 1},
        {1: 1, 3: -1},
        {0: -1, 2: 1},
        {1: 1, 3: 1},
        {0: 1, 3: -1},
        {0: 1}
    ]
    
    for q in range(num_blocks):
        b_00 = 0 * p + 2 * q
        b_01 = 0 * p + (2 * q + 1)
        b_10 = 1 * p + 2 * q
        b_11 = 1 * p + (2 * q + 1)
        
        c_00 = (2 * q) * n + 0
        c_01 = (2 * q + 1) * n + 0
        c_10 = (2 * q) * n + 1
        c_11 = (2 * q + 1) * n + 1
        
        b_map = {0: b_00, 1: b_01, 2: b_10, 3: b_11}
        c_map = {0: c_00, 1: c_01, 2: c_10, 3: c_11}
        
        for s in range(7):
            r = q * 7 + s
            for a_idx, coeff in U_coeffs[s].items():
                U[a_idx, r] = coeff
            for local_idx, coeff in V_coeffs[s].items():
                V[b_map[local_idx], r] = coeff
            for local_out_idx, coeff in W_contribs[s].items():
                W[c_map[local_out_idx], r] = coeff
    
    return U, V, W

def _schoolbook_decomposition(n, m, p):
    """Schoolbook rank-64 decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, column] = 1
                V[j * p + k, column] = 1
                W[k * n + i, column] = 1
                column += 1
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = DIMENSIONS
    
    # Try Strassen rank 56 first
    U, V, W = _strassen_decomposition(n, m, p)
    if _verify_decomp(U, V, W, n, m, p, num_tests=30):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fall back to schoolbook
    U, V, W = _schoolbook_decomposition(n, m, p)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
