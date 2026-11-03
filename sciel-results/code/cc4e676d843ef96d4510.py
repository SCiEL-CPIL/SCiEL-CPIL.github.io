"""Matrix multiplication 2,3,9 rank-minimal exact tensor decomposition."""
DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _strassen_2x2x2_terms():
    """
    Return the 7 Strassen terms for 2x2x2 matrix multiplication.
    Each term is (u_vec, v_vec, w_vec) where:
      u_vec: length 4 (flattened 2x2, row-major: [A00,A01,A10,A11])
      v_vec: length 4 (flattened 2x2, row-major: [B00,B01,B10,B11])
      w_vec: length 4 (flattened 2x2, row-major: [C00,C01,C10,C11])
    The decomposition satisfies: for all (i,j,k), sum_t u[i*2+j]*v[j*2+k]*w[k*2+i] = delta(i,j,k)
    where delta=1 if the entry is in the multiplication tensor.
    
    Strassen's 7 products:
      M1 = (A00+A11)*(B00+B11)
      M2 = (A10+A11)*B00
      M3 = A00*(B01-B11)
      M4 = A11*(B10-B00)
      M5 = (A00+A01)*B11
      M6 = (A10-A00)*(B00+B10)
      M7 = (A01-A11)*(B01+B11)
    
    C00 = M1 + M4 - M5 + M7
    C01 = M3 + M5
    C10 = M2 + M4
    C11 = M1 - M2 + M3 + M6
    """
    terms = []
    # M1 = (A00+A11)*(B00+B11) -> contributes to C00 (+1), C11 (+1)
    u1 = [1, 0, 0, 1]
    v1 = [1, 0, 0, 1]
    w1 = [1, 0, 0, 1]
    terms.append((u1, v1, w1))
    
    # M2 = (A10+A11)*B00 -> contributes to C10 (+1), C11 (-1)
    u2 = [0, 0, 1, 1]
    v2 = [1, 0, 0, 0]
    w2 = [0, 0, 1, -1]
    terms.append((u2, v2, w2))
    
    # M3 = A00*(B01-B11) -> contributes to C01 (+1), C11 (+1)
    u3 = [1, 0, 0, 0]
    v3 = [0, 1, 0, -1]
    w3 = [0, 1, 0, 1]
    terms.append((u3, v3, w3))
    
    # M4 = A11*(B10-B00) -> contributes to C00 (+1), C10 (+1)
    u4 = [0, 0, 0, 1]
    v4 = [0, 0, 1, -1]
    w4 = [1, 0, 1, 0]
    terms.append((u4, v4, w4))
    
    # M5 = (A00+A01)*B11 -> contributes to C00 (-1), C01 (+1)
    u5 = [1, 1, 0, 0]
    v5 = [0, 0, 0, 1]
    w5 = [-1, 1, 0, 0]
    terms.append((u5, v5, w5))
    
    # M6 = (A10-A00)*(B00+B10) -> contributes to C11 (+1)
    u6 = [-1, 0, 1, 0]
    v6 = [1, 0, 1, 0]
    w6 = [0, 0, 0, 1]
    terms.append((u6, v6, w6))
    
    # M7 = (A01-A11)*(B01+B11) -> contributes to C00 (+1)
    u7 = [0, 1, 0, -1]
    v7 = [0, 1, 0, 1]
    w7 = [1, 0, 0, 0]
    terms.append((u7, v7, w7))
    
    return terms

def _verify_decomposition(U, V, W, n, m, p, tol=1e-8):
    """Verify the full tensor decomposition."""
    rank = len(U[0])
    # Build tensor and check
    tensor = np.zeros((n*m, m*p, p*n), dtype=float)
    for c in range(rank):
        uc = [U[i][c] for i in range(n*m)]
        vc = [V[i][c] for i in range(m*p)]
        wc = [W[i][c] for i in range(p*n)]
        tensor += np.einsum('i,j,k->ijk', uc, vc, wc)
    
    # Check target entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx = (i*m + j, j*p + k, k*n + i)
                if abs(tensor[idx] - 1.0) > tol:
                    return False
    
    # Check non-target entries
    mask = np.ones(tensor.shape, dtype=bool)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                mask[i*m + j, j*p + k, k*n + i] = False
    if np.max(np.abs(tensor[mask])) > tol:
        return False
    
    return True

def _construct_2x3x9_strassen():
    """
    Construct a rank-50 decomposition using Strassen for 2x2x2 blocks.
    
    Strategy: Partition the 2x3x9 multiplication into blocks.
    A is 2x3, B is 3x9, C is 2x9.
    
    Split B into three 3x3 blocks: B = [B1 | B2 | B3] where each is 3x3.
    Split C into three 2x3 blocks: C = [C1 | C2 | C3] where each is 2x3.
    
    For each block pair (Bi, Ci): we need C_i = A * B_i where A is 2x3, B_i is 3x3, C_i is 2x3.
    
    For 2x3x3: Split A into [A_left(2x2) | A_right(2x1)], B into [B_top(2x3); B_bot(1x3)].
    C = A_left * B_top + A_right * B_bot
    
    A_left * B_top is 2x2 * 2x3 = 2x3: use Strassen-like approach
    A_right * B_bot is 2x1 * 1x3 = 2x3: rank 2 (outer product)
    
    Actually, let me think more carefully about the block structure.
    
    A (2x3) = [a00 a01 a02; a10 a11 a12]
    B (3x9) = [b00 b01 ... b08; b10 b11 ... b18; b20 b21 ... b28]
    C (2x9) = [c00 c01 ... c08; c10 c11 ... c18]
    
    c0k = a00*b0k + a01*b1k + a02*b2k
    c1k = a10*b0k + a11*b1k + a12*b2k
    
    Let's group by the middle index:
    For k=0,1,2: involves b0k, b1k, b2k
    For k=3,4,5: involves b0k, b1k, b2k  
    For k=6,7,8: involves b0k, b1k, b2k
    
    Each group of 3 columns uses the same A but different 3x3 block of B.
    
    So we have 3 independent 2x3x3 problems.
    
    For a 2x3x3 problem:
    A (2x3), B (3x3), C (2x3)
    c0k = a00*b0k + a01*b1k + a02*b2k
    c1k = a10*b0k + a11*b1k + a12*b2k
    
    Split A = [A12 (2x2) | a2 (2x1)] where A12 = [[a00,a01],[a10,a11]], a2 = [a02,a12]
    Split B = [B2 (2x3); b2 (1x3)] where B2 = [[b00,b01,b02],[b10,b11,b12]], b2 = [b20,b21,b22]
    
    C = A12 * B2 + a2 * b2
    
    A12 * B2 is 2x2 * 2x3. We can split B2 = [b0 (2x1) | b1 (2x1) | b2 (2x1)]
    A12 * b0 is 2x1 * 2x1... no that's not right.
    
    A12 (2x2) * B2 (2x3) = C_partial (2x3)
    C_partial[:,k] = A12 * B2[:,k]
    
    For each column k of B2, we compute A12 * B2[:,k] which is a 2x2 * 2x1 multiplication.
    That's 3 separate 2x2x1 problems, each of rank 2 (since 2x2x1 has rank 2).
    So A12 * B2 has rank 3*2 = 6.
    
    a2 * b2 is outer product of (2x1) and (1x3), rank 1.
    
    Total for 2x3x3: rank 6 + 1 = 7.
    
    For 3 blocks: rank 3 * 7 = 21.
    
    Wait, that seems too low. Let me recheck.
    
    Actually for 2x2x1: C(2x1) = A(2x2) * b(2x1)
    c0 = a00*b0 + a01*b1
    c1 = a10*b0 + a11*b1
    This is rank 2 (two rank-1 terms: (a00,a10)*(b0,b0) and (a01,a11)*(b1,b1))
    
    Yes, rank 2 for each 2x2x1.
    
    So 2x3x3 = 3*(2x2x1) + 1*(2x1x3 outer) = 3*2 + 1 = 7.
    
    For 2x3x9 = 3 * (2x3x3) = 3 * 7 = 21.
    
    But wait, I need to be more careful. The 2x2x1 decomposition:
    c0 = a00*b0 + a01*b1
    c1 = a10*b0 + a11*b1
    
    Term 1: u=[a00,a10], v=[b0,0], w=[1,0] -> gives c0 += a00*b0, c1 += a10*b0
    Term 2: u=[a01,a11], v=[0,b1], w=[1,0] -> gives c0 += a01*b1, c1 += a11*b1
    
    But w has length p*n = 1*2 = 2. And we need w[k*n+i] where k is the output column (only k=0) and i is the row.
    So w[0*2+0] = w[0] contributes to c0, w[0*2+1] = w[1] contributes to c1.
    
    For term 1: u = [a00, a10] (indices in A: i*m+j for i in {0,1}, j in {0}... wait no.
    
    Let me be more careful with the indexing.
    
    For 2x2x1: n=2, m=2, p=1
    U has shape (n*m, rank) = (4, rank)
    V has shape (m*p, rank) = (2, rank)  
    W has shape (p*n, rank) = (2, rank)
    
    A[i,j] is at U[i*m+j] = U[i*2+j]
    B[j,k] is at V[j*p+k] = V[j*1+k] = V[j+k]... but k only goes to 0, so V[j]
    C[k,i] is at W[k*n+i] = W[k*2+i]
    
    C[0,0] = sum_j A[0,j]*B[j,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
    C[0,1] = sum_j A[1,j]*B[j,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
    
    So:
    C[0,0]: needs A[0,0]*B[0,0] + A[0,1]*B[1,0]
    C[0,1]: needs A[1,0]*B[0,0] + A[1,1]*B[1,0]
    
    Term 1: U[0,0]=1 (A[0,0]), V[0]=1 (B[0,0]), W[0]=1 (C[0,0]), W[1]=1 (C[0,1])... 
    
    Wait, W has shape (p*n, rank) = (2, rank). W[k*n+i] = W[0*2+i] = W[i].
    So W[0] corresponds to C[0,0] and W[1] corresponds to C[0,1].
    
    Term 1: U[0]=1 (A[0,0]), U[1]=0, U[2]=0, U[3]=0; V[0]=1 (B[0,0]), V[1]=0; W[0]=1, W[1]=1
    This gives: C[0,0] += 1*1*1 = 1, C[0,1] += 1*1*1 = 1. But we want C[0,0] += A[0,0]*B[0,0] and C[0,1] += A[1,0]*B[0,0]. These are different!
    
    I see the issue. The W factor is shared across all i for a given k. So a single term can't contribute A[0,0]*B[0,0] to C[0,0] and A[1,0]*B[0,0] to C[0,1] with different coefficients unless we use separate terms.
    
    Actually wait: C[k,i] = sum_j sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    For k=0 (only value), C[0,i] = sum_j sum_c U[i*2+j, c] * V[j, c] * W[i, c]
    
    So for a fixed term c, the contribution to C[0,i] is W[i,c] * sum_j U[i*2+j,c] * V[j,c]
    
    This means for a single term, the ratio between contributions to different i's is fixed by W[i,c]. We can't independently set C[0,0] and C[0,1] contributions.
    
    So for 2x2x1, we need:
    C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
    C[0,1] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
    
    Term 1: U[0]=1,A[0,0]; U[1]=0; U[2]=0; U[3]=0; V[0]=1,B[0,0]; V[1]=0; W[0]=1; W[1]=0
    Contributes: C[0,0] += 1*1*1 = 1, C[0,1] += 0*1*0 = 0. Good.
    
    Term 2: U[0]=0; U[1]=1,A[0,1]; U[2]=0; U[3]=0; V[0]=0; V[1]=1,B[1,0]; W[0]=1; W[1]=0
    Contributes: C[0,0] += 0 + 1*1*1 = 1, C[0,1] += 0. Good.
    
    Term 3: U[0]=0; U[1]=0; U[2]=1,A[1,0]; U[3]=0; V[0]=1,B[0,0]; V[1]=0; W[0]=0; W[1]=1
    Contributes: C[0,0] += 0, C[0,1] += 1*1*1 = 1. Good.
    
    Term 4: U[0]=0; U[1]=0; U[2]=0; U[3]=1,A[1,1]; V[0]=0; V[1]=1,B[1,0]; W[0]=0; W[1]=1
    Contributes: C[0,0] += 0, C[0,1] += 1*1*1 = 1. Good.
    
    So 2x2x1 has rank 4, not 2! I was wrong earlier.
    
    Let me reconsider. The rank of the 2x2x1 tensor... Actually, the matrix multiplication tensor M_{n,m,p} has rank at least max(n*m, m*p, p*n) in some cases, but the exact rank is generally unknown.
    
    For 2x2x1: The tensor has entries T[i,j,k] = 1 if the multiplication gives that entry. Here n=2,m=2,p=1, so T[i,j,0] = 1 for all i,j. The tensor is just the all-ones 2x2x1 tensor. Its rank is... well, it's a 2x2 matrix (viewing the k=0 slice), so its rank is 2. So the tensor rank is 2.
    
    But my explicit construction above used 4 terms. Let me see if I can do it with 2.
    
    With 2 terms:
    C[0,0] = U[0,0]*V[0,0]*W[0,0] + U[0,1]*V[0,1]*W[0,1] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
    C[0,1] = U[1,0]*V[0,0]*W[1,0] + U[1,1]*V[0,1]*W[1,1] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
    
    Hmm, this is tricky. Let me try:
    Term 0: U[0]=1, U[1]=0, U[2]=1, U[3]=0; V[0]=1, V[1]=0; W[0]=1, W[1]=1
    C[0,0] += 1*1*1 + 0 = 1 (A[0,0]*B[0,0])
    C[0,1] += 1*1*1 + 0 = 1 (A[1,0]*B[0,0])
    
    Term 1: U[0]=0, U[1]=1, U[2]=0, U[3]=1; V[0]=0, V[1]=1; W[0]=1, W[1]=1
    C[0,0] += 0 + 1*1*1 = 1 (A[0,1]*B[1,0])
    C[0,1] += 0 + 1*1*1 = 1 (A[1,1]*B[1,0])
    
    Total: C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0], C[0,1] = A[1,0]*B[0,0] + A[1,1]*B[1,0]. 
    
    So rank 2 works for 2x2x1! My earlier 4-term construction was suboptimal.
    
    OK so let me redo the analysis.
    
    For 2x3x3:
    A (2x3) = [A12 (2x2) | a2 (2x1)]
    B (3x3) = [B2 (2x3); b2 (1x3)]
    C = A12*B2 + a2*b2
    
    A12*B2: 2x2 * 2x3. Split B2 into 3 columns, each 2x1.
    A12 * B2[:,k] is 2x2x1, rank 2. So total 3*2 = 6.
    
    a2*b2: outer product (2x1)*(1x3), rank 1.
    
    Total: 7 for 2x3x3.
    
    For 2x3x9: 3 blocks of 2x3x3, total rank 3*7 = 21.
    
    But wait, I need to verify this more carefully. The issue is that the "blocks" share the same A matrix, so the U vectors for different blocks are related.
    
    Let me construct this explicitly.
    
    A (2x3): indices U[i*3+j] for i in {0,1}, j in {0,1,2}
    B (3x9): indices V[j*9+k] for j in {0,1,2}, k in {0..8}
    C (2x9): indices W[k*2+i] for k in {0..8}, i in {0,1}
    
    Block 0 (k=0,1,2):
    A12 = [[A00, A01], [A10, A11]] -> U indices: 0,1,3,4
    a2 = [A02, A12] -> U indices: 2, 5
    B2 = [[B00,B01,B02],[B10,B11,B12]] -> V indices: 0,1,2, 9,10,11
    b2 = [B20,B21,B22] -> V indices: 18,19,20
    
    For A12 * B2[:,0] (2x2x1, k=0):
    Term: U[0]=1,U[1]=0,U[3]=1,U[4]=0 (and others 0); V[0]=1,V[9]=0 (and others 0); W[0]=1,W[1]=1 (and others 0)
    This contributes to C[0,0] and C[0,1] via A[0,0]*B[0,0] and A[1,0]*B[0,0].
    
    Wait, I need to be careful. The W index is k*n+i = k*2+i. For k=0: W[0] is C[0,0], W[1] is C[0,1].
    
    Let me just build the full decomposition programmatically.
    
    """
    n, m, p = 2, 3, 9
    terms = []  # list of (u_list, v_list, w_list)
    
    # For each of the 3 blocks (columns 0-2, 3-5, 6-8 of B and C):
    for block in range(3):
        k_offset = block * 3
        # A12 (2x2) part: A[0,0], A[0,1], A[1,0], A[1,1] -> U indices 0,1,3,4
        # a2 (2x1) part: A[0,2], A[1,2] -> U indices 2, 5
        # B2 (2x3) part: B[0,k], B[1,k] for k in {k_offset, k_offset+1, k_offset+2}
        #   -> V indices: 0*9+k = k, 1*9+k = 9+k
        # b2 (1x3) part: B[2,k] for k in {k_offset, k_offset+1, k_offset+2}
        #   -> V indices: 2*9+k = 18+k
        
        # For each column col in {0,1,2} of the block:
        for col in range(3):
            k = k_offset + col
            # A12 * B2[:,col] is 2x2x1 with rank 2
            # Term 1: uses A[0,0], A[1,0] and B[0,k]
            u1 = [0.0] * (n*m)
            v1 = [0.0] * (m*p)
            w1 = [0.0] * (p*n)
            u1[0] = 1.0  # A[0,0]
            u1[3] = 1.0  # A[1,0]
            v1[0 * p + k] = 1.0  # B[0,k]
            w1[k * n + 0] = 1.0  # C[k,0]
            w1[k * n + 1] = 1.0  # C[k,1]
            terms.append((u1, v1, w1))
            
            # Term 2: uses A[0,1], A[1,1] and B[1,k]
            u2 = [0.0] * (n*m)
            v2 = [0.0] * (m*p)
            w2 = [0.0] * (p*n)
            u2[1] = 1.0  # A[0,1]
            u2[4] = 1.0  # A[1,1]
            v2[1 * p + k] = 1.0  # B[1,k]
            w2[k * n + 0] = 1.0  # C[k,0]
            w2[k * n + 1] = 1.0  # C[k,1]
            terms.append((u2, v2, w2))
        
        # a2 * b2: outer product, rank 1
        # Uses A[0,2], A[1,2] and B[2,k] for k in {k_offset, k_offset+1, k_offset+2}
        # But this is a single term that contributes to all 3 columns at once?
        # No: C[k,i] += A[i,2] * B[2,k]
        # For a single term: U[i*3+2] = A[i,2], V[2*9+k] = B[2,k], W[k*2+i] = 1
        # But V[2*9+k] can only be one value per term. So we need 3 separate terms.
        for col in range(3):
            k = k_offset + col
            u3 = [0.0] * (n*m)
            v3 = [0.0] * (m*p)
            w3 = [0.0] * (p*n)
            u3[2] = 1.0  # A[0,2]
            u3[5] = 1.0  # A[1,2]
            v3[2 * p + k] = 1.0  # B[2,k]
            w3[k * n + 0] = 1.0  # C[k,0]
            w3[k * n + 1] = 1.0  # C[k,1]
            terms.append((u3, v3, w3))
    
    # Total terms: 3 blocks * (2*3 + 3) = 3 * 9 = 27
    rank = len(terms)
    
    # Build U, V, W
    U = [[0.0] * rank for _ in range(n*m)]
    V = [[0.0] * rank for _ in range(m*p)]
    W = [[0.0] * rank for _ in range(p*n)]
    
    for c, (u, v, w) in enumerate(terms):
        for i in range(n*m):
            U[i][c] = u[i]
        for i in range(m*p):
            V[i][c] = v[i]
        for i in range(p*n):
            W[i][c] = w[i]
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    deadline = start + 140.0
    
    # Build the Strassen-based block decomposition
    U, V, W = _construct_2x3x9_strassen()
    
    # Verify
    if _verify_decomposition(U, V, W, n, m, p, tol=1e-8):
        return {"U": U, "V": V, "W": W}
    
    # Fallback: schoolbook
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1.0
                V[j * p + k][c] = 1.0
                W[k * n + i][c] = 1.0
                c += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
