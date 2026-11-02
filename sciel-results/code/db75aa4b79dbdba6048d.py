import numpy as np
import time

DIMENSIONS = [2, 2, 10]

def construct():
    start_time = time.time()
    deadline = start_time + 110

    n, m, p = DIMENSIONS  # n=2, m=2, p=10

    # The multiplication tensor T[i,j,k] for C[i,k] = sum_j A[i,j] B[j,k]
    # We need to decompose: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    #
    # For 2x2x2, Strassen gives rank 7.
    # For 2x2x10, we can use 5 independent 2x2x2 blocks (each handling 2 columns of B and C),
    # giving rank 35. But we might do better by sharing structure.
    #
    # The known minimal rank for 2x2x10 is not well-established, but we know:
    # - Schoolbook: 2*2*10 = 40
    # - 5x Strassen: 5*7 = 35
    # - Potentially lower with shared structure
    #
    # Let's start with the 5x Strassen approach (rank 35) and see if we can improve.
    #
    # Strassen's 7 terms for 2x2x2:
    # P1 = (A00+A11)(B00+B11)
    # P2 = (A10+A11)B00
    # P3 = A00(B10-B11)
    # P4 = A11(B01-B00)
    # P5 = (A00+A01)B11
    # P6 = (A10+A11)(B01-B00)
    # P7 = (A01-A11)(B00+B10)
    #
    # C00 = P1 - P4 + P6 + P7
    # C10 = P3 + P4 - P5 + P7
    # C01 = P2 + P5 - P6 - P7
    # C11 = P1 + P2 - P3 + P5
    #
    # In tensor form, for each term r, we have:
    # U[i,j,r], V[j,k,r], W[k,i,r] such that sum_r U[i,j,r] V[j,k,r] W[k,i,r] = delta (the multiplication tensor)
    #
    # The multiplication tensor for 2x2x2 is:
    # T[i,j,k] = 1 if the term A[i,j]*B[j,k] contributes to C[i,k]
    # Actually, T[i,j,k] = 1 for all i,j,k in {0,1} because each A[i,j]*B[j,k] contributes to C[i,k]
    # Wait, no. The tensor is defined as T[i,j,k] = 1 if A[i,j] and B[j,k] multiply to give C[i,k].
    # Since C[i,k] = sum_j A[i,j] B[j,k], each term A[i,j]B[j,k] contributes to C[i,k].
    # So T[i,j,k] = 1 for all i,j,k.
    #
    # But that's the identity tensor, not the multiplication tensor.
    # The multiplication tensor is: M[i,j,k] = 1 if the product A[i,j]*B[j,k] appears in C[i,k].
    # Since C[i,k] = sum_j A[i,j] B[j,k], the term A[i,j]B[j,k] appears in C[i,k] for each j.
    # So M[i,j,k] = 1 for all i,j,k.
    #
    # Hmm, that doesn't seem right either. Let me think more carefully.
    #
    # The matrix multiplication tensor is defined as:
    # M = sum_{i,j,k} e_{ij} ⊗ e_{jk} ⊗ e_{ki}
    # where e_{ij} is the basis vector for the (i,j) entry of A, etc.
    #
    # So M[i,j,k] = 1 if we're looking at the term A[i,j]*B[j,k] which contributes to C[i,k].
    # This means M[i,j,k] = 1 for all i,j,k in the valid ranges.
    #
    # For 2x2x2: M[i,j,k] = 1 for i,j,k in {0,1}, so 8 entries all equal to 1.
    #
    # The rank of this tensor is 7 (Strassen).
    #
    # For 2x2x10: M[i,j,k] = 1 for i in {0,1}, j in {0,1}, k in {0,...,9}, so 40 entries all equal to 1.
    #
    # We can decompose this as 5 independent 2x2x2 tensors, each with rank 7, giving total rank 35.
    #
    # Can we do better? The structure is that for each k, the "slice" M[:,:,k] is a 2x2x2 tensor with all 1s.
    # But these slices are not independent in the tensor decomposition sense because the U, V, W factors
    # must be consistent across all k.
    #
    # Let's try the 5x Strassen approach first.

    # Strassen terms for 2x2x2
    # For each term r (0..6), we define U[i,j,r], V[j,k,r], W[k,i,r]
    # such that sum_r U[i,j,r] V[j,k,r] W[k,i,r] = 1 for all i,j,k in {0,1}
    
    # Using the standard Strassen decomposition:
    # P1 = (A00+A11)(B00+B11) -> U: A00+A11, V: B00+B11, W: 1
    # P2 = (A10+A11)B00       -> U: A10+A11, V: B00, W: 1
    # P3 = A00(B10-B11)       -> U: A00, V: B10-B11, W: 1
    # P4 = A11(B01-B00)       -> U: A11, V: B01-B00, W: 1
    # P5 = (A00+A01)B11       -> U: A00+A01, V: B11, W: 1
    # P6 = (A10+A11)(B01-B00) -> U: A10+A11, V: B01-B00, W: 1
    # P7 = (A01-A11)(B00+B10) -> U: A01-A11, V: B00+B10, W: 1
    
    # But we need the W factors to reconstruct C correctly.
    # C00 = P1 - P4 + P6 + P7
    # C10 = P3 + P4 - P5 + P7
    # C01 = P2 + P5 - P6 - P7
    # C11 = P1 + P2 - P3 + P5
    
    # So W[k,i,r] for each term r:
    # Term 0 (P1): W[0,0]=1, W[0,1]=0, W[1,0]=0, W[1,1]=1  -> C00 and C11
    # Term 1 (P2): W[0,0]=0, W[0,1]=1, W[1,0]=0, W[1,1]=1  -> C10 and C11
    # Term 2 (P3): W[0,0]=0, W[0,1]=1, W[1,0]=0, W[1,1]=-1 -> C10 and -C11
    # Term 3 (P4): W[0,0]=-1, W[0,1]=1, W[1,0]=0, W[1,1]=0 -> -C00 and C10
    # Term 4 (P5): W[0,0]=0, W[0,1]=-1, W[1,0]=1, W[1,1]=1 -> -C10, C01, C11
    # Term 5 (P6): W[0,0]=1, W[0,1]=0, W[1,0]=-1, W[1,1]=0 -> C00, -C01
    # Term 6 (P7): W[0,0]=1, W[0,1]=1, W[1,0]=-1, W[1,1]=-1 -> C00, C10, -C01, -C11
    
    # Let me verify:
    # C00 = P1 - P4 + P6 + P7 = 1*(-1) + 1*(1) + 1*(1) = 1-1+1+1 = 2? No, that's not right.
    # W[0,0,0]=1 (P1), W[0,0,3]=-1 (P4), W[0,0,5]=1 (P6), W[0,0,6]=1 (P7)
    # C00 = 1*1 + (-1)*(-1) + 1*1 + 1*1 = 1+1+1+1 = 4? Still not right.
    #
    # I think I'm confusing the W factors. Let me re-derive.
    #
    # The tensor decomposition is: M[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
    # For the multiplication tensor, M[i,j,k] = 1 for all i,j,k.
    #
    # So we need: sum_r U[i,j,r] V[j,k,r] W[k,i,r] = 1 for all i,j,k.
    #
    # This is a different constraint than what I was using. Let me look up the correct Strassen tensor decomposition.
    #
    # Actually, the standard Strassen algorithm computes C = A*B using 7 scalar multiplications.
    # The tensor decomposition of the multiplication tensor is related but not identical.
    #
    # Let me just use the known rank-35 decomposition (5x Strassen) and verify it works.
    
    # For 2x2x2, the multiplication tensor has entries M[i,j,k] = 1 for all i,j,k in {0,1}.
    # We need U, V, W of shape (4,7), (4,7), (4,7) such that:
    # sum_{r=0}^{6} U[i*2+j, r] * V[j*2+k, r] * W[k*2+i, r] = 1 for all i,j,k in {0,1}
    
    # Let me try to find such a decomposition.
    # Actually, the multiplication tensor for n x m x p has rank at most n*m*p.
    # For 2x2x2, the rank is 7 (Strassen).
    
    # Here's a known rank-7 decomposition for 2x2x2:
    # Using the formulas from the literature.
    
    # Let me try a different approach: use the fact that for 2x2x10, we can treat it as
    # 5 independent 2x2x2 problems if we can find a rank-7 decomposition for 2x2x2.
    
    # I'll use the following known decomposition for 2x2x2:
    # Reference: Various sources on tensor rank of matrix multiplication
    
    # The 7 terms (U, V, W) for 2x2x2:
    # U[i,j,r] is a 2x2 matrix for each r, flattened to 4 elements
    # V[j,k,r] is a 2x2 matrix for each r, flattened to 4 elements  
    # W[k,i,r] is a 2x2 matrix for each r, flattened to 4 elements
    
    # I'll use the decomposition from the "Strassen" style:
    # For each r in 0..6, define u_r, v_r, w_r as 2x2 matrices
    
    # Let me just hardcode a known working decomposition.
    # After some research, here's a rank-7 decomposition for 2x2x2:
    
    # U matrices (2x2 each):
    # U0 = [[1,0],[0,1]]
    # U1 = [[0,0],[1,1]]
    # U2 = [[1,0],[0,0]]
    # U3 = [[0,0],[0,1]]
    # U4 = [[1,1],[0,0]]
    # U5 = [[0,1],[0,-1]]
    # U6 = [[0,1],[0,-1]]  # Wait, this might not be right
    
    # Let me try a different approach. I'll use the fact that the current program works
    # with rank 35 and just clean it up, then try to see if we can do better.
    
    # Actually, let me just use the existing working decomposition from the current program
    # and see if there's a way to reduce the rank.
    
    # The current program uses:
    # terms = [
    #     ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
    #     ([0, 0, 1, 1], [1, 0, 0, 0], [0, 1, 0, -1]),
    #     ([1, 0, 0, 0], [0, 1, 0, -1], [0, 0, 1, 1]),
    #     ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 1, 0, 0]),
    #     ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 0, 1, 0]),
    #     ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
    #     ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0])
    # ]
    
    # Let me verify this decomposition for 2x2x2 first.
    
    terms_2x2 = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 1, 0, -1]),
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 0, 1, 1]),
        ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 1, 0, 0]),
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 0, 1, 0]),
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0])
    ]
    
    # Verify for 2x2x2: check that sum_r U[i*2+j,r] V[j*2+k,r] W[k*2+i,r] = 1 for all i,j,k
    def verify_2x2(terms):
        n2, m2, p2 = 2, 2, 2
        for i in range(n2):
            for j in range(m2):
                for k in range(p2):
                    total = 0.0
                    for r in range(len(terms)):
                        a_vec, b_vec, c_vec = terms[r]
                        u_val = a_vec[i*m2+j]
                        v_val = b_vec[j*p2+k]
                        w_val = c_vec[k*n2+i]
                        total += u_val * v_val * w_val
                    if abs(total - 1.0) > 1e-10:
                        return False, (i, j, k, total)
        return True, None
    
    ok, info = verify_2x2(terms_2x2)
    if not ok:
        # If the 2x2 decomposition doesn't work, fall back to schoolbook
        R = n * m * p
        U = np.zeros((n*m, R), dtype=float)
        V = np.zeros((m*p, R), dtype=float)
        W = np.zeros((p*n, R), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1.0
                    V[j*p + k, col] = 1.0
                    W[k*n + i, col] = 1.0
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Now use 5 blocks of 2x2x2 for 2x2x10
    num_blocks = p // 2  # 5 blocks
    R = 7 * num_blocks  # 35
    
    U = np.zeros((n*m, R), dtype=float)
    V = np.zeros((m*p, R), dtype=float)
    W = np.zeros((p*n, R), dtype=float)
    
    col = 0
    for q in range(num_blocks):
        for a_vec, b_vec, c_vec in terms_2x2:
            # U: index i*m + j
            for idx in range(4):
                i = idx // 2
                j = idx % 2
                U[i*m + j, col] = a_vec[idx]
            
            # V: index j*p + k, for block q, k = 2q, 2q+1
            V[0*p + 2*q, col] = b_vec[0]
            V[0*p + 2*q+1, col] = b_vec[1]
            V[1*p + 2*q, col] = b_vec[2]
            V[1*p + 2*q+1, col] = b_vec[3]
            
            # W: index k*n + i, for block q, k = 2q, 2q+1
            W[4*q + 0, col] = c_vec[0]
            W[4*q + 1, col] = c_vec[1]
            W[4*q + 2, col] = c_vec[2]
            W[4*q + 3, col] = c_vec[3]
            
            col += 1
    
    # Verify the full decomposition
    np.random.seed(12345)
    A = np.random.rand(n, m)
    B = np.random.rand(m, p)
    C_true = A @ B
    
    C_decomp = np.zeros((n, p))
    for i in range(n):
        for k in range(p):
            s = 0.0
            for r in range(R):
                u_val = 0.0
                for j in range(m):
                    u_val += U[i*m + j, r] * A[i, j]
                v_val = 0.0
                for jp in range(m):
                    v_val += V[jp*p + k, r] * B[jp, k]
                s += u_val * v_val * W[k*n + i, r]
            C_decomp[i, k] = s
    
    if not np.allclose(C_true, C_decomp, atol=1e-10):
        # Fallback to schoolbook
        R_school = n * m * p
        U = np.zeros((n*m, R_school), dtype=float)
        V = np.zeros((m*p, R_school), dtype=float)
        W = np.zeros((p*n, R_school), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1.0
                    V[j*p + k, col] = 1.0
                    W[k*n + i, col] = 1.0
                    col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
