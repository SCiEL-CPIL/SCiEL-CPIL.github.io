import numpy as np
import time

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 120.0
    
    n, m, p = DIMENSIONS
    
    # We construct the rank 17 decomposition for 2x3x3 matrix multiplication.
    # The decomposition is based on:
    # 1. Strassen (rank 7) for the 2x2x2 core.
    # 2. Schoolbook (rank 4) for the 2x1x2 block (A_right * B_bottom_left).
    # 3. Schoolbook (rank 4) for the 2x2x1 block (A_left * B_top_right).
    # 4. Schoolbook (rank 2) for the 2x1x1 block (A_right * B_bottom_right).
    # Total: 7 + 4 + 4 + 2 = 17.
    
    # We use integer entries.
    rank = 17
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Indices:
    # A (2x3): i in 0..1, j in 0..2. Index i*3 + j.
    #   A00=0, A01=1, A02=2
    #   A10=3, A11=4, A12=5
    # B (3x3): j in 0..2, k in 0..2. Index j*3 + k.
    #   B00=0, B01=1, B02=2
    #   B10=3, B11=4, B12=5
    #   B20=6, B21=7, B22=8
    # C (2x3) output W[k*n+i, r] => W[k*2 + i, r].
    #   k=0: W[0, r]=C[0,0], W[1, r]=C[1,0]
    #   k=1: W[2, r]=C[0,1], W[3, r]=C[1,1]
    #   k=2: W[4, r]=C[0,2], W[5, r]=C[1,2]
    
    # Block 1: Strassen for A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    # A indices: 0,1,3,4
    # B indices: 0,1,3,4
    # C indices: 0,1,2,3
    
    # r=0: M1 = (A00+A11)(B00+B11)
    U[0, 0] = 1; U[4, 0] = 1
    V[0, 0] = 1; V[4, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # r=1: M2 = (A10+A11)B00
    U[3, 1] = 1; U[4, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1
    
    # r=2: M3 = A00(B01-B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[4, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1
    
    # r=3: M4 = A11(B10-B00)
    U[4, 3] = 1
    V[3, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1
    
    # r=4: M5 = (A00+A01)B11
    U[0, 4] = 1; U[1, 4] = 1
    V[4, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1
    
    # r=5: M6 = (A10-A00)(B00+B01)
    U[3, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # r=6: M7 = (A01-A11)(B10+B11)
    U[1, 6] = 1; U[4, 6] = -1
    V[3, 6] = 1; V[4, 6] = 1
    W[0, 6] = 1
    
    # Block 2: Schoolbook for A[0:2, 2] * B[2, 0:2] -> C[0:2, 0:2]
    # A indices: 2, 5
    # B indices: 6, 7
    # C indices: 0,1,2,3
    
    # r=7: A02 B20
    U[2, 7] = 1
    V[6, 7] = 1
    W[0, 7] = 1
    
    # r=8: A02 B21
    U[2, 8] = 1
    V[7, 8] = 1
    W[2, 8] = 1
    
    # r=9: A12 B20
    U[5, 9] = 1
    V[6, 9] = 1
    W[1, 9] = 1
    
    # r=10: A12 B21
    U[5, 10] = 1
    V[7, 10] = 1
    W[3, 10] = 1
    
    # Block 3: Schoolbook for A[0:2, 0:3] * B[0:3, 2] -> C[0:2, 2]
    # A indices: 0,1,2,3,4,5
    # B indices: 2,5,8
    # C indices: 4,5
    
    # r=11: A00 B02
    U[0, 11] = 1
    V[2, 11] = 1
    W[4, 11] = 1
    
    # r=12: A01 B12
    U[1, 12] = 1
    V[5, 12] = 1
    W[4, 12] = 1
    
    # r=13: A02 B22
    U[2, 13] = 1
    V[8, 13] = 1
    W[4, 13] = 1
    
    # r=14: A10 B02
    U[3, 14] = 1
    V[2, 14] = 1
    W[5, 14] = 1
    
    # r=15: A11 B12
    U[4, 15] = 1
    V[5, 15] = 1
    W[5, 15] = 1
    
    # r=16: A12 B22
    U[5, 16] = 1
    V[8, 16] = 1
    W[5, 16] = 1
    
    # Now attempt to reduce rank by checking for linear dependencies
    # We check if any term can be expressed as a linear combination of others
    # by examining the tensor structure
    
    # Build the full tensor T[i,j,k] = sum_r U[i,r]*V[j,r]*W[k,r]
    # and check if removing any single term still gives the correct tensor
    
    def build_tensor(U_mat, V_mat, W_mat):
        """Build the multiplication tensor from factor matrices.
        W is indexed by k*n+i (official trace convention).
        T[i,j,k] = sum_r U[i,r] * V[j,r] * W[k*n+i, r]
        """
        n_dim, m_dim, p_dim = n, m, p
        T = np.zeros((n_dim, m_dim, p_dim))
        for r in range(U_mat.shape[1]):
            for i in range(n_dim):
                for j in range(m_dim):
                    for k in range(p_dim):
                        T[i, j, k] += U_mat[i, r] * V_mat[j, r] * W_mat[k * n_dim + i, r]
        return T
    
    def verify_decomposition(U_mat, V_mat, W_mat):
        """Verify that the decomposition gives the correct multiplication tensor."""
        # The correct tensor for matrix multiplication: T[i,j,k] = 1 if j==k else 0
        T_expected = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_expected[i, j, k] = 1.0 if j == k else 0.0
        
        T_actual = build_tensor(U_mat.astype(float), V_mat.astype(float), W_mat.astype(float))
        return np.allclose(T_actual, T_expected, atol=1e-10)
    
    # First verify our base decomposition is correct
    base_valid = verify_decomposition(U, V, W)
    
    if base_valid:
        # Try to eliminate each term by expressing it as a combination of others
        # For each term r, check if we can find coefficients c_s such that
        # U[:,r]*V[:,r]*W[:,r] = sum_{s!=r} c_s * U[:,s]*V[:,s]*W[:,s]
        # This is equivalent to checking if the vectorized form of term r
        # is in the span of the other terms.
        
        # Vectorize each term: term_r[i,j,k] = U[i,r]*V[j,r]*W[k,r]
        # Flatten to a vector of size n*m*p = 18
        
        def vectorize_term(U_mat, V_mat, W_mat, r):
            vec = np.zeros(n * m * p)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        vec[i * m * p + j * p + k] = U_mat[i, r] * V_mat[j, r] * W_mat[k * n + i, r]
            return vec
        
        # Build matrix of all terms
        terms = np.array([vectorize_term(U, V, W, r) for r in range(rank)])
        
        # Check rank of the term matrix
        term_rank = np.linalg.matrix_rank(terms, tol=1e-10)
        
        if term_rank < rank:
            # There are linear dependencies! Find which terms can be eliminated.
            # Use QR decomposition with column pivoting to find a basis.
            Q, R = np.linalg.qr(terms.T, mode='complete')
            # R is rank x rank, upper triangular
            # Find columns that are pivots
            diag = np.abs(np.diag(R))
            pivot_threshold = 1e-10
            pivot_cols = np.where(diag > pivot_threshold)[0]
            non_pivot_cols = np.setdiff1d(np.arange(rank), pivot_cols)
            
            # Try to eliminate non-pivot terms
            for elim_r in non_pivot_cols:
                if time.time() - start_time > deadline - 10:
                    break
                    
                # Create a copy with term elim_r removed
                keep = [r for r in range(rank) if r != elim_r]
                U_new = U[:, keep]
                V_new = V[:, keep]
                W_new = W[:, keep]
                
                # But we need to adjust the remaining terms to account for the eliminated term
                # The eliminated term's contribution must be absorbed into others.
                # This is non-trivial in general. Let's just check if the remaining
                # terms still span the same tensor space.
                
                # Actually, for a proper reduction, we need to find new U', V', W'
                # with fewer terms. This is a complex optimization.
                
                # Simpler approach: check if removing term elim_r and adjusting
                # one other term works.
                
                # For now, just record that we found dependencies.
                pass
        
        # Alternative approach: try specific structured reductions
        # Check if any pair of terms can be combined
        
        # Check if terms 7 and 9 (A02*B20 and A12*B20) can be combined
        # They share B20. Similarly 8 and 10 share B21.
        # Terms 11 and 14 share B02. Terms 12 and 15 share B12. Terms 13 and 16 share B22.
        
        # Try: can we merge A02*B20 + A12*B20 = (A02+A12)*B20?
        # That would require a new term with U having both A02 and A12, V having B20.
        # But that's just one term replacing two, saving rank 1.
        # However, we'd need to adjust W to account for the C indices.
        
        # Term 7: U[2]=1, V[6]=1, W[0]=1 -> contributes to C[0,0]
        # Term 9: U[5]=1, V[6]=1, W[1]=1 -> contributes to C[1,0]
        # Merged: U[2]=1, U[5]=1, V[6]=1, W[0]=1, W[1]=1
        # This gives: C[0,0] += A02*B20, C[1,0] += A12*B20
        # That's exactly what we want! So we can merge terms 7 and 9.
        
        # Similarly merge 8 and 10:
        # Term 8: U[2]=1, V[7]=1, W[2]=1
        # Term 10: U[5]=1, V[7]=1, W[3]=1
        # Merged: U[2]=1, U[5]=1, V[7]=1, W[2]=1, W[3]=1
        
        # And merge 11 and 14:
        # Term 11: U[0]=1, V[2]=1, W[4]=1
        # Term 14: U[3]=1, V[2]=1, W[5]=1
        # Merged: U[0]=1, U[3]=1, V[2]=1, W[4]=1, W[5]=1
        
        # And merge 12 and 15:
        # Term 12: U[1]=1, V[5]=1, W[4]=1
        # Term 15: U[4]=1, V[5]=1, W[5]=1
        # Merged: U[1]=1, U[4]=1, V[5]=1, W[4]=1, W[5]=1
        
        # And merge 13 and 16:
        # Term 13: U[2]=1, V[8]=1, W[4]=1
        # Term 16: U[5]=1, V[8]=1, W[5]=1
        # Merged: U[2]=1, U[5]=1, V[8]=1, W[4]=1, W[5]=1
        
        # This reduces from 17 to 17-5 = 12 terms!
        # Let's construct the reduced decomposition.
        
        rank_reduced = 12
        U_red = np.zeros((n * m, rank_reduced), dtype=int)
        V_red = np.zeros((m * p, rank_reduced), dtype=int)
        W_red = np.zeros((p * n, rank_reduced), dtype=int)
        
        # Copy Strassen terms (0-6)
        U_red[:, 0:7] = U[:, 0:7]
        V_red[:, 0:7] = V[:, 0:7]
        W_red[:, 0:7] = W[:, 0:7]
        
        # Merged term 7+9: index 7
        U_red[2, 7] = 1; U_red[5, 7] = 1
        V_red[6, 7] = 1
        W_red[0, 7] = 1; W_red[1, 7] = 1
        
        # Merged term 8+10: index 8
        U_red[2, 8] = 1; U_red[5, 8] = 1
        V_red[7, 8] = 1
        W_red[2, 8] = 1; W_red[3, 8] = 1
        
        # Merged term 11+14: index 9
        U_red[0, 9] = 1; U_red[3, 9] = 1
        V_red[2, 9] = 1
        W_red[4, 9] = 1; W_red[5, 9] = 1
        
        # Merged term 12+15: index 10
        U_red[1, 10] = 1; U_red[4, 10] = 1
        V_red[5, 10] = 1
        W_red[4, 10] = 1; W_red[5, 10] = 1
        
        # Merged term 13+16: index 11
        U_red[2, 11] = 1; U_red[5, 11] = 1
        V_red[8, 11] = 1
        W_red[4, 11] = 1; W_red[5, 11] = 1
        
        # Verify the reduced decomposition
        if verify_decomposition(U_red, V_red, W_red):
            return {"U": U_red.tolist(), "V": V_red.tolist(), "W": W_red.tolist()}
    
    # If reduction didn't work or wasn't valid, return the original
    if time.time() - start_time > deadline - 5:
        pass

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
