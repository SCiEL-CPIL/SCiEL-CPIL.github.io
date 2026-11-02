"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 5, 6
    start_time = time.time()
    deadline = start_time + 140.0

    def schoolbook(n, m, p):
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
        return U, V, W

    def strassen_2x2x2():
        """Strassen's 2x2x2 rank-7 decomposition."""
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # Strassen's algorithm:
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        # C11 = M1+M4-M5+M7
        # C12 = M3+M5
        # C21 = M2+M4
        # C22 = M1-M2+M3+M6
        
        # U: for A (2x2), indexed as A11,A12,A21,A22 -> indices 0,1,2,3
        # V: for B (2x2), indexed as B11,B12,B21,B22 -> indices 0,1,2,3
        # W: for C (2x2), indexed as C11,C12,C21,C22 -> indices 0,1,2,3
        
        # M1: (A11+A22)(B11+B22)
        U[0, 0] = 1; U[3, 0] = 1  # A11, A22
        V[0, 0] = 1; V[3, 0] = 1  # B11, B22
        W[0, 0] = 1  # C11
        W[3, 0] = 1  # C22
        
        # M2: (A21+A22)B11
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1  # C21
        W[3, 1] = -1  # C22
        
        # M3: A11(B12-B22)
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1  # C12
        W[3, 2] = 1  # C22
        
        # M4: A22(B21-B11)
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[0, 3] = 1  # C11
        W[2, 3] = 1  # C21
        
        # M5: (A11+A12)B22
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1  # C11
        W[1, 4] = 1  # C12
        
        # M6: (A21-A11)(B11+B12)
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1  # C22
        
        # M7: (A12-A22)(B21+B22)
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1  # C11
        
        return U, V, W

    def verify(U, V, W, n, m, p):
        """Verify the decomposition."""
        np.random.seed(42)
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        
        C_decomp = np.zeros((n, p))
        for r in range(U.shape[1]):
            M_r = np.zeros(n)
            N_r = np.zeros(p)
            for i in range(n):
                for j in range(m):
                    M_r[i] += U[i*m+j, r] * A[i, j]
            for k in range(p):
                for j in range(m):
                    N_r[k] += V[j*p+k, r] * B[j, k]
            for i in range(n):
                for k in range(p):
                    C_decomp[i, k] += W[k*n+i, r] * M_r[i] * N_r[k]
        
        return np.allclose(C_decomp, C_expected, atol=1e-10)

    # Strategy: Combine Strassen 2x2x2 blocks with schoolbook for remainders.
    # Dimensions: 3x5x6
    # Split 3 into 2+1, 5 into 2+2+1, 6 into 2+2+2.
    
    # We will build the decomposition by tiling.
    # Let's define a helper to add a block decomposition to the global matrices.
    
    # Total rows for U: 3*5 = 15
    # Total rows for V: 5*6 = 30
    # Total rows for W: 6*3 = 18
    
    U_global = np.zeros((15, 100)) # Pre-allocate with extra space, we'll trim
    V_global = np.zeros((30, 100))
    W_global = np.zeros((18, 100))
    
    current_rank = 0
    
    # We iterate over blocks of A (3x5) and B (5x6)
    # A is split into rows: 2, 1
    # B is split into cols: 2, 2, 2
    # Wait, the inner dimension (5) must match.
    # A is 3x5. Split rows of A: 2, 1. Split cols of A (and rows of B): 2, 2, 1.
    # B is 5x6. Split rows of B: 2, 2, 1. Split cols of B: 2, 2, 2.
    
    # So we have blocks:
    # A_ij (i in 0..1, j in 0..2)
    # B_jk (j in 0..2, k in 0..2)
    # C_ik = sum_j A_ij B_jk
    
    # Block sizes:
    # A_row_sizes = [2, 1]
    # A_col_sizes = [2, 2, 1]
    # B_row_sizes = [2, 2, 1] (must match A_col_sizes)
    # B_col_sizes = [2, 2, 2]
    
    A_row_sizes = [2, 1]
    A_col_sizes = [2, 2, 1]
    B_col_sizes = [2, 2, 2]
    
    # Indices
    # A row offsets: 0, 2
    # A col offsets: 0, 2, 4
    # B row offsets: 0, 2, 4
    # B col offsets: 0, 2, 4
    
    a_row_off = [0, 2]
    a_col_off = [0, 2, 4]
    b_row_off = [0, 2, 4]
    b_col_off = [0, 2, 4]
    
    # We need to sum contributions to C_ik.
    # C_ik is 3x6.
    # For each k (output col block), C_ik is sum over j of A_ij B_jk.
    # This is a sum of products of blocks.
    # This structure is exactly what the tensor decomposition handles.
    
    # Instead of summing, we treat each term A_ij B_jk as contributing to the rank.
    # However, the rank is not additive in that simple way for the sum.
    # But we can use the fact that if we have decompositions for each term,
    # we can combine them by concatenating the rank dimensions, provided we map the outputs correctly.
    # Wait, no. The tensor is T(A,B) = sum_k M_k(A) N_k(B) W_k.
    # If we split the sum over j: T = sum_j sum_k ...
    # Actually, the standard tensor decomposition for matrix multiplication is:
    # T = sum_r U_r (A) V_r (B) W_r
    # where U_r, V_r are linear forms on A and B, and W_r is the output matrix.
    # 
    # If we split A and B into blocks, we can view the problem as a sum of smaller matrix multiplications.
    # Let A = [A_1, A_2] (vertical stack? No, A is 3x5. Let's stick to the block partition).
    # A is partitioned into 2x3 blocks (2 row blocks, 3 col blocks).
    # B is partitioned into 3x2 blocks (3 row blocks, 2 col blocks).
    # C = A B is partitioned into 2x2 blocks (2 row blocks, 2 col blocks).
    # C_ik = sum_{j=0}^2 A_ij B_jk.
    #
    # Each term A_ij B_jk is a matrix product of size (n_i x m_j) * (m_j x p_k).
    # We can find a rank-r_ij_k decomposition for this specific product.
    # Let the decomposition be: A_ij B_jk = sum_{r=1}^{r_ij_k} U^{ij}_r(A_ij) V^{jk}_r(B_jk) W^{ijk}_r.
    # Here U^{ij}_r is a linear form on A_ij, V^{jk}_r on B_jk, and W^{ijk}_r is a matrix of size n_i x p_k.
    #
    # Then C_ik = sum_j sum_r U^{ij}_r(A_ij) V^{jk}_r(B_jk) W^{ijk}_r.
    #
    # We can combine all these terms into a single global decomposition.
    # The global linear form on A is: for a given global index (i, j, r), it is U^{ij}_r(A_ij) and 0 elsewhere.
    # Similarly for B.
    # The global output W is W^{ijk}_r placed in the (i, k) block of C.
    #
    # So the total rank is sum_{i,j,k} r_ij_k.
    #
    # We need to choose decompositions for each block product to minimize the sum.
    #
    # Block sizes (n_i, m_j, p_k):
    # i=0: n=2. i=1: n=1.
    # j=0: m=2. j=1: m=2. j=2: m=1.
    # k=0: p=2. k=1: p=2. k=2: p=2.
    
    # We have 2 * 3 * 2 = 12 block products.
    # We need rank for each (n, m, p).
    
    def get_block_decomp(n, m, p):
        """Get a decomposition for n x m x p."""
        if time.time() > deadline:
            return schoolbook(n, m, p)
            
        if n == 1 and m == 1 and p == 1:
            return np.ones((1,1)), np.ones((1,1)), np.ones((1,1))
            
        if n == 1 or m == 1 or p == 1:
            return schoolbook(n, m, p)
            
        if n == 2 and m == 2 and p == 2:
            return strassen_2x2x2()
            
        # Fallback to schoolbook for other small cases
        # We can add more cases if needed, but 2x2x2 is the main win.
        return schoolbook(n, m, p)

    # Accumulate the global decomposition
    U_list = []
    V_list = []
    W_list = []
    
    for i in range(2):
        n_i = A_row_sizes[i]
        a_row_start = a_row_off[i]
        
        for j in range(3):
            m_j = A_col_sizes[j]
            a_col_start = a_col_off[j]
            b_row_start = b_row_off[j]
            
            # Get decomposition for A_ij (n_i x m_j) * B_jk (m_j x p_k)
            # Wait, we need to loop over k too.
            # The decomposition for A_ij B_jk depends on p_k.
            # Actually, for a fixed i, j, we are multiplying A_ij (n_i x m_j) by B_jk (m_j x p_k).
            # The result is added to C_ik.
            # So for each (i, j), we have contributions to all k.
            # It is better to loop k inside.
            
            for k in range(2):
                p_k = B_col_sizes[k]
                b_col_start = b_col_off[k]
                
                # Get decomp for n_i x m_j x p_k
                U_blk, V_blk, W_blk = get_block_decomp(n_i, m_j, p_k)
                
                r_blk = U_blk.shape[1]
                
                # Expand U_blk to global U shape (15, r_blk)
                U_exp = np.zeros((15, r_blk))
                # Map local A_ij indices (n_i x m_j) to global A indices
                for li in range(n_i):
                    for lj in range(m_j):
                        global_i = a_row_start + li
                        global_j = a_col_start + lj
                        U_exp[global_i * 5 + global_j, :] = U_blk[li * m_j + lj, :]
                
                # Expand V_blk to global V shape (30, r_blk)
                V_exp = np.zeros((30, r_blk))
                # Map local B_jk indices (m_j x p_k) to global B indices
                for lj in range(m_j):
                    for lk in range(p_k):
                        global_j = b_row_start + lj
                        global_k = b_col_start + lk
                        V_exp[global_j * 6 + global_k, :] = V_blk[lj * p_k + lk, :]
                
                # Expand W_blk to global W shape (18, r_blk)
                W_exp = np.zeros((18, r_blk))
                # Map local C_ik indices (n_i x p_k) to global C indices
                # W convention: W[k*n + i, r]
                for li in range(n_i):
                    for lk in range(p_k):
                        global_i = a_row_start + li
                        global_k = b_col_start + lk
                        W_exp[global_k * 3 + global_i, :] = W_blk[lk * n_i + li, :]
                
                U_list.append(U_exp)
                V_list.append(V_exp)
                W_list.append(W_exp)
                
                if time.time() > deadline - 10:
                    break
            if time.time() > deadline - 10:
                break
        if time.time() > deadline - 10:
            break
    
    if not U_list:
        # Fallback to schoolbook if we timed out or failed
        U, V, W = schoolbook(3, 5, 6)
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    U_final = np.hstack(U_list)
    V_final = np.hstack(V_list)
    W_final = np.hstack(W_list)
    
    # Verify
    if not verify(U_final, V_final, W_final, 3, 5, 6):
        # Fallback
        U, V, W = schoolbook(3, 5, 6)
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}

# EVOLVE-BLOCK-END
