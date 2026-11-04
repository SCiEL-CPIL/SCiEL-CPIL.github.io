import numpy as np

def construct():
    n, m, p = 4, 4, 6
    # U: (n*m, R) = (16, R)
    # V: (m*p, R) = (24, R)
    # W: (p*n, R) = (24, R)
    
    # Strategy:
    # 1. Decompose A (4x4) and B (4x6) into 2x2 blocks.
    #    A blocks: A00, A01, A10, A11 (each 2x2)
    #    B blocks: B00, B01, B10, B11 (each 2x3)
    # 2. Apply Strassen's algorithm at the block level to compute C = A*B.
    #    This requires 7 block multiplications: S_k * T_k, where S_k is 2x2 and T_k is 2x3.
    # 3. For each block multiplication (2x2) * (2x3), decompose it into rank-1 terms.
    #    A (2x2)*(2x3) multiplication can be viewed as two independent (2x2)*(2x1) multiplications 
    #    (for the first 2 columns of the result) plus ... wait, no.
    #    Actually, a (2x2)*(2x3) product results in a 2x3 matrix.
    #    We can compute this using Strassen for the 2x2 part (columns 0,1) and schoolbook for column 2?
    #    Or we can treat it as 2x2 * 2x3 directly.
    #    
    #    Let's use a known rank decomposition for (2x2)*(2x3).
    #    The tensor for (2x2)*(2x3) has dimension 4*6*6 = 144? No, 4*2*6 = 48 entries in tensor?
    #    Indices: i (0..1) for A row, j (0..1) for A col, k (0..2) for B col? No.
    #    A is 2x2, B is 2x3.
    #    C[i, k] = sum_j A[i, j] B[j, k].
    #    Tensor T_{i, j, k} is not standard because B has 3 columns.
    #    We can split B into B_left (2x2) and B_right (2x1).
    #    C_left = A * B_left (2x2 result) -> Rank 7 (Strassen)
    #    C_right = A * B_right (2x1 result) -> Rank 4 (Schoolbook: 2x2 multiplications? No, 2*1*2 = 4 terms)
    #    Total rank per block multiplication: 7 + 4 = 11.
    #    
    #    Since there are 7 Strassen steps, total rank = 7 * 11 = 77.
    
    R = 77
    U = np.zeros((16, R))
    V = np.zeros((24, R))
    W = np.zeros((24, R))
    
    # Strassen coefficients for 2x2 blocks
    # A blocks: 0=A00, 1=A01, 2=A10, 3=A11
    # B blocks: 0=B00, 1=B01, 2=B10, 3=B11
    # Note: B blocks are 2x3 here.
    
    alpha = [
        {0: 1, 1: 0, 2: 0, 3: 1},  # S0 = A00 + A11
        {0: 0, 1: 0, 2: 1, 3: 1},  # S1 = A10 + A11
        {0: 1, 1: 0, 2: 0, 3: 0},  # S2 = A00
        {0: 0, 1: 0, 2: 0, 3: 1},  # S3 = A11
        {0: 1, 1: 1, 2: 0, 3: 0},  # S4 = A00 + A01
        {0: -1, 1: 0, 2: 1, 3: 0}, # S5 = A10 - A00
        {0: 0, 1: 1, 2: 0, 3: -1}  # S6 = A01 - A11
    ]
    beta = [
        {0: 1, 1: 0, 2: 0, 3: 1},  # T0 = B00 + B11
        {0: 1, 1: 0, 2: 0, 3: 0},  # T1 = B00
        {0: 0, 1: 1, 2: 0, 3: -1}, # T2 = B01 - B11
        {0: -1, 1: 0, 2: 1, 3: 0}, # T3 = B10 - B00
        {0: 0, 1: 0, 2: 0, 3: 1},  # T4 = B11
        {0: 1, 1: 1, 2: 0, 3: 0},  # T5 = B00 + B01
        {0: 0, 1: 0, 2: 1, 3: 1}   # T6 = B10 + B11
    ]
    # Reconstruction: C_ab = sum_k gamma_k * (S_k * T_k)
    # C00 = S0*T0 - S3*T3 + S6*T6
    # C01 = S2*T2 + S4*T4
    # C10 = S1*T1 + S3*T3
    # C11 = S0*T0 - S2*T2 + S5*T5
    # Note: The standard Strassen reconstruction for 2x2*2x2 is:
    # C00 = M1 - M4 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M3 + M6
    # Where:
    # M1 = (A00+A11)(B00+B11) -> alpha0, beta0
    # M2 = (A10+A11)B00       -> alpha1, beta1
    # M3 = A00(B01-B11)       -> alpha2, beta2
    # M4 = A11(B10-B00)       -> alpha3, beta3
    # M5 = (A00+A01)B11       -> alpha4, beta4
    # M6 = (A10-A00)(B00+B01) -> alpha5, beta5
    # M7 = (A01-A11)(B10+B11) -> alpha6, beta6
    
    recon = [
        [(0, 0, 1), (1, 1, 1)],     # M1 -> C00, C11
        [(1, 0, 1)],                # M2 -> C10
        [(0, 1, 1)],                # M3 -> C01
        [(0, 0, -1), (1, 0, 1)],    # M4 -> -C00, C10
        [(0, 1, 1)],                # M5 -> C01
        [(1, 1, 1)],                # M6 -> C11
        [(0, 0, 1)]                 # M7 -> C00
    ]
    
    # Helper to generate rank-1 terms for (2x2)*(2x3) multiplication
    # Returns lists of (u_indices, u_vals, v_indices, v_vals, w_indices, w_vals)
    # But we will directly populate U, V, W columns.
    
    def add_2x2x2x3_terms(col_offset, S_alpha, T_beta, C_recon):
        """
        Adds terms for computing S * T where S is 2x2, T is 2x3.
        S_alpha: dict mapping block_idx (0..3) to coeff for S.
                 S is constructed from A blocks.
                 But here S is already a linear combination of A blocks.
                 We need to map S's internal 2x2 structure to A's 4x4 structure.
                 S[i,j] = sum_{r} S_alpha[r] * A_r[i//2, i%2]? No.
                 Let's trace indices carefully.
                 
                 A is 4x4. Blocks A_r (r=0..3) are 2x2.
                 A_r is at rows 2*(r//2)..2*(r//2)+1, cols 2*(r%2)..2*(r%2)+1.
                 S is a 2x2 matrix formed by linear combination of A blocks.
                 S[i,j] = sum_{r} S_alpha[r] * A_r[i,j]  (where i,j in 0..1)
                 So S[i,j] corresponds to A[2*(r//2)+i, 2*(r%2)+j].
                 
                 Similarly for T (2x3).
                 T is formed from B blocks (2x3).
                 T[i,j] = sum_{c} T_beta[c] * B_c[i,j] (where i in 0..1, j in 0..2)
                 B_c is at rows 2*(c//2)..2*(c//2)+1, cols 3*(c%2)..3*(c%2)+2.
                 So T[i,j] corresponds to B[2*(c//2)+i, 3*(c%2)+j].
                 
                 The product P = S * T is 2x3.
                 P[i,k] = sum_j S[i,j] T[j,k].
                 
                 We decompose P into rank-1 terms.
                 We split T into T_left (2x2, cols 0,1) and T_right (2x1, col 2).
                 P_left = S * T_left (2x2). Use Strassen (rank 7).
                 P_right = S * T_right (2x1). Use Schoolbook (rank 4).
        """
        col = col_offset
        
        # --- Part 1: Strassen for S * T_left (2x2 * 2x2) ---
        # T_left consists of T[:, 0] and T[:, 1].
        # T[j, 0] = sum_c T_beta[c] * B_c[j, 0]
        # T[j, 1] = sum_c T_beta[c] * B_c[j, 1]
        # Note: B_c has 3 columns. B_c[j, 0] and B_c[j, 1] are the first two columns of block c.
        
        # Strassen terms for 2x2*2x2
        s_alpha = [
            {0: 1, 1: 0, 2: 0, 3: 1},  # (S00+S11)
            {0: 0, 1: 0, 2: 1, 3: 1},  # (S10+S11)
            {0: 1, 1: 0, 2: 0, 3: 0},  # S00
            {0: 0, 1: 0, 2: 0, 3: 1},  # S11
            {0: 1, 1: 1, 2: 0, 3: 0},  # (S00+S01)
            {0: -1, 1: 0, 2: 1, 3: 0}, # (S10-S00)
            {0: 0, 1: 1, 2: 0, 3: -1}  # (S01-S11)
        ]
        t_alpha = [
            {0: 1, 1: 0, 2: 0, 3: 1},  # (T00+T11)
            {0: 1, 1: 0, 2: 0, 3: 0},  # T00
            {0: 0, 1: 1, 2: 0, 3: -1}, # (T01-T11)
            {0: -1, 1: 0, 2: 1, 3: 0}, # (T10-T00)
            {0: 0, 1: 0, 2: 0, 3: 1},  # T11
            {0: 1, 1: 1, 2: 0, 3: 0},  # (T00+T01)
            {0: 0, 1: 0, 2: 1, 3: 1}   # (T10+T11)
        ]
        s_recon = [
            [(0, 0, 1), (1, 1, 1)],
            [(1, 0, 1)],
            [(0, 1, 1)],
            [(0, 0, -1), (1, 0, 1)],
            [(0, 1, 1)],
            [(1, 1, 1)],
            [(0, 0, 1)]
        ]
        
        for k in range(7):
            # U part: S entries
            # S[i,j] = sum_r S_alpha[k][r] * S_base[r][i,j]
            # S_base[r][i,j] = sum_a S_alpha_outer[a] * A_a[i,j] ? No.
            # S is defined by S_alpha_outer passed in.
            # S[i,j] = sum_{r in 0..3} S_alpha_outer[r] * A_r[i,j]
            # Strassen combines S entries: X_k[i,j] = sum_{r in 0..3} s_alpha[k][r] * S[r//2, r%2]
            # So X_k[i,j] = sum_r s_alpha[k][r] * sum_a S_alpha_outer[a] * A_a[r//2, r%2]
            # Wait, S[r//2, r%2] is the element at row r//2, col r%2 of the matrix S.
            # S is 2x2. Its elements are linear combinations of A blocks.
            # S[i,j] = sum_a S_alpha_outer[a] * A_a[i,j]
            # So X_k[i,j] = sum_r s_alpha[k][r] * (sum_a S_alpha_outer[a] * A_a[r//2, r%2])
            
            for r in range(4):
                if s_alpha[k][r] != 0:
                    sr, sc = r // 2, r % 2
                    for a in range(4):
                        if S_alpha[a] != 0:
                            # A_a[sr, sc]
                            ar, ac = a // 2, a % 2
                            u_idx = (ar * 2 + sr) * 4 + (ac * 2 + sc)
                            U[u_idx, col] += s_alpha[k][r] * S_alpha[a]
            
            # V part: T_left entries
            # T_left[i,j] (j in 0,1) = sum_c T_beta_outer[c] * B_c[i,j]
            # Strassen combines T_left entries: Y_k[i,j] = sum_c t_alpha[k][c] * T_left[c//2, c%2]
            # T_left[c//2, c%2] = sum_b T_beta_outer[b] * B_b[c//2, c%2]
            
            for c in range(4):
                if t_alpha[k][c] != 0:
                    tr, tc = c // 2, c % 2
                    for b in range(4):
                        if T_beta[b] != 0:
                            # B_b[tr, tc]
                            br, bc = b // 2, b % 2
                            v_idx = (br * 2 + tr) * 6 + (bc * 3 + tc)
                            V[v_idx, col] += t_alpha[k][c] * T_beta[b]
            
            # W part: Output of S * T_left
            # Z[i,j] = X_k[i,j] * Y_k[i,j] ? No, rank 1 term is u v^T w.
            # The term is X_k * Y_k^T? No, it's scalar multiplication.
            # The rank-1 term corresponds to the product of linear forms.
            # The output contribution is to P_left[i,j] where (i,j) is determined by s_recon[k].
            # s_recon[k] gives (out_i, out_j, gamma).
            # P_left[out_i, out_j] += gamma * (X_k * Y_k)
            # P_left is part of P (2x3). P_left[i,j] is P[i,j] for j=0,1.
            # P[i,j] contributes to C via C_recon.
            # C_recon is list of (block_i, block_j, gamma_outer).
            # C_block[block_i, block_j] += gamma_outer * P[out_i, out_j]
            # Global C index:
            # C is 4x6.
            # Block (bi, bj) is at rows 2*bi..2*bi+1, cols 3*bj..3*bj+2.
            # C[2*bi + out_i, 3*bj + out_j] += gamma_outer * gamma_inner
            
            for bi, bj, g_out in C_recon:
                for oi, oj, g_in in s_recon[k]:
                    w_idx = (bj * 3 + oj) * 4 + (bi * 2 + oi)
                    W[w_idx, col] += g_out * g_in
            
            col += 1
        
        # --- Part 2: Schoolbook for S * T_right (2x2 * 2x1) ---
        # T_right is 2x1. T_right[i, 0] = T[i, 2].
        # T[i, 2] = sum_c T_beta_outer[c] * B_c[i, 2]
        # P_right[i, 0] = sum_j S[i,j] T_right[j, 0]
        # 4 terms: (i,j) in {(0,0), (0,1), (1,0), (1,1)}
        
        for i in range(2):
            for j in range(2):
                # Term: S[i,j] * T[j, 2]
                
                # U part: S[i,j]
                for a in range(4):
                    if S_alpha[a] != 0:
                        ar, ac = a // 2, a % 2
                        u_idx = (ar * 2 + i) * 4 + (ac * 2 + j)
                        U[u_idx, col] += S_alpha[a]
                
                # V part: T[j, 2]
                for b in range(4):
                    if T_beta[b] != 0:
                        br, bc = b // 2, b % 2
                        v_idx = (br * 2 + j) * 6 + (bc * 3 + 2)
                        V[v_idx, col] += T_beta[b]
                
                # W part: P_right[i, 0]
                # P_right[i, 0] is P[i, 2]
                for bi, bj, g_out in C_recon:
                    w_idx = (bj * 3 + 2) * 4 + (bi * 2 + i)
                    W[w_idx, col] += g_out
                
                col += 1
        
        return col
    
    col = 0
    for k in range(7):
        S_alpha_k = alpha[k]
        T_beta_k = beta[k]
        C_recon_k = recon[k]
        
        col = add_2x2x2x3_terms(col, S_alpha_k, T_beta_k, C_recon_k)
    
    # Convert to lists for output
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# Verification (optional, for local testing)
if __name__ == "__main__":
    res = construct()
    U = np.array(res["U"])
    V = np.array(res["V"])
    W = np.array(res["W"])
    
    # Random test
    np.random.seed(42)
    A = np.random.rand(4, 4)
    B = np.random.rand(4, 6)
    
    # Flatten A and B
    a_flat = A.flatten()
    b_flat = B.flatten()
    
    # Compute C via decomposition
    # C = sum_k u_k v_k^T w_k^T ?
    # T = sum_k u_k \otimes v_k \otimes w_k
    # C_{i,k} = sum_j A_{i,j} B_{j,k}
    # Decomposition: C = sum_r (U[:,r] A^T) (V[:,r]^T B) W[:,r]?
    # Standard tensor decomposition:
    # C_{i,k} = sum_r U_{i,j?} ...
    # Let's check the contraction.
    # U is (16, R). A is 4x4 -> 16 flat.
    # V is (24, R). B is 4x6 -> 24 flat.
    # W is (24, R). C is 6x4? Or 4x6?
    # Task says W[k*n+i, r]. n=4, p=6.
    # C is 4x6.
    # Index in W: k*n + i. k in 0..5 (col), i in 0..3 (row).
    # So W is flattened C^T? Or C column-major?
    # "output factor follows the official trace convention W[k*n+i,r]"
    # Usually C[i,k]. If W index is k*n+i, then W is C stored column-major (k is outer loop).
    # Let's compute C_expected = A @ B.
    # C_expected[i,k].
    # We want to check if sum_r U[:,r]^T A_flat * V[:,r]^T B_flat * W[:,r] reconstructs C?
    # No, the tensor is T_{i,j,k} = delta?
    # Matrix mult tensor: T_{i,j,k} such that C_{i,k} = sum_j A_{i,j} B_{j,k} T_{i,j,k}?
    # T_{i,j,k} = 1 if indices match?
    # U_{idx, r} where idx = i*m + j.
    # V_{idx, r} where idx = j*p + k.
    # W_{idx, r} where idx = k*n + i.
    
    # Let's verify with a specific term.
    # Term r: u_r, v_r, w_r.
    # Contribution to C[i,k] is sum_j u_{i,j} v_{j,k} w_{k,i}?
    # No, w is a vector of length 24.
    # The decomposition is:
    # C = sum_r (U_r \otimes V_r \otimes W_r)
    # This means C_{i,k} = sum_r (sum_j U_{i*m+j, r} V_{j*p+k, r}) W_{k*n+i, r}
    
    C_calc = np.zeros((4, 6))
    for r in range(U.shape[1]):
        u = U[:, r]
        v = V[:, r]
        w = W[:, r]
        
        # u is 16, v is 24, w is 24
        # Sum over j
        # u[i*m+j] * v[j*p+k]
        # We can reshape u to 4x4, v to 4x6
        u_mat = u.reshape(4, 4)
        v_mat = v.reshape(4, 6)
        
        # Product P = u_mat^T v_mat?
        # P[i,k] = sum_j u_mat[j,i] v_mat[j,k]?
        # u index is i*m+j. So u_mat[i,j] = u[i*m+j].
        # v index is j*p+k. So v_mat[j,k] = v[j*p+k].
        # We want sum_j u[i,j] v[j,k].
        # This is (u_mat @ v_mat^T)?
        # (u_mat @ v_mat.T)[i,k] = sum_j u_mat[i,j] v_mat.T[j,k] = sum_j u_mat[i,j] v_mat[k,j].
        # No, v_mat is 4x6. v_mat.T is 6x4.
        # We need sum_j u[i,j] v[j,k].
        # This is element-wise product of u_mat and v_mat.T? No.
        # It's a contraction.
        # Let's just loop or use einsum.
        # np.einsum('ij,jk->ik', u_mat, v_mat)
        P = np.einsum('ij,jk->ik', u_mat, v_mat)
        
        # Now multiply by w
        # w index is k*n+i.
        # w_mat[k,i] = w[k*n+i].
        w_mat = w.reshape(6, 4).T # 4x6, w_mat[i,k] = w[k*4+i]
        
        C_calc += P * w_mat
    
    C_expected = A @ B
    
    error = np.max(np.abs(C_calc - C_expected))
    print(f"Max error: {error}")
    assert error < 1e-9, "Decomposition is incorrect"
    print("Decomposition verified successfully.")
