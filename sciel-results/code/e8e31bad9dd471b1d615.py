import time
import numpy as np

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 130.0  # Safety margin before 150s kill
    
    # First, try the constructive rank-16 decomposition
    # Strategy: Use Strassen on the top-left 2x2 block of B (rows 0-1, cols 0-1)
    # This handles C[0:2, 0:2] with 7 terms.
    # Then use direct products for the remaining parts.
    # Wait, this doesn't work directly because C[0:2, 2:4] depends on B[0:2, 2:4] and C[2:4, 0:4] depends on B[2:4, 0:4].
    # Let's re-evaluate the constructive approach.
    
    # The tensor T_{i,j,k} = 1 for all i,j,k.
    # We need rank 16.
    # Let's try to construct it explicitly.
    
    # Let's use the known rank 16 decomposition for 2x4x4.
    # It is known that mu(2,4,4) = 16.
    # One construction uses 8 scalar multiplications for the "right" part and 8 for the "left"?
    # Or maybe 7 Strassen + 9 something?
    
    # Let's try a specific known decomposition or find one via search if constructive fails.
    
    # Constructive attempt:
    # Split A (2x4) into A1 (2x2) and A2 (2x2).
    # Split B (4x4) into B11 (2x2), B12 (2x2), B21 (2x2), B22 (2x2).
    # C = A1 @ B1 + A2 @ B2 where B1 = [B11, B12] and B2 = [B21, B22].
    # C is 2x4.
    # C_left = A1 @ B11 + A2 @ B21 (2x2)
    # C_right = A1 @ B12 + A2 @ B22 (2x2)
    
    # If we compute C_left and C_right independently:
    # C_left requires multiplying two 2x2 matrices? No, A1 is 2x2, B11 is 2x2.
    # So C_left = A1 * B11 + A2 * B21.
    # This is a sum of two matrix products.
    # We can compute A1*B11 using Strassen (7 mults) and A2*B21 using Strassen (7 mults)?
    # Total 14 mults.
    # Similarly for C_right: A1*B12 (7 mults) + A2*B22 (7 mults) = 14 mults.
    # Total 28. That's the same as the current seed.
    
    # Can we do better?
    # Notice that A1 is shared in C_left and C_right.
    # A1*B11 and A1*B12 can be combined?
    # A1 * [B11, B12] = A1 * B1 (2x4).
    # Multiplying 2x2 by 2x4 has rank 8?
    # Let's check the rank of the 2x2x4 tensor.
    # It is known that mu(2,2,4) = 8.
    # So we can compute A1 * B1 with 8 mults.
    # And A2 * B2 with 8 mults.
    # Total 16 mults!
    
    # So the strategy is:
    # 1. Compute X1 = A1 * B1 (2x2 * 2x4 -> 2x4) using rank 8 decomposition.
    # 2. Compute X2 = A2 * B2 (2x2 * 2x4 -> 2x4) using rank 8 decomposition.
    # 3. C = X1 + X2.
    
    # We need the rank 8 decomposition for 2x2x4.
    # Let A' be 2x2, B' be 2x4.
    # A' = [[a00, a01], [a10, a11]]
    # B' = [[b00, b01, b02, b03], [b10, b11, b12, b13]]
    # C' = A' @ B' = 2x4.
    # C'_{i,k} = sum_j A'_{i,j} B'_{j,k}.
    
    # Rank 8 decomposition for 2x2x4:
    # We can view this as two independent 2x2x2 multiplications?
    # C'_{:, 0:2} = A' @ B'_{:, 0:2} (2x2 * 2x2). Rank 7 (Strassen) or 4 (standard)?
    # Standard 2x2x2 is rank 4.
    # If we use standard 4 for the first 2 cols and standard 4 for the next 2 cols, total 8.
    # This works!
    
    # So, rank 8 decomposition for 2x2x4 is just two separate standard 2x2x2 multiplications.
    # Term r=0..3: A' @ B'_{:, 0:2}
    # Term r=4..7: A' @ B'_{:, 2:4}
    
    # Let's construct the full rank 16 decomposition.
    # U, V, W of shape (8, 16), (16, 16), (8, 16) ? No.
    # U: (n*m, rank) = (2*4, 16) = (8, 16)
    # V: (m*p, rank) = (4*4, 16) = (16, 16)
    # W: (p*n, rank) = (4*2, 16) = (8, 16)
    
    # Indices:
    # A is 2x4. A[i, j] where i in 0..1, j in 0..3.
    # B is 4x4. B[j, k] where j in 0..3, k in 0..3.
    # C is 2x4. C[i, k] where i in 0..1, k in 0..3.
    
    # Decomposition:
    # Part 1: A1 (cols 0,1) * B1 (rows 0,1) -> contributes to C[:, 0:2]
    # A1 is A[:, 0:2]. B1 is B[0:2, :].
    # Wait, B1 needs to be 2x4 to match A1 (2x2) * B1 (2x4).
    # B1 = B[0:2, 0:4].
    # So Part 1 computes A[:, 0:2] @ B[0:2, 0:4].
    # This contributes to C[i, k] = sum_{j=0}^1 A[i, j] B[j, k].
    
    # Part 2: A2 (cols 2,3) * B2 (rows 2,3) -> contributes to C[:, 0:2]? No.
    # A2 is A[:, 2:4]. B2 is B[2:4, 0:4].
    # Part 2 computes A[:, 2:4] @ B[2:4, 0:4].
    # This contributes to C[i, k] = sum_{j=2}^3 A[i, j] B[j, k].
    
    # Total C = Part 1 + Part 2.
    
    # Now, decompose Part 1 (2x2 * 2x4) into 8 terms.
    # Let's use standard multiplication for the two 2x2 blocks of the output.
    # Output cols 0,1: C[:, 0:2] = A1 @ B[0:2, 0:2]
    # Output cols 2,3: C[:, 2:4] = A1 @ B[0:2, 2:4]
    
    # So Part 1 has 4 terms for cols 0,1 and 4 terms for cols 2,3.
    # Similarly for Part 2.
    
    # Total 16 terms.
    
    # Let's define the terms.
    # For a standard 2x2 * 2x2 multiplication:
    # C00 = A00 B00 + A01 B10
    # C01 = A00 B01 + A01 B11
    # C10 = A10 B00 + A11 B10
    # C11 = A10 B01 + A11 B11
    
    # The 4 terms are:
    # r=0: A00 B00 -> C00
    # r=1: A01 B10 -> C00
    # r=2: A00 B01 -> C01
    # r=3: A01 B11 -> C01
    # r=4: A10 B00 -> C10
    # r=5: A11 B10 -> C10
    # r=6: A10 B01 -> C11
    # r=7: A11 B11 -> C11
    
    # Wait, standard multiplication has 4 scalar multiplications:
    # M1 = A00 B00
    # M2 = A01 B10
    # M3 = A00 B01
    # M4 = A01 B11
    # M5 = A10 B00
    # M6 = A11 B10
    # M7 = A10 B01
    # M8 = A11 B11
    
    # Yes, 8 terms for 2x2 * 2x4.
    
    # Let's build the matrices.
    rank = 16
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Helper to add a term
    # Term (r) corresponds to:
    # A index (i, j)
    # B index (j', k) where j' is the row in the 2x2 block, mapped to global j
    # C index (i', k') where i' is the row in the 2x2 block, mapped to global i, k' is the col
    
    # Part 1: A[:, 0:2] * B[0:2, :]
    # Sub-part 1a: B[0:2, 0:2] -> C[:, 0:2]
    # Sub-part 1b: B[0:2, 2:4] -> C[:, 2:4]
    
    # Part 2: A[:, 2:4] * B[2:4, :]
    # Sub-part 2a: B[2:4, 0:2] -> C[:, 0:2]
    # Sub-part 2b: B[2:4, 2:4] -> C[:, 2:4]
    
    # Let's map indices.
    # Global A index: i (0..1), j (0..3)
    # Global B index: j (0..3), k (0..3)
    # Global C index: i (0..1), k (0..3)
    
    # For Part 1a (A1 * B11):
    # A1 indices: i (0..1), j_local (0..1) -> j_global = j_local
    # B11 indices: j_local (0..1) -> j_global = j_local, k_local (0..1) -> k_global = k_local
    # C indices: i (0..1), k_local (0..1) -> k_global = k_local
    
    # Terms for 2x2 * 2x2:
    # (i, j_local, k_local)
    # r=0: i=0, j=0, k=0 -> A(0,0), B(0,0), C(0,0)
    # r=1: i=0, j=1, k=0 -> A(0,1), B(1,0), C(0,0)
    # r=2: i=0, j=0, k=1 -> A(0,0), B(0,1), C(0,1)
    # r=3: i=0, j=1, k=1 -> A(0,1), B(1,1), C(0,1)
    # r=4: i=1, j=0, k=0 -> A(1,0), B(0,0), C(1,0)
    # r=5: i=1, j=1, k=0 -> A(1,1), B(1,0), C(1,0)
    # r=6: i=1, j=0, k=1 -> A(1,0), B(0,1), C(1,1)
    # r=7: i=1, j=1, k=1 -> A(1,1), B(1,1), C(1,1)
    
    # Let's generalize.
    # We have 4 blocks:
    # 1. A_cols 0-1, B_rows 0-1, C_cols 0-1 (8 terms)
    # 2. A_cols 0-1, B_rows 0-1, C_cols 2-3 (8 terms)
    # 3. A_cols 2-3, B_rows 2-3, C_cols 0-1 (8 terms)
    # 4. A_cols 2-3, B_rows 2-3, C_cols 2-3 (8 terms)
    
    # Wait, that's 32 terms.
    # My previous logic was:
    # Part 1: A1 (2x2) * B1 (2x4). Rank 8.
    # B1 is B[0:2, 0:4].
    # This produces C1 (2x4) which is added to C.
    # Part 2: A2 (2x2) * B2 (2x4). Rank 8.
    # B2 is B[2:4, 0:4].
    # This produces C2 (2x4) which is added to C.
    
    # So Part 1 handles ALL columns of C for the first 2 rows of B.
    # Part 2 handles ALL columns of C for the last 2 rows of B.
    
    # So:
    # Part 1: A[:, 0:2] * B[0:2, 0:4]
    # This is a 2x2 * 2x4 multiplication.
    # We decomposed this into two 2x2 * 2x2 multiplications:
    # 1a: A[:, 0:2] * B[0:2, 0:2] -> C[:, 0:2]
    # 1b: A[:, 0:2] * B[0:2, 2:4] -> C[:, 2:4]
    
    # Part 2: A[:, 2:4] * B[2:4, 0:4]
    # 2a: A[:, 2:4] * B[2:4, 0:2] -> C[:, 0:2]
    # 2b: A[:, 2:4] * B[2:4, 2:4] -> C[:, 2:4]
    
    # So we have 4 blocks, each 2x2 * 2x2.
    # Each 2x2 * 2x2 has 8 terms in the "standard" rank 8 decomposition?
    # No, standard 2x2 * 2x2 has 4 terms.
    # Wait, 2x2 * 2x2 tensor rank is 4.
    # So 4 blocks * 4 terms = 16 terms.
    
    # Let's verify the rank of 2x2 * 2x2.
    # T_{i,j,k} = 1.
    # C_{i,k} = sum_j A_{i,j} B_{j,k}.
    # This is the standard matrix multiplication.
    # The rank is 4 (the 4 scalar products).
    
    # So yes, 4 blocks of 4 terms = 16 terms.
    
    # Let's construct it.
    
    def add_block(a_col_offset, b_row_offset, c_col_offset, start_col):
        # a_col_offset: 0 or 2 (which 2 cols of A)
        # b_row_offset: 0 or 2 (which 2 rows of B)
        # c_col_offset: 0 or 2 (which 2 cols of C)
        # start_col: starting column index in U, V, W
        
        # The 4 terms for 2x2 * 2x2:
        # (i_local, j_local, k_local)
        # 0: 0, 0, 0
        # 1: 0, 1, 0
        # 2: 0, 0, 1
        # 3: 0, 1, 1
        # Wait, standard terms are:
        # M1 = A00 B00 -> C00
        # M2 = A01 B10 -> C00
        # M3 = A00 B01 -> C01
        # M4 = A01 B11 -> C01
        # M5 = A10 B00 -> C10
        # M6 = A11 B10 -> C10
        # M7 = A10 B01 -> C11
        # M8 = A11 B11 -> C11
        
        # Wait, I listed 8 terms above.
        # But rank is 4.
        # The 4 terms are:
        # r1: A00 B00 -> C00
        # r2: A01 B10 -> C00
        # r3: A00 B01 -> C01
        # r4: A01 B11 -> C01
        # r5: A10 B00 -> C10
        # r6: A11 B10 -> C10
        # r7: A10 B01 -> C11
        # r8: A11 B11 -> C11
        
        # This is 8 terms.
        # Why did I think rank is 4?
        # The rank of the 2x2x2 tensor is 4?
        # No, the rank of the 2x2x2 matrix multiplication tensor is 4?
        # Let's check.
        # C00 = A00 B00 + A01 B10
        # C01 = A00 B01 + A01 B11
        # C10 = A10 B00 + A11 B10
        # C11 = A10 B01 + A11 B11
        
        # Can we do it in 4 mults?
        # M1 = A00 B00
        # M2 = A01 B10
        # M3 = A00 B01
        # M4 = A01 B11
        # M5 = A10 B00
        # M6 = A11 B10
        # M7 = A10 B01
        # M8 = A11 B11
        
        # That's 8 mults.
        # Strassen does it in 7.
        # The rank of 2x2x2 is 7?
        # No, the rank of the 2x2x2 tensor is 7 (Strassen).
        # But we can also do it in 4? No, 4 is too low.
        # The trivial decomposition has 4 terms?
        # No, the trivial decomposition has 4 terms?
        # C00 = A00 B00 + A01 B10. This is 2 terms.
        # There are 4 entries in C.
        # Each entry is a sum of 2 products.
        # Total 8 products.
        # So the trivial rank is 8.
        
        # So 4 blocks * 8 terms = 32 terms.
        # That's worse than 28.
        
        # My mistake was assuming rank(2,2,4) = 8.
        # Let's check rank(2,2,4).
        # A is 2x2, B is 2x4.
        # C is 2x4.
        # C[:, 0:2] = A @ B[:, 0:2] (2x2 * 2x2). Rank 7.
        # C[:, 2:4] = A @ B[:, 2:4] (2x2 * 2x2). Rank 7.
        # Total 14?
        # But A is shared.
        # Can we save?
        # It is known that mu(2,2,4) = 8.
        # How?
        
        # Let's look up the rank of 2x2x4.
        # It is 8.
        # Decomposition:
        # M1 = A00 (B00 + B01)
        # M2 = A01 (B10 + B11)
        # M3 = A00 B01
        # M4 = A01 B11
        # ...
        
        # Actually, let's just use the search method from the seed program, but with a better initialization or method.
        # The seed program already tries rank 16.
        # It failed to find it in the time limit or the rounding failed.
        
        # Let's stick to the seed program's search but improve it.
        # Or, let's try to find the rank 16 decomposition online or derive it.
        
        # For now, let's return the seed program's logic but maybe with a faster check.
        
        pass

    def check_decomposition(U, V, W, rank, strict=True):
        if U.shape != (n * m, rank) or V.shape != (m * p, rank) or W.shape != (p * n, rank):
            return False
        # Check half-integer constraint
        if not np.allclose(2 * U, np.round(2 * U), atol=1e-5): return False
        if not np.allclose(2 * V, np.round(2 * V), atol=1e-5): return False
        if not np.allclose(2 * W, np.round(2 * W), atol=1e-5): return False
        
        # Check correctness
        np.random.seed(42)
        A = np.random.rand(n, m)
        B = np.random.rand(m, p)
        C = A @ B
        
        Cf = np.zeros((n, p))
        # Vectorized check
        # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
        # We need to compute sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        
        # Reshape for easier broadcasting
        # U_reshaped: (n, m, rank)
        U_r = U.reshape(n, m, rank)
        # V_reshaped: (m, p, rank)
        V_r = V.reshape(m, p, rank)
        # W_reshaped: (p, n, rank)
        W_r = W.reshape(p, n, rank)
        
        # Product term for fixed r: U_r[i,j,r] * V_r[j,k,r] * W_r[k,i,r]
        # Sum over r.
        
        # Let's compute T[i, j, k] = sum_r U_r[i,j,r] * V_r[j,k,r] * W_r[k,i,r]
        # This is a bit tricky with simple broadcasting.
        # Let's loop over r for the check since it's just 16-28 iterations.
        
        for r in range(rank):
            # u_r: (n, m)
            u_r = U_r[:, :, r]
            # v_r: (m, p)
            v_r = V_r[:, :, r]
            # w_r: (p, n)
            w_r = W_r[:, :, r]
            
            # term_r = u_r[i,j] * v_r[j,k] * w_r[k,i]
            # Sum over j.
            # temp[j, k] = sum_i u_r[i,j] * v_r[j,k] ... wait.
            # C[i,k] = sum_j A[i,j] B[j,k]
            # Decomposition: C[i,k] = sum_r (sum_j U[i,j,r] A[i,j]) (sum_k V[j,k,r] B[j,k]) W[k,i,r] ?? No.
            
            # The tensor definition is:
            # M_{i,j,k} = sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r}
            # We want M_{i,j,k} = delta_{j,k}? No.
            # M_{i,j,k} is the tensor for matrix multiplication.
            # M_{i,j,k} = 1 if the indices correspond to a valid multiplication term?
            # Actually, the matrix multiplication tensor T has entries T_{i,j,k} such that
            # C_{i,k} = sum_j A_{i,j} B_{j,k} = sum_{i,j,k} A_{i,j} B_{j,k} T_{i,j,k}
            # So T_{i,j,k} is 1 if the indices are valid? No, T is the structure tensor.
            # T_{i,j,k} = 1 if we are summing A_{i,j} B_{j,k}.
            # The indices are i (row of A, row of C), j (col of A, row of B), k (col of B, col of C).
            # So for every i, j, k, the term A_{i,j} B_{j,k} contributes to C_{i,k}.
            # Thus T_{i,j,k} = 1 for all valid i,j,k?
            # Wait, the decomposition is T = sum_r u^{(r)} v^{(r)} w^{(r)}.
            # T_{i,j,k} = sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r}
            # We need T_{i,j,k} to be 1 for all i,j,k?
            # No. The matrix multiplication tensor has non-zero entries for every (i,j,k) triple
            # because every A_{i,j} and B_{j,k} pair contributes to some C_{i,k}.
            # Specifically, C_{i,k} = sum_j A_{i,j} B_{j,k}.
            # So the tensor is T_{i,j,k} = 1 for all i,j,k.
            
            # Let's verify this interpretation.
            # If T_{i,j,k} = 1, then we need sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r} = 1 for all i,j,k.
            
            # Let's check the Strassen code.
            # In the seed code:
            # U[g_idx, col] += coeff where g_idx = r*m + c (A index)
            # V[g_idx, col] += coeff where g_idx = g_row*p + g_col (B index)
            # W[g_idx, col] += coeff where g_idx = g_col*n + g_row (C index? No, W index is k*n+i)
            
            # Let's re-read the seed code verification.
            # acc += U[ii * m + jj, cc] * V[jj * p + kk, cc] * W[kk * n + ii, cc]
            # This computes sum_cc U[ii,m+jj,cc] V[jj,p+kk,cc] W[kk,n+ii,cc]
            # This is the term for C[ii, kk].
            # And it sums over jj (the inner dimension).
            # So the full computation is:
            # Cf[ii, kk] = sum_jj ( sum_cc U[ii*m+jj, cc] V[jj*p+kk, cc] W[kk*n+ii, cc] )
            # This matches C[ii, kk] = sum_jj A[ii, jj] B[jj, kk]
            # IF the decomposition satisfies:
            # sum_cc U[ii*m+jj, cc] V[jj*p+kk, cc] W[kk*n+ii, cc] = 1 for all ii, jj, kk.
            
            # So yes, the target tensor T_{i,j,k} is 1 for all valid indices.
            
            # So for the check:
            # We need sum_r U[i,j,r] V[j,k,r] W[k,i,r] = 1 for all i,j,k.
            
            # Let's compute this sum.
            # u_r: (n, m)
            # v_r: (m, p)
            # w_r: (p, n)
            
            # We want S[i,j,k] = sum_r u_r[i,j] v_r[j,k] w_r[k,i]
            
            # For a fixed r, term_r[i,j,k] = u_r[i,j] * v_r[j,k] * w_r[k,i]
            # This is an outer product of sorts.
            
            # Let's just loop i, j, k for the check since n,m,p are small.
            pass
        
        # Efficient check:
        # Construct the tensor T_reconstructed
        T_rec = np.zeros((n, m, p))
        for r in range(rank):
            u_r = U.reshape(n, m, rank)[:, :, r] # (n, m)
            v_r = V.reshape(m, p, rank)[:, :, r] # (m, p)
            w_r = W.reshape(p, n, rank)[:, :, r] # (p, n)
            
            # We need to add u_r[i,j] * v_r[j,k] * w_r[k,i] to T_rec[i,j,k]
            # This is tricky to vectorize directly without large intermediate arrays.
            # But n,m,p are small (2,4,4).
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T_rec[i, j, k] += u_r[i, j] * v_r[j, k] * w_r[k, i]
                        
        # T_target is 1 everywhere
        T_target = np.ones((n, m, p))
        
        if not np.allclose(T_rec, T_target, atol=1e-5):
            return False
        return True

    # Strategy: Try to find a rank 16 decomposition.
    # If not found, fallback to rank 28 Strassen.
    
    best_result = None
    
    # Precompute indices for vectorized objective if we were to use gradient descent
    # But for now, let's try a simple random search or L-BFGS-B for rank 16.
    
    # The problem is non-convex.
    # Let's try L-BFGS-B with multiple restarts for rank 16.
    
    def solve_rank(rank):
        # Parameters: U, V, W flattened
        size_u = n * m * rank
        size_v = m * p * rank
        size_w = p * n * rank
        total_size = size_u + size_v + size_w
        
        # Objective: minimize sum_{i,j,k} (sum_r U[i,j,r] V[j,k,r] W[k,i,r] - 1)^2
        # This is a sum of squares of cubic terms.
        
        def objective(params):
            U_flat = params[:size_u].reshape(n, m, rank)
            V_flat = params[size_u:size_u+size_v].reshape(m, p, rank)
            W_flat = params[size_u+size_v:].reshape(p, n, rank)
            
            # Vectorized computation of S[i, j, k]
            # U: (n, m, rank)
            # V: (m, p, rank)
            # W: (p, n, rank)
            
            # term[i, j, k, i', r] = U[i, j, r] * V[j, k, r] * W[k, i', r]
            # We want S[i, j, k] = sum_r term[i, j, k, i, r]
            
            # Broadcasting
            # U: (n, m, 1, 1, rank)
            # V: (1, m, p, 1, rank)
            # W: (1, 1, p, n, rank)
            
            U_b = U_flat[:, :, np.newaxis, np.newaxis, :]
            V_b = V_flat[np.newaxis, :, :, np.newaxis, :]
            W_b = W_flat[np.newaxis, np.newaxis, :, :, :]
            
            prod = U_b * V_b * W_b # (n, m, p, n, rank)
            S = np.sum(prod, axis=4) # (n, m, p, n)
            
            # Extract diagonal in the first and last dimension
            # S[i, j, k, i]
            idx = np.arange(n)
            S_diag = S[idx, :, :, idx] # (n, m, p)
            
            err = np.sum((S_diag - 1.0) ** 2)
            return err

        def gradient(params):
            U_flat = params[:size_u].reshape(n, m, rank)
            V_flat = params[size_u:size_u+size_v].reshape(m, p, rank)
            W_flat = params[size_u+size_v:].reshape(p, n, rank)
            
            # Compute S and error term
            U_b = U_flat[:, :, np.newaxis, np.newaxis, :]
            V_b = V_flat[np.newaxis, :, :, np.newaxis, :]
            W_b = W_flat[np.newaxis, np.newaxis, :, :, :]
            
            prod = U_b * V_b * W_b # (n, m, p, n, rank)
            S = np.sum(prod, axis=4) # (n, m, p, n)
            
            idx = np.arange(n)
            S_diag = S[idx, :, :, idx] # (n, m, p)
            
            err_term = 2.0 * (S_diag - 1.0) # (n, m, p)
            
            grad_U = np.zeros_like(U_flat)
            grad_V = np.zeros_like(V_flat)
            grad_W = np.zeros_like(W_flat)
            
            # grad_U[i, j, r] = sum_k err_term[i, j, k] * V[j, k, r] * W[k, i, r]
            # err_term: (n, m, p)
            # V: (m, p, rank)
            # W: (p, n, rank)
            
            # We need to sum over k.
            # Let's reshape err_term to (n, m, p, 1, 1)
            # V to (1, m, p, 1, rank)
            # W to (1, 1, p, n, rank)
            
            E_b = err_term[:, :, :, np.newaxis, np.newaxis]
            V_b = V_flat[np.newaxis, :, :, np.newaxis, :]
            W_b = W_flat[np.newaxis, np.newaxis, :, :, :]
            
            # prod_grad_U[i, j, k, n, r] = E[i, j, k] * V[j, k, r] * W[k, n, r]
            # We want sum_k prod_grad_U[i, j, k, i, r]
            
            prod_gU = E_b * V_b * W_b # (n, m, p, n, rank)
            grad_U_flat = np.sum(prod_gU, axis=2) # (n, m, n, rank)
            grad_U = grad_U_flat[idx, :, idx, :] # (n, m, rank)
            
            # Similarly for V and W
            # grad_V[j, k, r] = sum_i err_term[i, j, k] * U[i, j, r] * W[k, i, r]
            # E_b: (n, m, p, 1, 1)
            # U_b: (n, m, 1, 1, rank)
            # W_b: (1, 1, p, n, rank)
            
            prod_gV = E_b * U_b * W_b # (n, m, p, n, rank)
            # We want sum_i prod_gV[i, j, k, i, r]
            # This is tricky with simple sum.
            # prod_gV is (n, m, p, n, rank).
            # We want to sum over the first axis (i) where the fourth axis (n) is equal to i.
            
            # Let's use einsum or a loop over i (n is small, 2)
            grad_V = np.zeros_like(V_flat)
            grad_W = np.zeros_like(W_flat)
            
            for i in range(n):
                # grad_V[j, k, r] += err_term[i, j, k] * U[i, j, r] * W[k, i, r]
                # err_term[i, :, :]: (m, p)
                # U[i, :, :]: (m, rank)
                # W[:, i, :]: (p, rank)
                
                E_i = err_term[i, :, :] # (m, p)
                U_i = U_flat[i, :, :] # (m, rank)
                W_i = W_flat[:, i, :] # (p, rank)
                
                # grad_V[j, k, r] = sum_i ...
                # Term for fixed i: E_i[j, k] * U_i[j, r] * W_i[k, r]
                
                # E_i: (m, p, 1)
                # U_i: (m, 1, rank)
                # W_i: (1, p, rank)
                
                term_V = E_i[:, :, np.newaxis] * U_i[:, np.newaxis, :] * W_i[np.newaxis, :, :]
                grad_V += np.sum(term_V, axis=2) # (m, p) ... wait, sum over r? No, sum over r is not here.
                # term_V is (m, p, rank). We want to add to grad_V[j, k, r].
                # So just add term_V.
                grad_V += term_V
                
                # grad_W[k, i, r] += err_term[i, j, k] * U[i, j, r] * V[j, k, r]
                # Sum over j.
                # E_i[j, k] * U_i[j, r] * V[j, k, r]
                # E_i: (m, p, 1)
                # U_i: (m, 1, rank)
                # V: (m, p, rank)
                
                term_W = E_i[:, :, np.newaxis] * U_i[:, np.newaxis, :] * V_flat[:, :, :]
                # term_W is (m, p, rank).
                # We want to sum over j (axis 0).
                # Result: (p, rank).
                # This corresponds to grad_W[k, i, r].
                grad_W[:, i, :] += np.sum(term_W, axis=0)
            
            return np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])

        best_params = None
        best_fun = np.inf
        
        for restart in range(20):
            if time.time() > deadline:
                break
            
            np.random.seed(rank * 1000 + restart)
            # Initialize with small random values
            params = np.random.randn(total_size) * 0.1
            
            try:
                from scipy.optimize import minimize
                res = minimize(objective, params, jac=gradient, method='L-BFGS-B',
                               options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-15})
                
                if res.fun < best_fun:
                    best_fun = res.fun
                    best_params = res.x
                    
                if best_fun < 1e-10:
                    break
                    
            except Exception as e:
                pass
        
        if best_params is not None and best_fun < 1e-8:
            U_opt = best_params[:size_u].reshape(n, m, rank)
            V_opt = best_params[size_u:size_u+size_v].reshape(m, p, rank)
            W_opt = best_params[size_u+size_v:].reshape(p, n, rank)
            
            # Round to half-integers
            U_int = np.round(2 * U_opt) / 2.0
            V_int = np.round(2 * V_opt) / 2.0
            W_int = np.round(2 * W_opt) / 2.0
            
            if check_decomposition(U_int, V_int, W_int, rank):
                return {"U": U_int.reshape(n*m, rank).tolist(), 
                        "V": V_int.reshape(m*p, rank).tolist(), 
                        "W": W_int.reshape(p*n, rank).tolist()}
            
            # If rounding failed, try to adjust
            # Sometimes the solution is close but not exactly on the grid.
            # We can try to snap to nearest half-integer and check.
            # Already done above.
            
        return None

    # Try rank 16
    result = solve_rank(16)
    if result:
        best_result = result
    else:
        # Try rank 18? 20?
        # The theoretical minimum is 16.
        # If 16 fails, try 17, 18... up to 27.
        for rank in range(17, 28):
            if time.time() > deadline:
                break
            result = solve_rank(rank)
            if result:
                best_result = result
                break

    if best_result:
        return best_result

    # Fallback to Strassen Rank 28
    rank = 28
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    strassen_terms = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1})
    ]
    
    def apply_strassen(a_col_offset, b_row_offset, b_col_offset, c_col_offset, start_col):
        for t_idx, (u_loc, v_loc, w_loc) in enumerate(strassen_terms):
            col = start_col + t_idx
            for loc_idx, coeff in u_loc.items():
                r = loc_idx // 2
                c = loc_idx % 2
                g_idx = r * m + (a_col_offset * 2 + c)
                U[g_idx, col] += coeff
            for loc_idx, coeff in v_loc.items():
                r = loc_idx // 2
                c = loc_idx % 2
                g_row = b_row_offset * 2 + r
                g_col = b_col_offset * 2 + c
                g_idx = g_row * p + g_col
                V[g_idx, col] += coeff
            for loc_idx, coeff in w_loc.items():
                r = loc_idx // 2
                c = loc_idx % 2
                g_row = r
                g_col = c_col_offset * 2 + c
                g_idx = g_col * n + g_row
                W[g_idx, col] += coeff

    apply_strassen(0, 0, 0, 0, 0)
    apply_strassen(1, 1, 0, 0, 7)
    apply_strassen(0, 0, 1, 1, 14)
    apply_strassen(1, 1, 1, 1, 21)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END

if __name__ == "__main__":
    start_time = time.time()
    res = construct()
    n, m, p = DIMENSIONS
    U = np.array(res["U"], dtype=float)
    V = np.array(res["V"], dtype=float)
    W = np.array(res["W"], dtype=float)
    
    # Test with random matrices
    np.random.seed(42)
    A = np.random.rand(n, m)
    B = np.random.rand(m, p)
    C = A @ B
    Cf = np.zeros((n, p))
    for ii in range(n):
        for kk in range(p):
            acc = 0.0
            for cc in range(U.shape[1]):
                for jj in range(m):
                    acc += U[ii * m + jj, cc] * V[jj * p + kk, cc] * W[kk * n + ii, cc]
            Cf[ii, kk] = acc
    
    if not np.allclose(C, Cf):
        print("Verification Failed")
        print("C:")
        print(C)
        print("Cf:")
        print(Cf)
    else:
        print("Verification Passed")
        
    print("Rank:", U.shape[1])
    print("Time taken:", time.time() - start_time)
