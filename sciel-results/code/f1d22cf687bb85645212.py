import time

def construct():
    t0 = time.time()
    tmax = t0 + 105.0

    n = m = p = 5

    def verify_decomposition(U, V, W):
        """Verify that the decomposition correctly computes matrix multiplication."""
        import numpy as np
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        rank = U.shape[1]
        
        # For random A, B, check that sum_r U[:,r] * V[:,r] * W[:,r] gives C = A*B
        # The decomposition means: C[i,k] = sum_j A[i,j] B[j,k]
        # In tensor form: T[i,j,k] decomposed as sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        
        for trial in range(10):
            A = np.random.randint(-5, 6, size=(n, m))
            B = np.random.randint(-5, 6, size=(m, p))
            C_expected = A @ B
            
            # Compute via decomposition
            # C[i,k] = sum_j A[i,j] B[j,k]
            # = sum_r (sum_j U[i*m+j,r] A[i,j]) * V_term * W_term
            # Actually: C[i,k] = sum_j sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] * A[i,j] * B[j,k]
            # No, the decomposition is: T[i,j,k] = sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r]
            # And C[i,k] = sum_j A[i,j] B[j,k] T[i,j,k]... no.
            #
            # The matrix multiplication tensor satisfies:
            # C[i,k] = sum_j A[i,j] B[j,k]
            # This is a bilinear computation: for each (i,k), C[i,k] = sum_j A[i,j]*B[j,k]
            # The rank-1 term for (i,j,k) contributes A[i,j]*B[j,k] to C[i,k]
            #
            # In the decomposition: for each r, we compute M_r = (sum_{i,j} U[i*m+j,r]*A[i,j]) * (sum_{j,k} V[j*p+k,r]*B[j,k])
            # And C[i,k] += M_r * W[k*n+i,r]
            
            C_computed = np.zeros((n, p), dtype=float)
            for r in range(rank):
                # Compute the A-linear combination
                a_val = 0.0
                for i in range(n):
                    for j in range(m):
                        if U[i*m+j, r] != 0:
                            a_val += U[i*m+j, r] * A[i, j]
                # Compute the B-linear combination
                b_val = 0.0
                for j in range(m):
                    for k in range(p):
                        if V[j*p+k, r] != 0:
                            b_val += V[j*p+k, r] * B[j, k]
                # Add contribution to C
                for i in range(n):
                    for k in range(p):
                        if W[k*n+i, r] != 0:
                            C_computed[i, k] += a_val * b_val * W[k*n+i, r]
            
            if not np.allclose(C_computed, C_expected, atol=1e-10):
                return False
        return True

    def build_decomposition(U_cols, V_cols, W_cols):
        rank = len(U_cols)
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        for r in range(rank):
            for idx, val in U_cols[r].items():
                U[idx][r] = val
            for idx, val in V_cols[r].items():
                V[idx][r] = val
            for idx, val in W_cols[r].items():
                W[idx][r] = val
        return U, V, W

    # Naive 5x5x5 decomposition (rank 125)
    def naive_5x5x5():
        U_cols = []
        V_cols = []
        W_cols = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_cols.append({i * m + j: 1})
                    V_cols.append({j * p + k: 1})
                    W_cols.append({k * n + i: 1})
        return U_cols, V_cols, W_cols

    # Try Strassen-based 4x4x4 core (rank 49) + naive borders
    def strassen_4x4_core():
        """
        Implement Strassen-based rank-49 decomposition for 4x4x4.
        Uses two levels: 4x4 split into 2x2 blocks, each 2x2 uses rank-7 Strassen.
        """
        # First, define the 7 rank-1 terms for 2x2x2 Strassen.
        # For 2x2x2, U index = i*2+j, V index = j*2+k, W index = k*2+i
        # (i,j,k in {0,1})
        
        # I need to carefully define the 7 Strassen terms.
        # Let me use the standard formulation:
        
        # The 2x2x2 multiplication tensor has rank 7. Here are the 7 rank-1 terms:
        # Each term is (U_coeffs, V_coeffs, W_coeffs) where each is a dict from index to value.
        
        # Using the standard Strassen decomposition:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)(B00)
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)(B11)
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # For each M_s, it's a product of linear combinations of A entries and B entries.
        # M_s = (sum_{i,j} alpha_{s,i,j} A[i,j]) * (sum_{j,k} beta_{s,j,k} B[j,k])
        # And M_s contributes to C entries: C[i,k] += gamma_{s,i,k} * M_s
        
        # So each M_s gives ONE rank-1 term in the tensor decomposition with:
        # U[i*2+j] = alpha_{s,i,j}
        # V[j*2+k] = beta_{s,j,k}
        # W[k*2+i] = gamma_{s,i,k}
        
        # Wait, that's not quite right either. A single M_s = a_val * b_val where
        # a_val = sum alpha*A, b_val = sum beta*B. Then M_s contributes gamma*M_s to each
        # C entry. So for each s, we have one "bilinear form" a_s * b_s, and it's added
        # to multiple C entries with different weights gamma.
        
        # In the rank-1 tensor decomposition sense, each M_s is NOT a single rank-1 term
        # unless it contributes to exactly one C entry. Since M_s contributes to multiple
        # C entries, we need to think of it differently.
        
        # Actually, in the tensor decomposition framework:
        # T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
        # 
        # For the bilinear computation C = A*B, we have:
        # C[i,k] = sum_j A[i,j] B[j,k]
        # 
        # A rank-r decomposition gives r bilinear forms:
        # M_r = (sum_{i,j} U[i*m+j,r] A[i,j]) * (sum_{j,k} V[j*p+k,r] B[j,k])
        # C[i,k] = sum_r W[k*n+i,r] * M_r
        #
        # So for Strassen's 7 products, each product M_s is one bilinear form.
        # But M_s = a_s * b_s where a_s depends only on A and b_s depends only on B.
        # Then C[i,k] = sum_s gamma_{s,i,k} M_s = sum_s gamma_{s,i,k} a_s b_s
        #
        # This means: U[i*2+j,s] = alpha_{s,i,j}, V[j*2+k,s] = beta_{s,j,k}, W[k*2+i,s] = gamma_{s,i,k}
        #
        # But wait, this would mean each s has potentially non-zero U, V, W entries for
        # multiple (i,j), (j,k), (k,i) pairs. That's fine - it's still one rank-1 term
        # in the sense that it's one bilinear form a_s * b_s weighted by gamma for each C entry.
        
        # Hmm, but actually no. In the standard tensor rank decomposition, each rank-1
        # term is a pure tensor e_i ⊗ e_j ⊗ e_k, meaning U, V, W each have exactly one
        # non-zero entry. The bilinear form interpretation is different.
        
        # Let me re-examine. The matrix multiplication tensor T ∈ F^{n×m} ⊗ F^{m×p} ⊗ F^{n×p}
        # is defined by T = sum_{i,j,k} e_{i,j}^A ⊗ e_{j,k}^B ⊗ e_{i,k}^C
        # where e_{i,j}^A is the standard basis for A (n×m matrix), etc.
        
        # A rank-1 decomposition: T = sum_r u_r ⊗ v_r ⊗ w_r
        # where u_r ∈ F^{n×m}, v_r ∈ F^{m×p}, w_r ∈ F^{n×p}
        
        # For the naive decomposition: T = sum_{i,j,k} e_{i,j}^A ⊗ e_{j,k}^B ⊗ e_{i,k}^C
        # Each term has u = e_{i,j}^A (one non-zero), v = e_{j,k}^B (one non-zero), w = e_{i,k}^C (one non-zero)
        
        # For Strassen, the rank-7 decomposition of 2×2×2 is:
        # T = sum_{s=1}^7 u_s ⊗ v_s ⊗ w_s
        # where u_s, v_s, w_s are 2×2 matrices (vectors in F^4).
        
        # So u_s can have multiple non-zero entries! It's a vector in F^4, not a basis vector.
        
        # OK so let me just define the 7 terms for 2x2x2 Strassen properly.
        
        # I'll use the well-known rank-7 decomposition. Let me look up the exact coefficients.
        
        # One standard formulation (from Strassen 1969):
        # Let A = [[a, b], [c, d]], B = [[e, f], [g, h]]
        # M1 = (a+d)(e+h)
        # M2 = (c+d)e
        # M3 = a(f-h)
        # M4 = d(g-e)
        # M5 = (a+b)h
        # M6 = (c-a)(e+f)
        # M7 = (b-d)(g+h)
        
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # Mapping to indices: A[i,j] with i,j in {0,1}
        # a=A[0,0], b=A[0,1], c=A[1,0], d=A[1,1]
        # e=B[0,0], f=B[0,1], g=B[1,0], h=B[1,1]
        
        # U index for A[i,j]: i*2+j
        # V index for B[j,k]: j*2+k  (note: B is indexed by [row, col] = [j, k])
        # W index for C[i,k]: k*2+i  (per the convention W[k*n+i] with n=2)
        
        # Term s=0 (M1): a_val = a+d = A[0,0]+A[1,1], b_val = e+h = B[0,0]+B[1,1]
        # U: {0: 1, 3: 1} (A[0,0], A[1,1])
        # V: {0: 1, 3: 1} (B[0,0], B[1,1])
        # W: C11 += 1, C22 += 1 -> C[0,0] += 1, C[1,1] += 1
        # W index for C[0,0]: 0*2+0 = 0, for C[1,1]: 1*2+1 = 3
        # W: {0: 1, 3: 1}
        
        # Term s=1 (M2): a_val = c+d = A[1,0]+A[1,1], b_val = e = B[0,0]
        # U: {2: 1, 3: 1}
        # V: {0: 1}
        # W: C21 += 1, C22 -= 1 -> C[1,0] += 1, C[1,1] -= 1
        # W index for C[1,0]: 0*2+1 = 1, for C[1,1]: 1*2+1 = 3
        # W: {1: 1, 3: -1}
        
        # Term s=2 (M3): a_val = a = A[0,0], b_val = f-h = B[0,1]-B[1,1]
        # U: {0: 1}
        # V: {1: 1, 3: -1}
        # W: C12 += 1, C22 += 1 -> C[0,1] += 1, C[1,1] += 1
        # W index for C[0,1]: 1*2+0 = 2, for C[1,1]: 1*2+1 = 3
        # W: {2: 1, 3: 1}
        
        # Term s=3 (M4): a_val = d = A[1,1], b_val = g-e = B[1,0]-B[0,0]
        # U: {3: 1}
        # V: {2: 1, 0: -1}
        # W: C11 += 1, C21 += 1 -> C[0,0] += 1, C[1,0] += 1
        # W index for C[0,0]: 0, for C[1,0]: 1
        # W: {0: 1, 1: 1}
        
        # Term s=4 (M5): a_val = a+b = A[0,0]+A[0,1], b_val = h = B[1,1]
        # U: {0: 1, 1: 1}
        # V: {3: 1}
        # W: C11 -= 1, C12 += 1 -> C[0,0] -= 1, C[0,1] += 1
        # W index for C[0,0]: 0, for C[0,1]: 2
        # W: {0: -1, 2: 1}
        
        # Term s=5 (M6): a_val = c-a = A[1,0]-A[0,0], b_val = e+f = B[0,0]+B[0,1]
        # U: {2: 1, 0: -1}
        # V: {0: 1, 1: 1}
        # W: C22 += 1 -> C[1,1] += 1
        # W index for C[1,1]: 3
        # W: {3: 1}
        
        # Term s=6 (M7): a_val = b-d = A[0,1]-A[1,1], b_val = g+h = B[1,0]+B[1,1]
        # U: {1: 1, 3: -1}
        # V: {2: 1, 3: 1}
        # W: C11 += 1 -> C[0,0] += 1
        # W index for C[0,0]: 0
        # W: {0: 1}
        
        s2x2 = [
            ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
            ({2: 1, 3: 1}, {0: 1}, {1: 1, 3: -1}),
            ({0: 1}, {1: 1, 3: -1}, {2: 1, 3: 1}),
            ({3: 1}, {2: 1, 0: -1}, {0: 1, 1: 1}),
            ({0: 1, 1: 1}, {3: 1}, {0: -1, 2: 1}),
            ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
            ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
        ]
        
        # Verify 2x2x2 Strassen
        import numpy as np
        for trial in range(5):
            A2 = np.random.randint(-5, 6, size=(2, 2))
            B2 = np.random.randint(-5, 6, size=(2, 2))
            C2_expected = A2 @ B2
            C2_computed = np.zeros((2, 2), dtype=float)
            for r in range(7):
                U_r, V_r, W_r = s2x2[r]
                a_val = sum(U_r.get(i*2+j, 0) * A2[i,j] for i in range(2) for j in range(2))
                b_val = sum(V_r.get(j*2+k, 0) * B2[j,k] for j in range(2) for k in range(2))
                for i in range(2):
                    for k in range(2):
                        C2_computed[i,k] += W_r.get(k*2+i, 0) * a_val * b_val
            if not np.allclose(C2_computed, C2_expected, atol=1e-10):
                # Skip - just use naive for now
                return None
        
        # Now build 4x4x4 from 2x2 blocks using Strassen.
        # 4x4 A = [[A00, A01], [A10, A11]] where each is 2x2
        # 4x4 B = [[B00, B01], [B10, B11]]
        # 4x4 C = [[C00, C01], [C10, C11]]
        
        # Block-level Strassen (same 7 terms but on 2x2 blocks):
        # M0 = (A00+A11)(B00+B11)
        # M1 = (A10+A11)B00
        # M2 = A00(B01-B11)
        # M3 = A11(B10-B00)
        # M4 = (A00+A01)B11
        # M5 = (A10-A00)(B00+B01)
        # M6 = (A01-A11)(B10+B11)
        
        # C00 = M0 + M3 - M4 + M6
        # C01 = M2 + M4
        # C10 = M1 + M3
        # C11 = M0 - M1 + M2 + M5
        
        # Each M_s is a 2x2 matrix product, decomposable into 7 rank-1 terms (from s2x2).
        # Total rank for 4x4x4: 7 * 7 = 49.
        
        U_cols = []
        V_cols = []
        W_cols = []
        
        # For each block-level term s (0..6) and each within-block term t (0..6):
        # We need to compute the contribution to each C block.
        
        # Block A indices: block (bi,bj) has 4x4 indices i=bi*2+ai, j=bj*2+aj
        # So U index = i*4+j = (bi*2+ai)*4 + (bj*2+aj)
        
        # Block B indices: block (bj2,bk2) has 4x4 indices j=bj2*2+aj2, k=bk2*2+ak2
        # So V index = j*4+k = (bj2*2+aj2)*4 + (bk2*2+ak2)
        
        # Block C indices: block (bi2,bk2) has 4x4 indices i=bi2*2+ai, k=bk2*2+ak
        # So W index = k*4+i = (bk2*2+ak)*4 + (bi2*2+ai)
        
        # Define which blocks each M_s involves:
        # For each s, we have:
        # A part: dict from (bi,bj) to coefficient
        # B part: dict from (bj2,bk2) to coefficient
        # C part: dict from (bi2,bk2) to coefficient (for the overall C combination)
        
        # But actually, the C part is not per-M_s; it's the combination:
        # C_block gets contributions from various M_s with certain signs.
        
        # Let me restructure. For the 4x4x4 decomposition:
        # C[i,k] = sum_{blocks} sum_s gamma_{s,block_C} * M_s
        # where M_s = a_s * b_s is a bilinear form on the A and B blocks.
        
        # For each s, a_s = sum_{(bi,bj)} alpha_{s,bi,bj} * A_block(bi,bj)
        # where A_block is a 2x2 matrix, and the sum means element-wise.
        # Similarly for b_s.
        
        # Then M_s = (sum_{(bi,bj),ai,aj} alpha_{s,bi,bj} A[bi*2+ai, bj*2+aj]) *
        #              (sum_{(bj2,bk2),aj2,ak2} beta_{s,bj2,bk2} B[bj2*2+aj2, bk2*2+ak2])
        
        # And C[bi2*2+ai, bk2*2+ak] += gamma_{s,bi2,bk2} * M_s
        
        # So for each (s, t) where t is the within-block Strassen term:
        # The within-block term t has coefficients for the 2x2 submatrices.
        # U for the 4x4: for each (bi,bj) with alpha_{s,bi,bj} != 0,
        #   and for each (ai,aj) in the 2x2 submatrix with coefficient u_t[ai*2+aj],
        #   U[(bi*2+ai)*4 + (bj*2+aj)] = alpha_{s,bi,bj} * u_t[ai*2+aj]
        
        # Similarly for V and W.
        
        # Let me define the block-level Strassen coefficients:
        
        # For M_s = a_s * b_s:
        # a_s involves A blocks with coefficients:
        A_block_coeffs = [
            {(0,0): 1, (1,1): 1},  # s=0: A00+A11
            {(1,0): 1, (1,1): 1},  # s=1: A10+A11
            {(0,0): 1},             # s=2: A00
            {(1,1): 1},             # s=3: A11
            {(0,0): 1, (0,1): 1},  # s=4: A00+A01
            {(1,0): 1, (0,0): -1}, # s=5: A10-A00
            {(0,1): 1, (1,1): -1}, # s=6: A01-A11
        ]
        
        B_block_coeffs = [
            {(0,0): 1, (1,1): 1},  # s=0: B00+B11
            {(0,0): 1},             # s=1: B00
            {(0,1): 1, (1,1): -1}, # s=2: B01-B11
            {(1,0): 1, (0,0): -1}, # s=3: B10-B00
            {(1,1): 1},             # s=4: B11
            {(0,0): 1, (0,1): 1},  # s=5: B00+B01
            {(1,0): 1, (1,1): 1},  # s=6: B10+B11
        ]
        
        # C combination: C_block(bi2,bk2) = sum_s gamma_{s,bi2,bk2} * M_s
        C_block_coeffs = {
            (0,0): {0: 1, 3: 1, 4: -1, 6: 1},   # C00 = M0 + M3 - M4 + M6
            (0,1): {2: 1, 4: 1},                  # C01 = M2 + M4
            (1,0): {1: 1, 3: 1},                  # C10 = M1 + M3
            (1,1): {0: 1, 1: -1, 2: 1, 5: 1},    # C11 = M0 - M1 + M2 + M5
        }
        
        # Now build the 49 rank-1 terms for 4x4x4.
        for s in range(7):
            for t in range(7):
                U_t, V_t, W_t = s2x2[t]
                
                U_col = {}
                V_col = {}
                W_col = {}
                
                # A part
                for (bi, bj), alpha in A_block_coeffs[s].items():
                    for (ai, aj), u_coeff in U_t.items():
                        u_4x4_idx = (bi*2+ai)*4 + (bj*2+aj)
                        val = alpha * u_coeff
                        U_col[u_4x4_idx] = U_col.get(u_4x4_idx, 0) + val
                
                # B part
                for (bj2, bk2), beta in B_block_coeffs[s].items():
                    for (aj2, ak2), v_coeff in V_t.items():
                        # V_t is indexed by j*2+k for the 2x2 case
                        # Here j corresponds to the row of B block, k to column
                        # B block (bj2, bk2) has elements B[bj2*2+aj2, bk2*2+ak2]
                        # V index in 4x4: j*4+k = (bj2*2+aj2)*4 + (bk2*2+ak2)
                        v_4x4_idx = (bj2*2+aj2)*4 + (bk2*2+ak2)
                        val = beta * v_coeff
                        V_col[v_4x4_idx] = V_col.get(v_4x4_idx, 0) + val
                
                # C part
                for (bi2, bk2), gamma in C_block_coeffs[(bi2, bk2)].items():
                    if gamma != 0 and s == bi2:  # Only process if this s contributes to this C block
                        pass  # We'll handle this below
                
                # Actually, I need to accumulate C contributions properly.
                # For each C block (bi2,bk2), the contribution from M_s is gamma_{s,bi2,bk2} * M_s.
                # M_s itself, when expanded via within-block term t, contributes to specific
                # elements of the 2x2 C block.
                
                # Let me redo: for each s, the C contribution to block (bi2,bk2) is
                # gamma_{s,bi2,bk2} * (sum over within-block terms t of W_t contributions)
                
                # Actually, the within-block Strassen already handles the C combination
                # within each 2x2 block. So for M_s contributing to C block (bi2,bk2)
                # with weight gamma, the within-block W coefficients get multiplied by gamma.
                
                # But the C block coefficients gamma depend on s, not on t.
                # So for a given s, the total C contribution to block (bi2,bk2) is
                # gamma_{s,bi2,bk2} * (sum_t W_t[...])
                
                # Hmm, but t is also a loop variable. Let me restructure.
                
                # For a fixed s, M_s = a_s * b_s. The contribution of M_s to C block (bi2,bk2)
                # is gamma_{s,bi2,bk2} * M_s. Since M_s is a 2x2 matrix product, its (ai,ak)
                # element is sum_j A_block[ai,j] B_block[j,ak].
                
                # The within-block Strassen decomposes this 2x2 product into 7 bilinear forms.
                # So for fixed s, the contribution to C block (bi2,bk2) element (ai,ak) is:
                # gamma_{s,bi2,bk2} * sum_t W_t[ak*2+ai] * (a_t) * (b_t)
                # where a_t and b_t are the t-th within-block bilinear forms.
                
                # So for the full 4x4x4 decomposition, for each (s,t) pair:
                # The A part is: sum_{(bi,bj)} alpha_{s,bi,bj} * A_block(bi,bj) expanded via U_t
                # The B part is: sum_{(bj2,bk2)} beta_{s,bj2,bk2} * B_block(bj2,bk2) expanded via V_t
                # The C part is: for each C block (bi2,bk2) with gamma_{s,bi2,bk2} != 0,
                #   W[(bk2*2+ak)*4 + (bi2*2+ai)] = gamma_{s,bi2,bk2} * W_t[ak*2+ai]
                
                # Let me just compute this directly.
                
                W_col = {}
                for (bi2, bk2) in C_block_coeffs:
                    gamma = C_block_coeffs[(bi2, bk2)].get(s, 0)
                    if gamma != 0:
                        for (ak, ai), w_coeff in W_t.items():
                            # C block (bi2,bk2) element (ai,ak)
                            # 4x4 C index: i=bi2*2+ai, k=bk2*2+ak
                            # W index: k*4+i = (bk2*2+ak)*4 + (bi2*2+ai)
                            w_4x4_idx = (bk2*2+ak)*4 + (bi2*2+ai)
                            val = gamma * w_coeff
                            W_col[w_4x4_idx] = W_col.get(w_4x4_idx, 0) + val
                
                # Remove zero entries
                U_col = {k: v for k, v in U_col.items() if v != 0}
                V_col = {k: v for k, v in V_col.items() if v != 0}
                W_col = {k: v for k, v in W_col.items() if v != 0}
                
                if U_col and V_col and W_col:
                    U_cols.append(U_col)
                    V_cols.append(V_col)
                    W_cols.append(W_col)
        
        return U_cols, V_cols, W_cols

    # Build the full 5x5x5 decomposition using Strassen 4x4 core + naive borders
    def strassen_based_5x5x5():
        core_U, core_V, core_W = strassen_4x4_core()
        if core_U is None:
            return None
        
        U_cols = []
        V_cols = []
        W_cols = []
        
        # Core 4x4x4 terms (indices 0..3 for i,j,k)
        for r in range(len(core_U)):
            U_cols.append(core_U[r])
            V_cols.append(core_V[r])
            W_cols.append(core_W[r])
        
        # Border terms (naive)
        # 1. i in 0..3, j=4, k in 0..3: A[i,4]*B[4,k] -> C[i,k]
        for i in range(4):
            for k in range(4):
                U_cols.append({i * 5 + 4: 1})
                V_cols.append({4 * 5 + k: 1})
                W_cols.append({k * 5 + i: 1})
        
        # 2. i in 0..3, j in 0..3, k=4: A[i,j]*B[j,4] -> C[i,4]
        for i in range(4):
            for j in range(4):
                U_cols.append({i * 5 + j: 1})
                V_cols.append({j * 5 + 4: 1})
                W_cols.append({4 * 5 + i: 1})
        
        # 3. i=4, j in 0..3, k in 0..3: A[4,j]*B[j,k] -> C[4,k]
        for j in range(4):
            for k in range(4):
                U_cols.append({4 * 5 + j: 1})
                V_cols.append({j * 5 + k: 1})
                W_cols.append({k * 5 + 4: 1})
        
        # 4. i in 0..3, j=4, k=4: A[i,4]*B[4,4] -> C[i,4]
        for i in range(4):
            U_cols.append({i * 5 + 4: 1})
            V_cols.append({4 * 5 + 4: 1})
            W_cols.append({4 * 5 + i: 1})
        
        # 5. i=4, j in 0..3, k=4: A[4,j]*B[j,4] -> C[4,4]
        for j in range(4):
            U_cols.append({4 * 5 + j: 1})
            V_cols.append({j * 5 + 4: 1})
            W_cols.append({4 * 5 + 4: 1})
        
        # 6. i=4, j=4, k in 0..3: A[4,4]*B[4,k] -> C[4,k]
        for k in range(4):
            U_cols.append({4 * 5 + 4: 1})
            V_cols.append({4 * 5 + k: 1})
            W_cols.append({k * 5 + 4: 1})
        
        # 7. i=4, j=4, k=4: A[4,4]*B[4,4] -> C[4,4]
        U_cols.append({4 * 5 + 4: 1})
        V_cols.append({4 * 5 + 4: 1})
        W_cols.append({4 * 5 + 4: 1})
        
        return U_cols, V_cols, W_cols

    # Try the Strassen-based decomposition first
    U_cols, V_cols, W_cols = strassen_based_5x5x5()
    
    if U_cols is None:
        # Fall back to naive
        U_cols, V_cols, W_cols = naive_5x5x5()
    
    U, V, W = build_decomposition(U_cols, V_cols, W_cols)
    
    # Verify
    if not verify_decomposition(U, V, W):
        # Fall back to naive if Strassen is wrong
        U_cols, V_cols, W_cols = naive_5x5x5()
        U, V, W = build_decomposition(U_cols, V_cols, W_cols)
    
    return {"U": U, "V": V, "W": W}
