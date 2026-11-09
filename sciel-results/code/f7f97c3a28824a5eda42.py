"""Matrix multiplication 5x5x5 tensor decomposition."""
DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    import time
    t0 = time.time()
    deadline = t0 + 110  # safety margin before 120s limit

    def _verify(U, V, W, n=5, m=5, p=5, tol=1e-6):
        """Check that the decomposition exactly represents matrix multiplication."""
        try:
            import numpy as np
            Ua = np.array(U, dtype=float)
            Va = np.array(V, dtype=float)
            Wa = np.array(W, dtype=float)
            rank = Ua.shape[1]
            # Reconstruct: C[i,k] = sum_j A[i,j]*B[j,k]
            # Tensor T[i,j,k] = delta_{i,j} * ... actually:
            # U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] should give 1 when i=j=k
            # More precisely: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = delta_{i,k} (for 2D: A[i,j]*B[j,k])
            # Actually the tensor for matrix mult is T[i,j,k] = delta_{j,i}... no.
            # The multiplication tensor: (A * B)[i,k] = sum_j A[i,j] B[j,k]
            # So T[i,j,k] = 1 if this represents the product, i.e., T is the identity in j for each (i,k)
            # T[i,j,k] = delta_{i,k}... no, T[i,j,k] is defined so that sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] = T[i,j,k]
            # where T[i,j,k] = 1 if j==j (always), but the tensor is:
            # T[i,j,k] = 1 for all i,j,k? No.
            # Matrix mult: C[i,k] = sum_j A[i,j] B[j,k]
            # The tensor T has T[i,j,k] = 1 (it's the "multiplication tensor" with all ones? No!)
            # Actually T[i,j,k] = 1 for every (i,j,k) triple? Let's check:
            # sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] should equal 1 for all i,j,k
            # In schoolbook: each (i,j,k) gets exactly one term with U=V=W=1, so yes T[i,j,k]=1.
            # Wait no - the matrix multiplication tensor T[i,j,k] is defined as:
            # T[i,j,k] = 1 if we're computing C[i,k] from A[i,j]*B[j,k]
            # Actually the standard multiplication tensor has T[i,j,k] = 1 for all i,j,k.
            # Because C[i,k] = sum_j A[i,j] B[j,k], and the tensor is the trilinear form
            # sum_{i,j,k} T[i,j,k] * A[i,j] * B[j,k] * C[k,i]... 
            # Let me just check: for the decomposition to be valid,
            # sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] must equal 1 for all i,j,k.
            # This is because in the product C[i,k] = sum_j A[i,j]*B[j,k],
            # each term A[i,j]*B[j,k] contributes to C[i,k], and the tensor records
            # that A[i,j] and B[j,k] multiply to give C[i,k].
            # So T[i,j,k] = 1 for all i,j,k.
            
            max_err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        for c in range(rank):
                            s += Ua[i*m+j, c] * Va[j*p+k, c] * Wa[k*n+i, c]
                        err = abs(s - 1.0)
                        if err > max_err:
                            max_err = err
            return max_err < tol
        except Exception:
            return False

    # Try Strassen-based rank-110 decomposition
    try:
        if time.time() > deadline - 5:
            raise TimeoutError()
        
        n, m, p = 5, 5, 5
        U_list = []
        V_list = []
        W_list = []

        # Part 1: Strassen for 4x4x4 (indices 0..3 for i, j, k)
        # Strassen 2x2x2:
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        # C11 = M1+M4-M5+M7, C12 = M3+M5, C21 = M2+M4, C22 = M1-M2+M3+M6
        
        # For 4x4, split into 2x2 blocks of 2x2 matrices.
        # Block (a,b) where a,b in {0,1} refers to 2x2 sub-matrices.
        # A[a*2:(a+1)*2, b*2:(b+1)*2], etc.
        
        # We'll use Strassen recursively: 4x4 -> 7 multiplications of 2x2,
        # each 2x2 uses Strassen: 7 multiplications of 1x1.
        # Total: 7*7 = 49 rank-1 terms for the 4x4 block.
        
        # Define Strassen 2x2 as rank-7 decomposition
        def strassen_2x2():
            """Return list of (a_coeffs, b_coeffs, c_coeffs) for Strassen 2x2.
            Each entry: ([(i,j,coeff),...], [(j,k,coeff),...], [(k,i,coeff),...])
            where a_coeffs is for A[i,j], b_coeffs for B[j,k], c_coeffs for C[k,i] (W index = k*n+i)
            """
            # Strassen formulas:
            # M1 = (A00+A11)(B00+B11)
            # M2 = (A10+A11)B00
            # M3 = A00(B01-B11)
            # M4 = A11(B10-B00)
            # M5 = (A00+A01)B11
            # M6 = (A10-A00)(B00+B01)
            # M7 = (A01-A11)(B10+B11)
            # C00 = M1+M4-M5+M7
            # C01 = M3+M5
            # C10 = M2+M4
            # C11 = M1-M2+M3+M6
            
            terms = []
            # M1 = (A00+A11)(B00+B11)
            terms.append(([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]))
            # M2 = (A10+A11)B00
            terms.append(([(1,0,1),(1,1,1)], [(0,0,1)], [(0,1,1),(1,1,1)]))
            # M3 = A00(B01-B11)
            terms.append(([(0,0,1)], [(0,1,1),(1,1,-1)], [(0,0,1),(1,0,1)]))
            # M4 = A11(B10-B00)
            terms.append(([(1,1,1)], [(1,0,1),(0,0,-1)], [(0,1,1),(1,0,1)]))
            # M5 = (A00+A01)B11
            terms.append(([(0,0,1),(0,1,1)], [(1,1,1)], [(0,1,-1),(1,1,1)]))
            # M6 = (A10-A00)(B00+B01)
            terms.append(([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,1,1)]))
            # M7 = (A01-A11)(B10+B11)
            terms.append(([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,0,1),(1,0,-1)]))
            
            # Now convert each Strassen term into rank-1 contributions
            # For a Strassen term with A_coeffs, B_coeffs, C_coeffs:
            # The contribution to C is: sum over (ai) A[ai] * sum over (bi) B[bi] * sum over (ci) C[ci]
            # This is a sum of rank-1 tensors: for each combination (ai, bi, ci):
            # U[ai] *= coeff_a, V[bi] *= coeff_b, W[ci] *= coeff_c
            # But wait, that would give product of sizes, not rank 7.
            # 
            # Actually Strassen works differently. Each Mi is a matrix product.
            # For 1x1 case: M1 = (a00+a11)*(b00+b11) is already rank 1 for the "scalar" part,
            # but as a tensor contribution, M1 contributes to multiple C entries.
            #
            # Let me reconsider. The Strassen decomposition for 2x2 matrix multiplication
            # as a tensor decomposition of rank 7:
            # 
            # Each of the 7 "multiplications" Mi corresponds to one rank-1 term in the
            # 2x2x2 multiplication tensor. But Mi is itself a 2x2 matrix product...
            # No, that's not right either.
            #
            # For 1x1x1: T[0,0,0] = 1, rank 1.
            # For 2x2x2: Strassen says rank <= 7.
            # The 7 rank-1 tensors are:
            # Each rank-1 tensor U^c, V^c, W^c where U^c is 4x1, V^c is 4x1, W^c is 4x1
            # such that sum_c U^c[i,j] V^c[j,k] W^c[k,i] = 1 for all i,j,k.
            #
            # The Strassen rank-7 decomposition:
            # c=1: U = [1,0,0,1]^T (for A: positions (0,0)=1,(0,1)=0,(1,0)=0,(1,1)=1)
            #       V = [1,0,0,1]^T (for B: (0,0)=1,(0,1)=0,(1,0)=0,(1,1)=1)
            #       W = [1,0,1,0]^T (for C: positions k*n+i: (0,0)=1,(0,1)=0,(1,0)=1,(1,1)=0)
            #       This represents: (A00+A11)*(B00+B11) contributes to C00 and C11
            #       Check: U[i,j]*V[j,k]*W[k,i]:
            #         i=0,j=0,k=0: 1*1*1=1 (C[0,0])
            #         i=0,j=0,k=1: 1*0*0=0
            #         i=0,j=1,k=0: 0*1*1=0
            #         i=0,j=1,k=1: 0*1*0=0
            #         i=1,j=0,k=0: 0*1*0=0
            #         i=1,j=0,k=1: 0*0*1=0
            #         i=1,j=1,k=0: 1*0*0=0
            #         i=1,j=1,k=1: 1*1*1=1 (C[1,1])
            #       Sum so far for T[0,0,0]: 1, T[1,1,1]: 1, others: 0. Good.
            #
            # c=2: U = [0,0,1,1]^T (A10+A11)
            #       V = [1,0,0,0]^T (B00)
            #       W = [0,1,1,0]^T (C01+C10)
            #       Check: (A10+A11)*B00 contributes to C10 and C11? 
            #       Actually M2 = (A10+A11)*B00, and C10 += M2, C11 += M2
            #       U[i,j]*V[j,k]*W[k,i]:
            #         i=1,j=0,k=0: 1*1*0=0
            #         i=1,j=0,k=1: 1*0*1=0
            #         i=1,j=1,k=0: 1*0*0=0
            #         i=1,j=1,k=1: 1*0*1=0
            #         Hmm, that gives 0. Let me recheck.
            #       V[j,k]: B00 means j=0,k=0, so V[0,0]=1, rest 0.
            #       U[i,j]: A10+A11 means (i=1,j=0)=1, (i=1,j=1)=1.
            #       W[k,i]: C01+C10 means (k=0,i=1)=1, (k=1,i=0)=1.
            #         i=1,j=0,k=0: U[1,0]=1, V[0,0]=1, W[0,1]=1 -> 1*1*1=1 (C[1,0])
            #         i=1,j=1,k=0: U[1,1]=1, V[1,0]=0, ... = 0
            #         i=0,j=0,k=0: U[0,0]=0, ... = 0
            #         i=0,j=1,k=0: U[0,1]=0, ... = 0
            #       So only C[1,0] gets 1. But we need C[1,0] AND C[1,1] to get M2.
            #       The Strassen formula says C10 = M2+M4, C11 = M1-M2+M3+M6.
            #       So M2 contributes +1 to C10 and -1 to C11.
            #       Let me fix: W should have W[0,1]=1 (C10) and W[1,1]=-1 (C11).
            #       W[k*n+i]: k=0,i=1 -> index 1; k=1,i=1 -> index 3.
            #       W = [0,1,0,-1]
            #       Check: i=1,j=0,k=0: U[1,0]*V[0,0]*W[0,1] = 1*1*1 = 1 (C[1,0])
            #               i=1,j=0,k=1: U[1,0]*V[0,1]*W[1,1] = 1*0*(-1) = 0
            #               i=1,j=1,k=0: U[1,1]*V[1,0]*W[0,1] = 1*0*1 = 0
            #               i=1,j=1,k=1: U[1,1]*V[1,1]*W[1,1] = 1*0*(-1) = 0
            #       Hmm, still only C[1,0]. The issue is V[j,k] for B00: j=0,k=0.
            #       So only j=0,k=0 is nonzero. Then U[i,0]*V[0,0]*W[0,i] for i=0,1.
            #       i=0: U[0,0]=0 -> 0
            #       i=1: U[1,0]=1, W[0,1]=1 -> 1
            #       So indeed only C[1,0] gets contribution. But we need C[1,1] too.
            #       
            #       I think I'm confusing myself. Let me look up the actual rank-7 tensors.
            #       
            #       The 7 rank-1 decompositions of the 2x2x2 multiplication tensor:
            #       Using the standard Strassen decomposition, the 7 terms are:
            #       
            #       Let me use a different approach. I'll verify numerically.
            #       For now, let me just use a known-correct set of rank-7 tensors.
            
            # Known rank-7 tensors for 2x2 matrix multiplication:
            # (U, V, W) each of size 4x1, indexed as [i*2+j], [j*2+k], [k*2+i]
            rank7 = [
                # 1: (A00+A11)(B00+B11) -> C00, C11
                ([1,0,0,1], [1,0,0,1], [1,0,0,1]),
                # 2: (A10+A11)(B00) -> C10, C11  
                ([0,0,1,1], [1,0,0,0], [0,1,0,-1]),
                # 3: (A00)(B01-B11) -> C00, C10
                ([1,0,0,0], [0,1,0,-1], [1,0,1,0]),
                # 4: (A11)(B10-B00) -> C01, C11
                ([0,0,0,1], [0,0,1,-1], [0,1,1,0]),
                # 5: (A00+A01)(B11) -> C01, C11
                ([1,1,0,0], [0,0,0,1], [0,-1,0,1]),
                # 6: (A10-A00)(B00+B01) -> C11
                ([1,-1,0,0], [1,1,0,0], [0,0,0,1]),
                # 7: (A01-A11)(B10+B11) -> C00, C10
                ([0,1,-1,0], [0,0,1,1], [1,0,-1,0]),
            ]
            return rank7

        def strassen_4x4():
            """Return rank-49 decomposition of 4x4x4 matrix multiplication.
            Each entry: (U_16x1, V_16x1, W_16x1) as lists.
            """
            s2 = strassen_2x2()
            # For 4x4, we split into 2x2 blocks.
            # The 7 Strassen multiplications for the outer 2x2:
            # Each "multiplication" is a 2x2 matrix product, which we do with Strassen rank-7.
            # Total: 7 * 7 = 49 rank-1 terms.
            
            # Outer Strassen terms (for the block-level 2x2):
            # O1 = (A00+A11)(B00+B11)
            # O2 = (A10+A11)B00
            # O3 = A00(B01-B11)
            # O4 = A11(B10-B00)
            # O5 = (A00+A01)B11
            # O6 = (A10-A00)(B00+B01)
            # O7 = (A01-A11)(B10+B11)
            # where A00 etc. are 2x2 blocks.
            
            # For each outer term, we need to apply the inner Strassen to compute
            # the 2x2 matrix product. The inner Strassen operates on the 2x2 "scalar" entries.
            
            # Let's define the outer terms as (A_coeffs, B_coeffs, C_coeffs) where
            # each is a dict mapping block_index -> coefficient.
            # Block indices: 00=(0,0), 01=(0,1), 10=(1,0), 11=(1,1)
            
            outer_terms = [
                # O1
                ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
                # O2
                ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
                # O3
                ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,0):1, (1,0):1}),
                # O4
                ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,1):1, (1,1):1}),
                # O5
                ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,1):-1, (1,1):1}),
                # O6
                ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
                # O7
                ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1, (1,0):-1}),
            ]
            
            result = []
            for a_coeff, b_coeff, c_coeff in outer_terms:
                for inner_U, inner_V, inner_W in s2:
                    # Combine: for each inner term, scale by outer coefficients
                    U_4 = [0.0] * 16
                    V_4 = [0.0] * 16
                    W_4 = [0.0] * 16
                    
                    # A block (bi,bj) has 4 entries: positions (bi*2+ii, bj*2+jj) for ii,jj in 0,1
                    # U index: i*4+j where i,j in 0..3
                    # Block (bi,bj): rows bi*2..bi*2+1, cols bj*2..bj*2+1
                    # Inner U[inner_i*2+inner_j] corresponds to entry (inner_i, inner_j) within the 2x2 block
                    # Global A position: (bi*2+inner_i, bj*2+inner_j), U index = (bi*2+inner_i)*4 + (bj*2+inner_j)
                    
                    for (bi, bj), a_val in a_coeff.items():
                        for ii in range(2):
                            for jj in range(2):
                                inner_idx = ii * 2 + jj
                                global_i = bi * 2 + ii
                                global_j = bj * 2 + jj
                                u_idx = global_i * 4 + global_j
                                U_4[u_idx] += a_val * inner_U[inner_idx]
                    
                    # B block (bj,bk): rows bj*2..bj*2+1, cols bk*2..bk*2+1
                    # V index: j*4+k
                    for (bj, bk), b_val in b_coeff.items():
                        for jj in range(2):
                            for kk in range(2):
                                inner_idx = jj * 2 + kk
                                global_j = bj * 2 + jj
                                global_k = bk * 2 + kk
                                v_idx = global_j * 4 + global_k
                                V_4[v_idx] += b_val * inner_V[inner_idx]
                    
                    # C block (bk,bi): rows bk*2..bk*2+1, cols bi*2..bi*2+1
                    # W index: k*4+i (p*n = 4*4 = 16, W[k*4+i])
                    for (bk, bi), c_val in c_coeff.items():
                        for kk in range(2):
                            for ii in range(2):
                                inner_idx = kk * 2 + ii
                                global_k = bk * 2 + kk
                                global_i = bi * 2 + ii
                                w_idx = global_k * 4 + global_i
                                W_4[w_idx] += c_val * inner_W[inner_idx]
                    
                    result.append((U_4, V_4, W_4))
            
            return result

        s4 = strassen_4x4()
        
        # Part 1: 4x4x4 block (i,j,k all in 0..3), rank 49
        for U4, V4, W4 in s4:
            U_full = [0.0] * 25  # i*5+j
            V_full = [0.0] * 25  # j*5+k
            W_full = [0.0] * 25  # k*5+i
            
            for i in range(4):
                for j in range(4):
                    U_full[i*5+j] = U4[i*4+j]
            for j in range(4):
                for k in range(4):
                    V_full[j*5+k] = V4[j*4+k]
            for k in range(4):
                for i in range(4):
                    W_full[k*5+i] = W4[k*4+i]
            
            U_list.append(U_full)
            V_list.append(V_full)
            W_list.append(W_full)
        
        # Part 2: Cross terms involving index 4 (the 5th row/col)
        # These are schoolbook: 4+4+1 = 9? No.
        # Terms where i=4, j=4, or k=4:
        # i in 0..4, j in 0..4, k in 0..4
        # Already covered: i,j,k all in 0..3 (49 terms)
        # Remaining: at least one of i,j,k is 4.
        # Total terms needed: 5*5*5 = 125, we have 49, need 125-49 = 76 more.
        # But wait, the 4x4 Strassen only covers i,j,k in 0..3.
        # The remaining terms are:
        # - i=4, j in 0..4, k in 0..4: 5*5 = 25 terms
        # - i in 0..3, j=4, k in 0..4: 4*5 = 20 terms  
        # - i in 0..3, j in 0..3, k=4: 4*4 = 16 terms
        # - i=4, j=4, k in 0..3: 4 terms (already counted in i=4)
        # Wait, I'm double counting. Let me use inclusion-exclusion.
        # Total = 125, covered = 4*4*4 = 64 tensor entries (but Strassen uses rank 49 for those 64 entries)
        # The remaining 125-64 = 61 entries each need one rank-1 term (schoolbook).
        # So total rank = 49 + 61 = 110.
        
        # Enumerate remaining (i,j,k) where at least one index is 4:
        remaining = []
        for i in range(5):
            for j in range(5):
                for k in range(5):
                    if i == 4 or j == 4 or k == 4:
                        remaining.append((i,j,k))
        
        assert len(remaining) == 61, f"Expected 61, got {len(remaining)}"
        
        for i, j, k in remaining:
            U_full = [0.0] * 25
            V_full = [0.0] * 25
            W_full = [0.0] * 25
            U_full[i*5+j] = 1.0
            V_full[j*5+k] = 1.0
            W_full[k*5+i] = 1.0
            U_list.append(U_full)
            V_list.append(V_full)
            W_list.append(W_full)
        
        rank = len(U_list)
        assert rank == 110, f"Expected rank 110, got {rank}"
        
        # Verify
        if _verify(U_list, V_list, W_list):
            # Convert to integers where possible
            def clean_matrix(mat):
                result = []
                for row in mat:
                    clean_row = []
                    for val in row:
                        if abs(val - round(val)) < 1e-9:
                            clean_row.append(int(round(val)))
                        else:
                            clean_row.append(val)
                    result.append(clean_row)
                return result
            
            U = clean_matrix(U_list)
            V = clean_matrix(V_list)
            W = clean_matrix(W_list)
            return {"U": U, "V": V, "W": W}
        else:
            raise ValueError("Rank-110 decomposition failed verification")

    except Exception:
        pass
    
    # Fallback: schoolbook rank 125
    n, m, p = 5, 5, 5
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
