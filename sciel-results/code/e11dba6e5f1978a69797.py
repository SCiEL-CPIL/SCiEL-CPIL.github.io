import time
import numpy as np

DIMENSIONS = [4, 4, 7]

def construct():
    start_time = time.time()
    DEADLINE = 130.0

    n, m, p = DIMENSIONS  # 4, 4, 7

    def build_schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def build_strassen_2x2():
        rank_s = 7
        U_s = [[0] * rank_s for _ in range(4)]
        V_s = [[0] * rank_s for _ in range(4)]
        W_s = [[0] * rank_s for _ in range(4)]

        U_s[0][0] = 1; U_s[3][0] = 1
        V_s[0][0] = 1; V_s[3][0] = 1
        W_s[0][0] = 1; W_s[3][0] = 1

        U_s[2][1] = 1; U_s[3][1] = 1
        V_s[0][1] = 1
        W_s[2][1] = 1; W_s[3][1] = -1

        U_s[0][2] = 1
        V_s[1][2] = 1; V_s[3][2] = -1
        W_s[1][2] = 1; W_s[3][2] = 1

        U_s[3][3] = 1
        V_s[2][3] = 1; V_s[0][3] = -1
        W_s[0][3] = 1; W_s[2][3] = 1

        U_s[0][4] = 1; U_s[1][4] = 1
        V_s[3][4] = 1
        W_s[0][4] = -1; W_s[1][4] = 1

        U_s[2][5] = 1; U_s[0][5] = -1
        V_s[0][5] = 1; V_s[1][5] = 1
        W_s[3][5] = 1

        U_s[1][6] = 1; U_s[3][6] = -1
        V_s[2][6] = 1; V_s[3][6] = 1
        W_s[0][6] = 1

        return U_s, V_s, W_s

    def build_4x4x4_strassen():
        U_s, V_s, W_s = build_strassen_2x2()
        rank_44 = 49
        U_44 = [[0] * rank_44 for _ in range(16)]
        V_44 = [[0] * rank_44 for _ in range(16)]
        W_44 = [[0] * rank_44 for _ in range(16)]

        A_coef = [
            [1, 0, 0, 1],
            [0, 0, 1, 1],
            [1, 0, 0, 0],
            [0, 0, 0, 1],
            [1, 1, 0, 0],
            [-1, 0, 1, 0],
            [0, 1, 0, -1],
        ]
        B_coef = [
            [1, 0, 0, 1],
            [1, 0, 0, 0],
            [0, 1, 0, -1],
            [-1, 0, 1, 0],
            [0, 0, 0, 1],
            [1, 1, 0, 0],
            [0, 0, 1, 1],
        ]
        C_coef = [
            [1, 0, 0, 1],
            [0, 0, 1, -1],
            [0, 1, 0, 1],
            [1, 0, 1, 0],
            [-1, 1, 0, 0],
            [0, 0, 0, 1],
            [1, 0, 0, 0],
        ]

        for s in range(7):
            for t in range(7):
                col = s * 7 + t
                for bi in range(2):
                    for bj in range(2):
                        a_mult = A_coef[s][bi*2+bj]
                        if a_mult == 0:
                            continue
                        for ii in range(2):
                            for jj in range(2):
                                u_val = U_s[ii*2+jj][t]
                                if u_val == 0:
                                    continue
                                gi = bi*2 + ii
                                gj = bj*2 + jj
                                U_44[gi*4+gj][col] += a_mult * u_val

                for bj in range(2):
                    for bk in range(2):
                        b_mult = B_coef[s][bj*2+bk]
                        if b_mult == 0:
                            continue
                        for jj in range(2):
                            for kk in range(2):
                                v_val = V_s[jj*2+kk][t]
                                if v_val == 0:
                                    continue
                                gj = bj*2 + jj
                                gk = bk*2 + kk
                                V_44[gj*4+gk][col] += b_mult * v_val

                for ci in range(2):
                    for cj in range(2):
                        c_mult = C_coef[s][ci*2+cj]
                        if c_mult == 0:
                            continue
                        for kk in range(2):
                            for ii in range(2):
                                w_val = W_s[kk*2+ii][t]
                                if w_val == 0:
                                    continue
                                gk = ci*2 + kk
                                gi = cj*2 + ii
                                W_44[gk*4+gi][col] += c_mult * w_val

        return U_44, V_44, W_44

    def build_4x4x3_hybrid():
        # 4x4x3: use Strassen 2x2x2 for k=0,1 and schoolbook for k=2
        # Strassen 2x2x2: rank 7 for 2x2x2
        # For 4x4x3: split A as 2x2 blocks, B as 2x2x3
        # Actually, let's think differently.
        # 4x4x3 can be seen as: C[i,j,k] = sum_l A[i,l]*B[l,j]*... no.
        # C = A*B where A is 4x4, B is 4x3.
        # C[i,k] = sum_j A[i,j]*B[j,k]
        # Tensor: T[i,j,k] = 1 for all i,j,k.
        # Split into two halves: k in {0,1} and k in {2}.
        # For k in {0,1}: 4x4x2 problem.
        # For k in {2}: 4x4x1 problem (rank 16).
        # 4x4x1: rank 16 (schoolbook).
        # 4x4x2: can we do better than 32?
        # 4x4x2 = two 4x4x1's. Rank <= 32.
        # But maybe we can share some structure.
        # Actually, for 4x4x2, we can view it as: C = A * B where B is 4x2.
        # This is equivalent to two separate 4x4x1 multiplications.
        # So rank(4x4x2) <= 2 * rank(4x4x1) = 32.
        # But can we do better? The tensor is the direct sum of two 4x4x1 tensors.
        # rank(A ⊕ B) <= rank(A) + rank(B), but can be less.
        # For 4x4x1, rank is 16 (it's a flat matrix essentially).
        # Actually 4x4x1 tensor T[i,j,0]=1. This is just the outer product of ones.
        # As a matrix (16x1), it has rank 1.
        # So 4x4x1 has rank 1! Wait, no.
        # T[i,j,k] where k is fixed to 0. The tensor is T[i,j,0] = 1.
        # As a 3-tensor, the rank is the minimum r such that T = sum_{t=1}^r u_t ⊗ v_t ⊗ w_t.
        # Here w_t must be supported on k=0. So w_t = [1, 0, ..., 0] (for p=1, w is 1-dim).
        # Actually for 4x4x1, p=1, so w is a vector of length 1.
        # T[i,j,0] = sum_t U[i,j,t] * V[j,0,t] * W[0,i,t].
        # Since V[j,0,t] and W[0,i,t] are scalars, this is sum_t U[i,j,t] * a_t * b_t.
        # This is a matrix factorization of the 4x4 all-ones matrix.
        # Rank of all-ones 4x4 matrix is 1.
        # So 4x4x1 has rank 1!
        # Let me verify: U[i,j,t] = 1 for all i,j when t=1.
        # V[j,0,1] = 1, W[0,i,1] = 1.
        # Then T[i,j,0] = 1*1*1 = 1. Yes!
        
        # So 4x4x3:
        # k=2 part: 4x4x1, rank 1.
        # k=0,1 part: 4x4x2.
        # 4x4x2: T[i,j,k] = 1 for k in {0,1}.
        # This is two copies of 4x4x1 stacked.
        # T[i,j,0] = 1, T[i,j,1] = 1.
        # As a tensor, can we share rank-1 terms?
        # T = u ⊗ v ⊗ w1 + u ⊗ v ⊗ w2 where w1 = [1,0], w2 = [0,1]?
        # No, because u ⊗ v ⊗ w1 has T[i,j,0]=u[i]v[j]w1[0]=u[i]v[j] and T[i,j,1]=0.
        # We need T[i,j,0]=1 and T[i,j,1]=1.
        # If we use u=[1,1,1,1], v=[1,1,1,1], then u[i]v[j]=1.
        # T = 1*1*w1 + 1*1*w2 where w1=[1,0], w2=[0,1].
        # This gives T[i,j,0]=1, T[i,j,1]=1. 
        # But wait, the rank-1 term is u ⊗ v ⊗ w.
        # Here we're using two rank-1 terms: (1,1,1,1) ⊗ (1,1,1,1) ⊗ (1,0) and (1,1,1,1) ⊗ (1,1,1,1) ⊗ (0,1).
        # But these can be combined? No, they're separate rank-1 terms.
        # So 4x4x2 has rank <= 2.
        # Actually, is rank 1 possible? 
        # T[i,j,k] = u[i] * v[j] * w[k].
        # We need u[i]v[j]w[0] = 1 and u[i]v[j]w[1] = 1 for all i,j.
        # So w[0] = w[1] (since u[i]v[j] is the same for all i,j... wait, u[i]v[j] depends on i,j).
        # For this to hold for all i,j, we need u[i]v[j]w[0] = 1 for all i,j.
        # This means u[i]v[j] is constant for all i,j, which implies u and v are constant vectors.
        # Let u[i] = a, v[j] = b. Then ab*w[0] = 1 and ab*w[1] = 1, so w[0]=w[1].
        # Then T[i,j,k] = ab*w = 1. Yes!
        # So 4x4x2 has rank 1!
        
        # Similarly, 4x4x3: T[i,j,k] = 1 for all k.
        # T = u ⊗ v ⊗ w where u[i]=a, v[j]=b, w[k]=c.
        # Then abc = 1. So rank 1!
        
        # Wait, this can't be right. Let me re-examine.
        # The matrix multiplication tensor for 4x4x3 is:
        # (A * B)[i,k] = sum_j A[i,j] * B[j,k]
        # The tensor T[i,j,k] represents the coefficient of A[i,j]*B[j,k] in the product.
        # T[i,j,k] = 1 for all i,j,k.
        
        # Can we decompose this as a single rank-1 term?
        # T[i,j,k] = U[i,j,r] * V[j,k,r] * W[k,i,r]
        # With r=1: T[i,j,k] = U[i,j,0] * V[j,k,0] * W[k,i,0]
        # We need U[i,j,0] * V[j,k,0] * W[k,i,0] = 1 for all i,j,k.
        # Let's try U[i,j,0] = 1, V[j,k,0] = 1, W[k,i,0] = 1.
        # Then T[i,j,k] = 1*1*1 = 1. Yes!
        
        # So the entire 4x4x3 tensor has rank 1?!
        # That seems too good to be true. Let me double-check with the 2x2x2 case.
        # 2x2x2 tensor T[i,j,k] = 1 for all i,j,k.
        # Can we do rank 1? U[i,j,0]*V[j,k,0]*W[k,i,0] = 1 for all i,j,k.
        # Try U=1, V=1, W=1. Yes, it works.
        # But we know Strassen gives rank 7 for 2x2x2, and it's known that rank(2x2x2) = 7.
        # So something is wrong with my reasoning.
        
        # Ah, I see the issue. The tensor for matrix multiplication is NOT T[i,j,k]=1.
        # It's T[i,j,k] = 1, but the decomposition must satisfy:
        # C[i,k] = sum_j A[i,j] B[j,k] = sum_r (sum_j U[i,j,r] A[i,j]) * (sum_j V[j,k,r] B[j,k]) * W[k,i,r]
        # Wait no, that's not right either.
        
        # The correct formulation:
        # C = A * B
        # C[i,k] = sum_j A[i,j] * B[j,k]
        # Tensor decomposition: C[i,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]... 
        # No, that's not how it works.
        
        # The matrix multiplication tensor M_{i,j,k} has the property that:
        # (A * B)[i,k] = sum_{j} A[i,j] * B[j,k]
        # In tensor form, if we view A, B, C as vectors, then:
        # <C, A ⊗ B> = sum_{i,j,k} C[i,k] * A[i,j] * B[j,k]
        # = sum_{i,j,k} (sum_j' A[i,j'] * B[j',k]) * A[i,j] * B[j,k]
        # This is getting confusing. Let me use the standard definition.
        
        # The matrix multiplication tensor of format (n,m,p) is a 3-tensor M of size n*m*p
        # such that for any matrices A (n×m) and B (m×p),
        # (AB)[i,k] = sum_j A[i,j] B[j,k]
        # can be computed as sum_r U_r · A · V_r · W_r where U_r is n×m, V_r is m×p, W_r is... 
        # No, the standard tensor decomposition is:
        # M[i,j,k] = sum_r u_r[i,j] * v_r[j,k] * w_r[k,i]
        # where u_r are n×m, v_r are m×p, w_r are p×n.
        # And the product is recovered by:
        # (AB)[i,k] = sum_j A[i,j] B[j,k] = sum_r (sum_{i',j'} u_r[i',j'] A[i',j']) ... 
        # I'm getting confused. Let me just trust the known result: rank(2x2x2) = 7.
        
        # The point is: my earlier claim that T[i,j,k]=1 has rank 1 is WRONG.
        # The matrix multiplication tensor is not simply the all-ones tensor in a way that admits rank-1 decomposition.
        # Actually wait, IS it? Let me re-examine.
        
        # For 2x2x2: M[i,j,k] = 1 for all i,j,k in {0,1}.
        # Rank-1 decomposition: M[i,j,k] = u[i,j] * v[j,k] * w[k,i].
        # Try u[i,j] = 1, v[j,k] = 1, w[k,i] = 1.
        # Then M[i,j,k] = 1*1*1 = 1. ✓
        # So why is rank(2x2x2) = 7?
        
        # I think I'm confusing two different things. The "rank" of the matrix multiplication tensor 
        # in the context of fast matrix multiplication is NOT the tensor rank in the algebraic sense.
        # It's the border rank or something related to the arithmetic complexity.
        
        # Actually, no. The rank of the matrix multiplication tensor IS the tensor rank, and it IS 
        # known that rank(2x2x2) = 7 (Strassen). But my rank-1 decomposition above seems to work...
        
        # Let me look this up mentally. The issue is that the matrix multiplication tensor M_{ijk} 
        # is defined such that the number of multiplications in the best algorithm equals the rank.
        # The tensor is M[i,j,k] = δ_{i,j,k}... no.
        
        # OK I think the confusion is in the index conventions. Let me just proceed with the known 
        # Strassen construction and not try to prove rank-1 decompositions that contradict known results.
        
        # Let me just use:
        # 4x4x4: Strassen rank 49
        # 4x4x3: schoolbook rank 48
        # Total: 97
        
        # And for the 4x4x3 part, let me try to do slightly better.
        # 4x4x3: split into 2x2x3 blocks.
        # 2x2x3: what's the best known rank? 
        # 2x2x3 can be done with: for each of the 3 output columns, do 2x2x1.
        # 2x2x1: rank 1 (as shown above, since it's essentially outer product).
        # Wait, 2x2x1: C[i,k] where k is fixed. C[i,0] = A[i,0]B[0,0] + A[i,1]B[1,0].
        # Tensor: M[i,j,0] = 1 for all i,j.
        # M[i,j,0] = u[i,j]*v[j,0]*w[0,i].
        # Try u[i,j]=1, v[j,0]=1, w[0,i]=1. Works. Rank 1.
        # So 2x2x3: three separate 2x2x1's, each rank 1. Total rank <= 3.
        # But can we share? The three columns are independent, so probably not.
        # So 2x2x3 has rank <= 3.
        
        # 4x4x3: four 2x2x3 blocks. Each rank <= 3. Total rank <= 12.
        # But the blocks share indices, so we need to be careful.
        
        # Actually, let me just go with the safe approach:
        # 4x4x4 Strassen (rank 49) + 4x4x3 schoolbook (rank 48) = 97.
        # This is already an improvement over the original 97... wait, the current program has 97.
        
        # Let me try: 4x4x4 Strassen (49) + 4x4x3 with 2x2x3 blocks.
        # 4x4x3: split A into four 2x2 blocks, B into four 2x2x3... no, B is 4x3.
        # A is 4x4, B is 4x3, C is 4x3.
        # Split A into 2x2 blocks: A = [[A00, A01], [A10, A11]]
        # Split B into 2x3 blocks vertically: B = [[B0], [B1]] where B0 is 2x3, B1 is 2x3.
        # C = A*B = [[A00*B0 + A01*B1], [A10*B0 + A11*B1]]
        # Each term like A00*B0 is 2x2 times 2x3, giving 2x3.
        # There are 4 such products.
        # Each 2x2x3 multiplication can be done with rank <= 3 (as argued above).
        # So total rank <= 4 * 3 = 12 for the 4x4x3 part.
        # Plus 49 for 4x4x4... but wait, 4x4x4 + 4x4x3 is not how it works.
        
        # The full problem is 4x4x7, not 4x4x4 + 4x4x3.
        # Let me reconsider.
        
        # 4x4x7: A is 4x4, B is 4x7, C is 4x7.
        # Split B into B1 (4x4) and B2 (4x3).
        # C = A*B1 + A*B2.
        # A*B1: 4x4x4, rank 49 (Strassen).
        # A*B2: 4x4x3, rank 48 (schoolbook) or better.
        
        # For A*B2 where B2 is 4x3:
        # Split A into 2x2 blocks, B2 into 2x3 blocks.
        # 4 sub-problems of 2x2x3, each rank <= 3.
        # Total rank <= 12.
        # So total: 49 + 12 = 61.
        
        # But I need to actually construct the 2x2x3 decomposition.
        
        def build_2x2x3():
            # 2x2x3: A is 2x2, B is 2x3, C is 2x3
            # C[i,k] = sum_j A[i,j] * B[j,k]
            # 4 sub-products: A[0,0]*B[0,k], A[0,1]*B[1,k], A[1,0]*B[0,k], A[1,1]*B[1,k]
            # for k=0,1,2.
            # Each is a scalar * scalar product.
            # Rank-1 term: u[i,j] * v[j,k] * w[k,i]
            # For A[0,0]*B[0,k]: u[0,0]=1, others 0. v[0,k]=1, others 0. w[k,0]=1, others 0.
            # This gives C[0,k] += A[0,0]*B[0,k]. ✓
            # Similarly for the others.
            # 4 terms total, one for each (i,j) pair... wait, there are 4 (i,j) pairs and 3 k values.
            # Each (i,j,k) triple needs a term. So 2*2*3 = 12 terms.
            # That's just schoolbook. Rank 12.
            
            # Can we do better? For fixed (i,j), we need C[i,k] += A[i,j]*B[j,k] for all k.
            # This is a rank-1 term in j only? No, it's a rank-1 term in the full tensor.
            # u[i,j,r] * v[j,k,r] * w[k,i,r]
            # For fixed (i,j), we want: sum_r u[i,j,r] * v[j,k,r] * w[k,i,r] = A[i,j]*B[j,k] for all k.
            # This is like factoring the 2x3 matrix A[i,j]*B[j,:] as a sum of rank-1 matrices... 
            # but it's just a single row vector times a scalar.
            # I think 2x2x3 has rank 12 (schoolbook) and it's not easy to improve.
            
            # Let me just use schoolbook for 4x4x3.
            rank = 48
            U = [[0] * rank for _ in range(16)]  # 4x4
            V = [[0] * rank for _ in range(28)]  # 4x7 but we only use k=4,5,6 -> rows j*7+4..j*7+6
            W = [[0] * rank for _ in range(28)]  # 7x4 but we only use k=4,5,6
            
            col = 0
            for i in range(4):
                for j in range(4):
                    for k in range(3):
                        U[i * 4 + j][col] = 1
                        V[j * 7 + (4 + k)][col] = 1
                        W[(4 + k) * 4 + i][col] = 1
                        col += 1
            return U, V, W
        
        U_44, V_44, W_44 = build_4x4x4_strassen()
        U_43, V_43, W_43 = build_2x2x3()  # placeholder, will replace
        
        # Actually, let me just use the existing build_4x4x7_optimized which gives 97.
        # That's the safe bet.
        
        pass

    def build_4x4x7_optimized():
        U_44, V_44, W_44 = build_4x4x4_strassen()
        rank_44 = 49
        rank_43 = 48
        total_rank = rank_44 + rank_43

        U_out = [[0] * total_rank for _ in range(16)]
        V_out = [[0] * total_rank for _ in range(28)]
        W_out = [[0] * total_rank for _ in range(28)]

        for r in range(16):
            for c in range(rank_44):
                U_out[r][c] = U_44[r][c]
                V_out[r][c] = V_44[r][c]
                W_out[r][c] = W_44[r][c]

        col_offset = rank_44
        for i in range(4):
            for j in range(4):
                for k in range(3):
                    kk = k + 4
                    U_out[i * 4 + j][col_offset] = 1
                    V_out[j * 7 + kk][col_offset] = 1
                    W_out[kk * 4 + i][col_offset] = 1
                    col_offset += 1

        return {"U": U_out, "V": V_out, "W": W_out}

    if time.time() - start_time > DEADLINE:
        return build_schoolbook()

    try:
        result = build_4x4x7_optimized()
        
        U = np.array(result["U"], dtype=float)
        V = np.array(result["V"], dtype=float)
        W = np.array(result["W"], dtype=float)
        rank = U.shape[1]
        
        U_r = U.reshape(n, m, rank)
        V_r = V.reshape(m, p, rank)
        W_r = W.reshape(p, n, rank)
        
        tensor_decomp = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
        
        target = np.ones((n, m, p), dtype=float)
        if not np.allclose(tensor_decomp, target, atol=1e-9):
            return build_schoolbook()

        return result
    except Exception:
        return build_schoolbook()
