import numpy as np
import time

def construct():
    n, m, p = 2, 4, 8
    start_time = time.time()
    deadline = 140  # safety margin
    
    # The strategy: embed 2x4 * 4x8 into 4x4 * 4x4 multiplication.
    # A (2x4) -> A_pad (4x4): top-left 2x4 block, rest zeros
    # B (4x8) -> B_pad (4x4): we need to map 4x8 into 4x4.
    # Actually, 4x8 doesn't fit in 4x4. Let me reconsider.
    
    # Alternative: Use the fact that 2x4 * 4x8 can be decomposed using
    # Strassen-like identities directly on the rectangular case.
    # 
    # Better approach: Use the known rank-19 algorithm for 4x4x4 matrix multiplication.
    # We can express 2x4 * 4x8 as a subset of a larger structured problem.
    #
    # Actually, let's think differently. The 2x4 * 4x8 multiplication has:
    # - 2 rows of A, 4 columns of A (= 4 rows of B), 8 columns of B
    # 
    # We can split B into two 4x4 blocks: B = [B1 | B2] where B1, B2 are 4x4
    # Then C = A*B = [A*B1 | A*B2], where A*B1 and A*B2 are both 2x4.
    # 
    # For each 2x4 * 4x4 product, we can use a rank-10 decomposition
    # (since 2x4*4x4 is a sub-problem of 4x4*4x4 which has rank 19,
    #  and the 2x4*4x4 part should have rank <= 10).
    #
    # Actually, let me just directly construct a low-rank decomposition.
    #
    # Key insight: For 2x4 * 4x8, we can use the following approach:
    # Split the 8 columns into two groups of 4: B = [B_left | B_right]
    # C = [C_left | C_right] where C_left = A*B_left (2x4), C_right = A*B_right (2x4)
    #
    # For 2x4 * 4x4 multiplication, the rank is at most 10 (from known results).
    # So total rank <= 20.
    #
    # Let me construct the 2x4 * 4x4 rank-10 decomposition explicitly.
    #
    # For 2x4 * 4x4: A is 2x4, B is 4x4, C is 2x4.
    # C[i,k] = sum_{j=0}^{3} A[i,j] * B[j,k]
    #
    # We can use the following identity (a variant of Strassen for rectangular):
    # Let's define 10 bilinear forms.
    #
    # Actually, let me use a more systematic approach.
    # The tensor for 2x4*4x4 has dimension 2*4*4 = 32 (schoolbook rank 32).
    # But we can do much better.
    #
    # Let me try to find a rank-10 decomposition for 2x4*4x4 using the following:
    # Write A = [a0, a1, a2, a3] (rows of the 2x4 matrix, each row is 4-vectors)
    # Actually A is 2x4: A[0,:] = (a00, a01, a02, a03), A[1,:] = (a10, a11, a12, a13)
    # B is 4x4: B[j,k] for j,k in 0..3
    # C is 2x4: C[i,k] = sum_j A[i,j]*B[j,k]
    #
    # I'll use a known construction. For m x n * n x p with small dimensions,
    # we can sometimes find explicit low-rank decompositions.
    #
    # Let me try a different structural approach: use the rank-19 4x4 algorithm
    # and extract the relevant parts.
    #
    # The 4x4*4x4 rank-19 algorithm (from Winograd et al.):
    # This is complex to write out. Let me instead try to directly verify
    # if a rank-19 solution exists for 2x4*4x8 by constructing it.
    #
    # Actually, the simplest structural approach: 
    # 2x4 * 4x8 can be seen as: compute 2x4 * 4x4 twice (for the two halves of B).
    # If I can find a rank-r decomposition for 2x4*4x4, total rank is 2r.
    #
    # For 2x4*4x4, let me try rank 10.
    # 
    # Here's a concrete construction for 2x4 * 4x4 with rank 10:
    # 
    # Let A = [[a00,a01,a02,a03],[a10,a11,a12,a13]]
    # Let B = [[b00,b01,b02,b03],[b10,b11,b12,b13],[b20,b21,b22,b23],[b30,b31,b32,b33]]
    # C[i,k] = sum_j A[i,j]*B[j,k]
    #
    # Define 10 products:
    # p0 = (a00+a10)*(b00+b10+b20+b30)  -- no, this mixes rows of A which is wrong.
    #
    # Wait, I need to be more careful. The standard approach for rectangular:
    # 
    # Let me just use the schoolbook for the 2x4*4x4 sub-problems but find
    # a smarter decomposition.
    #
    # Actually, let me try a completely different structural approach:
    # Use the fact that rank(2x4*4x8) <= rank(4x4*4x4) = 19 is NOT necessarily true
    # because the dimensions don't match. But we can embed.
    #
    # Let me try yet another approach: direct algebraic construction.
    #
    # For 2x4 * 4x8, I'll use 19 bilinear forms based on the following idea:
    # Group the 4 columns of A (and 4 rows of B) into pairs, and use Strassen-like
    # combinations.
    #
    # A = [A1 | A2] where A1, A2 are 2x2
    # B = [B1; B2] where B1, B2 are 2x8
    # C = A1*B1 + A2*B2 (2x8)
    #
    # Now A1*B1 is 2x2 * 2x8 = 2x8, and A2*B2 is 2x2 * 2x8 = 2x8.
    # Each 2x2*2x8 can be done with rank 7*4 = 28 (schoolbook for 2x2*2x8 is 2*2*8=32,
    # but using Strassen on the 2x2 part: rank 7 per 2x2*2x1, so 7*4=28 for 2x2*2x8).
    # Total: 56. That's the Strassen result we already have.
    #
    # To do better, we need to share bilinear forms between A1*B1 and A2*B2.
    #
    # Alternative: Use the full 4x4*4x4 rank-19 algorithm.
    # Embed A (2x4) into A_full (4x4) with zeros in bottom 2 rows.
    # Embed B (4x8) into... this doesn't work directly since 4x8 != 4x4.
    #
    # Let me try: B (4x8) = [B1 | B2] where B1, B2 are 4x4.
    # Then C = A*[B1|B2] = [A*B1 | A*B2]
    # 
    # For each A*B_i (2x4 * 4x4 -> 2x4), embed A into 4x4 (top 2 rows) and
    # use the 4x4*4x4 rank-19 algorithm, then extract the top 2 rows of the result.
    # The rank-19 4x4 algorithm gives 19 bilinear forms. For each B_i, we need
    # 19 forms, but some may be zero due to the zero padding of A.
    #
    # Actually, if A has zeros in rows 2,3 of the 4x4 embedding, then in the
    # rank-19 decomposition, some bilinear forms will involve only the zero rows
    # and will vanish. So the effective rank for 2x4*4x4 using the 4x4 algorithm
    # might be less than 19.
    #
    # This is getting complicated. Let me just implement the known rank-19 4x4
    # algorithm and see what rank we get for the embedded problem.
    #
    # The 4x4*4x4 rank-19 algorithm (from "An improved method for matrix multiplication" 
    # or similar). Actually, the best known for 4x4 is rank 19.
    #
    # Let me use a simpler approach: I know that for 2x2*2x2, Strassen gives rank 7.
    # For 2x4*4x8, let me try to directly construct a rank-19 solution.
    #
    # Here's my plan: I'll use the following explicit construction.
    # 
    # The 2x4*4x8 multiplication tensor can be decomposed as follows:
    # 
    # Let A[i,j] for i=0,1 and j=0,1,2,3
    # Let B[j,k] for j=0,1,2,3 and k=0,...,7
    # C[i,k] = sum_j A[i,j]*B[j,k]
    #
    # I'll define 19 bilinear forms M_r and output coefficients.
    #
    # Actually, let me just use a verified rank-19 construction for 4x4*4x4
    # and extract. Here's a known rank-19 for 4x4:
    #
    # From the literature, one rank-19 construction for 4x4*4x4 uses:
    # M1 = (A00+A10+A20+A30)*(B00+B10+B20+B30)  -- no, this isn't right for 4x4.
    #
    # Let me try a different tactic. I'll implement the Strassen-based rank-56
    # solution (which we know works) and then try to reduce it further using
    # a structured search over the 56 bilinear forms to find linear dependencies.
    #
    # Actually, the cleanest structural approach: 
    # Use the rank-19 4x4*4x4 algorithm from the following explicit formula.
    # 
    # I found a reference: the 4x4*4x4 rank-19 algorithm can be constructed
    # by applying the 2x2*2x2 Strassen algorithm in a specific way.
    #
    # Let me just implement a direct rank-19 construction for 2x4*4x8.
    #
    # Here's a concrete approach using the "bilinear complexity" theory:
    # 
    # For the 2x4*4x8 case, I'll use 19 terms as follows:
    # 
    # Define the following 19 bilinear forms (each is a product of a linear
    # form in A and a linear form in B):
    
    # Actually, let me just try to find ANY valid decomposition with rank < 64
    # using a structured algebraic approach, and fall back to schoolbook.
    
    # Let me implement the Strassen-based rank-56 solution first (known to work),
    # then try to improve.
    
    def get_strassen_56():
        """
        Use Strassen on 2x2 blocks to get rank 56 for 2x4*4x8.
        A (2x4) = [A11 A12] where A11, A12 are 2x2
        B (4x8) = [B11 B12; B21 B22] where each is 2x4
        C = [A11*A11 + A12*B21, A11*B12 + A12*B22]
        
        Wait, that's not right. Let me redo:
        A is 2x4: A = [[a00,a01,a02,a03],[a10,a11,a12,a13]]
        B is 4x8: B[j,k] for j=0..3, k=0..7
        
        Split A into two 2x2 blocks: A = [A_left | A_right]
        A_left = [[a00,a01],[a10,a11]], A_right = [[a02,a03],[a12,a13]]
        
        Split B into top and bottom 2x8: B = [B_top; B_bottom]
        B_top = B[0:2, :], B_bottom = B[2:4, :]
        
        C = A_left * B_top + A_right * B_bottom  (2x8)
        
        Each 2x2 * 2x8 product: use Strassen on the 2x2 part.
        2x2*2x8 = 2x2 * (2x4 blocks of 2x1)
        Actually, 2x2 * 2x8: treat B as 4 columns of 2x2.
        2x2 * 2x2 has rank 7 (Strassen).
        So 2x2 * 2x8 = 4 * (2x2*2x2) with rank 7*4 = 28.
        
        Total: 28 + 28 = 56.
        """
        rank = 56
        U = np.zeros((n * m, rank))  # (8, 56)
        V = np.zeros((m * p, rank))  # (32, 56)
        W = np.zeros((p * n, rank))  # (16, 56)
        
        # Strassen for 2x2 * 2x2:
        # Given P (2x2), Q (2x2), R = P*Q
        # M1 = (P00+P11)*(Q00+Q11)
        # M2 = (P10+P11)*Q00
        # M3 = P00*(Q10-Q11)
        # M4 = P11*(Q01-Q00)
        # M5 = (P00+P01)*Q11
        # M6 = (P10-P00)*(Q00+Q10)
        # M7 = (P01-P11)*(Q01+Q11)
        # R00 = M1+M4-M5+M7
        # R01 = M3+M5
        # R10 = M2+M4
        # R11 = M1-M2+M3+M6
        
        # For 2x2 * 2x8:
        # P is 2x2, Q is 2x8 = [Q_col0 | Q_col1 | Q_col2 | Q_col3] where each Q_coli is 2x2
        # Result R is 2x8 = [R_col0 | R_col1 | R_col2 | R_col3]
        # Each R_coli computed with 7 Strassen products.
        # Total: 28 products.
        
        def strassen_2x2_2x8(A_block, B_block, offset):
            """
            A_block: 2x2 matrix (values from A)
            B_block: 2x8 matrix (values from B)
            offset: starting index in the rank dimension
            Returns: list of (U_rows, V_rows, W_rows) for the 28 bilinear forms
            """
            # A_block[i,j] corresponds to A[i,j] in the original
            # B_block[j,k] corresponds to B[2*base_row+j, 2*base_col+k] in original
            # We need to track which original indices these correspond to.
            # This is getting complex. Let me just build it directly.
            pass
        
        # This is getting too complex to do symbolically. Let me just use the
        # schoolbook solution which we know works, and try a different approach
        # to find a lower rank.
        
        # Let me try a direct rank-19 construction.
        # 
        # I'll use the following known result: the 2x4*4x8 multiplication
        # can be done with rank 19 by embedding into 4x4*4x4.
        #
        # Here's a concrete rank-19 construction:
        # 
        # Embed A (2x4) into A4 (4x4):
        # A4 = [[a00, a01, a02, a03],
        #      [a10, a11, a12, a13],
        #      [0,   0,   0,   0  ],
        #      [0,   0,   0,   0  ]]
        #
        # Split B (4x8) into two 4x4 matrices: B1 = B[:,0:4], B2 = B[:,4:8]
        #
        # Use rank-19 4x4*4x4 for A4*B1 and A4*B2.
        # Since rows 2,3 of A4 are zero, many bilinear forms will vanish.
        #
        # The rank-19 4x4*4x4 algorithm: I'll use the one from
        # "A note on the bilinear complexity of matrix multiplication"
        # 
        # Actually, let me just hardcode a known rank-19 4x4 algorithm.
        # 
        # From https://en.wikipedia.org/wiki/Matrix_multiplication_algorithm
        # The 4x4*4x4 rank-19 algorithm is not as commonly listed as 2x2.
        #
        # Let me try a different approach: use the fact that we can verify
        # a decomposition numerically. I'll construct a candidate rank-19
        # decomposition using a structured pattern and verify it.
        
        return None  # Fall through to schoolbook
    
    # Let me just go with the schoolbook solution for now, but try a quick
    # numerical search for rank 19-30.
    
    def get_schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        r = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, r] = 1.0
                    V[j * p + k, r] = 1.0
                    W[k * n + i, r] = 1.0
                    r += 1
        return U, V, W
    
    def compute_error(U, V, W, A, B):
        U_reshaped = U.reshape(n, m, -1)
        V_reshaped = V.reshape(m, p, -1)
        W_reshaped = W.reshape(p, n, -1)
        X = np.einsum('ijr,ij->ir', U_reshaped, A)
        Y = np.einsum('jkr,jk->kr', V_reshaped, B)
        C_recon = np.einsum('ir,kr,ki->ik', X, Y, W_reshaped)
        C_true = A @ B
        return np.sum((C_recon - C_true) ** 2)
    
    def verify_decomposition(U, V, W, num_tests=20, seed=42):
        rng = np.random.RandomState(seed)
        max_err = 0
        for _ in range(num_tests):
            A = rng.randn(n, m)
            B = rng.randn(m, p)
            err = compute_error(U, V, W, A, B)
            max_err = max(max_err, err)
        return max_err
    
    # Try to find a low-rank decomposition using structured random search
    best_rank = n * m * p
    best_U, best_V, best_W = get_schoolbook()
    
    # Try ranks from 19 to 64
    for rank in range(19, min(40, best_rank + 1)):
        if time.time() > deadline:
            break
        
        found = False
        for trial in range(10):
            if time.time() > deadline:
                break
            
            rng = np.random.RandomState(rank * 7919 + trial * 104729)
            U = rng.randn(n * m, rank)
            V = rng.randn(m * p, rank)
            W = rng.randn(p * n, rank)
            
            # Use multiple random test matrices for more robust gradient
            A_tests = [rng.randn(n, m) for _ in range(3)]
            B_tests = [rng.randn(m, p) for _ in range(3)]
            
            lr = 0.05
            for step in range(50):
                if time.time() > deadline:
                    break
                
                total_grad_U = np.zeros_like(U)
                total_grad_V = np.zeros_like(V)
                total_grad_W = np.zeros_like(W)
                total_err = 0
                
                for A_t, B_t in zip(A_tests, B_tests):
                    U_reshaped = U.reshape(n, m, -1)
                    V_reshaped = V.reshape(m, p, -1)
                    W_reshaped = W.reshape(p, n, -1)
                    
                    X = np.einsum('ijr,ij->ir', U_reshaped, A_t)
                    Y = np.einsum('jkr,jk->kr', V_reshaped, B_t)
                    C_recon = np.einsum('ir,kr,ki->ik', X, Y, W_reshaped)
                    C_true = A_t @ B_t
                    diff = C_recon - C_true
                    
                    total_err += np.sum(diff ** 2)
                    
                    Z = np.einsum('ik,kr,ki->ir', diff, Y, W_reshaped)
                    dU = 2 * np.einsum('ij,ir->ijr', A_t, Z).reshape(n * m, rank)
                    
                    Z2 = np.einsum('ik,ir,ki->kr', diff, X, W_reshaped)
                    dV = 2 * np.einsum('jk,kr->jkr', B_t, Z2).reshape(m * p, rank)
                    
                    dW_reshaped = 2 * np.einsum('ik,ir,kr->ki', diff, X, Y)
                    dW = dW_reshaped.reshape(p * n, rank)
                    
                    total_grad_U += dU
                    total_grad_V += dV
                    total_grad_W += dW
                
                U -= lr * total_grad_U
                V -= lr * total_grad_V
                W -= lr * total_grad_W
                
                if total_err < 1e-12:
                    break
            
            if total_err < 1e-10:
                max_verify_err = verify_decomposition(U, V, W, num_tests=30)
                if max_verify_err < 1e-6:
                    best_rank = rank
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    found = True
                    break
        
        if found:
            break
    
    if best_rank < n * m * p:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    else:
        U, V, W = get_schoolbook()
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
