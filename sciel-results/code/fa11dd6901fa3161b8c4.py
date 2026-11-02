import numpy as np
import time
import sys

def construct():
    n, m, p = 2, 3, 6
    dim_U = n * m  # 6
    dim_V = m * p  # 18
    dim_W = p * n  # 12
    
    start_time = time.time()
    deadline = start_time + 140
    
    def verify(U, V, W):
        """Check if U,V,W form a valid tensor decomposition."""
        # Tensor T[i,j,k] should equal sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # We verify by checking that for all i,j,k:
        # sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] == 1
        rank = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(rank):
                        val += U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r]
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True
    
    def make_half_integer(U, V, W):
        """Round to nearest half-integer and verify."""
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        if verify(U_r, V_r, W_r):
            return U_r, V_r, W_r
        return None
    
    def schoolbook():
        """Return the schoolbook rank-36 decomposition."""
        rank = n * m * p
        U = np.zeros((dim_U, rank))
        V = np.zeros((dim_V, rank))
        W = np.zeros((dim_W, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W
    
    # Strategy: The 2x3x6 tensor can be viewed as two 2x3x3 blocks.
    # If we can find a rank-17 decomposition for 2x3x3, we get rank 34 for 2x3x6.
    # 
    # Key insight: We can construct a rank-17 solution for 2x3x3 by taking the
    # schoolbook rank-18 and finding one linear dependency. The trick is to use
    # a Strassen-like identity.
    #
    # For 2x3x3, let's use the following approach:
    # Consider the multiplication:
    #   C[i,k] = sum_j A[i,j] * B[j,k]  for i in {0,1}, j in {0,1,2}, k in {0,1,2}
    #
    # Schoolbook uses 18 terms: (i,j,k) -> A[i,j] * B[j,k]
    #
    # To reduce to 17, we can try to express one term as a linear combination
    # of others using an algebraic identity.
    #
    # Actually, let me try a different approach. The tensor for 2x3x6 is:
    # T[i,j,k] = 1 for all i in {0,1}, j in {0,1,2}, k in {0,...,5}
    #
    # We can split k into two groups: k=0,1,2 and k=3,4,5.
    # Let k' = k mod 3. Then T[i,j,k] = 1 depends only on (i,j,k').
    # So the tensor is T_2x3x3 ⊗ I_2 (Kronecker product with a 1x1x2 identity-like tensor).
    # Actually it's more like: T[i,j,k] = T_2x3x3[i,j,k mod 3] * delta(k // 3, block)
    # This means the rank is 2 * rank(2x3x3).
    #
    # So if rank(2x3x3) = 17, then rank(2x3x6) = 34.
    #
    # Now, how to get rank 17 for 2x3x3?
    # Let me try a constructive approach using a known algebraic identity.
    #
    # Consider the 2x3x3 multiplication. We can use the following:
    # Let A be 2x3, B be 3x3. C = A*B is 2x3.
    #
    # Define:
    #   M1 = A[0,:] * B[:,0]  (column 0 of C, row 0)
    #   M2 = A[0,:] * B[:,1]  (column 1 of C, row 0)
    #   M3 = A[0,:] * B[:,2]  (column 2 of C, row 0)
    #   M4 = A[1,:] * B[:,0]  (column 0 of C, row 1)
    #   M5 = A[1,:] * B[:,1]  (column 1 of C, row 1)
    #   M6 = A[1,:] * B[:,2]  (column 2 of C, row 1)
    #
    # Each M is a sum of 3 products, so total is 18 products.
    #
    # To reduce, we can use: (a+b)(c+d) = ac + ad + bc + bd - (a-b)(c-d) + 2*ac
    # Hmm, that doesn't help directly.
    #
    # Let me try a completely different strategy.
    # I'll use a randomized local search to find a rank-34 solution directly.
    # Start from schoolbook (rank 36), then try to find linear dependencies
    # by searching for 2 columns that can be replaced by a single column
    # with appropriate modifications to the other matrices.
    #
    # Actually, let me try the most direct approach: hardcode a rank-34 solution
    # by constructing it from the block structure.
    #
    # For 2x3x3, I'll use a rank-17 construction based on the following idea:
    # We can write the 2x3x3 tensor as a sum of 17 rank-1 tensors.
    # 
    # Here's a concrete approach: Use the fact that we can combine two of the
    # schoolbook terms using a linear algebra trick.
    #
    # Let me try yet another approach: use a small integer grid search.
    # For 2x3x3, the schoolbook has 18 terms. I'll try to find coefficients
    # for 17 rank-1 tensors that sum to the identity tensor.
    #
    # This is a system of 18 equations (one for each (i,j,k) triple) with
    # 17*3 = 51 unknowns (but constrained to be integers/half-integers).
    #
    # Actually, let me just try to construct it explicitly.
    #
    # For 2x3x3, define the following 17 rank-1 tensors:
    # 
    # The key idea from literature: rank(2,3,3) <= 17 can be achieved by
    # using a "Strassen-type" reduction on the 3x3 part.
    #
    # Let me use a different, more reliable strategy:
    # I'll construct a rank-34 solution for 2x3x6 directly using a 
    # structured approach that is guaranteed to work.
    #
    # Approach: Use the Kronecker product structure.
    # T_{2,3,6} = T_{2,3,3} \otimes T_{1,1,2} where T_{1,1,2}[0,0,k]=delta(k,0)+delta(k,1)...
    # Actually, T_{2,3,6}[i,j,k] = 1 for all valid indices.
    # And T_{2,3,3}[i,j,k'] = 1 for all valid i,j,k'.
    # So T_{2,3,6}[i,j,k] = T_{2,3,3}[i,j, k%3] * 1_{k//3 == k//3}
    # This is just the Kronecker product with a 1x1x2 tensor that has all ones.
    #
    # So if I have a rank-R decomposition for 2x3x3, I can create a rank-2R
    # decomposition for 2x3x6 by duplicating each term for the two k-blocks.
    #
    # Now I need rank(2,3,3) <= 17.
    #
    # Let me try to explicitly construct a rank-17 solution for 2x3x3.
    # 
    # I'll use the following strategy: start from schoolbook (18 terms) and
    # try to find a linear dependency. I'll search over small perturbations.
    
    # Let me try a direct construction for rank 17 of 2x3x3.
    # 
    # Consider: we need 17 rank-1 tensors whose sum equals the 2x3x3 all-ones tensor.
    # Each rank-1 tensor is u ⊗ v ⊗ w where u is in R^2, v in R^3, w in R^3.
    # The sum must equal the all-ones tensor.
    #
    # This is equivalent to: sum_r u_r[i] * v_r[j] * w_r[k] = 1 for all i,j,k.
    #
    # Let me try:
    # For r = 0,...,16:
    #   u_r = (1, 1) or (1, 0) or (0, 1) etc.
    #   v_r, w_r chosen to satisfy the constraint.
    #
    # This is a hard constraint satisfaction problem. Let me use a different tactic.
    #
    # I'll use a numerical optimization approach to find a rank-17 solution,
    # then round to half-integers and verify.
    
    # Let me try a completely different and more reliable approach:
    # Use a known algebraic identity to reduce rank.
    #
    # For 2x3x3, here's a rank-17 construction:
    # 
    # Let the 18 schoolbook terms be indexed by (i,j,k) with i in {0,1}, j in {0,1,2}, k in {0,1,2}.
    # 
    # We can use the identity:
    # (a+b+c)(x+y+z) = ax + ay + az + bx + by + bz + cx + cy + cz
    # = (a+b)(x+y) + a*z + b*z + c*(x+y+z) - c*(x+y) 
    # Hmm, this is getting complicated.
    #
    # Let me try the most pragmatic approach: use a randomized search with 
    # local improvement, constrained to half-integers.
    
    # Actually, let me try a specific construction.
    # 
    # For 2x3x3, consider:
    #   C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0] + A[0,2]*B[2,0]
    #   C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1] + A[0,2]*B[2,1]
    #   C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2] + A[0,2]*B[2,2]
    #   C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0] + A[1,2]*B[2,0]
    #   C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1] + A[1,2]*B[2,1]
    #   C[1,2] = A[1,0]*B[0,2] + A[1,1]*B[1,2] + A[1,2]*B[2,2]
    #
    # Schoolbook: 18 products.
    #
    # Strassen-like trick for 2x3x3:
    #   Let S1 = A[0,0] + A[0,1]
    #   Let S2 = A[0,1] + A[0,2]  
    #   Let S3 = A[1,0] + A[1,1]
    #   Let S4 = A[1,1] + A[1,2]
    #   Let T1 = B[0,0] + B[0,1]
    #   Let T2 = B[0,1] + B[0,2]
    #   Let T3 = B[1,0] + B[1,1]
    #   Let T4 = B[1,1] + B[1,2]
    #   ...
    #
    # This is getting too complex. Let me use a numerical approach.
    
    # I'll use scipy's optimization to find a rank-34 solution.
    # But first, let me try a simpler structural approach.
    #
    # Key insight: The tensor T_{2,3,6} has a special structure because p=6=2*3.
    # We can write it as a block matrix:
    #   B = [B1 | B2] where B1, B2 are 3x3.
    #   C = A * B = [A*B1 | A*B2]
    #
    # So C[0,:] = [A*B1 | A*B2] and C[1,:] = [A*B1_row1 | A*B2_row1]
    #
    # If we can compute A*B1 with rank r1 and A*B2 with rank r2, total rank is r1+r2.
    # By symmetry, r1 = r2 = rank(2,3,3).
    #
    # So I need rank(2,3,3). Let me search for it.
    
    # Let me try a randomized search for a rank-17 solution of 2x3x3.
    # I'll parameterize 17 rank-1 tensors and minimize the error.
    
    import random
    random.seed(42)
    np.random.seed(42)
    
    def search_rank_17_2x3x3(max_iters=50000, time_budget=100):
        """Search for a rank-17 decomposition of 2x3x3 using random restarts + local search."""
        n2, m2, p2 = 2, 3, 3
        dim_U2 = n2 * m2  # 6
        dim_V2 = m2 * p2  # 9
        dim_W2 = p2 * n2  # 6
        rank = 17
        
        # Target tensor
        T = np.ones((n2, m2, p2))
        
        best_error = float('inf')
        best_U = None
        best_V = None
        best_W = None
        
        t0 = time.time()
        for it in range(max_iters):
            if time.time() - t0 > time_budget:
                break
            
            # Random initialization with small values
            U = np.random.randn(dim_U2, rank) * 0.5
            V = np.random.randn(dim_V2, rank) * 0.5
            W = np.random.randn(dim_W2, rank) * 0.5
            
            # Local improvement using gradient-free method
            for sub_iter in range(50):
                # Compute error
                err = 0.0
                for i in range(n2):
                    for j in range(m2):
                        for k in range(p2):
                            val = 0.0
                            for r in range(rank):
                                val += U[i*m2+j, r] * V[j*p2+k, r] * W[k*n2+i, r]
                            err += (val - 1.0) ** 2
                
                if err < best_error:
                    best_error = err
                    best_U = U.copy()
                    best_V = V.copy()
                    best_W = W.copy()
                    if err < 1e-10:
                        return best_U, best_V, best_W
                
                if time.time() - t0 > time_budget:
                    break
                
                # Perturb a random entry
                mat_idx = random.randint(0, 2)
                if mat_idx == 0:
                    r, c = random.randint(0, dim_U2-1), random.randint(0, rank-1)
                    U[r, c] += random.gauss(0, 0.1)
                elif mat_idx == 1:
                    r, c = random.randint(0, dim_V2-1), random.randint(0, rank-1)
                    V[r, c] += random.gauss(0, 0.1)
                else:
                    r, c = random.randint(0, dim_W2-1), random.randint(0, rank-1)
                    W[r, c] += random.gauss(0, 0.1)
        
        return best_U, best_V, best_W
    
    # Try the search
    U17, V17, W17 = search_rank_17_2x3x3(max_iters=30000, time_budget=90)
    
    if U17 is not None:
        # Check if it's close enough to exact
        if best_error < 1e-8:
            # Round to half-integers
            U17_r = np.round(U17 * 2) / 2
            V17_r = np.round(V17 * 2) / 2
            W17_r = np.round(W17 * 2) / 2
            
            # Verify for 2x3x3
            n2, m2, p2 = 2, 3, 3
            valid = True
            for i in range(n2):
                for j in range(m2):
                    for k in range(p2):
                        val = 0.0
                        for r in range(17):
                            val += U17_r[i*m2+j, r] * V17_r[j*p2+k, r] * W17_r[k*n2+i, r]
                        if abs(val - 1.0) > 1e-9:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                # Now construct 2x3x6 by duplicating
                rank_total = 34
                U34 = np.zeros((dim_U, rank_total))
                V34 = np.zeros((dim_V, rank_total))
                W34 = np.zeros((dim_W, rank_total))
                
                # First block: k=0,1,2
                # Second block: k=3,4,5
                for r in range(17):
                    # First copy (k in {0,1,2})
                    U34[:, r] = U17_r[:, r]
                    V34[:9, r] = V17_r[:, r]
                    V34[9:, r] = 0
                    W34[:6, r] = W17_r[:, r]
                    W34[6:, r] = 0
                    
                    # Second copy (k in {3,4,5})
                    U34[:, r+17] = U17_r[:, r]
                    V34[:9, r+17] = 0
                    V34[9:, r+17] = V17_r[:, r]
                    W34[:6, r+17] = 0
                    W34[6:, r+17] = W17_r[:, r]
                
                # Verify the full 2x3x6 decomposition
                if verify(U34, V34, W34):
                    return {
                        "U": U34.tolist(),
                        "V": V34.tolist(),
                        "W": W34.tolist()
                    }
    
    # Fallback: schoolbook rank 36
    U, V, W = schoolbook()
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
