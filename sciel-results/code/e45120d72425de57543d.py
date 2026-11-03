"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import itertools

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
import numpy as np

def _verify(U, V, W, n=3, m=4, p=4):
    """Check if U,V,W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    # Build the tensor T of shape (n*m, m*p, p*n)
    # T[i*m+j, j*p+k, k*n+i] should be 1 for valid indices
    # Actually, the bilinear map is: C[k*n+i] = sum_j A[i*m+j] * B[j*p+k]
    # So the tensor entry T[a,b,c] = 1 if there exist i,j,k such that
    # a = i*m+j, b = j*p+k, c = k*n+i
    # Otherwise 0.
    
    # For exact verification, check that for all i,k:
    # sum_r U[i*m+j, r] * V[j*p+k, r] summed over j equals 1 for the right c
    # Actually: C[i,k] = sum_j A[i,j] * B[j,k]
    # In our indexing: A has index i*m+j, B has index j*p+k, C has index k*n+i
    # The decomposition says: for each r, we compute U_col_r · A * V_col_r · B,
    # then W_col_r scales into C.
    # More precisely: C = sum_r (W_col_r) * ((U_col_r)^T A) * (V_col_r^T B)
    # where (U_col_r)^T A is a 1xm vector, (V_col_r)^T B is a mxp vector... 
    
    # Let me think more carefully.
    # The bilinear form: C_{i,k} = sum_j A_{i,j} B_{j,k}
    # Tensor T has T_{i,j,k} = 1 (it's the identity in the middle index)
    # Rank-r decomposition: T = sum_{r=1}^{r} u_r ⊗ v_r ⊗ w_r
    # where u_r ∈ R^{n*m}, v_r ∈ R^{m*p}, w_r ∈ R^{p*n}
    # This means: for all i,j,k: sum_r u_r[i*m+j] * v_r[j*p+k] * w_r[k*n+i] = 1
    
    # Equivalently, define for each r:
    # U_r as an n×m matrix (reshaped from u_r)
    # V_r as an m×p matrix (reshaped from v_r)  
    # W_r as a p×n matrix (reshaped from w_r)
    # Then: sum_r (U_r^T ⊙ something)... 
    
    # Actually the standard formulation: 
    # C = sum_r w_r * (u_r^T A) * (v_r^T B)
    # where u_r^T A is a row vector (1×m), v_r^T B is... wait.
    
    # Let me use the standard matrix multiplication tensor decomposition:
    # We want: for each output C_{i,k}, C_{i,k} = sum_j A_{i,j} B_{j,k}
    # A rank-1 term contributes: alpha_r * (sum_j u_{r,i,j} A_{i,j}) * (sum_j v_{r,j,k} B_{j,k})
    # No wait, that's not right either.
    
    # The correct formulation for bilinear algorithms:
    # Preprocess: compute linear combinations of A entries: a_r = sum_{i,j} U_{i,j,r} A_{i,j}
    # Preprocess: compute linear combinations of B entries: b_r = sum_{j,k} V_{j,k,r} B_{j,k}
    # Multiply: m_r = a_r * b_r
    # Postprocess: C_{i,k} = sum_r W_{i,k,r} m_r
    # 
    # In our indexing:
    # U is (n*m) × rank, so U_{i*m+j, r}
    # V is (m*p) × rank, so V_{j*p+k, r}
    # W is (p*n) × rank, so W_{k*n+i, r}
    #
    # The condition is: for all i, j, k:
    # sum_r U_{i*m+j, r} * V_{j*p+k, r} * W_{k*n+i, r} = 1
    # (because the tensor T_{i,j,k} = 1 for the matmul tensor)
    
    # Let's verify this condition.
    # For efficiency, we can check: for each r, form matrices:
    # U_r (n×m), V_r (m×p), W_r (p×n) reshaped from the columns
    # Then the contribution of term r to C_{i,k} is:
    # W_r[k,i] * (sum_j U_r[i,j] * A_{i,j}) * ... 
    
    # Hmm, actually I think the correct interpretation is:
    # a_r = sum_{i,j} U_{i*m+j, r} * A_{i,j}  (scalar)
    # b_r = sum_{j,k} V_{j*p+k, r} * B_{j,k}  (scalar)
    # m_r = a_r * b_r
    # C_{i,k} = sum_r W_{k*n+i, r} * m_r
    #
    # For this to equal sum_j A_{i,j} B_{j,k}, we need:
    # sum_r W_{k*n+i, r} * (sum_{i',j'} U_{i'*m+j', r} A_{i',j'}) * (sum_{j'',k''} V_{j''*p+k'', r} B_{j'',k''})
    # = sum_j A_{i,j} B_{j,k}
    #
    # This is a very specific condition. Let me just check it numerically with random matrices.
    
    # Use random test matrices
    rng = np.random.RandomState(42)
    for trial in range(3):
        A = rng.randn(n, m)
        B = rng.randn(m, p)
        C_expected = A @ B
        
        # Compute using decomposition
        U_arr = np.array(U)  # (n*m, rank)
        V_arr = np.array(V)  # (m*p, rank)
        W_arr = np.array(W)  # (p*n, rank)
        
        # a_r = U^T A_reshaped for each r
        A_flat = A.flatten()  # (n*m,) in row-major: A[i,j] -> index i*m+j
        B_flat = B.flatten()  # (m*p,) in row-major: B[j,k] -> index j*p+k
        
        # a = U^T @ A_flat, shape (rank,)
        a = U_arr.T @ A_flat
        b = V_arr.T @ B_flat
        m_vals = a * b  # (rank,)
        
        # C_flat = W @ m_vals, shape (p*n,)
        C_flat = W_arr @ m_vals
        C_computed = C_flat.reshape(p, n).T  # W is indexed k*n+i, so reshape to (p,n) then transpose
        
        if not np.allclose(C_computed, C_expected, atol=1e-9):
            return False
    return True

def _schoolbook(n=3, m=4, p=4):
    """Return the schoolbook decomposition with rank = n*m*p."""
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

def _try_strassen_2x2_block(n=3, m=4, p=4):
    """
    Try a Strassen-inspired decomposition.
    
    Split A (3×4) into top (2×4) and bottom (1×4).
    Split B (4×4) into left (4×2) and right (4×2).
    
    C = [A_top * B_left, A_top * B_right;
         A_bot * B_left, A_bot * B_right]
    
    For the 2×2 blocks (top-left, top-right), we can potentially use Strassen
    if we can share computations. But the bottom row only needs 2 mults each.
    
    Actually, let's try a different partitioning.
    
    Alternative: Think of it as multiplying a 3×4 by 4×4.
    We can pad to 4×4 and use a 4×4 decomposition, but that's wasteful.
    
    Let's try: partition the 4 inner dimension into 2+2.
    A = [A1 | A2] where A1 is 3×2, A2 is 3×2
    B = [B1; B2] where B1 is 2×4, B2 is 2×4
    C = A1*B1 + A2*B2
    
    Each term A1*B1 is 3×2 * 2×4 = 3×4, needs 3*2*4=24 mults schoolbook.
    Same for A2*B2. Total: 48. No improvement.
    
    What if we use Strassen on the 2×2 "blocks" in a clever way?
    
    Let me try yet another approach: use the fact that rank(<1,n,n>) = n.
    We can write the 3×4 × 4×4 as a sum of 3 rank-1 (in the output) multiplications:
    C_i = A_i * B for i=0,1,2, where A_i is 1×4 and B is 4×4.
    Each requires 4 mults, total 12. But wait, that's not how it works.
    
    Actually C_i,k = sum_j A_i,j B_j,k. For fixed i, this is 4 mults (one per j).
    Total: 3*4 = 12 mults. But that IS the schoolbook for this case!
    
    Wait no. Schoolbook is n*m*p = 3*4*4 = 48. Why is there a discrepancy?
    
    Oh I see my confusion. The "schoolbook" algorithm for matrix multiplication
    does n*m*p scalar multiplications. For 3×4 * 4×4, that's 3*4*4 = 48.
    
    But I just said "for fixed i, this is 4 mults" - that's wrong. For fixed i,
    C_i,k = sum_{j=0}^{3} A_i,j * B_j,k. For each k (4 values), we need 4 mults.
    So for fixed i, we need 4*4 = 16 mults. Total: 3*16 = 48. Correct.
    
    So the schoolbook is indeed 48. Can we do better?
    
    The lower bound is max(nm, mp, np) = max(12, 16, 12) = 16.
    
    Let me try a constructive approach with rank around 24-32.
    
    One idea: use the "Four Russians" or block methods.
    
    Another idea: exploit that n=3 is small. Write:
    C = A * B where A is 3×4, B is 4×4.
    
    Let's use a decomposition based on partitioning the inner dimension:
    Split j into {0,1} and {2,3}:
    C = A[:,0:2] * B[0:2,:] + A[:,2:4] * B[2:4,:]
    
    First term: 3×2 * 2×4 = 3×4, schoolbook rank = 3*2*4 = 24
    Second term: same, 24
    Total: 48. Still no improvement.
    
    What if we use Strassen on the 2×2 structure?
    
    Let A1 = A[:,0:2] (3×2), B1 = B[0:2,:] (2×4)
    A2 = A[:,2:4] (3×2), B2 = B[2:4,:] (2×4)
    
    C = A1*B1 + A2*B2
    
    If we could compute A1*B1 + A2*B2 using fewer than 48 mults...
    
    One approach: define intermediate quantities:
    M1 = (A1+A2) * (B1+B2)  -- this gives A1B1 + A1B2 + A2B1 + A2B2
    M2 = A1 * B1
    M3 = A2 * B2
    
    Then C = M1 - A1B2 - A2B1 + M2 + M3... this doesn't simplify nicely.
    
    Actually, let me try a completely different strategy: random search with verification.
    """
    return None  # Placeholder - we'll implement random search instead

def _random_search(n=3, m=4, p=4, max_rank=32, n_trials=50, seed=42):
    """
    Random search for low-rank decomposition.
    
    Strategy: Initialize U, V, W with random values, then try to solve
    for a rank-r decomposition using iterative methods.
    
    Actually, a better approach: use the fact that we need
    sum_r U[:,r] * V[:,r] * W[:,r] = T (the matmul tensor)
    
    This is a system of nonlinear equations. We can try:
    1. Fix U and V, solve for W (linear in W for each entry)
    2. Fix U and W, solve for V
    3. Fix V and W, solve for U
    
    This is like alternating least squares for tensor decomposition.
    """
    rng = np.random.RandomState(seed)
    best = None
    best_rank = n * m * p
    
    for target_rank in range(16, min(max_rank + 1, n*m*p + 1)):
        for trial in range(n_trials):
            # Initialize randomly
            U = rng.randn(n*m, target_rank) * 0.1
            V = rng.randn(m*p, target_rank) * 0.1
            W = rng.randn(p*n, target_rank) * 0.1
            
            # Alternating minimization
            for it in range(100):
                # Update W: for each output index c = k*n+i,
                # we need sum_r W[c,r] * (U[:,r] * V[:,r])[corresponding entry] = T[c]
                # Actually, the condition is:
                # For all i,j,k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
                
                # This is complex. Let me use a different formulation.
                # 
                # Think of it as: for each r, define:
                # u_r as n×m matrix, v_r as m×p matrix, w_r as p×n matrix
                # The contribution to C_{i,k} from term r is:
                # w_r[k,i] * (u_r @ diag(a_i) @ ... ) 
                
                # I think the cleanest way is:
                # a_r = sum_{i,j} U[i*m+j, r] * A[i,j]
                # b_r = sum_{j,k} V[j*p+k, r] * B[j,k]
                # C[i,k] = sum_r W[k*n+i, r] * a_r * b_r
                #
                # For this to work for ALL A and B, we need the tensor identity.
                # This is equivalent to saying that the 3-way tensor formed by
                # (U[:,r], V[:,r], W[:,r]) for r=1..rank sums to the matmul tensor.
                
                # Let me just do a simple gradient-free approach:
                # Try to find rank-1 tensors that sum to the matmul tensor.
                
                # Actually, let me try a different approach entirely.
                # I'll use the known structure and try to find a decomposition
                # with specific patterns.
                break  # Skip the ALS for now, use structured approach
            
            # Skip - use structured approach below
            break
        
        if best is not None and best_rank < target_rank:
            break
    
    return best

def _structured_search(n=3, m=4, p=4):
    """
    Structured search based on block decompositions.
    
    Key insight: We can partition the problem into sub-problems.
    
    Approach 1: Use the fact that <3,4,4> can be embedded in <4,4,4>.
    The rank of <4,4,4> is at most 49 (from known bounds), but we don't have
    an explicit decomposition.
    
    Approach 2: Direct construction.
    
    Let's try to build a decomposition with rank = 24 by using
    a "diagonal" structure.
    
    For each output position (i,k), we need:
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    If we assign one multiplication to each (i,j,k) triple, we get rank 48.
    To reduce rank, we need to share multiplications across different (i,k) pairs
    or different j values.
    
    One way: if for a given r, U[:,r] is nonzero only for a specific i (row of A),
    and V[:,r] is nonzero only for a specific k (column of B), then
    the multiplication a_r * b_r contributes to only one output C[i,k].
    This is the schoolbook structure.
    
    To share, we need U[:,r] to be nonzero for multiple i values, or
    V[:,r] to be nonzero for multiple k values.
    
    Example: Suppose for some r, U[:,r] has U[i1*m+j, r] = 1 and U[i2*m+j, r] = 1
    for the same j. Then a_r = A[i1,j] + A[i2,j].
    If V[:,r] has V[j*p+k1, r] = 1 and V[j*p+k2, r] = 1, then
    b_r = B[j,k1] + B[j,k2].
    Then a_r * b_r = (A[i1,j]+A[i2,j])*(B[j,k1]+B[j,k2])
    = A[i1,j]B[j,k1] + A[i1,j]B[j,k2] + A[i2,j]B[j,k1] + A[i2,j]B[j,k2]
    
    This single multiplication contributes to 4 output entries!
    But we need to subtract the cross terms...
    
    This is essentially the Strassen trick. Let me try to implement it.
    
    Strassen for 2×2 matrices:
    M1 = (A11+A22)(B11+B22)
    M2 = (A21+A22)B11
    M3 = A11(B21-B22)
    M4 = A22(B11-B12)
    M5 = (A11+A12)B22
    M6 = (A21-A11)(B11+B12)
    M7 = (A12-A22)(B21+B22)
    
    C11 = M1+M4-M5+M7
    C12 = M3+M5
    C21 = M2+M4
    C22 = M1-M2+M3+M6
    
    For our problem, we have 3×4 * 4×4. Let's try to apply a similar idea.
    
    Partition A into rows: A_top (2×4) and A_bot (1×4)
    Partition B into columns: B_left (4×2) and B_right (4×2)
    
    C_top_left = A_top * B_left (2×2) -- use Strassen: 7 mults
    C_top_right = A_top * B_right (2×2) -- use Strassen: 7 mults
    C_bot_left = A_bot * B_left (1×2) -- 2 mults
    C_bot_right = A_bot * B_right (1×2) -- 2 mults
    
    Total: 7+7+2+2 = 18 mults? But the inputs are shared!
    
    The issue: Strassen for C_top_left needs combinations of A_top and B_left.
    Strassen for C_top_right needs combinations of A_top and B_right.
    These share A_top but use different B parts.
    
    Let me write out the Strassen formulas:
    For C_top_left = A_top * B_left:
    A_top = [[a00, a01, a02, a03], [a10, a11, a12, a13]]
    B_left = [[b00, b01], [b10, b11], [b20, b21], [b30, b31]]
    
    Wait, Strassen is for 2×2 matrices. Here A_top is 2×4 and B_left is 4×2.
    The product is 2×2. We can write:
    A_top * B_left = A_top[:,0:2] * B_left[0:2,:] + A_top[:,2:4] * B_left[2:4,:]
    
    Each term is 2×2 * 2×2, which can use Strassen.
    
    Term 1: A_top[:,0:2] * B_left[0:2,:] -- 7 mults (Strassen)
    Term 2: A_top[:,2:4] * B_left[2:4,:] -- 7 mults (Strassen)
    Total for C_top_left: 14 mults
    
    Similarly for C_top_right: 14 mults
    C_bot_left: A_bot * B_left = A_bot[0:2] * B_left[0:2,:] + A_bot[2:4] * B_left[2:4,:]
    Each is 1×2 * 2×2 = 1×2, needs 2 mults each, total 4 mults.
    C_bot_right: similarly 4 mults.
    
    Total: 14+14+4+4 = 36 mults. Better than 48!
    
    But can we do better by sharing? The Strassen intermediates for different
    blocks might share some computations if the submatrices overlap.
    
    Actually, the submatrices are disjoint in the index j, so there's no sharing
    between the 4 blocks. The total is 36.
    
    Can we do even better? What if we use a different partitioning?
    
    Let's try partitioning j into 1+1+1+1 (trivial) or 2+2 (which we did).
    
    What about using the "Toom-Cook" style for the inner dimension?
    
    Actually, let me just implement the 36-rank decomposition and verify it.
    """
    
    # Build the 36-rank decomposition using Strassen on 2×2 blocks.
    # 
    # Partition: j = {0,1} and {2,3}
    # A = [A_left | A_right] where A_left is 3×2, A_right is 3×2
    # B = [B_top; B_bot] where B_top is 2×4, B_bot is 2×4
    # C = A_left * B_top + A_right * B_bot
    #
    # A_left * B_top: 3×2 * 2×4 = 3×4
    # Split into:
    #   A_left[0:2] * B_top[:,0:2] (2×2 * 2×2 = 2×2) -- Strassen: 7
    #   A_left[0:2] * B_top[:,2:4] (2×2 * 2×2 = 2×2) -- Strassen: 7
    #   A_left[2:3] * B_top[:,0:2] (1×2 * 2×2 = 1×2) -- 2 mults
    #   A_left[2:3] * B_top[:,2:4] (1×2 * 2×2 = 1×2) -- 2 mults
    # Total for A_left*B_top: 7+7+2+2 = 18
    #
    # Similarly for A_right*B_bot: 18
    # Grand total: 36
    
    # Let me implement this carefully.
    #
    # For the Strassen 2×2*2×2 part:
    # Given P (2×2) and Q (2×2), compute P*Q using 7 mults:
    # M1 = (P00+P11)*(Q00+Q11)
    # M2 = (P10+P11)*Q00
    # M3 = P00*(Q10-Q11)
    # M4 = P11*(Q00-Q01)
    # M5 = (P00+P01)*Q11
    # M6 = (P10-P00)*(Q00+Q01)
    # M7 = (P01-P11)*(Q10+Q11)
    #
    # R00 = M1+M4-M5+M7
    # R01 = M3+M5
    # R10 = M2+M4
    # R11 = M1-M2+M3+M6
    #
    # In our tensor decomposition framework:
    # Each "multiplication" M corresponds to one rank-1 term.
    # For M1 = (P00+P11)*(Q00+Q11):
    #   U part: P00 and P11 get coefficient 1
    #   V part: Q00 and Q11 get coefficient 1
    #   W part: contributes to R00, R01, R10, R11 with appropriate signs
    
    # Let me build the full decomposition.
    
    U = []
    V = []
    W = []
    
    def add_term(u_dict, v_dict, w_dict):
        """Add a rank-1 term. u_dict maps (i,j)->coeff, v_dict maps (j,k)->coeff, w_dict maps (k,i)->coeff"""
        u_vec = [0] * (n*m)
        v_vec = [0] * (m*p)
        w_vec = [0] * (p*n)
        for (i, j), c in u_dict.items():
            u_vec[i*m+j] += c
        for (j, k), c in v_dict.items():
            v_vec[j*p+k] += c
        for (k, i), c in w_dict.items():
            w_vec[k*n+i] += c
        U.append(u_vec)
        V.append(v_vec)
        W.append(w_vec)
    
    # Block 1: A_left[0:2] * B_top[:,0:2] -> C[0:2, 0:2]
    # P = A[0:2, 0:2] = [[A00, A01], [A10, A11]]
    # Q = B[0:2, 0:2] = [[B00, B01], [B10, B11]]
    # (using 0-based: A[i,j] with i in {0,1}, j in {0,1}; B[j,k] with j in {0,1}, k in {0,1})
    
    # Strassen M1 = (A00+A11)*(B00+B11)
    add_term({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,0):1, (0,1):1, (1,1):1})
    # Wait, I need to be more careful. The W part should map to the output C[i,k].
    # M1 contributes to:
    # C[0,0] += M1
    # C[0,1] += 0 (M1 doesn't directly contribute to C01 in Strassen)
    # Actually in Strassen:
    # R00 = M1+M4-M5+M7
    # R01 = M3+M5
    # R10 = M2+M4
    # R11 = M1-M2+M3+M6
    
    # So M1 contributes to R00 with +1 and R11 with +1.
    # R00 is C[0,0] (output row 0, col 0)
    # R11 is C[1,1] (output row 1, col 1)
    
    # Let me redo this more carefully.
    # I'll clear and rebuild.
    
    U = []
    V = []
    W = []
    
    # For a Strassen block computing P*Q where P is 2×2 (rows i1,i2, cols j1,j2)
    # and Q is 2×2 (rows j1,j2, cols k1,k2), producing R (2×2, rows i1,i2, cols k1,k2):
    
    def strassen_block(i1, i2, j1, j2, k1, k2):
        """Add 7 Strassen terms for the block P=A[i1:i2+1, j1:j2+1] * Q=B[j1:j2+1, k1:k2+1]"""
        # M1 = (P00+P11)*(Q00+Q11)
        # P00 = A[i1,j1], P11 = A[i2,j2]
        # Q00 = B[j1,k1], Q11 = B[j2,k2]
        add_term(
            {(i1,j1):1, (i2,j2):1},
            {(j1,k1):1, (j2,k2):1},
            {(k1,i1):1, (k2,i2):1}  # contributes to C[i1,k1] and C[i2,k2]
        )
        # M2 = (P10+P11)*Q00
        # P10 = A[i2,j1], P11 = A[i2,j2], Q00 = B[j1,k1]
        add_term(
            {(i2,j1):1, (i2,j2):1},
            {(j1,k1):1},
            {(k1,i2):1}  # contributes to C[i2,k1]
        )
        # M3 = P00*(Q10-Q11)
        # P00 = A[i1,j1], Q10 = B[j2,k1], Q11 = B[j2,k2]
        add_term(
            {(i1,j1):1},
            {(j2,k1):1, (j2,k2):-1},
            {(k1,i1):1, (k2,i1):1}  # contributes to C[i1,k1] and C[i1,k2]
        )
        # M4 = P11*(Q00-Q01)
        # P11 = A[i2,j2], Q00 = B[j1,k1], Q01 = B[j1,k2]
        add_term(
            {(i2,j2):1},
            {(j1,k1):1, (j1,k2):-1},
            {(k1,i1):1, (k1,i2):1}  # contributes to C[i1,k1] and C[i2,k1]
        )
        # M5 = (P00+P01)*Q11
        # P00 = A[i1,j1], P01 = A[i1,j2], Q11 = B[j2,k2]
        add_term(
            {(i1,j1):1, (i1,j2):1},
            {(j2,k2):1},
            {(k2,i1):-1, (k2,i2):1}  # contributes to C[i1,k2] (negative) and C[i2,k2]
        )
        # M6 = (P10-P00)*(Q00+Q01)
        # P10 = A[i2,j1], P00 = A[i1,j1], Q00 = B[j1,k1], Q01 = B[j1,k2]
        add_term(
            {(i2,j1):1, (i1,j1):-1},
            {(j1,k1):1, (j1,k2):1},
            {(k2,i2):1}  # contributes to C[i2,k2]
        )
        # M7 = (P01-P11)*(Q10+Q11)
        # P01 = A[i1,j2], P11 = A[i2,j2], Q10 = B[j2,k1], Q11 = B[j2,k2]
        add_term(
            {(i1,j2):1, (i2,j2):-1},
            {(j2,k1):1, (j2,k2):1},
            {(k1,i1):1, (k2,i1):-1}  # contributes to C[i1,k1] and C[i1,k2] (negative)
        )
    
    # Block 1: rows 0-1, cols 0-1 of A; rows 0-1, cols 0-1 of B -> C[0:2, 0:2]
    strassen_block(0, 1, 0, 1, 0, 1)
    
    # Block 2: rows 0-1, cols 0-1 of A; rows 0-1, cols 2-3 of B -> C[0:2, 2:4]
    strassen_block(0, 1, 0, 1, 2, 3)
    
    # Block 3: row 2, cols 0-1 of A; rows 0-1, cols 0-1 of B -> C[2, 0:2]
    # This is 1×2 * 2×2 = 1×2, needs 2 mults: A[2,0]*B[0,0:2] + A[2,1]*B[1,0:2]
    # Mult 1: A[2,0] * (B[0,0]+B[1,0])... no, just do it directly.
    # C[2,0] = A[2,0]*B[0,0] + A[2,1]*B[1,0]
    # C[2,1] = A[2,0]*B[0,1] + A[2,1]*B[1,1]
    # We can do: M1 = A[2,0]*B[0,0], M2 = A[2,0]*B[0,1], M3 = A[2,1]*B[1,0], M4 = A[2,1]*B[1,1]
    # That's 4 mults. Or we can share: M1 = A[2,0]*(B[0,0]+B[0,1]), but then we can't separate.
    # Just use 4 separate mults (or 2 if we group).
    # Actually for 1×2 * 2×2, the rank is 2 (since it's <1,2,2> which has rank 2).
    # C[2,k] = sum_j A[2,j]*B[j,k] for k=0,1
    # We can compute: M1 = A[2,0]*B[0,0] + A[2,1]*B[1,0] = C[2,0] (1 mult if we precompute)
    # But in bilinear form, each "multiplication" is a scalar product of two linear combos.
    # For rank 2: 
    #   M1 = (A[2,0]) * (B[0,0]) -> contributes to C[2,0]
    #   M2 = (A[2,1]) * (B[1,0]) -> contributes to C[2,0]
    #   M3 = (A[2,0]) * (B[0,1]) -> contributes to C[2,1]
    #   M4 = (A[2,1]) * (B[1,1]) -> contributes to C[2,1]
    # That's 4. But we can do better:
    #   M1 = (A[2,0]+A[2,1]) * (B[0,0]+B[1,0]) -- gives A[2,0]B[0,0] + A[2,0]B[1,0] + A[2,1]B[0,0] + A[2,1]B[1,0]
    # This doesn't directly help.
    # 
    # The rank of <1,2,2> is 2. Let me verify:
    # C0 = A0*B0 + A1*B1
    # C1 = A0*B2 + A1*B3
    # We need 2 mults. One way:
    # M1 = (A0+A1)*(B0+B2) = A0B0 + A0B2 + A1B0 + A1B2
    # M2 = A0*B1 ... hmm this doesn't work cleanly.
    #
    # Actually <1,n,n> has rank n. So <1,2,2> has rank 2.
    # C0 = A0*B0 + A1*B1
    # C1 = A0*B2 + A1*B3
    # M1 = A0 * (B0 + B2) -> contributes B0 to C0 and B2 to C1
    # M2 = A1 * (B1 + B3) -> contributes B1 to C0 and B3 to C1
    # C0 = M1_part0 + M2_part0, C1 = M1_part1 + M2_part1
    # But M1 = A0*(B0+B2) is a single scalar, we can't extract B0 and B2 separately from it.
    #
    # I think I'm confusing myself. Let me use the standard result:
    # <1,n,n> has rank n. The decomposition is:
    # For j=0..n-1: M_j = A_j * B_j (where B_j is the j-th row of B, i.e., B[j,0..n-1])
    # Wait, that gives n mults, each producing a 1×n vector? No.
    #
    # Let me just use 4 mults for each 1×2 * 2×2 block. Total for 4 such blocks: 16.
    # Plus 7*4 = 28 for the Strassen blocks. Total: 44. Still better than 48.
    #
    # Actually wait, I have 4 Strassen blocks (2×2 * 2×2) and 4 small blocks (1×2 * 2×2).
    # 4*7 + 4*4 = 28 + 16 = 44.
    
    # Let me just use direct multiplication for the 1×2 * 2×2 blocks.
    def small_block(i, j1, j2, k1, k2):
        """Compute A[i, j1:j2+1] * B[j1:j2+1, k1:k2+1] using (j2-j1+1)*(k2-k1+1) mults"""
        for j in range(j1, j2+1):
            for k in range(k1, k2+1):
                add_term(
                    {(i,j):1},
                    {(j,k):1},
                    {(k,i):1}
                )
    
    # Block 3: row 2, cols 0-1 of A; rows 0-1, cols 0-1 of B
    small_block(2, 0, 1, 0, 1)
    
    # Block 4: row 2, cols 0-1 of A; rows 0-1, cols 2-3 of B
    small_block(2, 0, 1, 2, 3)
    
    # Block 5: rows 0-1, cols 2-3 of A; rows 2-3, cols 0-1 of B
    strassen_block(0, 1, 2, 3, 0, 1)
    
    # Block 6: rows 0-1, cols 2-3 of A; rows 2-3, cols 2-3 of B
    strassen_block(0, 1, 2, 3, 2, 3)
    
    # Block 7: row 2, cols 2-3 of A; rows 2-3, cols 0-1 of B
    small_block(2, 2, 3, 0, 1)
    
    # Block 8: row 2, cols 2-3 of A; rows 2-3, cols 2-3 of B
    small_block(2, 2, 3, 2, 3)
    
    rank = len(U)
    result = {"U": U, "V": V, "W": W}
    
    # Verify
    if _verify(U, V, W):
        return result
    else:
        return None

def construct():
    """
    Attempt to find a low-rank exact tensor decomposition for 3x4 * 4x4 matrix multiplication.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 100  # 100 second deadline (safety margin from 150s limit)
    
    # Start with schoolbook
    best_result = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try structured Strassen-based decomposition
    if time.time() < deadline:
        structured = _structured_search(n, m, p)
        if structured is not None:
            r = len(structured["U"][0])
            if r < best_rank:
                best_result = structured
                best_rank = r
    
    # Try to improve further with additional search if time permits
    if time.time() < deadline and best_rank > 20:
        # Try random perturbations of the structured solution
        rng = np.random.RandomState(123)
        U = np.array(best_result["U"], dtype=float)
        V = np.array(best_result["V"], dtype=float)
        W = np.array(best_result["W"], dtype=float)
        
        # Try to merge or simplify terms
        # This is complex; skip for now.
        pass
    
    return best_result
# EVOLVE-BLOCK-END

