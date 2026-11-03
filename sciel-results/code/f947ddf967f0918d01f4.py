import numpy as np
import time
import itertools

DIMENSIONS = [2, 4, 5]

def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=5
    deadline = time.time() + 145
    
    # Build the rank-36 Strassen solution
    U36, V36, W36 = _build_strassen_36(n, m, p)
    U36 = np.array(U36, dtype=np.float64)
    V36 = np.array(V36, dtype=np.float64)
    W36 = np.array(W36, dtype=np.float64)
    R36 = 36
    
    best_rank = R36
    best_U, best_V, best_W = U36, V36, W36
    
    # Strategy: Try to reduce rank by finding linear dependencies among columns
    # For rank 35: check if any pair of columns can be merged
    # A rank-36 decomposition can be reduced to rank 35 iff there exists
    # a non-trivial linear combination of the rank-1 tensors that is zero,
    # OR we can find a rank-35 decomposition directly.
    
    # Approach 1: For each pair of columns (a, b), try to merge them.
    # If columns a and b of U, V, W are linearly dependent in a certain sense,
    # we can potentially reduce rank.
    
    # More concretely: given rank-36 factors, for each pair (a,b), 
    # check if there exists alpha such that replacing columns a,b with a single
    # column (U_a + alpha*U_b, V_a + alpha*V_b, W_a + alpha*W_b) still works.
    # This requires: for all (i,j,k),
    #   U[i*m+j,a]*V[j*p+k,a]*W[k*n+i,a] + U[i*m+j,b]*V[j*p+k,b]*W[k*n+i,b]
    #   = (U[i*m+j,a]+alpha*U[i*m+j,b])*(V[j*p+k,a]+alpha*V[j*p+k,b])*(W[k*n+i,a]+alpha*W[k*n+i,b])
    
    # This is complex. Let's try a different approach.
    
    # Approach 2: Randomized perturbation + solve
    # Start with rank-36, perturb two columns slightly, solve for W, check validity.
    
    # Approach 3: Direct structured search for rank 35
    # Use the known structure: the rank-36 solution has 4 Strassen blocks (7 each) 
    # and one schoolbook block (4 terms) = 28 + 4 = 32... wait that's 32 not 36.
    # Let me recount: 4 Strassen blocks * 7 = 28, plus schoolbook for k=4: 2*4=8, total 36. Yes.
    
    # Let me try: can we merge the schoolbook block with Strassen blocks?
    # The schoolbook terms for k=4 are: for i in {0,1}, j in {0,1,2,3}:
    #   U[i*4+j, col] = 1, V[j*5+4, col] = 1, W[4*2+i, col] = 1
    # These 8 terms might be mergeable with Strassen terms.
    
    # Let's try a more systematic approach:
    # For rank 35, we need to find U (8x35), V (20x35), W (10x35) such that
    # sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for all i,j,k
    
    # Use the rank-36 solution as initialization and try to null out one column.
    
    np.random.seed(42)
    
    # Try to find rank 35 by:
    # 1. Take the rank-36 factors
    # 2. For each pair of columns (a, b), try to express column b as a linear combo
    #    of the others, then eliminate it.
    
    # Actually, let's try a simpler approach: 
    # Randomly initialize U and V with small integers, solve for W, check if valid.
    # But use the structure from the rank-36 solution to guide initialization.
    
    # Better approach: Use the rank-36 solution, and for each column index c,
    # try to set that column to zero in U, V, W and see if the remaining 35 columns
    # can be adjusted to still give the correct result. This is a nonlinear problem.
    
    # Let's try: for rank 35, use random U, V from {-1, 0, 1} with the constraint
    # that they are "close" to the rank-36 solution structure.
    
    # Actually, the most promising approach: 
    # The matmul tensor for 2x4x5 might have rank 35 or less.
    # Let's do a more targeted search.
    
    # Key insight: For 2x2xn, rank is known to be 7 + (n-2) = n+5 for n >= 2.
    # For 2x2x5, rank = 10.
    # For 2x4x5, we can split into two 2x2x5 problems:
    #   Problem 1: i in {0,1}, j in {0,1}, k in {0,1,2,3,4} -> rank 10
    #   Problem 2: i in {0,1}, j in {2,3}, k in {0,1,2,3,4} -> rank 10
    # Total: rank 20? But wait, the A and B matrices are different for each problem.
    # Problem 1 uses A[:,0:2] and B[0:2,:]
    # Problem 2 uses A[:,2:4] and B[2:4,:]
    # These are independent, so total rank = 10 + 10 = 20!
    
    # Wait, but the output C is the same for both. Let me think again.
    # C[i,k] = sum_j A[i,j]*B[j,k]
    # = sum_{j=0}^{1} A[i,j]*B[j,k] + sum_{j=2}^{3} A[i,j]*B[j,k]
    # The first sum is a 2x2x5 matmul (A is 2x2, B is 2x5, C is 2x5)
    # The second sum is another 2x2x5 matmul (A is 2x2, B is 2x5, C is 2x5)
    # Both contribute to the same C matrix (2x5).
    # So we need: C = C1 + C2 where C1 is from first block and C2 from second.
    # If we decompose each with rank 10, total rank = 20.
    # But the W factors must add: W_total = W1 + W2 (element-wise for same k,n+i indices)
    # Actually, the W indices are (k*n+i) which is the same for both blocks.
    # So we can just add the W contributions.
    
    # Let me construct the 2x2x5 rank-10 decomposition!
    
    def build_2x2x5_rank10():
        """Build rank-10 decomposition for 2x2x5 matmul tensor.
        U: (2*2, 10) = (4, 10)
        V: (2*5, 10) = (10, 10)  
        W: (5*2, 10) = (10, 10)
        such that sum_r U[i*2+j,r] * V[j*5+k,r] * W[k*2+i,r] = 1
        """
        rank = 10
        U = np.zeros((4, rank))
        V = np.zeros((10, rank))
        W = np.zeros((10, rank))
        
        # Known construction for 2x2xn with rank n+5:
        # Use Strassen for the "core" and add terms for extra k values.
        # 
        # The 7 Strassen terms handle a 2x2x2 subproblem.
        # For 2x2x5, we need to handle k=0,1,2,3,4.
        # 
        # Standard result: rank(2x2xn) = n+5 for n >= 2.
        # Construction:
        # Terms 0-6: Generalized Strassen incorporating all n k-values
        # Terms 7 to n+4: Additional terms
        #
        # Let me use a known explicit construction.
        # From literature: the 2x2xn tensor can be written as:
        # 7 terms (Strassen-like) + (n-2) terms
        #
        # For n=5: 7 + 3 = 10 terms.
        
        # Here's a construction I'll derive:
        # M1 = (A00+A11) * (B00+B11)  -> C00 += M1, C11 += M1
        # M2 = (A10+A11) * B00         -> C10 += M2, C11 -= M2
        # M3 = A00 * (B01-B11)         -> C01 += M3, C11 += M3
        # M4 = A11 * (B10-B00)         -> C00 += M4, C10 += M4
        # M5 = (A00+A01) * B11         -> C00 -= M5, C01 += M5
        # M6 = (A10-A00) * (B00+B01)   -> C11 += M6
        # M7 = (A01-A11) * (B10+B11)   -> C00 += M7
        
        # These 7 terms handle k=0,1. For k=2,3,4, we need:
        # C00 += A00*B0k + A10*B1k  (wait, no: C[i,k] = sum_j A[i,j]*B[j,k])
        # C00 += A00*B0k, C01 += A01*B0k, C10 += A10*B1k, C11 += A11*B1k
        # Wait, I'm confusing indices. Let me be careful.
        # C[i,k] = sum_{j=0}^{1} A[i,j] * B[j,k]
        # So C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
        # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
        # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
        # C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1]
        # C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k] for k=2,3,4
        # C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k] for k=2,3,4
        
        # The 7 Strassen terms above handle k=0,1 correctly.
        # For k=2,3,4, we need 4 terms per k (schoolbook) = 12 terms.
        # Total would be 7+12=19, not 10.
        
        # The rank-10 construction must be more clever, sharing terms across k values.
        
        # Let me try a different known construction.
        # Actually, I recall that the correct formula is:
        # rank(2x2xn) = 7 + (n-2) for n >= 2, giving 10 for n=5.
        
        # Here's how it works:
        # The 7 Strassen terms can be "extended" to handle more k values by
        # modifying the B factors to include all k values.
        
        # M1 = (A00+A11) * sum_k B_k  where B_k involves B[0,k] and B[1,k]
        # But this doesn't directly work because different C entries need different combinations.
        
        # Let me just try the direct approach: use 5 terms for the "diagonal" 
        # and 5 terms for the "off-diagonal" or something like that.
        
        # Actually, let me try a completely different approach.
        # For 2x2xn, we can write:
        # C = A * B
        # Let D = A + J*A*J where J is the swap matrix [[0,1],[1,0]]
        # Then use 5 terms involving D and B, plus n terms.
        
        # I'm spending too much time deriving this. Let me just use the 
        # straightforward approach: for 2x4x5, split into two 2x2x5 blocks.
        # For each 2x2x5 block, use a rank-13 decomposition (7 Strassen + 6 direct).
        # Total: 2*13 = 26.
        
        # Actually wait - let me reconsider. The 2x2x5 rank might actually be 10.
        # Let me just try to find it numerically.
        
        return None
    
    # Given the complexity of deriving the exact rank-10 construction,
    # let me try a different strategy: direct search for rank 35.
    
    # The key question: is rank(2x4x5) <= 35?
    # We know rank(2x4x5) <= 36 from the Strassen construction.
    # Lower bounds: rank >= max(2*4, 4*5, 5*2) = max(8,20,10) = 20 (trivial)
    # Better lower bounds exist but let's just search.
    
    # Strategy: Use the rank-36 solution as a guide.
    # Try to find rank-35 by:
    # 1. Taking the rank-36 factors
    # 2. For each column c, try to "absorb" it into other columns
    # 3. Or: random search with structure
    
    # Let's try: initialize U, V with the rank-36 structure but with one 
    # column removed, then solve for W.
    
    found = False
    for R in [35]:
        if time.time() > deadline:
            break
        
        # Try multiple random initializations
        for trial in range(20000):
            if time.time() > deadline:
                break
            
            # Random U and V with small integer entries
            U_try = np.random.randint(-2, 3, size=(n * m, R)).astype(np.float64)
            V_try = np.random.randint(-2, 3, size=(m * p, R)).astype(np.float64)
            
            # Build the linear system for W
            # For each (i,j,k): sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
            n_eq = n * m * p  # 40
            n_var = p * n * R  # 10*R = 350
            
            A_mat = np.zeros((n_eq, n_var), dtype=np.float64)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        eq_idx = i * (m * p) + j * p + k
                        var_base = (k * n + i) * R
                        A_mat[eq_idx, var_base:var_base + R] = (
                            U_try[i * m + j, :] * V_try[j * p + k, :]
                        )
            
            B_vec = np.ones(n_eq)
            
            try:
                W_flat, residuals, rank_sol, sv = np.linalg.lstsq(A_mat, B_vec, rcond=None)
            except:
                continue
            
            if residuals is None or len(residuals) == 0:
                continue
            if residuals[0] > 1e-10:
                continue
            
            # Check if W can be rounded to half-integers
            W_rounded = np.round(W_flat * 2) / 2.0
            W_mat = W_rounded.reshape((p * n, R))
            
            # Verify
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        sum_val = np.dot(
                            U_try[i * m + j, :],
                            V_try[j * p + k, :] * W_mat[k * n + i, :]
                        )
                        if abs(sum_val - 1.0) > 1e-9:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                best_rank = R
                best_U = U_try
                best_V = V_try
                best_W = W_mat
                found = True
                break
        
        if found:
            break
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}


def _build_strassen_36(n, m, p):
    """Build Strassen rank-36 decomposition for 2x4 by 4x5 matrix multiplication."""
    rank = 36
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    col = 0

    def add_strassen_222(i0, j0, k0, col_start):
        nonlocal col
        c = col_start
        U[i0 * m + j0, c] += 1
        U[(i0 + 1) * m + (j0 + 1), c] += 1
        V[j0 * p + k0, c] += 1
        V[(j0 + 1) * p + (k0 + 1), c] += 1
        W[k0 * n + i0, c] += 1
        W[(k0 + 1) * n + (i0 + 1), c] += 1
        col += 1

        c = col_start + 1
        U[(i0 + 1) * m + j0, c] += 1
        U[(i0 + 1) * m + (j0 + 1), c] += 1
        V[j0 * p + k0, c] += 1
        W[k0 * n + (i0 + 1), c] += 1
        W[(k0 + 1) * n + (i0 + 1), c] -= 1
        col += 1

        c = col_start + 2
        U[i0 * m + j0, c] += 1
        V[j0 * p + (k0 + 1), c] += 1
        V[(j0 + 1) * p + (k0 + 1), c] -= 1
        W[(k0 + 1) * n + i0, c] += 1
        W[(k0 + 1) * n + (i0 + 1), c] += 1
        col += 1

        c = col_start + 3
        U[(i0 + 1) * m + (j0 + 1), c] += 1
        V[(j0 + 1) * p + k0, c] += 1
        V[j0 * p + k0, c] -= 1
        W[k0 * n + i0, c] += 1
        W[k0 * n + (i0 + 1), c] += 1
        col += 1

        c = col_start + 4
        U[i0 * m + j0, c] += 1
        U[i0 * m + (j0 + 1), c] += 1
        V[(j0 + 1) * p + (k0 + 1), c] += 1
        W[k0 * n + i0, c] -= 1
        W[(k0 + 1) * n + i0, c] += 1
        col += 1

        c = col_start + 5
        U[(i0 + 1) * m + j0, c] += 1
        U[i0 * m + j0, c] -= 1
        V[j0 * p + k0, c] += 1
        V[j0 * p + (k0 + 1), c] += 1
        W[(k0 + 1) * n + (i0 + 1), c] += 1
        col += 1

        c = col_start + 6
        U[i0 * m + (j0 + 1), c] += 1
        U[(i0 + 1) * m + (j0 + 1), c] -= 1
        V[(j0 + 1) * p + k0, c] += 1
        V[(j0 + 1) * p + (k0 + 1), c] += 1
        W[k0 * n + i0, c] += 1
        col += 1

        return col_start + 7

    def add_schoolbook_2x4x1(i0, j0, k0, col_start):
        c = col_start
        for di in range(2):
            for dj in range(4):
                U[(i0 + di) * m + (j0 + dj), c] = 1
                V[(j0 + dj) * p + k0, c] = 1
                W[k0 * n + (i0 + di), c] = 1
                c += 1
        return c

    col = add_strassen_222(0, 0, 0, 0)
    col = add_strassen_222(0, 2, 0, col)
    col = add_strassen_222(0, 0, 2, col)
    col = add_strassen_222(0, 2, 2, col)
    col = add_schoolbook_2x4x1(0, 0, 4, col)

    assert col == rank, f"Expected rank {rank}, got {col}"
    return U, V, W
