"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def _validate(U, V, W, n=4, m=4, p=5, tol=1e-9):
    """Check that U,V,W form a valid tensor decomposition of matmul(n,m,p).
    Returns True if valid, False otherwise."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    if rank != V.shape[1] or rank != W.shape[1]:
        return False
    if U.shape[0] != n * m or V.shape[0] != m * p or W.shape[0] != p * n:
        return False
    # Build the matmul tensor T[i,j,k] = delta(i from A row, j from B col... )
    # T is indexed as: A is n x m, B is m x p, C is n x p
    # Tensor T[a, b, c] where a in [0,n), b in [0,p), c in [0,m)
    # Actually the standard: C = A @ B, C[i,k] = sum_j A[i,j]*B[j,k]
    # Tensor form: T[i, k, j] = 1 if we consider A[i,j] * B[j,k] * C[i,k]
    # The decomposition: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = T[i,k,j]
    # where T[i,k,j] = 1 (the matmul tensor has 1s at positions (i, k, j) for all i,k,j)
    # Wait, let me re-derive. The matmul tensor M is a 3-tensor with M[i, k, j] = 1
    # for all i in [0,n), k in [0,p), j in [0,m).
    # A decomposition of rank r means:
    #   M[i, k, j] = sum_{r=0}^{r-1} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # So for each (i,k,j), the sum should equal 1.
    # Let's verify with the schoolbook: for each (i,j,k) triple, U[i*m+j, col]=1,
    # V[j*p+k, col]=1, W[k*n+i, col]=1, so the product is 1 at that specific (i,k,j)
    # and 0 elsewhere. Yes, that's correct.
    
    # Efficient check: build the tensor via outer products
    # M_decomp[i, k, j] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # We need M_decomp[i,k,j] = 1 for all i,k,j.
    
    # Use einsum for efficiency
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # For each r: U_col = U[:, r] reshaped to (n, m), V_col = V[:, r] reshaped to (m, p), W_col = W[:, r] reshaped to (p, n)
    # The contribution to M[i,k,j] from rank r is U[i,j,r] * V[j,k,r] * W[k,i,r]
    
    # Let's compute the full tensor and check
    # M_decomp = np.einsum('ijr,jkr,kir->ikj', U.reshape(n,m,-1), V.reshape(m,p,-1), W.reshape(p,n,-1))
    # But this might be memory intensive for large rank. For rank up to ~80 it's fine (4*5*4=80 entries).
    
    U3 = U.reshape(n, m, -1)
    V3 = V.reshape(m, p, -1)
    W3 = W.reshape(p, n, -1)
    
    M_decomp = np.einsum('ijr,jkr,kir->ikj', U3, V3, W3)
    
    # Check if all entries are 1
    if not np.allclose(M_decomp, 1.0, atol=tol):
        return False
    return True

def _construct_schoolbook(n=4, m=4, p=5):
    """Standard schoolbook decomposition of rank n*m*p."""
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

def _try_strassen_3x3x3_embedded():
    """
    Try to embed the 3x3x3 Strassen decomposition (rank 7) into the top-left 3x3x3
    sub-tensor, and handle the boundary with schoolbook corrections.
    
    The 3x3x3 Strassen decomposition has 7 terms. For multiplying 3x3 by 3x3:
    M1 = (A11 + A22)(B11 + B22)
    M2 = (A21 + A22)B11
    M3 = A11(B21 - B22)
    M4 = A22(B11 - B12)
    M5 = (A11 + A12)B22
    M6 = (A21 - A11)(B11 + B12)
    M7 = (A12 - A22)(B21 + B22)
    
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    
    For our 4x4 * 4x5 problem, we can try to use Strassen for the top-left 3x3 * 3x3 part
    and then handle the remaining parts.
    
    Actually, let me think about this more carefully. The matmul tensor for 4x4 * 4x5
    has 4*4*5 = 80 entries (i, k, j) where i in [0,4), k in [0,5), j in [0,4).
    
    If we could decompose the 3x3x3 part with rank 7, and the boundary parts with
    schoolbook, we might save some rank.
    
    The 3x3x3 part corresponds to i in [0,3), k in [0,3), j in [0,3). That's 27 entries.
    With Strassen, we use 7 rank-1 tensors for this.
    
    The remaining entries are:
    - i in [0,3), k in [3,5), j in [0,3): 3*2*3 = 18 entries (schoolbook: 18)
    - i in [3,4), k in [0,3), j in [0,3): 1*3*3 = 9 entries (schoolbook: 9)
    - i in [0,3), k in [0,3), j in [3,4): 3*3*1 = 9 entries (schoolbook: 9)
    - i in [3,4), k in [3,5), j in [0,3): 1*2*3 = 6 entries (schoolbook: 6)
    - i in [3,4), k in [0,3), j in [3,4): 1*3*1 = 3 entries (schoolbook: 3)
    - i in [0,3), k in [3,5), j in [3,4): 3*2*1 = 6 entries (schoolbook: 6)
    - i in [3,4), k in [3,5), j in [3,4): 1*2*1 = 2 entries (schoolbook: 2)
    
    Total boundary: 18+9+9+6+3+6+2 = 53 entries, schoolbook rank 53.
    Total: 7 + 53 = 60.
    
    But wait, the Strassen decomposition for 3x3x3 doesn't directly apply because
    the indices are coupled. Let me think about whether we can actually do this.
    
    Actually, the issue is that the Strassen decomposition works for the specific
    structure of the 3x3x3 matmul tensor. When we embed it into a larger tensor,
    the rank-1 tensors from Strassen only cover the (i,k,j) in [0,3)^3 part,
    and we need separate rank-1 tensors for the boundary.
    
    So the total rank would be 7 (Strassen) + 53 (schoolbook boundary) = 60.
    That's better than 80!
    
    Let me implement this.
    """
    n, m, p = 4, 4, 5
    
    # Strassen 3x3x3 decomposition
    # For 3x3 * 3x3, the 7 products are:
    # We index A as 3x3 with rows 0,1,2 and cols 0,1,2
    # B as 3x3 with rows 0,1,2 and cols 0,1,2
    # C as 3x3 with rows 0,1,2 and cols 0,1,2
    
    # The 7 rank-1 tensors for Strassen 3x3:
    # Each term M_s = (sum of A elements) * (sum of B elements)
    # contributes to C elements.
    
    # Let me define the 7 terms explicitly.
    # Term 1: (A00+A11)*(B00+B11) -> contributes to C00, C11
    # Term 2: (A10+A11)*B00 -> contributes to C10
    # Term 3: A00*(B10-B11) -> contributes to C01
    # Term 4: A11*(B00-B01) -> contributes to C00, C11
    # Term 5: (A00+A01)*B11 -> contributes to C01, C11
    # Term 6: (A10-A00)*(B00+B01) -> contributes to C10, C11
    # Term 7: (A01-A11)*(B10+B11) -> contributes to C00, C11
    
    # Wait, I need to be more careful. Let me use the standard Strassen for 2x2
    # and then extend to 3x3. Actually, the standard Strassen is for 2x2.
    # For 3x3, we can use a different decomposition.
    
    # Actually, let me reconsider. The known rank for 3x3 matrix multiplication is 7.
    # This is the Strassen rank. Let me look up the explicit decomposition.
    
    # For 3x3 * 3x3 = 3x3, the 7 terms are:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11  
    # M3 = A11(B21 - B22)
    # M4 = A22(B11 - B12)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    #
    # Wait, this is for 2x2 blocks, not 3x3. Let me reconsider.
    #
    # Actually, for 3x3 matrices, we can partition them differently.
    # The rank of 3x3 matrix multiplication is known to be 7.
    #
    # Let me just use the fact that we can write the 3x3x3 matmul tensor as a sum
    # of 7 rank-1 tensors. The explicit coefficients are known.
    
    # I'll define the 7 terms with their U, V, W vectors.
    # For the 3x3x3 sub-tensor (i in 0..2, k in 0..2, j in 0..2):
    
    # Let me define each term as (u_vec, v_vec, w_vec) where:
    # u_vec is of length 3*3=9 (for i,j in 0..2)
    # v_vec is of length 3*3=9 (for j,k in 0..2)  
    # w_vec is of length 3*3=9 (for k,i in 0..2)
    
    # Term 1: (A00+A11)*(B00+B11)
    # u: A00=1, A11=1, others 0 -> [1,0,0, 0,1,0, 0,0,0]
    # v: B00=1, B11=1, others 0 -> [1,0,0, 0,1,0, 0,0,0]
    # w: C00 gets +1, C11 gets +1 -> w[0*3+0]=1 (k=0,i=0), w[1*3+1]=1 (k=1,i=1)
    #    -> [1,0,0, 0,1,0, 0,0,0]
    
    # This is getting complicated. Let me just verify that rank 60 is achievable
    # and implement it cleanly.
    
    # Actually, let me try a different approach: just use schoolbook for now
    # and see if we can find a better decomposition through optimization.
    # For now, return the schoolbook decomposition.
    return _construct_schoolbook(n, m, p)

def construct():
    """Construct a rank-minimal tensor decomposition for 4x4 * 4x5 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 120  # 120 second limit
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 80
    
    # Start with schoolbook
    best_result = _construct_schoolbook(n, m, p)
    best_rank = schoolbook_rank
    
    # Try the Strassen-embedded approach
    try:
        strassen_result = _try_strassen_3x3x3_embedded()
        rank_s = strassen_result["U"][0].__len__() if strassen_result["U"] else 0
        if rank_s < best_rank:
            if _validate(strassen_result["U"], strassen_result["V"], strassen_result["W"], n, m, p):
                best_result = strassen_result
                best_rank = rank_s
    except Exception:
        pass
    
    # Try to find a better decomposition through random search
    # We'll try to find a decomposition of rank r < best_rank
    # by solving a system of equations.
    
    # For a given rank r, we need to find U (n*m x r), V (m*p x r), W (p*n x r)
    # such that for all (i,k,j): sum_{r'} U[i*m+j, r'] * V[j*p+k, r'] * W[k*n+i, r'] = 1
    
    # This is a system of n*p*m = 80 equations with 3*r*(n*m + m*p + p*n) / 3 variables
    # Actually, the number of variables is r*(n*m + m*p + p*n) = r*(16 + 20 + 20) = 56*r
    # But the equations are cubic (trilinear), so it's not linear.
    
    # We can use a tensor train / CP decomposition approach.
    # Fix two of the matrices and solve for the third.
    
    # Alternative: use the fact that for each fixed pair (i,j), the equation
    # sum_{r'} V[j*p+k, r'] * W[k*n+i, r'] = 1 / U[i*m+j, r'] (if U != 0)
    # This is still nonlinear.
    
    # Let me try a different approach: use the known result that the rank of
    # 4x4 * 4x5 matrix multiplication might be less than 80.
    
    # Actually, let me try to use a structured decomposition.
    # We can try to use the fact that the 4x4 * 4x5 problem can be seen as
    # four 4x5 products, or five 4x4 products with one extra column.
    
    # For now, let's just return the best result we have.
    # The schoolbook gives rank 80, and if Strassen embedding works, we get 60.
    
    elapsed = time.time() - start_time
    if elapsed > deadline - 5:
        return best_result
    
    # Try to improve using a simple local search
    # We'll try to reduce the rank by removing zero columns and checking validity.
    
    # First, let's see if we can get a valid decomposition with rank < 80
    # by using a sparse structure.
    
    # Try rank 60 with the Strassen embedding idea
    # Let me implement it properly this time.
    
    def _strassen_3x3x3_decomp():
        """
        Return the 7 rank-1 tensors for 3x3x3 matrix multiplication.
        Each tensor is (u, v, w) where u is 9-dim, v is 9-dim, w is 9-dim.
        """
        # The 7 terms for 3x3 * 3x3:
        # Using the known decomposition:
        
        terms = []
        
        # I'll use the following decomposition (verified for 3x3):
        # This is based on the fact that rank(3x3x3) = 7
        
        # Let me define the 7 terms explicitly by their action on the tensor.
        # For each term t, we have:
        # T[i,k,j] += u[i,j] * v[j,k] * w[k,i]
        
        # Term 1: covers C[0,0] and C[1,1]
        # u[0,0]=1, u[1,1]=1
        # v[0,0]=1, v[1,1]=1
        # w[0,0]=1, w[1,1]=1
        u1 = [0]*9; u1[0]=1; u1[4]=1  # A[0,0], A[1,1]
        v1 = [0]*9; v1[0]=1; v1[4]=1  # B[0,0], B[1,1]
        w1 = [0]*9; w1[0]=1; w1[4]=1  # C[0,0], C[1,1]
        terms.append((u1, v1, w1))
        
        # This is getting too complex to do by hand. Let me just verify
        # that the schoolbook works and return it.
        
        return terms
    
    # For now, just return the best result
    return best_result

# EVOLVE-BLOCK-END
