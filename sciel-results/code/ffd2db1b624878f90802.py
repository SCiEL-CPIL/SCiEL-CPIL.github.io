"""Matrix multiplication 3x4x7 - rank-minimal exact tensor decomposition."""
DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction

def _verify_decomposition(U, V, W, n, m, p, rank):
    """Verify that the decomposition exactly computes matrix multiplication."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # The matrix multiplication tensor: C[i,k] = sum_j A[i,j] * B[j,k]
    # In the decomposition: C = sum_r U_r^T * A * V_r * W_r^T (in some sense)
    # More precisely, the bilinear map is:
    # C[i,k] = sum_r (sum_j U[i*m+j, r] * A[i,j]) * (sum_j' V[j'*p+k, r] * B[j',k]) * W[k*n+i, r]
    # Wait, that's not right. Let me think more carefully.
    #
    # The standard tensor decomposition of matrix multiplication:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # 
    # In the Kronecker product form used by the evaluator:
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # 
    # The decomposition means: for all A (n x m) and B (m x p),
    # C[i,k] = sum_r U[i*m + ?, r] ... 
    #
    # Actually, the standard formulation is:
    # Let's define the linear maps:
    # L_A: A -> vector of size n*m (row-major: A[i,j] at index i*m+j)
    # L_B: B -> vector of size m*p (row-major: B[j,k] at index j*p+k)
    # L_C: C -> vector of size p*n (row-major: C[i,k] at index k*n+i)
    #
    # The bilinear map f(A,B) = C satisfies:
    # L_C(C)[k*n+i] = sum_j A[i,j] B[j,k]
    #
    # In tensor decomposition form:
    # L_C(C)[k*n+i] = sum_r U[i*m+j, r] ... no, that's not right either.
    #
    # The correct formulation: The matrix multiplication tensor T has entries
    # T[i,j,k] = 1 if the product A[i,j]*B[j,k] contributes to C[i,k].
    # But actually T[i,j,k] should be such that:
    # C[i,k] = sum_{j,j'} T[i,j,k,j'] A[i,j] B[j',k]
    # where T[i,j,k,j'] = delta[j,j'] (i.e., T is just the identity in the j dimension).
    #
    # For the bilinear decomposition with factor matrices U, V, W:
    # The condition is that for all i, k:
    # sum_j A[i,j] B[j,k] = sum_r (sum_j U[i*m+j, r] A[i,j]) * (sum_j' V[j'*p+k, r] B[j',k]) * W[k*n+i, r]
    #
    # Wait, this doesn't look right dimensionally. Let me reconsider.
    #
    # Actually, I think the correct interpretation is:
    # U[i*m+j, r] is the coefficient for A[i,j] in the r-th "term"
    # V[j*p+k, r] is the coefficient for B[j,k] in the r-th "term"  
    # W[k*n+i, r] is the coefficient for C[k*n+i] in the r-th "term"
    #
    # And the decomposition says:
    # C[k*n+i] = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] * A[i,j]) * (sum_j' V[j'*p+k, r] * B[j',k])
    #
    # Hmm, but that would make it a product of two linear forms times a scalar, which is
    # a rank-1 term in the bilinear map sense. Let me verify with schoolbook.
    #
    # Schoolbook: for each (i,j,k), we have one term where:
    # U[i*m+j, r] = 1, V[j*p+k, r] = 1, W[k*n+i, r] = 1
    # Then: C[k*n+i] = sum_r W[k*n+i,r] * (sum_j' U[i*m+j',r] A[i,j']) * (sum_j'' V[j''*p+k,r] B[j'',k])
    # For the term where r corresponds to (i,j,k):
    # W[k*n+i, r] = 1
    # sum_j' U[i*m+j',r] A[i,j'] = A[i,j] (since U[i*m+j,r]=1 and 0 otherwise)
    # sum_j'' V[j''*p+k,r] B[j'',k] = B[j,k] (since V[j*p+k,r]=1 and 0 otherwise)
    # So this term contributes 1 * A[i,j] * B[j,k] = A[i,j]*B[j,k] to C[k*n+i]
    # Summing over all r, we get C[k*n+i] = sum_{i',j,k'} A[i',j]*B[j,k'] * W[k*n+i, r_{i',j,k'}]
    # But W[k*n+i, r_{i',j,k'}] = 1 only if k'=k and i'=i, so we get sum_j A[i,j]*B[j,k]. Correct!
    
    # Let's verify by checking on random matrices
    np.random.seed(42)
    for trial in range(5):
        A = np.random.randint(-5, 6, size=(n, m)).astype(float)
        B = np.random.randint(-5, 6, size=(m, p)).astype(float)
        C_expected = A @ B
        
        # Compute using decomposition
        C_computed = np.zeros((n, p), dtype=float)
        for r in range(rank):
            u_col = U[:, r]
            v_col = V[:, r]
            w_col = W[:, r]
            # u_col has shape (n*m,), reshape to (n, m)
            u_mat = u_col.reshape(n, m)
            # v_col has shape (m*p,), reshape to (m, p)
            v_mat = v_col.reshape(m, p)
            # w_col has shape (p*n,), reshape to (p, n)
            w_mat = w_col.reshape(p, n)
            
            # The r-th term contributes: w_col[k*n+i] * (u_mat @ A?) ... 
            # Actually: sum_j U[i*m+j, r] A[i,j] = (u_mat * A).sum() element-wise? No.
            # sum_j u_mat[i,j] * A[i,j] = (u_mat * A).sum() but that's a scalar.
            # Wait, I need to be more careful.
            
            # For each output position (i, k):
            # contribution from term r = W[k*n+i, r] * (sum_j U[i*m+j, r] * A[i,j]) * (sum_j' V[j'*p+k, r] * B[j',k])
            
        # Let me compute it properly
        C_computed = np.zeros((n, p), dtype=float)
        for i in range(n):
            for k in range(p):
                total = 0.0
                for r in range(rank):
                    # sum_j U[i*m+j, r] * A[i,j]
                    u_sum = sum(U[i*m+j, r] * A[i, j] for j in range(m))
                    # sum_j' V[j'*p+k, r] * B[j',k]
                    v_sum = sum(V[j*p+k, r] * B[j, k] for j in range(m))
                    total += W[k*n+i, r] * u_sum * v_sum
                C_computed[i, k] = total
        
        if not np.allclose(C_computed, C_expected, atol=1e-9):
            return False
    return True

def _try_strassen_like(n, m, p, deadline):
    """
    Attempt to find a lower-rank decomposition using a structured approach.
    For 3x4x7, we explore whether any rank < 84 exists.
    
    Strategy: Use the fact that for small matrices, we can try to find
    decompositions with rank significantly below schoolbook by exploiting
    algebraic structure.
    
    For 3x4x7: n=3, m=4, p=7. Schoolbook = 84.
    Known bounds: R(3,4,7) <= 84 (schoolbook).
    
    Let's try a different approach: use the Laderman-like or Winograd-like
    techniques adapted for rectangular matrices.
    
    Actually, for very small cases, let's try to search for a decomposition
    with rank r by solving a system of polynomial equations.
    """
    start = time.time()
    
    # For 3x4x7, let's try ranks from a reasonable upper bound down.
    # The theoretical lower bound is max(n*p, m*p, n*m) = max(21, 28, 12) = 28.
    # But practical lower bounds are higher. Let's try from 84 down to see
    # if we can find anything better.
    
    # Actually, a more principled approach: try to use the structure of
    # the problem. For 3x4x7, we can try to partition and use sub-decompositions.
    
    # Let's try a direct approach: assume rank r and try to find U, V, W
    # with entries in {0, 1/2, 1, -1/2, -1} or similar small rational values.
    
    # Given the time constraint, let's try a few candidate ranks.
    # If we can't find anything better quickly, return schoolbook.
    
    candidates = [84, 80, 76, 72, 68, 64, 60, 56, 52, 48, 44, 40, 36, 32, 28]
    
    for r in candidates:
        if time.time() > deadline:
            break
        if time.time() - start > 5:  # Spend at most 5s per candidate rank
            break
        result = _search_rank(n, m, p, r, deadline - (time.time() - start))
        if result is not None:
            return result
    
    return None

def _search_rank(n, m, p, r, time_budget):
    """
    Search for a rank-r decomposition using random initialization and
    iterative refinement. This is a heuristic search.
    """
    start = time.time()
    
    # Try random initializations with sparse structure
    for attempt in range(20):
        if time.time() - start > time_budget:
            break
        
        # Initialize with random sparse matrices
        np.random.seed(attempt * 1000 + r)
        
        U = np.zeros((n * m, r), dtype=float)
        V = np.zeros((m * p, r), dtype=float)
        W = np.zeros((p * n, r), dtype=float)
        
        # Random sparse initialization
        for col in range(r):
            # Each column of U has a few non-zeros
            u_idx = np.random.choice(n * m, size=min(3, n*m), replace=False)
            U[u_idx, col] = np.random.choice([-1, 1], size=len(u_idx))
            
            v_idx = np.random.choice(m * p, size=min(3, m*p), replace=False)
            V[v_idx, col] = np.random.choice([-1, 1], size=len(v_idx))
            
            w_idx = np.random.choice(p * n, size=min(3, p*n), replace=False)
            W[w_idx, col] = np.random.choice([-1, 1], size=len(w_idx))
        
        # Check if this happens to work (very unlikely for random init)
        if _verify_decomposition(U, V, W, n, m, p, r):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
        # Try a structured initialization based on schoolbook but with merging
        # For each output (i,k), we need sum_j A[i,j]B[j,k]
        # Group terms to reduce rank
        
        # Alternative: try to find a decomposition where some columns
        # handle multiple (i,j,k) triples
        U2 = np.zeros((n * m, r), dtype=float)
        V2 = np.zeros((m * p, r), dtype=float)
        W2 = np.zeros((p * n, r), dtype=float)
        
        # Start with schoolbook and try to merge columns
        # This is a greedy approach
        pass
    
    return None

def construct():
    """
    Returns a tensor decomposition for matrix multiplication 3x4 by 4x7.
    Attempts to find a rank-minimal decomposition, falling back to schoolbook.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 84
    
    # First, build the schoolbook decomposition as fallback
    U_school = np.zeros((n * m, schoolbook_rank), dtype=np.int8)
    V_school = np.zeros((m * p, schoolbook_rank), dtype=np.int8)
    W_school = np.zeros((p * n, schoolbook_rank), dtype=np.int8)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    # Try to find a better decomposition
    better = _try_strassen_like(n, m, p, deadline)
    
    if better is not None:
        # Verify the better decomposition
        U_b = np.array(better["U"], dtype=float)
        V_b = np.array(better["V"], dtype=float)
        W_b = np.array(better["W"], dtype=float)
        if _verify_decomposition(U_b, V_b, W_b, n, m, p, U_b.shape[1]):
            return better
    
    # Return schoolbook
    return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
# EVOLVE-BLOCK-END
