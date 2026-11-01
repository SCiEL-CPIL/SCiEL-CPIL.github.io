"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [2, 2, 4]

# EVOLVE-BLOCK-START
import numpy as np
from numpy.linalg import svd, norm

def _build_rank14():
    """Return the verified rank-14 Strassen-based decomposition."""
    n, m, p = 2, 2, 4
    rank = 14
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # A indices: A[0]=A00, A[1]=A01, A[2]=A10, A[3]=A11
    # B indices: B[0]=B00, B[1]=B01, B[2]=B02, B[3]=B03, B[4]=B10, B[5]=B11, B[6]=B12, B[7]=B13
    # C indices (W): C[0]=C00, C[1]=C10, C[2]=C01, C[3]=C11, C[4]=C02, C[5]=C12, C[6]=C03, C[7]=C13
    
    # Left block (cols 0,1) - Strassen
    # M1 = (A00+A11)(B00+B11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[5][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2 = (A10+A11)B00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1
    
    # M3 = A00(B01-B11)
    U[0][2] = 1
    V[1][2] = 1; V[5][2] = -1
    W[2][2] = 1; W[3][2] = 1
    
    # M4 = A11(B10-B00)
    U[3][3] = 1
    V[4][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1
    
    # M5 = (A00+A01)B11
    U[0][4] = 1; U[1][4] = 1
    V[5][4] = 1
    W[0][4] = -1; W[2][4] = 1
    
    # M6 = (A10-A00)(B00+B01)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # M7 = (A01-A11)(B10+B11)
    U[1][6] = 1; U[3][6] = -1
    V[4][6] = 1; V[5][6] = 1
    W[0][6] = 1

    # Right block (cols 2,3) - Strassen
    # N1 = (A00+A11)(B02+B13)
    U[0][7] = 1; U[3][7] = 1
    V[2][7] = 1; V[7][7] = 1
    W[4][7] = 1; W[7][7] = 1
    
    # N2 = (A10+A11)B02
    U[2][8] = 1; U[3][8] = 1
    V[2][8] = 1
    W[5][8] = 1; W[7][8] = -1
    
    # N3 = A00(B03-B13)
    U[0][9] = 1
    V[3][9] = 1; V[7][9] = -1
    W[6][9] = 1; W[7][9] = 1
    
    # N4 = A11(B12-B02)
    U[3][10] = 1
    V[6][10] = 1; V[2][10] = -1
    W[4][10] = 1; W[5][10] = 1
    
    # N5 = (A00+A01)B13
    U[0][11] = 1; U[1][11] = 1
    V[7][11] = 1
    W[4][11] = -1; W[6][11] = 1
    
    # N6 = (A10-A00)(B02+B03)
    U[2][12] = 1; U[0][12] = -1
    V[2][12] = 1; V[3][12] = 1
    W[7][12] = 1
    
    # N7 = (A01-A11)(B12+B13)
    U[1][13] = 1; U[3][13] = -1
    V[6][13] = 1; V[7][13] = 1
    W[4][13] = 1

    return {"U": U, "V": V, "W": W}


def _build_constraint_matrix():
    """
    Build the linear constraint matrix for the bilinear multiplication tensor.
    
    For matrix multiplication C = A * B where A is n×m, B is m×p, C is n×p:
    C[i,j] = sum_k A[i,k] * B[k,j]
    
    The tensor T has entries T[i,j,k] = 1 if the product A[i,k]*B[k,j] contributes to C[i,j].
    
    We want to find U (n*m × r), V (m*p × r), W (p*n × r) such that:
    sum_r U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r] = T[i,j,k]
    
    This is a bilinear constraint in (U,V,W). To linearize for a fixed rank r,
    we can fix two of the three matrices and solve for the third, or use
    the null-space approach.
    
    Alternative: For a given rank r, we need vectors u_r, v_r, w_r such that:
    sum_r u_r ⊗ v_r ⊗ w_r = T
    
    The constraint is: for all (i,j,k):
    sum_r U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r] = delta(i,j,k)
    where delta = 1 if k is the connecting index (always true in standard matmul).
    
    Actually, T[i,j,k] = 1 for all valid (i,j,k) in standard matrix multiplication
    since C[i,j] = sum_k A[i,k] B[k,j].
    
    For the linear system approach with fixed rank r:
    We have r unknown columns in U, V, W. The constraint is bilinear, not linear.
    
    However, we can use the following approach:
    Fix the structure by treating it as: find r triples (u_r, v_r, w_r) such that
    the sum of their outer products equals T.
    
    For the null-space approach:
    The space of all valid decompositions of rank r can be parameterized.
    But a simpler approach: use random initialization and gradient-based optimization
    to minimize ||sum_r u_r ⊗ v_r ⊗ w_r - T||_F^2 subject to sparsity.
    
    For now, let's use a simpler approach: 
    1. Start with the known rank-14 solution
    2. Try to merge pairs of rank-1 terms to reduce rank
    3. Or: use SVD-like decomposition on the tensor
    """
    n, m, p = 2, 2, 4
    
    # Build the multiplication tensor T of shape (n, p, m)
    # T[i, j, k] = 1 (since C[i,j] = sum_k A[i,k]*B[k,j])
    T = np.zeros((n, p, m))
    for i in range(n):
        for j in range(p):
            for k in range(m):
                T[i, j, k] = 1.0
    
    return T, n, m, p


def _verify_decomposition(U, V, W, tol=1e-10):
    """Verify that the decomposition correctly computes matrix multiplication."""
    n, m, p = 2, 2, 4
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    r = U.shape[1]
    
    # Check shapes
    if U.shape != (n * m, r):
        return False
    if V.shape != (m * p, r):
        return False
    if W.shape != (p * n, r):
        return False
    
    # For each rank term, compute the contribution
    # U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r] should sum to 1 for the matmul tensor
    # Actually, the decomposition means:
    # C[i,j] = sum_r (sum_k U[i*m+k, r] * A[i,k]) * (sum_k V[k*p+j, r] * B[k,j]) * W[j*n+i, r]
    # Wait, that's not right either. Let me reconsider.
    
    # The standard tensor decomposition for matmul:
    # C[i,j] = sum_r M_r * W[j*n+i, r]
    # where M_r = (sum_k U[i*m+k, r] * A[i,k]) * (sum_k V[k*p+j, r] * B[k,j])
    
    # Actually, the bilinear form is:
    # C[i,j] = sum_r (U[:, r]^T * A_vec) * (V[:, r]^T * B_vec) * W[j*n+i, r]
    
    # But this is a trilinear form, not bilinear. The correct formulation:
    # The multiplication tensor T has entries T[i,j,k] for the bilinear map.
    # A decomposition of rank r means:
    # T[i,j,k] = sum_r U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r]
    
    # So we need: sum_r U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r] = 1 for all (i,j,k)
    
    max_err = 0.0
    for i in range(n):
        for j in range(p):
            for k in range(m):
                val = 0.0
                for r in range(r):
                    val += U[i * m + k, r] * V[k * p + j, r] * W[j * n + i, r]
                if abs(val - 1.0) > max_err:
                    max_err = abs(val - 1.0)
    
    return max_err < tol


def _search_lower_rank(deadline):
    """
    Search for a lower-rank decomposition using null-space and sparsification.
    Returns a dict if found, else None.
    """
    T, n, m, p = _build_constraint_matrix()
    target_rank = 13  # Try to find rank 13
    
    # Approach: Use the known rank-14 solution as a starting point
    # and try to find a linear combination that has a zero column,
    # effectively reducing the rank.
    
    base = _build_rank14()
    U14 = np.array(base['U'], dtype=float)
    V14 = np.array(base['V'], dtype=float)
    W14 = np.array(base['W'], dtype=float)
    
    # The space of all valid decompositions of rank <= 14 that are linear
    # combinations of the 14 rank-1 terms can be explored.
    # If there exists a non-trivial linear relation among the 14 terms
    # such that one coefficient becomes 0, we can reduce rank.
    
    # Alternative: try random linear combinations of the 14 terms
    # and check if any column becomes (nearly) zero.
    
    np.random.seed(42)
    
    for trial in range(5000):
        if time.time() > deadline - 10:
            break
        
        # Random coefficients for combining the 14 rank-1 terms
        coeffs = np.random.randn(14)
        
        # New U, V, W are linear combinations of the original columns
        # U_new[:, j] = sum_i coeffs[i] * U14[:, i] for some permutation
        # This doesn't directly reduce rank.
        
        # Better: try to find if any subset of 13 terms spans the same tensor.
        # For each term to remove, check if the remaining 13 can still represent T.
        # This requires solving a system of equations.
        
        pass
    
    # Try a different approach: use SVD on the matricized tensor
    # to find a low-rank approximation, then round to integers.
    
    # Matricize T into a matrix of shape (n*p, m) or similar
    # T is (n, p, m) = (2, 4, 2)
    # Reshape to (n*p, m) = (8, 2)
    T_mat = T.reshape(n * p, m)
    
    # SVD
    U_svd, s_vals, Vt = svd(T_mat, full_matrices=False)
    
    # The rank of T_mat is at most 2, which is much less than 14.
    # This means the tensor has a low-rank structure in this matricization.
    
    # But this doesn't directly give us the bilinear rank.
    
    # For now, return None if no improvement found.
    return None


def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit
    
    # First, try to search for a lower rank
    result = _search_lower_rank(deadline)
    
    if result is not None:
        # Verify the found decomposition
        if _verify_decomposition(result['U'], result['V'], result['W']):
            return result
    
    # Fall back to the known rank-14 decomposition
    return _build_rank14()
# EVOLVE-BLOCK-END
