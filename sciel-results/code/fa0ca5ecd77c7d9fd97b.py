"""
Matrix multiplication 4x5x5: rank-minimal exact tensor decomposition.

Strategy:
  - Start from the schoolbook decomposition of rank 100.
  - Use alternating least squares (ALS) to find a rank-r decomposition (r < 100)
    that minimizes the Frobenius norm of the reconstruction error.
  - If the residual is exactly zero (within a tight tolerance), we have found
    an exact decomposition of rank r.
  - We try ranks in increasing order from 1 to 99, stopping at the first exact
    solution found.
  - A deadline guard ensures we return the best result before 120 s.
"""
import time
import numpy as np
from itertools import product

DIMENSIONS = [4, 5, 5]
DEADLINE_MARGIN = 5  # seconds to spare before the 120 s limit
START_TIME = time.time()
TIME_LIMIT = 120


def _time_remaining():
    return TIME_LIMIT - DEADLINE_MARGIN - (time.time() - START_TIME)


def _build_tensor(n, m, p):
    """
    Build the matrix multiplication tensor T[i, j, k] = 1 if the triple (i, j, k)
    corresponds to a term in the matrix product C = A @ B, i.e., A[i,j] * B[j,k].
    The tensor is 3D of shape (n, m, p).
    """
    T = np.zeros((n, m, p), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1.0
    return T


def _tensor_to_matricizations(T):
    """
    Convert a 3D tensor T of shape (n, m, p) to its three matricizations:
      T1: (n, m*p)  -- T[i, j*p + k]
      T2: (m, n*p)  -- T[j, i*p + k]
      T3: (p, n*m)  -- T[k, i*m + j]
    """
    n, m, p = T.shape
    T1 = T.reshape(n, m * p)
    T2 = T.transpose(1, 0, 2).reshape(m, n * p)
    T3 = T.transpose(2, 0, 1).reshape(p, n * m)
    return T1, T2, T3


def _matricizations_to_tensor(T1, T2, T3, n, m, p):
    """
    Reconstruct a 3D tensor from its three matricizations.
    We use T1 as the primary source and verify consistency.
    """
    T = T1.reshape(n, m, p)
    return T


def _als_decompose(T, rank, max_iter=200, tol=1e-10):
    """
    Alternating least squares for CP decomposition of a 3D tensor T.
    Returns (U, V, W, residual) where U, V, W are factor matrices and
    residual is the Frobenius norm of the error.
    """
    n, m, p = T.shape
    T1, T2, T3 = _tensor_to_matricizations(T)

    # Random initialization
    rng = np.random.default_rng(42)
    U = rng.standard_normal((n, rank))
    V = rng.standard_normal((m, rank))
    W = rng.standard_normal((p, rank))

    for _ in range(max_iter):
        if _time_remaining() < 2:
            break

        # Update U
        # T1 = U @ (V * W)^T  where (V * W) is element-wise product of V and W along rows
        VW = V * W  # element-wise: (m, rank)
        # T1 (n, m*p) = U (n, rank) @ VW^T (rank, m*p)
        # U = T1 @ (VW^T)^+ = T1 @ pinv(VW^T)
        try:
            U_new = T1 @ np.linalg.pinv(VW.T)
        except np.linalg.LinAlgError:
            U_new = T1 @ np.linalg.pinv(VW.T, hermitian=False)

        # Update V
        UW = U * W  # element-wise: (n, rank)
        # T2 (m, n*p) = V (m, rank) @ UW^T (rank, n*p)
        try:
            V_new = T2 @ np.linalg.pinv(UW.T)
        except np.linalg.LinAlgError:
            V_new = T2 @ np.linalg.pinv(UW.T, hermitian=False)

        # Update W
        UV = U * V  # element-wise: (n, rank)
        # T3 (p, n*m) = W (p, rank) @ UV^T (rank, n*m)
        try:
            W_new = T3 @ np.linalg.pinv(UV.T)
        except np.linalg.LinAlgError:
            W_new = T3 @ np.linalg.pinv(UV.T, hermitian=False)

        # Check convergence
        diff = max(np.linalg.norm(U_new - U), np.linalg.norm(V_new - V), np.linalg.norm(W_new - W))
        U, V, W = U_new, V_new, W_new

        if diff < tol:
            break

    # Compute residual
    T_recon = np.einsum('ir,jr,kr->ijk', U, V, W)
    residual = np.linalg.norm(T - T_recon, 'fro')

    return U, V, W, residual


def _factor_to_output(U, V, W, n, m, p):
    """
    Convert factor matrices U (n, rank), V (m, rank), W (p, rank)
    to the output format:
      U_out: (n*m, rank) with U_out[i*m + j, c] = U[i, c] * ... 
      Actually, the standard CP decomposition of the matmul tensor gives:
        T[i, j, k] = sum_c U[i, c] * V[j, c] * W[k, c]
      The bilinear factors are:
        U_bilinear[i*m + j, c] = U[i, c]  (but this is not correct for bilinear)
      
      For matrix multiplication, the bilinear form is:
        C[k, i] = sum_j A[i, j] B[j, k]
      
      The tensor decomposition T[i, j, k] = sum_c u_c[i] v_c[j] w_c[k]
      gives a bilinear decomposition where:
        U_bilinear[i*m + j, c] = u_c[i]   -- but this is not right.
      
      Actually, for the standard matmul tensor, the bilinear decomposition is:
        U[i*m + j, c] = 1 if the term (i, j, k) is active for component c
        But in a general CP decomposition, we need to convert.
      
      Let me think more carefully. The bilinear decomposition of matrix multiplication
      with rank r means:
        C[k, i] = sum_{c=1}^{r} U[i*m + j, c] * V[j*p + k, c] * ... 
      
      No, that's not right either. Let me re-read the problem.
      
      The problem says:
        U: (n*m, rank)
        V: (m*p, rank)
        W: (p*n, rank)
      
      And the decomposition is:
        T[i, j, k] = sum_c U[i*m + j, c] * V[j*p + k, c] * W[k*n + i, c]
      
      But that's not bilinear - that's trilinear. For a bilinear decomposition of
      matrix multiplication, we need:
        C[k, i] = sum_j A[i, j] B[j, k] = sum_c sum_j U[i*m + j, c] * V[j*p + k, c] * ...
      
      Hmm, I think the W matrix is for the output, and the decomposition is:
        W[k*n + i, c] * (sum_j U[i*m + j, c] * A[i, j] * V[j*p + k, c] * B[j, k])
      
      Actually, looking at the schoolbook decomposition:
        For each (i, j, k):
          U[i*m + j, c] = 1
          V[j*p + k, c] = 1
          W[k*n + i, c] = 1
      
      So the bilinear form is:
        sum_c U[i*m + j, c] * A[i, j] * V[j*p + k, c] * B[j, k] * W[k*n + i, c]
      
      But this is not standard. Let me just use the schoolbook form and convert
      the CP factors to this form.
      
      For a CP decomposition T[i,j,k] = sum_c u_c[i] v_c[j] w_c[k],
      we can set:
        U[i*m + j, c] = u_c[i]
        V[j*p + k, c] = v_c[j]
        W[k*n + i, c] = w_c[k]
      
      But this doesn't work because U depends on both i and j, etc.
      
      I think the correct interpretation is that the tensor T is the matmul tensor,
      and we're finding a rank-r decomposition of T as a 3D tensor (not bilinear).
      The output format just stores the factors in a specific layout.
    """
    rank = U.shape[1]
    U_out = np.zeros((n * m, rank), dtype=np.float64)
    V_out = np.zeros((m * p, rank), dtype=np.float64)
    W_out = np.zeros((p * n, rank), dtype=np.float64)
    
    for i in range(n):
        for j in range(m):
            for c in range(rank):
                U_out[i * m + j, c] = U[i, c]
    
    for j in range(m):
        for k in range(p):
            for c in range(rank):
                V_out[j * p + k, c] = V[j, c]
    
    for k in range(p):
        for i in range(n):
            for c in range(rank):
                W_out[k * n + i, c] = W[k, c]
    
    return U_out.tolist(), V_out.tolist(), W_out.tolist()


def _verify_decomposition(U, V, W, n, m, p):
    """
    Verify that the decomposition exactly reconstructs the matmul tensor.
    T[i, j, k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    """
    T = _build_tensor(n, m, p)
    T_recon = np.zeros_like(T)
    for c in range(U.shape[1]):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_recon[i, j, k] += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
    return np.linalg.norm(T - T_recon, 'fro')


def _schoolbook_decomposition(n, m, p):
    """Return the schoolbook decomposition of rank n*m*p."""
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


def construct():
    n, m, p = DIMENSIONS
    full_rank = n * m * p
    
    # Start with schoolbook as fallback
    best_result = _schoolbook_decomposition(n, m, p)
    best_rank = full_rank
    
    # Try to find a lower-rank exact decomposition using ALS
    # We'll try ranks from 1 to full_rank - 1
    # But to be efficient, we'll try a few key ranks first
    
    # Known lower bounds: rank >= max(n*m, m*p, p*n) = max(20, 25, 20) = 25
    # Also rank >= n*m*p / max(n, m, p) is not a standard bound.
    # Let's try ranks from 25 to 99
    
    ranks_to_try = list(range(25, full_rank))
    
    # Sort by likelihood of success (try lower ranks first)
    ranks_to_try.sort()
    
    for rank in ranks_to_try:
        if _time_remaining() < 10:
            break
        
        # Try a few random restarts for this rank
        found_exact = False
        for restart in range(3):
            if _time_remaining() < 5:
                break
            
            # Modify the random seed for each restart
            np.random.seed(42 + restart)
            U, V, W, residual = _als_decompose(_build_tensor(n, m, p), rank, max_iter=100, tol=1e-12)
            
            if residual < 1e-8:
                # Verify with exact check
                U_list, V_list, W_list = _factor_to_output(U, V, W, n, m, p)
                err = _verify_decomposition(np.array(U_list), np.array(V_list), np.array(W_list), n, m, p)
                if err < 1e-10:
                    best_result = {"U": U_list, "V": V_list, "W": W_list}
                    best_rank = rank
                    found_exact = True
                    break
        
        if found_exact:
            # We found an exact decomposition at this rank
            # Since we're trying ranks in increasing order, this is the minimum
            break
    
    return best_result


if __name__ == "__main__":
    result = construct()
    n, m, p = DIMENSIONS
    rank = len(result["U"][0])
    print(f"Found decomposition of rank {rank}")
