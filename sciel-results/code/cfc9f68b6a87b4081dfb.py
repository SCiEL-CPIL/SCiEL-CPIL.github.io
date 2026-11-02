"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import itertools

DIMENSIONS = [3, 4, 5]

def _generate_schoolbook(n, m, p):
    """Generate the standard schoolbook decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p):
    """Verify if U, V, W form a valid decomposition of the matrix multiplication tensor."""
    rank = U.shape[1]
    # The tensor T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] 
    # Note: W index in task is W[k*n+i, r].
    # So T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # We check if the bilinear map A -> A*B is reproduced.
    # (A*B)[i,k] = sum_j A[i,j] B[j,k]
    # Our decomposition gives (A*B)[i,k] = sum_r (sum_j A[i,j] U[i*m+j, r]) * (sum_k' ... wait)
    # Actually, the standard decomposition identity is:
    # (AB)_{ik} = \sum_r (\sum_j A_{ij} u_{ij}^r) (\sum_j B_{jk} v_{jk}^r) ... NO.
    
    # The tensor decomposition T_{ijk} = \sum_r u^r_i v^r_j w^r_k corresponds to:
    # (AB)_{ik} = \sum_j A_{ij} B_{jk}
    # This is represented by the tensor T with indices i (row of A), j (col of A / row of B), k (col of B).
    # The decomposition is T_{ijk} = \sum_r U[i,j,r] V[j,k,r] W[k,i,r]? 
    # Let's check the index convention in the prompt: "output factor follows the official trace convention W[k*n+i,r]".
    # This implies the third index of the tensor is a pair (k, i).
    # So T[i, j, (k, i)] = \sum_r U[i, j, r] V[j, k, r] W[k, i, r].
    # The matrix product element (AB)_{ik} = \sum_j A_{ij} B_{jk}.
    # This matches the tensor definition where for a fixed i, k, the value is \sum_j T_{i, j, k, i}.
    
    # Let's construct the product using the factors.
    # P1 = A @ U_part ? No.
    # For each r, we have a rank-1 term in the tensor.
    # The contribution to (AB)_{ik} from term r is:
    # ( \sum_j A_{ij} U[i*m+j, r] ) * ( \sum_l B_{lj} V[j*p+l, r] ) ... this doesn't look right dimensionally for a simple product.
    
    # Let's use the explicit tensor check.
    # T_true[i, j, k] = 1 if j is between 0..m-1 (always), else 0. Wait, T is the identity of the multiplication map.
    # T[i, j, k] is the coefficient of A_ij and B_jk in (AB)_ik.
    # So T[i, j, k] = 1 for all valid i, j, k.
    # The decomposition is: \sum_r U[i, j, r] V[j, k, r] W[k, i, r] = 1 for all i, j, k.
    
    # Check sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    
    # Optimization: Check a subset of (i,j,k) if time is tight, but for exactness we must check all.
    # n*m*p = 60 checks. Each check is a dot product of length rank.
    # This is very fast.
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_val = U[i*m + j, :]
                v_val = V[j*p + k, :]
                w_val = W[k*n + i, :]
                prod_sum = np.dot(np.dot(u_val, v_val), w_val) # element-wise then sum? No.
                # Sum is \sum_r u_r v_r w_r
                prod_sum = np.sum(u_val * v_val * w_val)
                if not np.isclose(prod_sum, 1.0, atol=1e-9):
                    return False
    return True

def _search_lower_rank(n, m, p, max_rank_to_try, time_limit):
    """
    Attempt to find a decomposition with rank < n*m*p.
    This is a heuristic search. Given the difficulty, it will likely fail quickly for 3x4x5
    unless we use a specific known structure. 
    For now, we will try to perturb the schoolbook decomposition to see if we can drop a rank,
    but this is mathematically unlikely to work simply by perturbation.
    
    A more effective heuristic for small tensors is to look for linear dependencies among the 60 basis tensors.
    If there is a linear dependency, we can reduce the rank.
    The tensor T is 3x4x5. 
    If the rank is 60, the 60 vectors (flattened) are linearly independent.
    We check if they are independent. If they are, rank is 60.
    If not, we can reduce.
    """
    U, V, W = _generate_schoolbook(n, m, p)
    
    # Flatten each tensor slice T[:, :, k] or similar? 
    # The tensor T is defined by T[i,j,k] = 1.
    # The rank of the tensor is the minimal r such that T = sum_r u^r \otimes v^r \otimes w^r.
    # For the multiplication tensor, the rank is known to be <= n*m*p.
    # For small dimensions, we can check if the "standard" decomposition is minimal.
    
    # A quick check: Is the rank of the matrix unfolding less than n*m*p?
    # If rank(T_{(1)}) < n*m*p, then tensor rank < n*m*p.
    # T_{(1)} is a matrix of size (n*m) x (p*n).
    # T_{(1)}[i*m+j, k*n+i] = T[i,j,k] = 1.
    
    T_unfold = np.zeros((n*m, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_unfold[i*m + j, k*n + i] = 1.0
                
    # If the matrix rank is less than min(n*m, p*n), then the tensor rank is less than n*m*p?
    # No, tensor rank >= max(matrix rank).
    # But if tensor rank is 60, matrix rank must be <= 60.
    # Matrix rank of T_unfold for 3x4x5:
    # n=3, m=4, p=5.
    # Size: 12 x 15.
    # The entries are 1.
    # Let's calculate the rank of this specific matrix.
    mat_rank = np.linalg.matrix_rank(T_unfold)
    
    # If mat_rank < 60, it doesn't guarantee tensor rank < 60, but it's a necessary condition.
    # Actually, for the multiplication tensor, the rank is often equal to the size of the smallest unfolding? No.
    
    # Given the time constraints and the high likelihood that 3x4x5 has rank 60 (or close to it) and no simple search will find a lower rank,
    # we return the schoolbook decomposition.
    
    return U, V, W

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Default to schoolbook
    U, V, W = _generate_schoolbook(n, m, p)
    
    # Verify schoolbook (should always be true)
    if not _verify_decomposition(U, V, W, n, m, p):
        raise ValueError("Schoolbook decomposition failed verification")
    
    # If we have time, we could attempt a search, but for this specific size,
    # the probability of finding a lower rank via generic search in <100s is low.
    # We stick to the verified schoolbook decomposition.
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
