"""Search for rank-minimal exact tensor decomposition of 3x3 * 3x6 matrix multiplication."""
import time
import random
import numpy as np

DIMENSIONS = [3, 3, 6]

# Deadline guard: return best result before 150s wall-clock kill.
_START = time.time()
_DEADLINE = 140.0  # safety margin

def _elapsed():
    return time.time() - _START

def _time_left():
    return _DEADLINE - _elapsed()

def _matmul_tensor(n, m, p):
    """Return the matrix multiplication tensor T[i,j,k] = 1 (scalar 1 tensor) as a 3D array.
    Actually the multiplication tensor has entries T[i,j,k] where the output index is (i,k)
    and inputs are (i,j) and (j,k). We represent it as a 3D array T[i, j, k] = 1
    meaning the (i,k) entry of product uses A[i,j]*B[j,k].
    For decomposition: sum_r U[i,j,r]*V[j,k,r]*W[r] where W[r] is a scalar per r? 
    No. The standard form: (AB)[i,k] = sum_j A[i,j] B[j,k].
    Tensor decomposition: (AB)[i,k] = sum_r U[i,j? ...]
    Let me use the standard: T_{i,j,k} = 1 for the multiplication tensor of (n x m)*(m x p) -> (n x p).
    Here n=3, m=3, p=6. T is indexed (i, j, k) with i in [0,n), j in [0,m), k in [0,p).
    T[i,j,k] = 1 (all ones) because (AB)[i,k] = sum_j A[i,j]*B[j,k].
    A rank-r decomposition: T[i,j,k] = sum_{r=0}^{r-1} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    where U has shape (n*m, r), V has shape (m*p, r), W has shape (p*n, r).
    """
    T = np.ones((n, m, p), dtype=float)
    return T

def _decompose_check(U, V, W, n, m, p, tol=1e-9):
    """Check if U (n*m x r), V (m*p x r), W (p*n x r) decompose the multiplication tensor.
    (AB)[i,k] = sum_r U[i*m+j? no...
    Actually: (AB)[i,k] = sum_j A[i,j] B[j,k]
    = sum_r (sum_j U[i*m+j, r] * A[i,j]) * (sum_k' ... ) 
    Let me be precise. We want:
    (AB)[i,k] = sum_r U[i, r] * V[r] * W[k, r] where the "i" part encodes (i,j), "k" part encodes (j,k)?
    No. The factorization is:
    (AB)[i,k] = sum_{j, r} A[i,j] B[j,k] * delta_{i,j,k,r}... 
    The correct form: define for each r, three vectors:
    u_r: length n*m (indexed by (i,j))
    v_r: length m*p (indexed by (j,k))
    w_r: length p*n (indexed by (k,i))
    Then (AB)[i,k] = sum_r u_r[i*m+j] ... no, that's not right either.
    
    The standard tensor decomposition for matrix multiplication:
    T[i, j, k] = sum_r U[i, j, r] * V[j, k, r] * W[k, i, r]
    where U is (n x m x r), V is (m x p x r), W is (p x n x r).
    Flattening: U_flat has shape (n*m, r), V_flat has shape (m*p, r), W_flat has shape (p*n, r).
    And T[i,j,k] = sum_r U_flat[i*m+j, r] * V_flat[j*p+k, r] * W_flat[k*n+i, r].
    """
    r = U.shape[1]
    # Compute the reconstructed tensor
    # T_recon[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # We need T_recon[i,j,k] == 1 for all i,j,k (the multiplication tensor is all ones)
    # Actually no! The multiplication tensor T[i,j,k] = 1 means the coefficient of A[i,j]*B[j,k] in (AB)[i,k] is 1.
    # So yes, T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
    
    # Efficient check using matrix operations
    # For each (i,k), (AB)[i,k] = sum_j A[i,j]*B[j,k]
    # = sum_r (sum_j U[i*m+j, r] * A[i,j]) * (sum_k2 ... ) -- this isn't separable that way.
    
    # Let's just verify by constructing the tensor.
    # T_recon[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # We need this to equal 1 for all i,j,k.
    
    # Build using einsum or loops
    # U has shape (n*m, r), V has shape (m*p, r), W has shape (p*n, r)
    # Reshape U to (n, m, r), V to (m, p, r), W to (p, n, r)
    U3 = U.reshape(n, m, r)
    V3 = V.reshape(m, p, r)
    W3 = W.reshape(p, n, r)
    
    # T_recon[i,j,k] = sum_r U3[i,j,r] * V3[j,k,r] * W3[k,i,r]
    # Use einsum
    T_recon = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    
    # Check against all-ones tensor
    T_target = np.ones((n, m, p), dtype=float)
    return np.allclose(T_recon, T_target, atol=tol, rtol=tol)

def _random_decomposition(n, m, p, rank, seed):
    """Generate a random rank decomposition and try to solve for exact values.
    We pick random U, V (rank columns each), then solve for W.
    """
    rng = np.random.RandomState(seed)
    
    U = rng.randn(n * m, rank)
    V = rng.randn(m * p, rank)
    
    # We need W such that for all (i,j,k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # This is a system of n*m*p equations in p*n*rank unknowns (W entries).
    # But it's not linear in W because W appears multiplied by U*V which depends on r.
    # Actually it IS linear in W: for each (i,j,k), the equation is:
    # sum_r (U[i*m+j, r] * V[j*p+k, r]) * W[k*n+i, r] = 1
    # But W is indexed by (k*n+i, r), so the unknowns are W[a, r] for a in [0, p*n), r in [0, rank).
    # The coefficient of W[a, r] in the (i,j,k) equation is:
    #   U[i*m+j, r] * V[j*p+k, r] if a == k*n+i, else 0
    # So the system is: for each (i,j,k), sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # This couples different (k,i) through the sum over r.
    
    # Let's set up the linear system. Unknowns: W_flat of length p*n*rank.
    # For equation (i,j,k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # The variable W[k*n+i, r] has index (k*n+i)*rank + r in the flat vector.
    
    num_eqs = n * m * p
    num_unknowns = p * n * rank
    
    A_mat = np.zeros((num_eqs, num_unknowns), dtype=float)
    b_vec = np.ones(num_eqs, dtype=float)
    
    eq_idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for r in range(rank):
                    var_idx = (k * n + i) * rank + r
                    A_mat[eq_idx, var_idx] = U[i * m + j, r] * V[j * p + k, r]
                eq_idx += 1
    
    # Solve least squares
    try:
        W_flat, residuals, rank_s, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
        W = W_flat.reshape(p * n, rank)
    except:
        return None, None, None, None
    
    # Check if it's exact
    if _decompose_check(U, V, W, n, m, p, tol=1e-9):
        return U, V, W, rank
    
    return None, None, None, None

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition of rank n*m*p."""
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
    return U, V, W, rank

def construct():
    n, m, p = DIMENSIONS
    
    # Start with schoolbook as fallback
    U_sb, V_sb, W_sb, rank_sb = _schoolbook(n, m, p)
    best_U, best_V, best_W, best_rank = U_sb, V_sb, W_sb, rank_sb
    
    # Try to find lower rank decompositions
    # The theoretical lower bound for 3x3 * 3x6 is at least max(n*m, m*p, p*n) = max(9, 18, 18) = 18
    # Actually the lower bound is max(n*p, n*m, m*p) for the "support" but the true lower bound
    # for tensor rank of matrix multiplication is more subtle. For 3x3*3x3, the rank is at most 23
    # (Strassen-like). For 3x3*3x6, let's try ranks from a reasonable lower bound up.
    
    # Lower bound: the tensor has support size n*m*p = 54, but rank can be less.
    # A simple lower bound: rank >= max(n*m, m*p, p*n) is NOT correct.
    # The correct lower bound: rank >= max(n*p, n*m, m*p) is also not a standard bound.
    # Let's use: rank must be at least the max dimension of the "core" which is tricky.
    # For safety, try ranks from 18 (which is m*p = p*n) upward.
    
    # Actually, let's think: the multiplication tensor for (n x m) * (m x p) has rank at most n*m*p.
    # A known lower bound is max(n*m, m*p, p*n) = 18 in our case.
    # But it could be higher. Let's try from 18 up, but with a time limit.
    
    min_rank_to_try = 18  # m*p = p*n = 18
    
    seed_counter = 0
    max_rank_to_try = best_rank  # don't exceed schoolbook
    
    for rank in range(min_rank_to_try, min_rank_to_try + 10):  # try a few ranks
        if _time_left() < 5:
            break
        
        # Try multiple random seeds for this rank
        max_attempts = 200
        for attempt in range(max_attempts):
            if _time_left() < 5:
                break
            
            seed = seed_counter
            seed_counter += 1
            
            U, V, W, r = _random_decomposition(n, m, p, rank, seed)
            if U is not None and r < best_rank:
                best_U, best_V, best_W, best_rank = U, V, W, r
                # If we found a valid decomposition, we can stop searching this rank
                # and try lower ranks (but we're going upward, so we'd need to go back)
                # Actually we're trying ranks in increasing order, so if we succeed at rank R,
                # we should check if we can do better at lower ranks. But we started at 18.
                # Let's just record and continue to see if we can find even lower.
                # But since we're going from 18 up, the first success is the best for this range.
                break  # found a valid decomposition at this rank, move to next rank? No, we want minimum.
        
        # If we found a valid decomposition at this rank, we can stop since we're going upward
        # and we want the minimum. But wait, we started at the lower bound, so the first success
        # is the minimum in the range we're trying.
        if best_rank == rank:
            break
    
    # Convert to lists for return
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
