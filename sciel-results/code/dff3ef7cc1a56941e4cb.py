"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from itertools import product

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def _schoolbook():
    """Return the schoolbook (rank n*m*p) decomposition."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}


def _verify(U_list, V_list, W_list, n=2, m=3, p=6, tol=1e-6):
    """Verify that the decomposition exactly reconstructs the multiplication tensor."""
    U = np.array(U_list, dtype=float)
    V = np.array(V_list, dtype=float)
    W = np.array(W_list, dtype=float)
    rank = U.shape[1]
    # Build the tensor T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    # a in [0, n*m), b in [0, m*p), c in [0, p*n)
    # The multiplication tensor has T[a,b,c] = 1 iff:
    #   a = i*m + j, b = j*p + k, c = k*n + i for some i in [0,n), j in [0,m), k in [0,p)
    # Otherwise T[a,b,c] = 0.
    T = np.einsum('ar,br,cr->abc', U, V, W)
    # Build expected tensor
    T_exp = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_exp[i*m+j, j*p+k, k*n+i] = 1.0
    return np.allclose(T, T_exp, atol=tol, rtol=0)


def _try_als_search(max_rank=35, max_iters=200, seed=42, time_limit=120):
    """
    Try to find a rank-<schoolbook decomposition via ALS (alternating least squares).
    We attempt ranks from 18 (lower bound) up to max_rank.
    """
    n, m, p = DIMENSIONS
    rng = np.random.RandomState(seed)
    best_result = None
    best_rank = n * m * p  # schoolbook rank as fallback
    
    # Precompute the tensor for verification
    T_exp = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_exp[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Try ranks from lower bound up
    for target_rank in range(18, min(max_rank + 1, best_rank)):
        U = None
        V = None
        W = None
        success = False
        
        # Multiple random restarts for each rank
        n_restarts = 5
        for restart in range(n_restarts):
            U = rng.randn(n*m, target_rank)
            V = rng.randn(m*p, target_rank)
            W = rng.randn(p*n, target_rank)
            # Normalize columns to prevent blow-up
            for r in range(target_rank):
                norm = max(np.linalg.norm(U[:, r]), np.linalg.norm(V[:, r]), np.linalg.norm(W[:, r]), 1e-10)
                U[:, r] /= norm
                V[:, r] /= norm
                W[:, r] /= norm
            
            for it in range(max_iters):
                # Update U: for each a, solve min ||sum_r U[a,r]*V_r*W_r - T[a,:,:]||
                # T[a, b, c] = sum_r U[a,r] * V[b,r] * W[c,r]
                # For fixed V, W: U[a,:] solves (V^T W) as Gram-like
                # Actually: T[a, :, :] = sum_r U[a,r] * outer(V[:,r], W[:,r])
                # So T[a, :, :] (flattened) = U[a,:] @ (V ⊙ W) where (V⊙W)[b,c,r] = V[b,r]*W[c,r]
                # Let M[b,c,r] = V[b,r]*W[c,r], shape (m*p, p*n, target_rank)
                # Then T[a, :, :] flattened = U[a, :] @ M_flat where M_flat has shape (target_rank, m*p*p*n)
                # More efficiently:
                # T[a, :, :] = sum_r U[a,r] * np.outer(V[:,r], W[:,r])
                # Vectorize: let VW[r, b, c] = V[b,r]*W[c,r], shape (target_rank, m*p, p*n)
                # T[a, :, :] = (U[a, :] @ VW).reshape(m*p, p*n)
                # So we solve: U[a, :] @ VW = T[a, :, :].reshape(-1)
                # where VW is (target_rank, m*p*p*n)
                
                VW = np.einsum('br,cr->bcr', V, W).transpose(2, 0, 1)  # (p*n, target_rank, m*p) -> need (target_rank, m*p*p*n)
                # Actually let's do it differently.
                # T[a, b, c] = sum_r U[a,r] * V[b,r] * W[c,r]
                # For fixed b,c: T[:, b, c] = U @ (V[b,:] * W[c,:])
                # For fixed a: T[a, :, :] = sum_r U[a,r] * V[:,r] * W[:,r]
                
                # Solve for U:
                # T[a, :, :] = sum_r U[a,r] * outer(V[:,r], W[:,r])
                # Flatten: T[a].flatten() = U[a,:] @ (V*W) where (V*W)[r, b*p*n+c] = V[b,r]*W[c,r]
                VW_matrix = np.einsum('br,cr->rbc', V, W).reshape(target_rank, m*p*p*n)
                try:
                    U = np.linalg.lstsq(VW_matrix.T, T_exp.reshape(n*m, -1).T, rcond=None)[0].T
                except:
                    break
                
                # Solve for V:
                # T[a, b, c] = sum_r V[b,r] * U[a,r] * W[c,r]
                # For fixed a: T[a, :, c] = sum_r V[:,r] * U[a,r] * W[c,r]
                # T[a, :, :].flatten() = V @ (U*W) where (U*W)[r, a*(p*n)+c] = U[a,r]*W[c,r]
                UW_matrix = np.einsum('ar,cr->rac', U, W).reshape(target_rank, n*m*p*n)
                try:
                    V = np.linalg.lstsq(UW_matrix.T, T_exp.reshape(m*p, -1).T, rcond=None)[0].T
                except:
                    break
                
                # Solve for W:
                # T[a, b, c] = sum_r W[c,r] * U[a,r] * V[b,r]
                # For fixed b: T[:, b, c] = sum_r W[:,r] * U[:,r] * V[b,r]
                # T[:, b, :].flatten() = W @ (U*V) where (U*V)[r, a*m*p+b] ... 
                # Actually T[a, b, c] = sum_r W[c,r] * (U[a,r] * V[b,r])
                # T.reshape(-1, p*n) with rows indexed by (a,b): T[(a,b), c] = sum_r W[c,r] * (U*V)[r, a*(m*p)+b]
                UV_matrix = np.einsum('ar,br->rab', U, V).reshape(target_rank, n*m*m*p)
                try:
                    W = np.linalg.lstsq(UV_matrix.T, T_exp.reshape(-1, p*n).T, rcond=None)[0].T
                except:
                    break
                
                # Check convergence
                T_rec = np.einsum('ar,br,cr->abc', U, V, W)
                err = np.linalg.norm(T_rec - T_exp)
                if err < 1e-5:
                    success = True
                    break
            
            if success:
                # Convert to list
                best_result = {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
                best_rank = target_rank
                break
        
        if success:
            break
    
    if best_result is not None:
        return best_result
    return None


def _try_strassen_based():
    """
    Attempt a Strassen-inspired block decomposition.
    Split the 2x3 matrix A into [A1; A2] where A1 is 1x3, A2 is 1x3.
    Split the 3x6 matrix B into [B1 | B2] where B1 is 3x3, B2 is 3x3.
    Then C = A @ B = [A1@B1 + A2@B1 | A1@B2 + A2@B2]... 
    Actually C is 2x6: C[i, k] = sum_j A[i,j] * B[j,k]
    
    Let's try a different decomposition. The key idea from the strategy:
    Treat 3x6 as two 3x3 blocks, so we have two 2x3x3 multiplications combined.
    R(2,3,3) for 2x3x3: schoolbook is 18, but we can do better.
    Actually for 2x3x3, the rank is known to be at most... let's just use schoolbook for sub-problems
    but combine them cleverly.
    
    Alternative: use the fact that we're multiplying (2x3) @ (3x6) = (2x6).
    We can write this as a composition: first multiply 2x3 by 3x3 (giving 2x3), then by 3x2 (giving 2x6)?
    No, that changes the problem.
    
    Let's try a direct structured approach:
    C[i,k] = sum_j A[i,j] * B[j,k] for i in {0,1}, j in {0,1,2}, k in {0..5}
    
    We can use the following trick: since n=2 is small, we can exploit linearity in the 2 rows.
    For each pair of columns (k1, k2) in B, we have:
    C[:, k1] = A @ B[:, k1]
    C[:, k2] = A @ B[:, k2]
    
    This doesn't immediately help reduce rank.
    
    Let's try a different approach: use the known rank of 2x3x3.
    If R(2,3,3) <= 9 (using some method), then we could potentially do 2 * 9 = 18? No, that's the lower bound.
    
    Actually, let's just return None and rely on schoolbook or ALS.
    """
    return None


def construct():
    start_time = time.time()
    deadline = 120.0  # evaluator kills at 150s, keep margin
    
    # Try Strassen-based structured decomposition
    result = _try_strassen_based()
    if result is not None:
        if _verify(result['U'], result['V'], result['W']):
            return result
    
    # Try ALS search for lower rank
    elapsed = time.time() - start_time
    if elapsed < 100:  # leave margin
        time_limit = min(100 - elapsed, 120 - elapsed - 10)
        if time_limit > 5:
            result = _try_als_search(max_rank=35, max_iters=50, seed=42, time_limit=time_limit)
            if result is not None:
                if _verify(result['U'], result['V'], result['W']):
                    return result
    
    # Fallback: schoolbook
    return _schoolbook()
# EVOLVE-BLOCK-END
