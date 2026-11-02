import time
import numpy as np
from scipy.optimize import nnls

# Deadline guard: must finish well before 120s; use 110s as internal limit.
_DEADLINE = 110.0
_start_time = time.time()

def _time_left():
    return _DEADLINE - (time.time() - _start_time)

def _verify_236(U, V, W, tol=1e-9):
    """
    Verify that U, V, W decompose the 2x3x6 multiplication tensor.
    U: (6, r), V: (18, r), W: (12, r)
    """
    n, m, p = 2, 3, 6
    r = U.shape[1]
    
    # Reshape to tensor forms
    # U: (n, m, r)
    U_mat = U.reshape(n, m, r)
    # V: (m, p, r)
    V_mat = V.reshape(m, p, r)
    # W: (p, n, r) -> (n, p, r)
    W_raw = W.reshape(p, n, r)
    W_mat = np.transpose(W_raw, (1, 0, 2)) # (n, p, r)
    
    # Compute term[i, j, k] = sum_c U[i, j, c] * V[j, k, c] * W[i, k, c]
    # prod_UV: (n, m, p, r)
    prod_UV = U_mat[:, :, np.newaxis, :] * V_mat[np.newaxis, :, :, :]
    # Multiply by W[i, k, c]
    # W_mat: (n, p, r) -> (n, 1, p, r)
    prod = prod_UV * W_mat[:, np.newaxis, :, :]
    
    result = np.sum(prod, axis=3) # (n, m, p)
    
    # The target tensor for matrix multiplication is 1 for all i,j,k
    if not np.allclose(result, 1.0, atol=tol):
        return False
    return True

def _construct_236_rank_30():
    """
    Construct a rank-30 decomposition for 2x3x6.
    Strategy:
    1. Fix W to a block-diagonal structure that decouples the two 2x3x3 blocks.
       W has shape (12, 30).
       For the first block (k=0..2), we want W[k, i, c] to be non-zero only for c in [0, 14].
       For the second block (k=3..5), we want W[k, i, c] to be non-zero only for c in [15, 29].
       
       Specifically, let's try a simple structure for W:
       For c in [0, 14]:
         W[k*n + i, c] = 1 if k < 3 else 0.
       For c in [15, 29]:
         W[k*n + i, c] = 1 if k >= 3 else 0.
       
       This makes the problem two independent 2x3x3 problems.
       However, we need the decomposition to be exact.
       If W is fixed this way, the equation becomes:
       sum_{c=0}^{14} U[i,j,c] V[j,k,c] = 1 for k<3
       sum_{c=15}^{29} U[i,j,c] V[j,k,c] = 1 for k>=3
       
       This is equivalent to two separate rank-15 decompositions of the 2x3x3 identity tensor.
       It is known that the rank of the 2x3x3 matrix multiplication tensor is at most 15 (schoolbook).
       Can we do better? The border rank is 12, but integer rank might be higher.
       Actually, for 2x3x3, the schoolbook rank is 18.
       Wait, the current program tries to find rank 15 for 2x3x3.
       If we assume rank 15 is possible for 2x3x3 with half-integers, then rank 30 is possible for 2x3x6.
       
       However, we can try to mix the two blocks to potentially lower the rank.
       But for now, let's ensure we can produce a valid rank-30 decomposition by solving for U and V
       given a fixed W.
    """
    n, m, p = 2, 3, 6
    rank = 30
    
    # Initialize W
    W = np.zeros((p * n, rank))
    
    # Block 1: k=0,1,2 use columns 0..14
    # W[k*n + i, c]
    for k in range(3):
        for i in range(n):
            W[k*n + i, 0:15] = 1.0
    
    # Block 2: k=3,4,5 use columns 15..29
    for k in range(3, 6):
        for i in range(n):
            W[k*n + i, 15:30] = 1.0
            
    # Now solve for U and V.
    # The condition is:
    # For each i, j, k:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # This is a system of linear equations in the products U*V? No, it's bilinear.
    # However, if we fix U, we can solve for V linearly.
    # Or, we can use the fact that we just need *a* solution.
    # Let's try to find U and V such that they work.
    # Actually, if we set U and V to be identity-like, it might not work directly because of the W structure.
    
    # Let's try a different approach:
    # We know that for 2x3x3, the schoolbook decomposition has rank 18.
    # Let's construct a rank-18 schoolbook decomposition for 2x3x3 and use it twice.
    # Rank 18 * 2 = 36. That's worse than 30.
    # We need a rank-15 decomposition for 2x3x3.
    
    # Is there a known rank-15 decomposition for 2x3x3?
    # The 2x2x2 has rank 7.
    # The 2x3x3 tensor is "small".
    # Let's try to find a rank-15 decomposition using a linear solver with a fixed W structure that allows mixing.
    
    # Alternative: Use the known rank-15 decomposition if it exists.
    # A paper by Bläser et al. or similar might have it.
    # If not, we can try to solve the bilinear system with random initialization and gradient descent, but that was the failed strategy.
    
    # Let's try a structured search for U and V assuming W is fixed to the block-diagonal form.
    # If we can find U and V of rank 15 (each half) that satisfy the 2x3x3 multiplication, we are done.
    # The 2x3x3 multiplication tensor $T_{233}$ has entries $T_{i,j,k} = 1$.
    # We need $U_{i,j,c} V_{j,k,c} = 1$ for all i,j,k when summed over c.
    # This is equivalent to finding a rank-15 decomposition of the identity matrix in a specific tensor form?
    # No, it's the standard matrix multiplication tensor.
    
    # Let's try to solve for U and V with a random W initialization and gradient descent, but with a tighter constraint:
    # We will fix W to be 0 or 1, and optimize U and V.
    
    # Actually, let's just return a valid rank-36 schoolbook decomposition if we can't find a better one quickly.
    # But the inspiration says rank 34 is possible.
    # Rank 34 = 17 + 17? Or 15 + 19?
    
    # Let's try to find a rank-17 decomposition for 2x3x3.
    # If we can find a rank-17 decomposition for 2x3x3, we can use it twice to get rank 34 for 2x3x6.
    
    def _find_233_decomp(rank):
        """
        Find a rank-r decomposition for 2x3x3.
        Returns U, V, W or None.
        """
        n, m, p = 2, 3, 3
        # We fix W to be random 0/1 to break symmetry, then solve for U, V.
        # Actually, let's use a simple gradient descent on U, V, W but with a strong bias towards integers/half-integers.
        
        best_obj = np.inf
        best_U, best_V, best_W = None, None, None
        
        for restart in range(20):
            if _time_left() < 5.0:
                break
            
            np.random.seed(restart * 100 + rank)
            # Initialize with small random values
            U = np.random.randn(n*m, rank) * 0.1
            V = np.random.randn(m*p, rank) * 0.1
            W = np.random.randn(p*n, rank) * 0.1
            
            # Scale W to be closer to 1
            W /= np.linalg.norm(W, axis=0, keepdims=True)
            W *= np.sqrt(n*m*p) # rough normalization
            
            obj = np.inf
            for step in range(100):
                if _time_left() < 2.0:
                    break
                
                # Compute objective
                U_mat = U.reshape(n, m, rank)
                V_mat = V.reshape(m, p, rank)
                W_raw = W.reshape(p, n, rank)
                W_mat = np.transpose(W_raw, (1, 0, 2))
                
                prod_UV = U_mat[:, :, np.newaxis, :] * V_mat[np.newaxis, :, :, :]
                prod = prod_UV * W_mat[:, np.newaxis, :, :]
                result = np.sum(prod, axis=3)
                
                diff = result - 1.0
                obj = np.sum(diff * diff)
                
                if obj < best_obj:
                    best_obj = obj
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    if best_obj < 1e-8:
                        # Snap to half-integers
                        U_s = np.round(U * 2) / 2
                        V_s = np.round(V * 2) / 2
                        W_s = np.round(W * 2) / 2
                        # Verify
                        if _verify_233(U_s, V_s, W_s, tol=1e-6):
                            return U_s, V_s, W_s
                
                # Gradient computation (simplified)
                # dE/dU[i,j,c] = 2 * sum_k diff[i,j,k] V[j,k,c] W[i,k,c]
                temp_U = diff[:, :, :, np.newaxis] * V_mat[np.newaxis, :, :, :]
                grad_U_mat = np.sum(temp_U * W_mat[:, np.newaxis, :, :], axis=2)
                grad_U = 2.0 * grad_U_mat.reshape(n*m, rank)
                
                temp_V = diff[:, :, :, np.newaxis] * U_mat[:, :, np.newaxis, :]
                grad_V_mat = np.sum(temp_V * W_mat[:, np.newaxis, :, :], axis=0)
                grad_V = 2.0 * grad_V_mat.reshape(m*p, rank)
                
                temp_W = diff[:, :, :, np.newaxis] * U_mat[:, :, np.newaxis, :]
                grad_W_mat = np.sum(temp_W * V_mat[np.newaxis, :, :, :], axis=1)
                grad_W_raw = np.transpose(grad_W_mat, (1, 0, 2))
                grad_W = 2.0 * grad_W_raw.reshape(p*n, rank)
                
                lr = 0.005
                U -= lr * grad_U
                V -= lr * grad_V
                W -= lr * grad_W
                
                # Snap periodically
                if step % 20 == 19:
                    U = np.round(U * 2) / 2
                    V = np.round(V * 2) / 2
                    W = np.round(W * 2) / 2
        
        # Final check
        if best_obj < 1e-6:
            U_s = np.round(best_U * 2) / 2
            V_s = np.round(best_V * 2) / 2
            W_s = np.round(best_W * 2) / 2
            if _verify_233(U_s, V_s, W_s, tol=1e-6):
                return U_s, V_s, W_s
        return None

    # Try to find a rank-15 decomposition
    decomp_15 = _find_233_decomp(15)
    if decomp_15:
        U1, V1, W1 = decomp_15
        # Combine two copies
        U = np.hstack([U1, U1])
        V = np.hstack([V1, V1])
        W = np.hstack([W1, W1])
        # Wait, the V and W indices for the second block need to be shifted in the 2x3x6 context.
        # The combination logic in the current program handles this.
        return _combine_233_to_236(U1, V1, W1, U1, V1, W1)
    
    # Try rank 17
    decomp_17 = _find_233_decomp(17)
    if decomp_17:
        U1, V1, W1 = decomp_17
        return _combine_233_to_236(U1, V1, W1, U1, V1, W1)
        
    # Fallback to schoolbook rank 36
    return _schoolbook_236()

def _schoolbook_236():
    """Return schoolbook rank-36 decomposition for 2x3x6 as lists."""
    n, m, p = 2, 3, 6
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

def _combine_233_to_236(U1, V1, W1, U2, V2, W2):
    """
    Combine two 2x3x3 rank-r decompositions into a 2x3x6 rank-2r decomposition.
    """
    n, m, p = 2, 3, 6
    rank1, rank2 = U1.shape[1], U2.shape[1]
    total_rank = rank1 + rank2
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Fill first half (columns 0..rank1-1)
    for r in range(rank1):
        for i in range(n):
            for j in range(m):
                U[i * m + j][r] = U1[i * m + j][r]
        for j in range(m):
            for k in range(3):
                V[j * p + k][r] = V1[j * 3 + k][r]
        for k in range(3):
            for i in range(n):
                W[k * n + i][r] = W1[k * n + i][r]
    
    # Fill second half (columns rank1..total_rank-1)
    offset = rank1
    for r in range(rank2):
        for i in range(n):
            for j in range(m):
                U[i * m + j][offset + r] = U2[i * m + j][r]
        for j in range(m):
            for k in range(3):
                V[j * p + (k + 3)][offset + r] = V2[j * 3 + k][r]
        for k in range(3):
            for i in range(n):
                W[(k + 3) * n + i][offset + r] = W2[k * n + i][r]
    
    return {"U": U, "V": V, "W": W}

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 2x3x6 matrix multiplication.
    """
    global _start_time
    _start_time = time.time()
    
    result = _construct_236_rank_30()
    
    # Verify the result
    if result is not None:
        U = np.array(result['U'])
        V = np.array(result['V'])
        W = np.array(result['W'])
        if _verify_236(U, V, W, tol=1e-6):
            return result
        else:
            # If verification fails, return schoolbook
            return _schoolbook_236()
    else:
        return _schoolbook_236()
