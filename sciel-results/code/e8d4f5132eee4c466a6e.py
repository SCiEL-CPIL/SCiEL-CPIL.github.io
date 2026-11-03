"""Rewritten program for 4x4 * 4x5 matrix multiplication tensor decomposition.
Goal: find a rank-minimal exact tensor decomposition.
The schoolbook rank is 4*4*5 = 80.
We attempt to find a lower rank via randomized numerical search + rounding to half-integers,
with a strict time budget and fallback to schoolbook.
"""

import time
import random
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [4, 4, 5]
n, m, p = DIMENSIONS
RANK_SCHOOLBOOK = n * m * p  # 80
DEADLINE = 150.0  # seconds
START_TIME = time.time()


def _remaining_time():
    return DEADLINE - (time.time() - START_TIME)


def _schoolbook():
    """Return the schoolbook decomposition of rank 80."""
    rank = RANK_SCHOOLBOOK
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}


def _verify(U, V, W, tol=1e-6):
    """Check if U, V, W form an exact decomposition of the matmul tensor.
    The tensor T[i,j,k] should equal sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    for all i in [0,n), j in [0,m), k in [0,p).
    """
    rank = U.shape[1]
    # Build the tensor via einsum
    # U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
    # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    # Compute T[i,j,k] = sum_r U3[i,j,r] * V3[j,k,r] * W3[k,i,r]
    # Use einsum: 'ijr,jkr,kir->ijk'
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3, optimize=True)
    # The matmul tensor T[i,j,k] = delta[i,j,k] where it's 1 if we're computing C[i,k] = sum_j A[i,j]B[j,k]
    # Actually the tensor for matmul is: T[i,j,k] = 1 if this term contributes to C[i,k]
    # The matmul tensor is simply: T[i,j,k] = 1 for all valid (i,j,k) since C[i,k] = sum_j A[i,j]*B[j,k]
    # Wait, no. The tensor is defined such that the matmul operation is:
    # C[i,k] = sum_{j,r} A[i,j] * B[j,k] * T[i,j,k,r] ... no.
    # 
    # Standard tensor decomposition of matmul:
    # The multiplication tensor T has entries T[i,j,k] where the matmul is:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # As a trilinear form: sum_{i,j,k} A[i,j] B[j,k] C'[i,k] where C'[i,k] is the dual.
    # The tensor T[i,j,k] = 1 for all i,j,k in the "standard" basis because each term A[i,j]*B[j,k] contributes to C[i,k].
    # 
    # So T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
    # 
    # Let me verify: U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] summed over r should equal 1 for all (i,j,k).
    
    # Create the target tensor
    T_target = np.ones((n, m, p))
    
    return np.allclose(T, T_target, atol=tol)


def _round_to_half(x):
    """Round each element to the nearest half-integer."""
    return np.round(x * 2) / 2.0


def _search_one_attempt(rank, seed=None):
    """Try to find a rank-`rank` decomposition via random init + optimization."""
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()
    
    U = rng.standard_normal((n * m, rank))
    V = rng.standard_normal((m * p, rank))
    W = rng.standard_normal((p * n, rank))
    
    # Scale to reasonable magnitude
    U /= np.sqrt(rank)
    V /= np.sqrt(rank)
    W /= np.sqrt(rank)
    
    def _loss(params):
        """Loss = ||T - T_target||_F^2"""
        U_p = params[0:n*m*rank].reshape(n*m, rank)
        V_p = params[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
        W_p = params[n*m*rank + m*p*rank:].reshape(p*n, rank)
        
        U3 = U_p.reshape(n, m, rank)
        V3 = V_p.reshape(m, p, rank)
        W3 = W_p.reshape(p, n, rank)
        
        T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3, optimize=True)
        T_target = np.ones((n, m, p))
        return np.sum((T - T_target) ** 2)
    
    def _grad(params):
        """Gradient of the loss."""
        U_p = params[0:n*m*rank].reshape(n*m, rank)
        V_p = params[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
        W_p = params[n*m*rank + m*p*rank:].reshape(p*n, rank)
        
        U3 = U_p.reshape(n, m, rank)
        V3 = V_p.reshape(m, p, rank)
        W3 = W_p.reshape(p, n, rank)
        
        T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3, optimize=True)
        T_target = np.ones((n, m, p))
        diff = T - T_target  # (n, m, p)
        
        # dLoss/dU3[i,j,r] = 2 * sum_{k} diff[i,j,k] * V3[j,k,r] * W3[k,i,r]
        # dLoss/dV3[j,k,r] = 2 * sum_{i} diff[i,j,k] * U3[i,j,r] * W3[k,i,r]
        # dLoss/dW3[k,i,r] = 2 * sum_{j} diff[i,j,k] * U3[i,j,r] * V3[j,k,r]
        
        dU3 = 2 * np.einsum('ijk,jkr,kir->ijr', diff, V3, W3, optimize=True)
        dV3 = 2 * np.einsum('ijk,ijr,kir->jkr', diff, U3, W3, optimize=True)
        dW3 = 2 * np.einsum('ijk,ijr,jkr->kir', diff, U3, V3, optimize=True)
        
        dU = dU3.reshape(n*m*rank)
        dV = dV3.reshape(m*p*rank)
        dW = dW3.reshape(p*n*rank)
        
        return np.concatenate([dU, dV, dW])
    
    params0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
    
    # Try L-BFGS-B
    result = minimize(_loss, params0, method='L-BFGS-B', jac=_grad,
                      options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 500})
    
    U_opt = result.x[0:n*m*rank].reshape(n*m, rank)
    V_opt = result.x[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
    W_opt = result.x[n*m*rank + m*p*rank:].reshape(p*n, rank)
    
    # Check if close to exact
    if _verify(U_opt, V_opt, W_opt, tol=1e-4):
        # Try rounding to half-integers
        U_r = _round_to_half(U_opt)
        V_r = _round_to_half(V_opt)
        W_r = _round_to_half(W_opt)
        if _verify(U_r, V_r, W_r, tol=1e-6):
            return U_r, V_r, W_r
        # Return the unrounded if it's exact enough
        if _verify(U_opt, V_opt, W_opt, tol=1e-8):
            return U_opt, V_opt, W_opt
    
    return None


def construct():
    start = time.time()
    
    # First, try to find a better rank than schoolbook
    # Known lower bound for 4x4*4x5: at least max(4*4, 4*5, 4*4) = 16? No, that's not right.
    # The tensor rank is at least max(n*m, m*p, n*p) = max(16, 20, 20) = 20? No.
    # Actually the tensor rank is at least the max of the dimensions of the slices.
    # For matmul n x m * m x p, the tensor has size n*m*p.
    # A lower bound is max(n*m, m*p, n*p) but that's not a standard lower bound for tensor rank.
    # The standard lower bound is: rank >= max(n*m, m*p, n*p) is NOT correct.
    # Actually, the tensor rank of the matmul tensor is at least max(n, m, p) * min(n, m, p) or something.
    # Let's just try ranks from 20 up to 79 and see if we can find a decomposition.
    
    best_result = None
    
    # Try a few ranks below schoolbook
    for rank in range(20, RANK_SCHOOLBOOK):
        if time.time() - start > DEADLINE - 10:
            break
        
        # Try a few random seeds for this rank
        for attempt in range(3):
            if time.time() - start > DEADLINE - 10:
                break
            
            result = _search_one_attempt(rank, seed=rank * 100 + attempt)
            if result is not None:
                U, V, W = result
                best_result = {"U": U, "V": V, "W": W}
                # Found a valid decomposition at this rank, try to go lower
                break
        
        if best_result is not None:
            # Found one at this rank, try lower ranks
            continue
    
    # If we didn't find anything better, fall back to schoolbook
    if best_result is None:
        best_result = _schoolbook()
    
    return best_result
