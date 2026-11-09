"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [3, 3, 5]
_TIME_LIMIT = 135  # seconds; leaves margin before 150s kill

# EVOLVE-BLOCK-START
def _schoolbook():
    n, m, p = DIMENSIONS
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


def _verify(U, V, W, tol=1e-10):
    """Verify that U, V, W exactly decompose the matmul tensor."""
    n, m, p = DIMENSIONS
    rank = len(U[0]) if U else 0
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)

    # Reshape to (n,m,rank), (m,p,rank), (p,n,rank)
    U3 = Ua.reshape(n, m, rank)
    V3 = Va.reshape(m, p, rank)
    W3 = Wa.reshape(p, n, rank)

    # The matmul tensor: T[i,j,k] = 1 iff the entry (i,k) of (A*B) is computed from a[i,j]*b[j,k]
    # Actually the tensor for matmul is: T[i,j,k] = delta(i,j,k) where it's 1 for each (i,j,k)
    # The decomposition says: sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = 1 for all (i,j,k)
    # Wait, the convention is W[k*n+i, r], so W index is (k, i)
    # The tensor being decomposed is the matmul tensor T[i,j,k] = 1 (for all i in [0,n), j in [0,m), k in [0,p))
    # Actually no. The matmul tensor has T[i,j,k] = 1 for all valid (i,j,k), and the decomposition is:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = T[i,j,k] = 1
    # This is the standard bilinear decomposition.

    # Compute sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # Use einsum: 'ijr,jkr,kir->ijk'
    result = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)

    # The target tensor is all 1s of shape (n, m, p)
    target = np.ones((n, m, p))
    return np.allclose(result, target, atol=tol)


def _try_random_rank(target_rank, time_budget, rng):
    """Try to find a decomposition of target_rank using random search with Newton iterations."""
    n, m, p = DIMENSIONS
    start = time.time()
    
    for _ in range(50):  # up to 50 restarts
        if time.time() - start > time_budget:
            break
        
        # Random initialization
        U = rng.standard_normal((n * m, target_rank))
        V = rng.standard_normal((m * p, target_rank))
        W = rng.standard_normal((p * n, target_rank))
        
        # Newton iterations to satisfy sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = 1
        for it in range(30):
            if time.time() - start > time_budget:
                break
            
            U3 = U.reshape(n, m, target_rank)
            V3 = V.reshape(m, p, target_rank)
            W3 = W.reshape(p, n, target_rank)
            
            # residual: R[i,j,k] = sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] - 1
            prod = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
            R = prod - 1.0
            
            if np.max(np.abs(R)) < 1e-12:
                break
            
            # Newton step: solve for deltaU, deltaV, deltaW
            # Linearize: R ≈ prod + sum_r (dU[i,j,r]*V[j,k,r]*W[k,i,r] + U[i,j,r]*dV[j,k,r]*W[k,i,r] + U[i,j,r]*V[j,k,r]*dW[k,i,r])
            # We do a simplified Gauss-Newton: update one factor at a time
            
            # Update U: for each (i,j), U[i,j,:] *= (1 / sqrt(prod[i,j,:].mean() if >0 else 1))
            # Better: simple multiplicative update
            for i in range(n):
                for j in range(m):
                    # U[i,j,:] should be scaled so that sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] is closer to 1
                    # Compute for this (i,j): sum_r U[i,j,r] * (V[j,:,r] * W[:,i,r])
                    # V[j,:,r] has shape (p, rank), W[:,i,r] has shape (p, rank)
                    # So (V[j,:,r] * W[:,i,r]) has shape (p, rank), sum over k gives shape (rank,)
                    # Actually we need: for each r, V[j,k,r]*W[k,i,r] summed over k
                    vw = np.einsum('kr,kr->r', V3[j, :, :], W3[:, i, :])
                    # Current contribution: U[i,j,:] * vw (element-wise)
                    # We want sum_r U[i,j,r] * vw[r] = 1 (approximately, since other terms also contribute)
                    # Actually the full equation is sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = 1 for ALL k
                    # So this is more complex. Let's just do a simple gradient step.
                    
                    # Gradient of (prod[i,j,:] - 1)^2 with respect to U[i,j,r]:
                    # 2*(prod[i,j,k]-1)*V[j,k,r]*W[k,i,r] summed over k
                    grad = np.einsum('k,kr,kr->r', (prod[i, j, :] - 1), V3[j, :, :], W3[:, i, :])
                    U[i, j, :] -= 0.1 * grad
            
            for j in range(m):
                for k in range(p):
                    # Similar for V[j,k,:]
                    prod_jk = prod[:, j, k]  # shape (n,)
                    # grad wrt V[j,k,r]: sum_i (prod[i,j,k]-1) * U[i,j,r] * W[k,i,r]
                    grad = np.einsum('i,ir,ki->r', (prod[:, j, k] - 1), U3[:, j, :], W3[k, :, :])
                    V[j, k, :] -= 0.1 * grad
            
            for k in range(p):
                for i in range(n):
                    # Similar for W[k,i,:]
                    prod_ik = prod[i, :, k]  # shape (m,)
                    # grad wrt W[k,i,r]: sum_j (prod[i,j,k]-1) * U[i,j,r] * V[j,k,r]
                    grad = np.einsum('j,ir,jk->r', (prod[i, :, k] - 1), U3[i, :, :], V3[:, k, :])
                    W[k, i, :] -= 0.1 * grad
        
        # Check
        if _verify(U.tolist(), V.tolist(), W.tolist()):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return None


def construct():
    start = time.time()
    
    # First, try to find a lower rank decomposition
    best_result = None
    
    # Try ranks from 44 down to some minimum
    # The known lower bound for 3x3x5 matmul rank: 
    # For 3x3x3, rank is 23. For 3x3x5, it should be higher.
    # Schoolbook is 45. Let's try 44, 43, ...
    
    time_per_rank = min(10, (130 - (time.time() - start)) / 10)
    
    for target_rank in range(44, 20, -1):
        if time.time() - start > 130:
            break
        remaining = 130 - (time.time() - start)
        if remaining < 5:
            break
        
        budget = min(time_per_rank, remaining / 5)
        if budget < 1:
            break
        
        rng = random.Random(42 + target_rank)
        np.random.seed(42 + target_rank)
        
        result = _try_random_rank(target_rank, budget, rng)
        if result is not None:
            best_result = result
            break
    
    if best_result is not None:
        return best_result
    
    # Fallback to schoolbook
    return _schoolbook()
# EVOLVE-BLOCK-END

