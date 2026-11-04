"""Tensor rank minimization for 2x3 * 3x6 matrix multiplication."""
import time
import random
import numpy as np

DIMENSIONS = [2, 3, 6]
_START_TIME = time.time()
_DEADLINE = 110.0  # safety margin before 120s kill

def _make_tensor(n, m, p):
    """Build the matrix multiplication tensor T[i,j,k] for A[i,j]*B[j,k]->C[i,k]."""
    T = np.zeros((n, m, p), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1.0
    return T

def _tensor_to_flat(T):
    n, m, p = T.shape
    return T.reshape(n * m * p)

def _decompose_to_UVW(flat, n, m, p, rank):
    """
    Given a flat vector of length n*m*p*rank (rank factorizations stacked),
    unpack into U (n*m x rank), V (m*p x rank), W (p*n x rank).
    
    Convention: T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    We store flat as concatenation over c: flat[c*n*m*p : (c+1)*n*m*p]
    where within each c, the order is i, j, k (i major, then j, then k).
    
    So flat[c*n*m*p + i*m*p + j*p + k] = U[i,j,c] * V[j,k,c] * W[k,i,c]
    """
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    for c in range(rank):
        base = c * n * m * p
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = flat[base + i * m * p + j * p + k]
                    # We need U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = val
                    # Strategy: set U=1, V=val^(1/2), W=val^(1/2) if val>=0, etc.
                    # But that doesn't work generally. Instead, use a fixed scheme:
                    # U[i,j,c] = 1, V[j,k,c] = val, W[k,i,c] = 1 if we want exact product.
                    # But this only works if we can factor val as a product of three terms.
                    # Simpler: U[i,j,c]=1, V[j,k,c]=val, W[k,i,c]=1 gives product val.
                    U[i * m + j, c] = 1.0
                    V[j * p + k, c] = val
                    W[k * n + i, c] = 1.0
    return U, V, W

def _verify_rank(flat, n, m, p, rank, tol=1e-6):
    """Check that the decomposition of given rank exactly reconstructs T."""
    U, V, W = _decompose_to_UVW(flat, n, m, p, rank)
    T = _make_tensor(n, m, p)
    # Reconstruct
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                if abs(val - T[i, j, k]) > tol:
                    return False
    return True

def _random_rank_attempt(n, m, p, rank, n_iter=200, lr_base=0.01):
    """
    Try to find a rank-`rank` decomposition via random gradient descent on a 
    smooth surrogate. We parameterize the tensor flat vector and try to make it
    factorable as sum of rank-1 tensors.
    
    We use the fact that a vector y is a sum of `rank` rank-1 tensors iff the 
    tensor Y (reshaped) has tensor rank <= rank. We minimize the difference 
    between Y and the nearest tensor of tensor-rank <= rank using ALS.
    """
    T = _make_tensor(n, m, p)
    T_flat = _tensor_to_flat(T)
    
    # Initialize U, V, W randomly
    np.random.seed(None)
    best_flat = None
    
    for trial in range(3):
        U = np.random.randn(n * m, rank)
        V = np.random.randn(m * p, rank)
        W = np.random.randn(p * n, rank)
        
        # ALS: iteratively fix two, solve for the third
        for it in range(n_iter):
            if time.time() - _START_TIME > _DEADLINE - 10:
                break
            
            # Update U: for each c, U[:,c] is fixed, V[:,c] and W[:,c] fixed
            # T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
            # For fixed c, the contribution to T is U[:,c] * outer(V[:,c], W[:,c]) reshaped
            # This is complex for ALS. Instead, let's do a simpler approach:
            # Minimize ||flatten(T) - sum_c vec(U_c * V_c * W_c)||^2 via gradient descent on flat params.
            
            # Build current estimate
            est = np.zeros(n * m * p)
            for c in range(rank):
                for i in range(n):
                    for j in range(m):
                        uval = U[i * m + j, c]
                        for k in range(p):
                            est[i * m * p + j * p + k] += uval * V[j * p + k, c] * W[k * n + i, c]
            
            # Compute gradient w.r.t. U, V, W
            diff = est - T_flat
            # Grad_U[i,j,c] = V[j,k,c]*W[k,i,c] summed over k... this is expensive
            # For small dims, just do it
            dU = np.zeros((n * m, rank))
            dV = np.zeros((m * p, rank))
            dW = np.zeros((p * n, rank))
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        idx = i * m * p + j * p + k
                        dval = diff[idx]
                        for c in range(rank):
                            uval = U[i * m + j, c]
                            vval = V[j * p + k, c]
                            wval = W[k * n + i, c]
                            dU[i * m + j, c] += dval * vval * wval
                            dV[j * p + k, c] += dval * uval * wval
                            dW[k * n + i, c] += dval * uval * vval
            
            lr = lr_base / (1 + it * 0.01)
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
            
            # Check convergence
            if it % 10 == 0:
                est2 = np.zeros(n * m * p)
                for c in range(rank):
                    for i in range(n):
                        for j in range(m):
                            uval = U[i * m + j, c]
                            for k in range(p):
                                est2[i * m * p + j * p + k] += uval * V[j * p + k, c] * W[k * n + i, c]
                err = np.linalg.norm(est2 - T_flat)
                if err < 1e-4:
                    # Build flat representation
                    flat = np.zeros(n * m * p * rank)
                    for c in range(rank):
                        for i in range(n):
                            for j in range(m):
                                for k in range(p):
                                    flat[c * n * m * p + i * m * p + j * p + k] = \
                                        U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                    if best_flat is None or np.linalg.norm(flat - T_flat) < 1e-5:
                        best_flat = flat
                        if np.linalg.norm(flat - T_flat) < 1e-5:
                            return flat
        
        if best_flat is not None and np.linalg.norm(best_flat - T_flat) < 1e-4:
            return best_flat
    
    return best_flat

def _strassen_style_split(n, m, p):
    """
    Try a Strassen-like recursive split. For m=3, split into m1=1, m2=2.
    A = A1 | A2 where A1 is n x 1, A2 is n x 2
    B = B1; B2 where B1 is 1 x p, B2 is 2 x p
    C = A1*B1 + A2*B2
    
    We can use Strassen's idea to reduce rank if the sub-problems allow.
    For A1*B1: rank n*1*p = n*p
    For A2*B2: rank n*2*p = 2*n*p
    Total naive: 3*n*p
    
    But if we can combine, maybe reduce. Let's try to find explicit low-rank decompositions.
    """
    T = _make_tensor(n, m, p)
    T_flat = _tensor_to_flat(T)
    
    # Try ranks from known lower bound up to n*m*p
    # Theoretical lower bound for 2x3x6 is at least max(2*3, 3*6, 2*6) = 18? 
    # Actually lower bound is max(n*m, m*p, p*n) = max(6, 18, 12) = 18
    # Known: rank(2x3x6) is known to be 16 or less? Let's search.
    
    for rank in range(18, n * m * p + 1):
        if time.time() - _START_TIME > _DEADLINE - 20:
            break
        result = _random_rank_attempt(n, m, p, rank, n_iter=100)
        if result is not None and _verify_rank(result, n, m, p, rank):
            U, V, W = _decompose_to_UVW(result, n, m, p, rank)
            # Convert to lists with rounded values for cleanliness
            U_list = [[float(U[i][j]) for j in range(rank)] for i in range(n * m)]
            V_list = [[float(V[i][j]) for j in range(rank)] for i in range(m * p)]
            W_list = [[float(W[i][j]) for j in range(range(p * n))] for i in range(p * n)]
            return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback to schoolbook
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

def construct():
    n, m, p = DIMENSIONS
    
    # Try to find a better decomposition
    if time.time() - _START_TIME < _DEADLINE - 30:
        try:
            result = _strassen_style_split(n, m, p)
            # Verify the result
            U = np.array(result["U"])
            V = np.array(result["V"])
            W = np.array(result["W"])
            T = _make_tensor(n, m, p)
            ok = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0.0
                        for c in range(U.shape[1]):
                            val += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                        if abs(val - T[i, j, k]) > 1e-6:
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    break
            if ok:
                return result
        except Exception:
            pass
    
    # Schoolbook fallback
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
