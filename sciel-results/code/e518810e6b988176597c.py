import time
import numpy as np

DIMENSIONS = [3, 3, 5]
_START_TIME = time.time()
_DEADLINE = 150
_SAFETY_MARGIN = 10  # seconds before the kill

def _schoolbook_decomposition():
    """
    Rank-45 exact tensor decomposition for 3x3 * 3x5 matrix multiplication.
    Schoolbook algorithm with the specified index conventions.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p  # 3 * 3 * 5 = 45

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

def _verify_decomposition(U, V, W, n, m, p):
    """
    Verify that U, V, W exactly decompose the matrix multiplication tensor.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    The tensor T[i,j,k] = 1 for all (i,j,k) should equal
    sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    """
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if abs(s - 1) > 1e-9:
                    return False
    return True

def _try_strassen_like_search(n, m, p, max_rank, time_budget):
    """
    Attempt to find a rank < max_rank exact decomposition using a structured
    search inspired by Strassen's algorithm. We parameterize each rank-1 term
    by three vectors (u_r, v_r, w_r) and use iterative refinement.
    
    Returns (U, V, W) if a valid decomposition is found, else None.
    """
    nm = n * m
    mp = m * p
    pn = p * n
    
    best_result = None
    
    # Try various ranks from low to high, up to max_rank-1
    # Theoretical lower bound is roughly n*m*p / (some factor), but for 3x3x3 it is 23.
    # For 3x3x5, let's search from 23 up to 44.
    for rank in range(23, min(max_rank, 45)):
        if time.time() - _START_TIME > _DEADLINE - _SAFETY_MARGIN:
            break
        
        # Try multiple random restarts
        n_restarts = 5
        for restart in range(n_restarts):
            if time.time() - _START_TIME > _DEADLINE - _SAFETY_MARGIN:
                break
            
            np.random.seed(rank * 1000 + restart * 100 + 42)
            
            # Initialize factors randomly
            U = np.random.randn(nm, rank) * 0.5
            V = np.random.randn(mp, rank) * 0.5
            W = np.random.randn(pn, rank) * 0.5
            
            # Reshape for vectorized operations
            # U: (n, m, R)
            # V: (m, p, R)
            # W: (p, n, R)
            U_r = U.reshape(n, m, rank)
            V_r = V.reshape(m, p, rank)
            W_r = W.reshape(p, n, rank)
            
            max_iters = 500
            lr = 0.05
            for it in range(max_iters):
                if time.time() - _START_TIME > _DEADLINE - _SAFETY_MARGIN:
                    break
                
                # Compute Error Tensor E[i,j,k] = 1 - sum_r U[i,j,r] V[j,k,r] W[k,i,r]
                # Use einsum for efficiency
                # term_r = U_r[i,j,r] * V_r[j,k,r] * W_r[k,i,r]
                # Sum over r
                # E = 1 - np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
                E = 1.0 - np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
                
                err_norm = np.linalg.norm(E)
                if err_norm < 1e-10:
                    # Found a valid decomposition!
                    # Round to nearest half-integer and verify
                    U_rounded = np.round(U * 2) / 2
                    V_rounded = np.round(V * 2) / 2
                    W_rounded = np.round(W * 2) / 2
                    
                    U_list = U_rounded.tolist()
                    V_list = V_rounded.tolist()
                    W_list = W_rounded.tolist()
                    
                    if _verify_decomposition(U_list, V_list, W_list, n, m, p):
                        return {"U": U_list, "V": V_list, "W": W_list}
                    else:
                        # Try without rounding (keep as float)
                        U_list = U.tolist()
                        V_list = V.tolist()
                        W_list = W.tolist()
                        if _verify_decomposition(U_list, V_list, W_list, n, m, p):
                            return {"U": U_list, "V": V_list, "W": W_list}
                
                # Gradient steps using einsum
                # dE/dU[i,j,r] = - sum_k E[i,j,k] V[j,k,r] W[k,i,r]
                # We want to minimize sum E^2, so grad U = -2 * sum E * V * W
                # Update: U -= lr * grad_U
                # grad_U = np.einsum('ijk,jkr,kir->ijr', E, V_r, W_r)
                grad_U = np.einsum('ijk,jkr,kir->ijr', E, V_r, W_r)
                U_r -= lr * grad_U
                
                # dE/dV[j,k,r] = - sum_i E[i,j,k] U[i,j,r] W[k,i,r]
                # grad_V = np.einsum('ijk,ijr,kir->jkr', E, U_r, W_r)
                grad_V = np.einsum('ijk,ijr,kir->jkr', E, U_r, W_r)
                V_r -= lr * grad_V
                
                # dE/dW[k,i,r] = - sum_j E[i,j,k] U[i,j,r] V[j,k,r]
                # grad_W = np.einsum('ijk,ijr,jkr->kir', E, U_r, V_r)
                grad_W = np.einsum('ijk,ijr,jkr->kir', E, U_r, V_r)
                W_r -= lr * grad_W
                
                # Decay learning rate
                lr *= 0.999
            
            # After optimization, check final error
            E_final = 1.0 - np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
            if np.linalg.norm(E_final) < 1e-8:
                U_rounded = np.round(U * 2) / 2
                V_rounded = np.round(V * 2) / 2
                W_rounded = np.round(W * 2) / 2
                U_list = U_rounded.tolist()
                V_list = V_rounded.tolist()
                W_list = W_rounded.tolist()
                if _verify_decomposition(U_list, V_list, W_list, n, m, p):
                    return {"U": U_list, "V": V_list, "W": W_list}
                
                U_list = U.tolist()
                V_list = V.tolist()
                W_list = W.tolist()
                if _verify_decomposition(U_list, V_list, W_list, n, m, p):
                    return {"U": U_list, "V": V_list, "W": W_list}
        
        # If we get here, no valid decomposition found for this rank
        continue
    
    return None

def construct():
    """
    Entry point with deadline guard. Attempts to find a lower-rank decomposition
    using a Strassen-like search, falling back to the schoolbook rank-45 decomposition.
    """
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 45
    
    # Check if we have time for a search
    elapsed = time.time() - _START_TIME
    if elapsed > _DEADLINE - _SAFETY_MARGIN:
        # Not enough time, return schoolbook
        return _schoolbook_decomposition()
    
    # Try to find a lower-rank decomposition
    result = _try_strassen_like_search(n, m, p, schoolbook_rank, _DEADLINE - _SAFETY_MARGIN - elapsed)
    
    if result is not None:
        return result
    
    # Fallback to schoolbook
    return _schoolbook_decomposition()
