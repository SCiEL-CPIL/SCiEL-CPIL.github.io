import numpy as np
from scipy.optimize import minimize
import time

DIMENSIONS = [3, 4, 5]

def _build_target_tensor():
    """Build the matrix multiplication tensor Y[a, b, c] where a=i*m+j, b=j*p+k, c=k*n+i."""
    n, m, p = DIMENSIONS
    Y = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                Y[a, b, c] = 1.0
    return Y

def _verify_decomposition(U, V, W, Y, tol=1e-9):
    """Verify that the decomposition exactly matches the target tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    T = np.einsum('ar,br,cr->abc', U, V, W)
    return np.max(np.abs(T - Y)) < tol

def _schoolbook():
    n, m, p = DIMENSIONS
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

def _get_strassen_222_vectors():
    """
    Returns U, V, W for 2x2x2 matrix multiplication tensor decomposition of rank 7.
    Uses a small optimization to find integer solutions.
    """
    n2, m2, p2 = 2, 2, 2
    Y2 = np.zeros((n2*m2, m2*p2, p2*n2))
    for i in range(n2):
        for j in range(m2):
            for k in range(p2):
                a = i*m2 + j
                b = j*p2 + k
                c = k*n2 + i
                Y2[a, b, c] = 1.0
        
    rank = 7
    best_result = None
    
    for restart in range(5):
        np.random.seed(123 + restart)
        U = np.random.randn(4, rank)
        V = np.random.randn(4, rank)
        W = np.random.randn(4, rank)
        
        def obj(params):
            Uf = params[:28].reshape(4,7)
            Vf = params[28:56].reshape(4,7)
            Wf = params[56:].reshape(4,7)
            T = np.einsum('ar,br,cr->abc', Uf, Vf, Wf)
            return np.sum((T-Y2)**2)
        
        x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        try:
            res = minimize(obj, x0, method='L-BFGS-B', options={'maxiter': 50, 'maxfun': 50})
            if res.fun < 1e-10:
                Uf = res.x[:28].reshape(4,7)
                Vf = res.x[28:56].reshape(4,7)
                Wf = res.x[56:].reshape(4,7)
                
                Ui = np.round(Uf)
                Vi = np.round(Vf)
                Wi = np.round(Wf)
                
                T = np.einsum('ar,br,cr->abc', Ui, Vi, Wi)
                if np.allclose(T, Y2):
                    return Ui, Vi, Wi
        except Exception:
            pass
            
    return None

def construct():
    n, m, p = DIMENSIONS
    Y = _build_target_tensor()
    
    U2, V2, W2 = _get_strassen_222_vectors()
    
    if U2 is None:
        return _schoolbook()
        
    U_full = np.zeros((12, 58))
    V_full = np.zeros((20, 58))
    W_full = np.zeros((15, 58))
    
    rank_offset = 0
    
    # Block 1: i in {0,1}, j in {0,1}, k in {0,1}
    u_map = [0, 1, 4, 5]
    v_map = [0, 1, 5, 6]
    w_map = [0, 1, 3, 4]
    
    for local_idx, global_idx in enumerate(u_map):
        U_full[global_idx, rank_offset:rank_offset+7] = U2[local_idx, :]
    for local_idx, global_idx in enumerate(v_map):
        V_full[global_idx, rank_offset:rank_offset+7] = V2[local_idx, :]
    for local_idx, global_idx in enumerate(w_map):
        W_full[global_idx, rank_offset:rank_offset+7] = W2[local_idx, :]
        
    rank_offset += 7
    
    # Block 2: i in {0,1}, j in {2,3}, k in {0,1}
    u_map2 = [2, 3, 6, 7]
    v_map2 = [10, 11, 15, 16]
    w_map2 = [0, 1, 3, 4]
    
    for local_idx, global_idx in enumerate(u_map2):
        U_full[global_idx, rank_offset:rank_offset+7] = U2[local_idx, :]
    for local_idx, global_idx in enumerate(v_map2):
        V_full[global_idx, rank_offset:rank_offset+7] = V2[local_idx, :]
    for local_idx, global_idx in enumerate(w_map2):
        W_full[global_idx, rank_offset:rank_offset+7] = W2[local_idx, :]
        
    rank_offset += 7
    
    # Schoolbook for the rest
    for i in range(3):
        for j in range(4):
            for k in range(5):
                if i < 2 and k < 2:
                    continue
                
                a = i * 4 + j
                b = j * 5 + k
                c = k * 3 + i
                
                U_full[a, rank_offset] = 1
                V_full[b, rank_offset] = 1
                W_full[c, rank_offset] = 1
                rank_offset += 1
                
    if _verify_decomposition(U_full, V_full, W_full, Y):
        return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
    else:
        return _schoolbook()
