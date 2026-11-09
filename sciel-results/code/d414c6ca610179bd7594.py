import numpy as np
import time

DIMENSIONS = [3, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    nm, mp, np_ = n*m, m*p, p*n
    
    T = np.zeros((nm, mp, np_))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    start = time.time()
    deadline = start + 130
    
    # Try greedy SVD decomposition
    def greedy_decompose(max_terms=75):
        U_list = []
        V_list = []
        W_list = []
        R = T.copy()
        
        for r in range(max_terms):
            if time.time() > deadline - 15:
                break
            
            # Matricize and SVD
            R_mat = R.reshape(nm, mp * np_)
            try:
                U_svd, s, Vt_svd = np.linalg.svd(R_mat, full_matrices=False)
            except:
                break
            
            if len(s) == 0 or s[0] < 1e-12:
                break
            
            u = U_svd[:, 0]
            vw = Vt_svd[0].reshape(mp, np_)
            
            try:
                U_vw, s_vw, Vt_vw = np.linalg.svd(vw, full_matrices=False)
            except:
                break
            
            if len(s_vw) == 0 or s_vw[0] < 1e-12:
                break
            
            v = U_vw[:, 0]
            w = Vt_vw[0]
            
            U_list.append(u)
            V_list.append(v)
            W_list.append(w)
            
            R -= np.einsum('a,b,c->abc', u, v, w)
            
            if np.max(np.abs(R)) < 1e-12:
                break
        
        if len(U_list) == 0:
            return None
        
        # Check if we got an exact decomposition
        T_check = np.einsum('ar,br,cr->abc', np.array(U_list).T, np.array(V_list).T, np.array(W_list).T)
        if np.max(np.abs(T_check - T)) > 1e-10:
            return None
        
        # Try to round to half-integers
        U_arr = np.array(U_list).T
        V_arr = np.array(V_list).T
        W_arr = np.array(W_list).T
        
        U_r = np.round(U_arr * 2) / 2
        V_r = np.round(V_arr * 2) / 2
        W_r = np.round(W_arr * 2) / 2
        
        T_check2 = np.einsum('ar,br,cr->abc', U_r, V_r, W_r)
        if np.max(np.abs(T_check2 - T)) < 1e-9:
            return U_r, V_r, W_r
        
        # Try integer rounding
        U_i = np.round(U_arr)
        V_i = np.round(V_arr)
        W_i = np.round(W_arr)
        T_check3 = np.einsum('ar,br,cr->abc', U_i, V_i, W_i)
        if np.max(np.abs(T_check3 - T)) < 1e-9:
            return U_i.astype(float), V_i.astype(float), W_i.astype(float)
        
        return None
    
    result = greedy_decompose()
    if result is not None:
        U, V, W = result
        return {
            "U": [[float(x) for x in row] for row in U],
            "V": [[float(x) for x in row] for row in V],
            "W": [[float(x) for x in row] for row in W]
        }
    
    # Fallback: schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(np_)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
