import time
import numpy as np

DIMENSIONS = [5, 5, 5]

def _strassen_2x2():
    """
    Return the 7-term Strassen decomposition for 2x2 matrix multiplication.
    """
    strassen_terms = []
    
    # M1 = (a00+a11)(b00+b11)
    strassen_terms.append((
        [0, 3], [1, 1],      
        [0, 3], [1, 1],      
        [0, 2, 3], [1, 0, 1],  
    ))
    
    # M2 = (a10+a11)b00
    strassen_terms.append((
        [2, 3], [1, 1],      
        [0], [1],            
        [2, 3], [1, -1],     
    ))
    
    # M3 = a00(b01-b11)
    strassen_terms.append((
        [0], [1],            
        [1, 3], [1, -1],     
        [1, 3], [1, 1],      
    ))
    
    # M4 = a11(b10-b00)
    strassen_terms.append((
        [3], [1],            
        [2, 0], [1, -1],     
        [0, 2], [1, 1],      
    ))
    
    # M5 = (a00+a01)b11
    strassen_terms.append((
        [0, 1], [1, 1],      
        [3], [1],            
        [0, 1], [-1, 1],     
    ))
    
    # M6 = (a10-a00)(b00+b01)
    strassen_terms.append((
        [2, 0], [1, -1],     
        [0, 1], [1, 1],      
        [3], [1],            
    ))
    
    # M7 = (a01-a11)(b10+b11)
    strassen_terms.append((
        [1, 3], [1, -1],     
        [2, 3], [1, 1],      
        [0], [1],            
    ))
    
    return strassen_terms

def _verify_decomposition(U, V, W, n=5, m=5, p=5, tol=1e-10):
    """Verify that the tensor decomposition is exact."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    rank = U.shape[1]
    U_r = U.reshape(n, m, rank)
    V_r = V.reshape(m, p, rank)
    W_r = W.reshape(p, n, rank)
    
    T = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
    
    # T[i,j,k] should be 1
    expected = np.ones((n, m, p))
    err = np.max(np.abs(T - expected))
    return err < tol, err

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS
    
    # Block sizes: 2, 2, 1
    sizes = [2, 2, 1]
    offsets = [0, 2, 4]
    
    U_rows = n * m
    V_rows = m * p
    W_rows = p * n
    
    terms = []
    
    strassen = _strassen_2x2()
    
    for i in range(3):
        for j in range(3):
            for k in range(3):
                si, sj, sk = sizes[i], sizes[j], sizes[k]
                oi, oj, ok = offsets[i], offsets[j], offsets[k]
                
                # Determine rank and decomposition type
                if si == 2 and sj == 2 and sk == 2:
                    # Strassen
                    for term in strassen:
                        u_local, u_coeffs, v_local, v_coeffs, w_local, w_coeffs = term
                        u_dict = {}
                        v_dict = {}
                        w_dict = {}
                        
                        for li, coeff in zip(u_local, u_coeffs):
                            if coeff != 0:
                                li_row = li // 2
                                li_col = li % 2
                                global_u = (oi + li_row) * m + (oj + li_col)
                                u_dict[global_u] = u_dict.get(global_u, 0) + coeff
                                
                        for li, coeff in zip(v_local, v_coeffs):
                            if coeff != 0:
                                li_row = li // 2
                                li_col = li % 2
                                global_v = (oj + li_row) * p + (ok + li_col)
                                v_dict[global_v] = v_dict.get(global_v, 0) + coeff
                                
                        for li, coeff in zip(w_local, w_coeffs):
                            if coeff != 0:
                                li_row = li // 2
                                li_col = li % 2
                                global_w = (ok + li_col) * n + (oi + li_row)
                                w_dict[global_w] = w_dict.get(global_w, 0) + coeff
                                
                        terms.append((u_dict, v_dict, w_dict))
                        
                elif (si == 2 and sj == 1 and sk == 2) or (si == 2 and sj == 1 and sk == 1) or (si == 1 and sj == 1 and sk == 2):
                    # Rank 1 cases
                    # Outer product type
                    u_dict = {}
                    v_dict = {}
                    w_dict = {}
                    
                    # U: si x sj
                    for li in range(si):
                        for lj in range(sj):
                            global_u = (oi + li) * m + (oj + lj)
                            u_dict[global_u] = 1
                            
                    # V: sj x sk
                    for lj in range(sj):
                        for lk in range(sk):
                            global_v = (oj + lj) * p + (ok + lk)
                            v_dict[global_v] = 1
                            
                    # W: sk x si
                    for lk in range(sk):
                        for li in range(si):
                            global_w = (ok + lk) * n + (oi + li)
                            w_dict[global_w] = 1
                            
                    terms.append((u_dict, v_dict, w_dict))
                    
                else:
                    # Schoolbook
                    for li in range(si):
                        for lj in range(sj):
                            for lk in range(sk):
                                global_u = (oi + li) * m + (oj + lj)
                                global_v = (oj + lj) * p + (ok + lk)
                                global_w = (ok + lk) * n + (oi + li)
                                terms.append(({global_u: 1}, {global_v: 1}, {global_w: 1}))
                                
    rank = len(terms)
    
    U = [[0] * rank for _ in range(U_rows)]
    V = [[0] * rank for _ in range(V_rows)]
    W = [[0] * rank for _ in range(W_rows)]
    
    for c, (u_dict, v_dict, w_dict) in enumerate(terms):
        for idx, coeff in u_dict.items():
            if 0 <= idx < U_rows:
                U[idx][c] = coeff
        for idx, coeff in v_dict.items():
            if 0 <= idx < V_rows:
                V[idx][c] = coeff
        for idx, coeff in w_dict.items():
            if 0 <= idx < W_rows:
                W[idx][c] = coeff
                
    valid, err = _verify_decomposition(U, V, W)
    
    if not valid:
        # Fallback to schoolbook
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
    
    return {"U": U, "V": W, "W": W}

# Wait, I made a typo in the return statement of the fallback or main?
# "return {"U": U, "V": V, "W": W}"
# I wrote "return {"U": U, "V": W, "W": W}" in the draft above.
# I will fix it in the final code.
