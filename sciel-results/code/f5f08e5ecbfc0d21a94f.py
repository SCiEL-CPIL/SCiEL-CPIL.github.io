import numpy as np

DIMENSIONS = [4, 4, 9]

def _strassen_2x2():
    """Return (U, V, W) lists for Strassen 2x2 decomposition.
    Each is a list of 7 vectors of length 4.
    u index: i*2 + j
    v index: j*2 + k
    w index: k*2 + i
    """
    U_list = []
    V_list = []
    W_list = []
    
    # Term 1: (A11+A22)(B11+B22)
    u = np.array([1, 0, 0, 1], dtype=float)
    v = np.array([1, 0, 0, 1], dtype=float)
    w = np.array([1, 0, 0, 1], dtype=float)
    U_list.append(u.copy()); V_list.append(v.copy()); W_list.append(w.copy())
    
    # Term 2: (A21+A22)B11
    u = np.array([0, 0, 1, 1], dtype=float)
    v = np.array([1, 0, 0, 0], dtype=float)
    w = np.array([0, 0, 1, 0], dtype=float)
    U_list.append(u.copy()); V_list.append(v.copy()); W_list.append(w.copy())
    
    # Term 3: A11(B12-B22)
    u = np.array([1, 0, 0, 0], dtype=float)
    v = np.array([0, 1, 0, -1], dtype=float)
    w = np.array([0, 1, 0, 0], dtype=float)
    U_list.append(u.copy()); V_list.append(v.copy()); W_list.append(w.copy())
    
    # Term 4: A22(B21-B11)
    u = np.array([0, 0, 0, 1], dtype=float)
    v = np.array([0, 0, 1, -1], dtype=float)
    w = np.array([1, 0, 1, 0], dtype=float)
    U_list.append(u.copy()); V_list.append(v.copy()); W_list.append(w.copy())
    
    # Term 5: (A11+A12)B22
    u = np.array([1, 1, 0, 0], dtype=float)
    v = np.array([0, 0, 0, 1], dtype=float)
    w = np.array([-1, 1, 0, 0], dtype=float)
    U_list.append(u.copy()); V_list.append(v.copy()); W_list.append(w.copy())
    
    # Term 6: (A21-A11)(B11+B12)
    u = np.array([0, 0, 1, -1], dtype=float)
    v = np.array([1, 1, 0, 0], dtype=float)
    w = np.array([0, 0, 1, 0], dtype=float)
    U_list.append(u.copy()); V_list.append(v.copy()); W_list.append(w.copy())
    
    # Term 7: (A12-A22)(B21+B22)
    u = np.array([0, 1, 0, -1], dtype=float)
    v = np.array([0, 0, 1, 1], dtype=float)
    w = np.array([0, 0, 0, 1], dtype=float)
    U_list.append(u.copy()); V_list.append(v.copy()); W_list.append(w.copy())
    
    return U_list, V_list, W_list

def _strassen_4x4x4():
    """Return (U, V, W) lists for Strassen 4x4x4 decomposition using two levels of Strassen.
    Each is a list of 49 vectors of length 16.
    u index: i*4 + j
    v index: j*4 + k
    w index: k*4 + i
    """
    S_U, S_V, S_W = _strassen_2x2()
    
    U_list = []
    V_list = []
    W_list = []
    
    for t in range(7):
        u_outer = S_U[t]
        v_outer = S_V[t]
        w_outer = S_W[t]
        
        for t2 in range(7):
            u_inner = S_U[t2]
            v_inner = S_V[t2]
            w_inner = S_W[t2]
            
            u = np.zeros(16, dtype=float)
            v = np.zeros(16, dtype=float)
            w = np.zeros(16, dtype=float)
            
            # u: i*4+j where i = 2*i0+i1, j = 2*j0+j1
            # u_outer[i0*2+j0] * u_inner[i1*2+j1]
            for i0 in range(2):
                for j0 in range(2):
                    for i1 in range(2):
                        for j1 in range(2):
                            i = 2*i0 + i1
                            j = 2*j0 + j1
                            u[i*4+j] += u_outer[i0*2+j0] * u_inner[i1*2+j1]
            
            # v: j*4+k where j = 2*j0+j1, k = 2*k0+k1
            # v_outer[j0*2+k0] * v_inner[j1*2+k1]
            for j0 in range(2):
                for k0 in range(2):
                    for j1 in range(2):
                        for k1 in range(2):
                            j = 2*j0 + j1
                            k = 2*k0 + k1
                            v[j*4+k] += v_outer[j0*2+k0] * v_inner[j1*2+k1]
            
            # w: k*4+i where k = 2*k0+k1, i = 2*i0+i1
            # w_outer[k0*2+i0] * w_inner[k1*2+i1]
            for k0 in range(2):
                for i0 in range(2):
                    for k1 in range(2):
                        for i1 in range(2):
                            k = 2*k0 + k1
                            i = 2*i0 + i1
                            w[k*4+i] += w_outer[k0*2+i0] * w_inner[k1*2+i1]
            
            U_list.append(u)
            V_list.append(v)
            W_list.append(w)
    
    return U_list, V_list, W_list

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 9
    
    U_rows = n * m  # 16
    V_rows = m * p  # 36
    W_rows = p * n  # 36
    
    U_all = []
    V_all = []
    W_all = []
    
    def add_term(u, v, w):
        U_all.append(u)
        V_all.append(v)
        W_all.append(w)
    
    # Get Strassen 4x4x4 decomposition
    S4_U, S4_V, S4_W = _strassen_4x4x4()
    s4_rank = len(S4_U)  # should be 49
    
    # Split p=9 into [4, 4, 1]
    col_groups = [4, 4, 1]
    col_offset = 0
    
    for q in col_groups:
        if q == 4:
            # Use Strassen 4x4x4 for this block
            for t in range(s4_rank):
                u = np.zeros(U_rows, dtype=float)
                v = np.zeros(V_rows, dtype=float)
                w = np.zeros(W_rows, dtype=float)
                
                # u is already in the right format (i*4+j for i,j in 0..3)
                u[:] = S4_U[t]
                
                # v: j*4+k_local -> j*p + (col_offset + k_local)
                for j in range(4):
                    for k_local in range(4):
                        v[j*p + (col_offset + k_local)] = S4_V[t][j*4 + k_local]
                
                # w: k_local*4+i_local -> (col_offset + k_local)*n + i_local
                for k_local in range(4):
                    for i_local in range(4):
                        w[(col_offset + k_local)*n + i_local] = S4_W[t][k_local*4 + i_local]
                
                add_term(u, v, w)
        else:
            # q == 1: 4x4 x 4x1, schoolbook rank 16
            for i in range(n):
                for j in range(m):
                    u = np.zeros(U_rows, dtype=float)
                    v = np.zeros(V_rows, dtype=float)
                    w = np.zeros(W_rows, dtype=float)
                    u[i * m + j] = 1.0
                    v[j * p + col_offset] = 1.0
                    w[col_offset * n + i] = 1.0
                    add_term(u, v, w)
        
        col_offset += q
    
    rank = len(U_all)
    U_arr = np.array(U_all).T
    V_arr = np.array(V_all).T
    W_arr = np.array(W_all).T
    
    U = [[float(x) for x in row] for row in U_arr]
    V = [[float(x) for x in row] for row in V_arr]
    W = [[float(x) for x in row] for row in W_arr]
    
    return {"U": U, "V": V, "W": W}
