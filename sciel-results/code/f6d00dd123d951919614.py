import numpy as np
import time

DIMENSIONS = [5, 5, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 100.0 # 100s limit to be safe
    
    n, m, p = DIMENSIONS
    parts = [2, 2, 1]
    offsets = [0, 2, 4]
    
    # Pre-allocate arrays if we knew the rank, but we can just collect columns
    U_cols = []
    V_cols = []
    W_cols = []
    
    # Strassen factors for 2x2x2
    # Indices u,v,w in 0..1
    # Returns list of (u_vec, v_vec, w_vec) for local 2x2x2
    def strassen_factors():
        factors = []
        # M1: (A00+A11)(B00+B11) -> C00, C11
        u = np.zeros(4); v = np.zeros(4); w = np.zeros(4)
        u[0] = 1; u[3] = 1
        v[0] = 1; v[3] = 1
        w[0] = 1; w[3] = 1
        factors.append((u, v, w))
        
        # M2: (A10+A11)B00 -> C10, -C11
        u = np.zeros(4); v = np.zeros(4); w = np.zeros(4)
        u[2] = 1; u[3] = 1
        v[0] = 1
        w[2] = 1; w[3] = -1
        factors.append((u, v, w))
        
        # M3: A00(B01-B11) -> C01, C11
        u = np.zeros(4); v = np.zeros(4); w = np.zeros(4)
        u[0] = 1
        v[1] = 1; v[3] = -1
        w[1] = 1; w[3] = 1
        factors.append((u, v, w))
        
        # M4: A11(B10-B00) -> C00, C10
        u = np.zeros(4); v = np.zeros(4); w = np.zeros(4)
        u[3] = 1
        v[2] = 1; v[0] = -1
        w[0] = 1; w[2] = 1
        factors.append((u, v, w))
        
        # M5: (A00+A01)B11 -> -C00, C01
        u = np.zeros(4); v = np.zeros(4); w = np.zeros(4)
        u[0] = 1; u[1] = 1
        v[3] = 1
        w[0] = -1; w[1] = 1
        factors.append((u, v, w))
        
        # M6: (A10-A00)(B00+B01) -> C11
        u = np.zeros(4); v = np.zeros(4); w = np.zeros(4)
        u[2] = 1; u[0] = -1
        v[0] = 1; v[1] = 1
        w[3] = 1
        factors.append((u, v, w))
        
        # M7: (A01-A11)(B10+B11) -> C00
        u = np.zeros(4); v = np.zeros(4); w = np.zeros(4)
        u[1] = 1; u[3] = -1
        v[2] = 1; v[3] = 1
        w[0] = 1
        factors.append((u, v, w))
        
        return factors

    strassen = strassen_factors()
    
    for a in range(3):
        for b in range(3):
            for c in range(3):
                if time.time() > deadline:
                    break
                size_a = parts[a]
                size_b = parts[b]
                size_c = parts[c]
                off_a = offsets[a]
                off_b = offsets[b]
                off_c = offsets[c]
                
                if size_a == 2 and size_b == 2 and size_c == 2:
                    # Use Strassen
                    for u_loc, v_loc, w_loc in strassen:
                        u_vec = np.zeros(25)
                        v_vec = np.zeros(25)
                        w_vec = np.zeros(25)
                        
                        # Map local 0,1 to global
                        # Local index l maps to off + l
                        # u_vec index: (off_a + u) * 5 + (off_b + v)
                        # u_loc has 4 entries corresponding to (0,0), (0,1), (1,0), (1,1)
                        # Let's iterate over the 4 local indices
                        for lu in range(2):
                            for lv in range(2):
                                idx_u = lu * 2 + lv
                                val = u_loc[idx_u]
                                if val != 0:
                                    gi = off_a + lu
                                    gj = off_b + lv
                                    u_vec[gi * 5 + gj] = val
                                    
                        for lv in range(2):
                            for lw in range(2):
                                idx_v = lv * 2 + lw
                                val = v_loc[idx_v]
                                if val != 0:
                                    gj = off_b + lv
                                    gk = off_c + lw
                                    v_vec[gj * 5 + gk] = val
                                    
                        for lw in range(2):
                            for lu in range(2):
                                idx_w = lw * 2 + lu
                                val = w_loc[idx_w]
                                if val != 0:
                                    gk = off_c + lw
                                    gi = off_a + lu
                                    w_vec[gk * 5 + gi] = val
                                    
                        U_cols.append(u_vec)
                        V_cols.append(v_vec)
                        W_cols.append(w_vec)
                else:
                    # Schoolbook
                    for u in range(size_a):
                        for v in range(size_b):
                            for w in range(size_c):
                                u_vec = np.zeros(25)
                                v_vec = np.zeros(25)
                                w_vec = np.zeros(25)
                                
                                gi = off_a + u
                                gj = off_b + v
                                gk = off_c + w
                                
                                u_vec[gi * 5 + gj] = 1
                                v_vec[gj * 5 + gk] = 1
                                w_vec[gk * 5 + gi] = 1
                                
                                U_cols.append(u_vec)
                                V_cols.append(v_vec)
                                W_cols.append(w_vec)
                                
    rank = len(U_cols)
    U = np.array(U_cols).T
    V = np.array(V_cols).T
    W = np.array(W_cols).T
    
    # Convert to list of lists as required by interface?
    # "factor matrices of shapes (n*m, rank) ... with integer/half-integer entries"
    # The return type in the seed is {'U': ..., 'V': ..., 'W': ...}
    # The seed returned lists of lists.
    # I will return lists of lists to be safe, or numpy arrays if allowed.
    # The prompt says "factor matrices of shapes...".
    # Usually numpy arrays are accepted.
    # But the seed used lists.
    # Let's check the seed return:
    # return {"U": U, "V": V, "W": W}
    # where U was a list of lists.
    # I will return lists of lists to match the seed interface exactly.
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
