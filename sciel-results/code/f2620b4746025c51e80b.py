def _build_optimized_decomposition():
    n, m, p = 3, 3, 7
    
    U = []
    V = []
    W = []
    
    def make_U(indices):
        u = [0.0] * (n * m)
        for idx, val in indices:
            u[idx] = val
        return u
    
    def make_V(indices):
        v = [0.0] * (m * p)
        for idx, val in indices:
            v[idx] = val
        return v
    
    def make_W(indices):
        w = [0.0] * (p * n)
        for idx, val in indices:
            w[idx] = val
        return w
    
    def u_idx(i, j):
        return i * m + j
    
    def v_idx(j, k):
        return j * p + k
    
    def w_idx(k, i):
        return k * n + i
    
    # Strassen for A[0:2,0:2] @ B[0:2,0:2] -> C[0:2,0:2]
    # Using the same Strassen as in the current code
    ua11 = u_idx(0, 0)
    ua12 = u_idx(0, 1)
    ua21 = u_idx(1, 0)
    ua22 = u_idx(1, 1)
    
    vb11 = v_idx(0, 0)
    vb12 = v_idx(0, 1)
    vb21 = v_idx(1, 0)
    vb22 = v_idx(1, 1)
    
    wc11 = w_idx(0, 0)
    wc12 = w_idx(1, 0)
    wc21 = w_idx(0, 1)
    wc22 = w_idx(1, 1)
    
    # M1 = (a11+a22)(b11+b22) -> C11 += 1, C22 += 1
    U.append(make_U([(ua11, 1), (ua22, 1)]))
    V.append(make_V([(vb11, 1), (vb22, 1)]))
    W.append(make_W([(wc11, 1), (wc22, 1)]))
    
    # M2 = (a21+a22)b11 -> C21 += 1, C22 -= 1
    U.append(make_U([(ua21, 1), (ua22, 1)]))
    V.append(make_V([(vb11, 1)]))
    W.append(make_W([(wc21, 1), (wc22, -1)]))
    
    # M3 = a11(b21-b22) -> C12 += 1, C22 += 1
    U.append(make_U([(ua11, 1)]))
    V.append(make_V([(vb21, 1), (vb22, -1)]))
    W.append(make_W([(wc12, 1), (wc22, 1)]))
    
    # M4 = a22(b12-b11) -> C11 += 1, C21 += 1
    U.append(make_U([(ua22, 1)]))
    V.append(make_V([(vb12, 1), (vb11, -1)]))
    W.append(make_W([(wc11, 1), (wc21, 1)]))
    
    # M5 = (a11+a12)b22 -> C11 -= 1, C12 += 1
    U.append(make_U([(ua11, 1), (ua12, 1)]))
    V.append(make_V([(vb22, 1)]))
    W.append(make_W([(wc11, -1), (wc12, 1)]))
    
    # M6 = (a21-a11)(b11+b12) -> C22 += 1
    U.append(make_U([(ua21, 1), (ua11, -1)]))
    V.append(make_V([(vb11, 1), (vb12, 1)]))
    W.append(make_W([(wc22, 1)]))
    
    # M7 = (a12-a22)(b21+b22) -> C11 += 1, C22 += 1
    U.append(make_U([(ua12, 1), (ua22, -1)]))
    V.append(make_V([(vb21, 1), (vb22, 1)]))
    W.append(make_W([(wc11, 1), (wc22, 1)]))
    
    # C[0:2, 2] = A[0:2,0:2] @ B[0:2,2] + A[0:2,2] @ B[2,2]
    for i in [0, 1]:
        for j in [0, 1]:
            U.append(make_U([(u_idx(i, j), 1)]))
            V.append(make_V([(v_idx(j, 2), 1)]))
            W.append(make_W([(w_idx(2, i), 1)]))
    for i in [0, 1]:
        U.append(make_U([(u_idx(i, 2), 1)]))
        V.append(make_V([(v_idx(2, 2), 1)]))
        W.append(make_W([(w_idx(2, i), 1)]))
    
    # C[2, 0:2] = A[2,0:2] @ B[0:2,0:2] + A[2,2] @ B[2,0:2]
    for j in [0, 1]:
        for k in [0, 1]:
            U.append(make_U([(u_idx(2, j), 1)]))
            V.append(make_V([(v_idx(j, k), 1)]))
            W.append(make_W([(w_idx(k, 2), 1)]))
    for k in [0, 1]:
        U.append(make_U([(u_idx(2, 2), 1)]))
        V.append(make_V([(v_idx(2, k), 1)]))
        W.append(make_W([(w_idx(k, 2), 1)]))
    
    # C[2, 2] = A[2,0:2] @ B[0:2,2] + A[2,2] @ B[2,2]
    for j in [0, 1]:
        U.append(make_U([(u_idx(2, j), 1)]))
        V.append(make_V([(v_idx(j, 2), 1)]))
        W.append(make_W([(w_idx(2, 2), 1)]))
    U.append(make_U([(u_idx(2, 2), 1)]))
    V.append(make_V([(v_idx(2, 2), 1)]))
    W.append(make_W([(w_idx(2, 2), 1)]))
    
    # Columns 3-6: schoolbook
    for j in range(3):
        for k in range(3, 7):
            u_entries = [(u_idx(i, j), 1) for i in range(3)]
            U.append(make_U(u_entries))
            V.append(make_V([(v_idx(j, k), 1)]))
            w_entries = [(w_idx(k, i), 1) for i in range(3)]
            W.append(make_W(w_entries))
    
    return U, V, W
