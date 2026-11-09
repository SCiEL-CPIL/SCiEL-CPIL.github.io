    def get_snapped_solution(U, V, W):
        U_snap = np.round(U * 2) / 2
        V_snap = np.round(V * 2) / 2
        W_snap = np.round(W * 2) / 2
        return U_snap, V_snap, W_snap

    def check_and_snap(U, V, W, r):
        # Check if close to half-integers
        if not (np.allclose(U * 2, np.round(U * 2), atol=1e-4) and
                np.allclose(V * 2, np.round(V * 2), atol=1e-4) and
                np.allclose(W * 2, np.round(W * 2), atol=1e-4)):
            return None
            
        U_snap, V_snap, W_snap = get_snapped_solution(U, V, W)
        
        # Check exact decomposition with snapped values
        T_target = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = i * m + j
                    v = j * p + k
                    w = k * n + i
                    T_target[u, v, w] = 1.0
        
        T_calc = np.einsum('ur,vr,wr->uvw', U_snap, V_snap, W_snap)
        
        if np.allclose(T_calc, T_target, atol=1e-9):
            return {"U": U_snap.tolist(), "V": V_snap.tolist(), "W": W_snap.tolist()}
        else:
            return None
