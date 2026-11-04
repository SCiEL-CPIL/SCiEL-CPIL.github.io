import time
import numpy as np

def construct():
    n, m, p = 3, 4, 4
    dim_u = n * m  # 12
    dim_v = m * p  # 16
    dim_w = p * n  # 12

    start_time = time.time()
    deadline = start_time + 110

    def build_rank44():
        rank = 44
        U = [[0] * rank for _ in range(dim_u)]
        V = [[0] * rank for _ in range(dim_v)]
        W = [[0] * rank for _ in range(dim_w)]

        col = 0
        def add_strassen(X_u, Y_v, Z_w, col):
            # Strassen's 7-term algorithm for 2x2x2
            # X_u = [x0, x1, x2, x3] rows of first 2x2 block
            # Y_v = [y0, y1, y2, y3] cols of second 2x2 block
            # Z_w = [z0, z1, z2, z3] positions in output
            # M1 = (x0+x3)(y0+y3)
            U[X_u[0]][col] += 1; U[X_u[3]][col] += 1
            V[Y_v[0]][col] += 1; V[Y_v[3]][col] += 1
            W[Z_w[0]][col] += 1; W[Z_w[3]][col] += 1
            col += 1
            # M2 = x3(y2-y0)
            U[X_u[3]][col] += 1
            V[Y_v[2]][col] += 1; V[Y_v[0]][col] -= 1
            W[Z_w[0]][col] += 1; W[Z_w[2]][col] += 1
            col += 1
            # M3 = (x2-x0)(y0+y1)
            U[X_u[2]][col] += 1; U[X_u[0]][col] -= 1
            V[Y_v[0]][col] += 1; V[Y_v[1]][col] += 1
            W[Z_w[3]][col] += 1
            col += 1
            # M4 = x0(y3-y1)
            U[X_u[0]][col] += 1
            V[Y_v[3]][col] += 1; V[Y_v[1]][col] -= 1
            W[Z_w[1]][col] += 1; W[Z_w[3]][col] += 1
            col += 1
            # M5 = (x1+x0)(y0+y2)
            U[X_u[1]][col] += 1; U[X_u[0]][col] += 1
            V[Y_v[0]][col] += 1; V[Y_v[2]][col] += 1
            W[Z_w[0]][col] -= 1; W[Z_w[1]][col] += 1
            col += 1
            # M6 = (x3-x1)(y1+y3)
            U[X_u[3]][col] += 1; U[X_u[1]][col] -= 1
            V[Y_v[1]][col] += 1; V[Y_v[3]][col] += 1
            W[Z_w[0]][col] += 1
            col += 1
            # M7 = (x2-x3)(y3-y2)
            U[X_u[2]][col] += 1; U[X_u[3]][col] -= 1
            V[Y_v[3]][col] += 1; V[Y_v[2]][col] -= 1
            W[Z_w[2]][col] += 1; W[Z_w[3]][col] -= 1
            col += 1
            return col

        # Block structure:
        # A is 3x4, B is 4x4, C is 4x3
        # Split A into rows [0:2], [2:3] (sizes 2, 1)
        # Split B into cols [0:2], [2:4] (sizes 2, 2) and rows [0:2], [2:4] (sizes 2, 2)
        # Split C into cols [0:2], [2:3] (sizes 2, 1)
        
        # Top-left 2x2x2: A[0:2,:], B[:,0:2], C[0:2,:]
        # A rows: 0,1 -> indices 0,1 in U
        # B cols: 0,1 -> indices 0,1 in V (for first 2 rows of B)
        # C cols: 0,1 -> indices 0,1 in W
        col = add_strassen([0,1,4,5], [0,1,4,5], [0,1,4,5], col)
        
        # Top-right 2x2x2: A[0:2,:], B[:,2:4], C[0:2,:]
        # A rows: 0,1 -> indices 0,1 in U
        # B cols: 2,3 -> indices 2,3 in V (for first 2 rows of B)
        # C cols: 0,1 -> indices 0,1 in W
        col = add_strassen([0,1,4,5], [2,3,6,7], [0,1,4,5], col)
        
        # Bottom-left 2x2x2: A[2:3,:], B[:,0:2], C[2:3,:]
        # A rows: 2,3 -> indices 2,3 in U (row 2 is index 8, row 3 is index 9)
        # Wait, let me reconsider the indexing.
        # A is 3x4: rows 0,1,2; cols 0,1,2,3
        # U indices: i*m+j where i in [0,2], j in [0,3]
        # So row 0 of A: U[0], U[1], U[2], U[3]
        # Row 1 of A: U[4], U[5], U[6], U[7]
        # Row 2 of A: U[8], U[9], U[10], U[11]
        
        # B is 4x4: rows 0,1,2,3; cols 0,1,2,3
        # V indices: j*p+k where j in [0,3], k in [0,3]
        # Row 0 of B: V[0], V[1], V[2], V[3]
        # Row 1 of B: V[4], V[5], V[6], V[7]
        # Row 2 of B: V[8], V[9], V[10], V[11]
        # Row 3 of B: V[12], V[13], V[14], V[15]
        
        # C is 4x3: rows 0,1,2,3; cols 0,1,2
        # W indices: k*n+i where k in [0,3], i in [0,2]
        # Row 0 of C: W[0], W[1], W[2]
        # Row 1 of C: W[3], W[4], W[5]
        # Row 2 of C: W[6], W[7], W[8]
        # Row 3 of C: W[9], W[10], W[11]
        
        # Let me redo this more carefully.
        # The tensor T[i,j,k] = sum_l A[i,l] * B[l,k] * C[k,i]
        # U has indices (i*m+j), V has indices (j*p+k), W has indices (k*n+i)
        
        # For a 2x2x2 Strassen block, we need:
        # Two rows of A (i values), two cols of B (k values), and two rows of C (k values)
        # But actually, the structure is:
        # A[i,l] for fixed i in {i1,i2}, l in {l1,l2}
        # B[l,k] for fixed l in {l1,l2}, k in {k1,k2}
        # C[k,i] for fixed k in {k1,k2}, i in {i1,i2}
        
        # So for a Strassen block with A rows {i1,i2}, B cols {k1,k2}, C rows {k1,k2}:
        # U indices: i1*m+l1, i1*m+l2, i2*m+l1, i2*m+l2
        # V indices: l1*p+k1, l1*p+k2, l2*p+k1, l2*p+k2
        # W indices: k1*n+i1, k1*n+i2, k2*n+i1, k2*n+i2
        
        # Let me restart with proper indexing.
        rank = 44
        U = [[0] * rank for _ in range(dim_u)]
        V = [[0] * rank for _ in range(dim_v)]
        W = [[0] * rank for _ in range(dim_w)]

        col = 0
        def add_strassen2(i1, i2, k1, k2, col):
            # A rows i1,i2; B cols k1,k2; C rows k1,k2
            # U indices for A[i,l]: i*m+l
            # For 2x2, we need l1,l2 such that B[l,k] connects
            # Actually, l must be the same for A's cols and B's rows
            # Let's say we're using B rows l1,l2
            # U: i1*m+l1, i1*m+l2, i2*m+l1, i2*m+l2
            # V: l1*p+k1, l1*p+k2, l2*p+k1, l2*p+k2
            # W: k1*n+i1, k1*n+i2, k2*n+i1, k2*n+i2
            return col
        
        # This is getting complicated. Let me use a different approach.
        # I'll build the rank-44 decomposition directly.
        
        # Clear and rebuild
        U = [[0] * rank for _ in range(dim_u)]
        V = [[0] * rank for _ in range(dim_v)]
        W = [[0] * rank for _ in range(dim_w)]
        col = 0
        
        # Strassen for (i in {0,1}, k in {0,1}) with l in {0,1}
        # U: 0*4+0=0, 0*4+1=1, 1*4+0=4, 1*4+1=5
        # V: 0*4+0=0, 0*4+1=1, 1*4+0=4, 1*4+1=5
        # W: 0*3+0=0, 0*3+1=1, 1*3+0=3, 1*3+1=4
        Xu = [0, 1, 4, 5]
        Yv = [0, 1, 4, 5]
        Zw = [0, 1, 3, 4]
        col = add_strassen(Xu, Yv, Zw, col)
        
        # Strassen for (i in {0,1}, k in {2,3}) with l in {0,1}
        # U: 0, 1, 4, 5
        # V: 0*4+2=2, 0*4+3=3, 1*4+2=6, 1*4+3=7
        # W: 2*3+0=6, 2*3+1=7, 3*3+0=9, 3*3+1=10
        Xu = [0, 1, 4, 5]
        Yv = [2, 3, 6, 7]
        Zw = [6, 7, 9, 10]
        col = add_strassen(Xu, Yv, Zw, col)
        
        # Strassen for (i in {0,1}, k in {0,1}) with l in {2,3}
        # U: 0*4+2=2, 0*4+3=3, 1*4+2=6, 1*4+3=7
        # V: 2*4+0=8, 2*4+1=9, 3*4+0=12, 3*4+1=13
        # W: 0, 1, 3, 4
        Xu = [2, 3, 6, 7]
        Yv = [8, 9, 12, 13]
        Zw = [0, 1, 3, 4]
        col = add_strassen(Xu, Yv, Zw, col)
        
        # Strassen for (i in {0,1}, k in {2,3}) with l in {2,3}
        # U: 2, 3, 6, 7
        # V: 2*4+2=10, 2*4+3=11, 3*4+2=14, 3*4+3=15
        # W: 6, 7, 9, 10
        Xu = [2, 3, 6, 7]
        Yv = [10, 11, 14, 15]
        Zw = [6, 7, 9, 10]
        col = add_strassen(Xu, Yv, Zw, col)
        
        # Now for i=2 (the third row of A), we need schoolbook for k in {0,1,2,3}
        # A[2,l] * B[l,k] * C[k,2]
        # U: 2*4+l = 8+l for l in {0,1,2,3}
        # V: l*4+k for l in {0,1,2,3}, k in {0,1,2,3}
        # W: k*3+2 for k in {0,1,2,3}
        for k in range(4):
            for l in range(4):
                U[8+l][col] += 1
                V[l*4+k][col] += 1
                W[k*3+2][col] += 1
                col += 1
        
        return U, V, W

    def verify(U, V, W, n, m, p):
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        U_int = np.round(U_arr * 2).astype(np.int64)
        V_int = np.round(V_arr * 2).astype(np.int64)
        W_int = np.round(W_arr * 2).astype(np.int64)

        T_expected = np.zeros((dim_u, dim_v, dim_w), dtype=np.int64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_expected[i*m+j, j*p+k, k*n+i] = 8

        T_actual = np.einsum('ir,jr,kr->ijk', U_int, V_int, W_int)
        return np.array_equal(T_actual, T_expected)

    def build_schoolbook():
        rank = 48
        U = [[0] * rank for _ in range(dim_u)]
        V = [[0] * rank for _ in range(dim_v)]
        W = [[0] * rank for _ in range(dim_w)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return U, V, W

    # Build the rank-44 decomposition
    U, V, W = build_rank44()
    
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)

    # Verify the base decomposition is correct
    if not verify(U, V, W, n, m, p):
        U, V, W = build_schoolbook()
        return {"U": U, "V": V, "W": W}

    current_U = U_np.copy()
    current_V = V_np.copy()
    current_W = W_np.copy()
    current_rank = 44

    # Precompute expected tensor
    T_expected = np.zeros((dim_u, dim_v, dim_w), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_expected[i*m+j, j*p+k, k*n+i] = 8

    def verify_against_expected(U_arr, V_arr, W_arr):
        U_int = np.round(U_arr * 2).astype(np.int64)
        V_int = np.round(V_arr * 2).astype(np.int64)
        W_int = np.round(W_arr * 2).astype(np.int64)
        T_actual = np.einsum('ir,jr,kr->ijk', U_int, V_int, W_int)
        return np.array_equal(T_actual, T_expected)

    # Phase 1: Try removing single columns
    for attempt in range(80):
        if time.time() > deadline:
            break
        if current_rank <= 1:
            break

        found = False
        for r_idx in range(current_rank):
            if time.time() > deadline:
                break
            
            keep_indices = [i for i in range(current_rank) if i != r_idx]
            
            U_test = current_U[:, keep_indices]
            V_test = current_V[:, keep_indices]
            W_test = current_W[:, keep_indices]
            
            if verify_against_expected(U_test, V_test, W_test):
                current_U = U_test
                current_V = V_test
                current_W = W_test
                current_rank -= 1
                found = True
                break
        
        if not found:
            break

    # Phase 2: Try removing pairs of columns
    pair_deadline = min(deadline, time.time() + 40)
    if current_rank > 1:
        found_pair = False
        for r1 in range(current_rank):
            if time.time() > pair_deadline:
                break
            for r2 in range(r1+1, current_rank):
                if time.time() > pair_deadline:
                    break
                
                keep_indices = [i for i in range(current_rank) if i != r1 and i != r2]
                
                U_test = current_U[:, keep_indices]
                V_test = current_V[:, keep_indices]
                W_test = current_W[:, keep_indices]
                
                if verify_against_expected(U_test, V_test, W_test):
                    current_U = U_test
                    current_V = V_test
                    current_W = W_test
                    current_rank -= 2
                    found_pair = True
                    break
            if found_pair:
                break
        if found_pair:
            for attempt in range(40):
                if time.time() > deadline:
                    break
                if current_rank <= 1:
                    break
                found = False
                for r_idx in range(current_rank):
                    if time.time() > deadline:
                        break
                    keep_indices = [i for i in range(current_rank) if i != r_idx]
                    U_test = current_U[:, keep_indices]
                    V_test = current_V[:, keep_indices]
                    W_test = current_W[:, keep_indices]
                    if verify_against_expected(U_test, V_test, W_test):
                        current_U = U_test
                        current_V = V_test
                        current_W = W_test
                        current_rank -= 1
                        found = True
                        break
                if not found:
                    break

    # Final verification
    if current_rank < 44:
        if verify_against_expected(current_U, current_V, current_W):
            return {"U": current_U.tolist(), "V": current_V.tolist(), "W": current_W.tolist()}

    return {"U": U, "V": V, "W": W}
