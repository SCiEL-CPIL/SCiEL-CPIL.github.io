import time
import numpy as np

def construct():
    n, m, p = 3, 4, 4
    dim_u = n * m  # 12
    dim_v = m * p  # 16
    dim_w = p * n  # 12

    start_time = time.time()
    deadline = start_time + 110

    def verify(U, V, W):
        """Exact verification using integer arithmetic (multiply by 2 to handle half-integers)."""
        U_arr = np.array(U)
        V_arr = np.array(V)
        W_arr = np.array(W)
        
        # Multiply by 2 to convert to integers
        U_int = np.round(U_arr * 2).astype(np.int64)
        V_int = np.round(V_arr * 2).astype(np.int64)
        W_int = np.round(W_arr * 2).astype(np.int64)
        
        # Check that rounding was exact
        if not np.allclose(U_arr * 2, U_int, atol=1e-9):
            return False
        if not np.allclose(V_arr * 2, V_int, atol=1e-9):
            return False
        if not np.allclose(W_arr * 2, W_int, atol=1e-9):
            return False
        
        # Build expected tensor: T[i,j,k] = 1 if i in [0,n), j in [0,m), k in [0,p)
        # U index: i*m + j, V index: j*p + k, W index: k*n + i
        T_expected = np.zeros((dim_u, dim_v, dim_w), dtype=np.int64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_expected[i*m+j, j*p+k, k*n+i] = 8  # 2*2*2 = 8 from scaling
        
        T_actual = np.einsum('ir,jr,kr->ijk', U_int, V_int, W_int)
        return np.array_equal(T_actual, T_expected)

    def build_rank44():
        """Build a rank-44 decomposition using Strassen for 2x2x2 blocks."""
        rank = 44
        U = np.zeros((dim_u, rank), dtype=float)
        V = np.zeros((dim_v, rank), dtype=float)
        W = np.zeros((dim_w, rank), dtype=float)

        def add_strassen_block(U_idx, V_idx, W_idx, col):
            """Add 7 Strassen terms for 2x2x2 block.
            U_idx = [u0, u1, u2, u3] for 2x2 block
            V_idx = [v0, v1, v2, v3]
            W_idx = [w0, w1, w2, w3]
            """
            # M1 = (A00+A11)(B00+B11)
            U[U_idx[0]][col] += 1; U[U_idx[3]][col] += 1
            V[V_idx[0]][col] += 1; V[V_idx[3]][col] += 1
            W[W_idx[0]][col] += 1; W[W_idx[3]][col] += 1
            col += 1
            # M2 = A11(B10-B00)
            U[U_idx[3]][col] += 1
            V[V_idx[1]][col] += 1; V[V_idx[0]][col] -= 1
            W[W_idx[2]][col] += 1; W[W_idx[3]][col] += 1
            col += 1
            # M3 = (A00-A11)(B01+B11)
            U[U_idx[0]][col] += 1; U[U_idx[3]][col] -= 1
            V[V_idx[2]][col] += 1; V[V_idx[3]][col] += 1
            W[W_idx[1]][col] += 1; W[W_idx[3]][col] += 1
            col += 1
            # M4 = A00(B11-B01)
            U[U_idx[0]][col] += 1
            V[V_idx[3]][col] += 1; V[V_idx[2]][col] -= 1
            W[W_idx[0]][col] += 1; W[W_idx[2]][col] += 1
            col += 1
            # M5 = (A01+A11)(B00-B10)
            U[U_idx[1]][col] += 1; U[U_idx[3]][col] += 1
            V[V_idx[0]][col] += 1; V[V_idx[1]][col] -= 1
            W[W_idx[0]][col] += 1; W[W_idx[1]][col] += 1
            col += 1
            # M6 = (A10-A00)(B01+B11)
            U[U_idx[2]][col] += 1; U[U_idx[0]][col] -= 1
            V[V_idx[2]][col] += 1; V[V_idx[3]][col] += 1
            W[W_idx[2]][col] += 1; W[W_idx[3]][col] += 1
            col += 1
            # M7 = (A01-A10)(B10+B11)
            U[U_idx[1]][col] += 1; U[U_idx[2]][col] -= 1
            V[V_idx[1]][col] += 1; V[V_idx[3]][col] += 1
            W[W_idx[1]][col] += 1; W[W_idx[3]][col] += 1
            col += 1
            return col

        col = 0
        # Block (i=0,1) x (j=0,1) x (k=0,1): U[0,1,4,5], V[0,1,4,5], W[0,3,1,4]
        col = add_strassen_block([0,1,4,5], [0,1,4,5], [0,3,1,4], col)
        # Block (i=0,1) x (j=0,1) x (k=2,3): U[0,1,4,5], V[2,3,6,7], W[6,9,7,10]
        col = add_strassen_block([0,1,4,5], [2,3,6,7], [6,9,7,10], col)
        # Block (i=0,1) x (j=2,3) x (k=0,1): U[0,1,4,5], V[8,9,12,13], W[0,3,1,4]
        col = add_strassen_block([0,1,4,5], [8,9,12,13], [0,3,1,4], col)
        # Block (i=0,1) x (j=2,3) x (k=2,3): U[0,1,4,5], V[10,11,14,15], W[6,9,7,10]
        col = add_strassen_block([0,1,4,5], [10,11,14,15], [6,9,7,10], col)
        # Block (i=2) x (j=0,1) x (k=0,1): schoolbook 4 terms
        for j in [0,1]:
            for k in [0,1]:
                U[8+j][col] = 1  # i=2, j=j
                V[j*4+k][col] = 1
                W[k*3+2][col] = 1  # k*3+i = k*3+2
                col += 1
        # Block (i=2) x (j=0,1) x (k=2,3): schoolbook 4 terms
        for j in [0,1]:
            for k in [2,3]:
                U[8+j][col] = 1
                V[j*4+k][col] = 1
                W[k*3+2][col] = 1
                col += 1
        # Block (i=2) x (j=2,3) x (k=0,1): schoolbook 4 terms
        for j in [2,3]:
            for k in [0,1]:
                U[8+j][col] = 1
                V[j*4+k][col] = 1
                W[k*3+2][col] = 1
                col += 1
        # Block (i=2) x (j=2,3) x (k=2,3): schoolbook 4 terms
        for j in [2,3]:
            for k in [2,3]:
                U[8+j][col] = 1
                V[j*4+k][col] = 1
                W[k*3+2][col] = 1
                col += 1

        assert col == rank, f"Expected {rank} columns, got {col}"
        return U, V, W

    def build_rank48():
        """Build a rank-48 schoolbook decomposition."""
        rank = 48
        U = np.zeros((dim_u, rank), dtype=float)
        V = np.zeros((dim_v, rank), dtype=float)
        W = np.zeros((dim_w, rank), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        assert col == rank
        return U, V, W

    # Build rank-44 decomposition
    U, V, W = build_rank44()
    
    # Verify
    if not verify(U, V, W):
        # Fall back to rank-48
        U, V, W = build_rank48()
        if verify(U, V, W):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        # Should not happen, but return anyway
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    current_U = U.copy()
    current_V = V.copy()
    current_W = W.copy()
    current_rank = 44

    # Try to reduce rank by removing linearly dependent columns
    # Strategy: try removing each column and check if decomposition still valid
    max_removal_attempts = 30
    
    for attempt in range(max_removal_attempts):
        if time.time() > deadline - 10:
            break
        
        if current_rank <= 20:
            break
        
        found = False
        for r_idx in range(current_rank):
            if time.time() > deadline - 10:
                break
            
            keep_indices = [i for i in range(current_rank) if i != r_idx]
            
            U_test = current_U[:, keep_indices]
            V_test = current_V[:, keep_indices]
            W_test = current_W[:, keep_indices]
            
            if verify(U_test, V_test, W_test):
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
        if verify(current_U, current_V, current_W):
            return {"U": current_U.tolist(), "V": current_V.tolist(), "W": current_W.tolist()}

    # Return the rank-44 decomposition
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
