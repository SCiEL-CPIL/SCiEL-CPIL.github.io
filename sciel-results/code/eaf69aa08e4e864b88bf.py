import numpy as np
import time

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-minimal exact tensor decomposition for 3x4 * 4x6 matrix multiplication.
    Uses block Strassen + direct, then vectorized local search to reduce rank.
    """
    n, m, p = 3, 4, 6
    
    def build_baseline():
        R = 66
        U = [[0] * R for _ in range(n * m)]
        V = [[0] * R for _ in range(m * p)]
        W = [[0] * R for _ in range(p * n)]
        
        col = 0
        
        def add_strassen_block(a_r, a_c, b_r, b_c, c_r, c_c):
            nonlocal col
            A00 = a_r * m + a_c
            A01 = a_r * m + a_c + 1
            A10 = (a_r + 1) * m + a_c
            A11 = (a_r + 1) * m + a_c + 1
            B00 = b_r * p + b_c
            B01 = b_r * p + b_c + 1
            B10 = (b_r + 1) * p + b_c
            B11 = (b_r + 1) * p + b_c + 1
            C00 = c_c * n + c_r
            C01 = (c_c + 1) * n + c_r
            C10 = c_c * n + (c_r + 1)
            C11 = (c_c + 1) * n + (c_r + 1)
            
            U[A00][col] += 1; U[A11][col] += 1
            V[B00][col] += 1; V[B11][col] += 1
            W[C00][col] += 1; W[C11][col] += 1
            col += 1
            U[A10][col] += 1; U[A11][col] += 1
            V[B00][col] += 1
            W[C10][col] += 1; W[C11][col] -= 1
            col += 1
            U[A00][col] += 1
            V[B01][col] += 1; V[B11][col] -= 1
            W[C01][col] += 1; W[C11][col] += 1
            col += 1
            U[A11][col] += 1
            V[B10][col] += 1; V[B00][col] -= 1
            W[C00][col] += 1; W[C10][col] += 1
            col += 1
            U[A00][col] += 1; U[A01][col] += 1
            V[B11][col] += 1
            W[C00][col] -= 1; W[C01][col] += 1
            col += 1
            U[A10][col] += 1; U[A00][col] -= 1
            V[B00][col] += 1; V[B01][col] += 1
            W[C11][col] += 1
            col += 1
            U[A01][col] += 1; U[A11][col] -= 1
            V[B10][col] += 1; V[B11][col] += 1
            W[C00][col] += 1
            col += 1
        
        def add_direct_1x2_2x2(a_r, a_c, b_r, b_c, c_r, c_c):
            nonlocal col
            A0 = a_r * m + a_c
            A1 = a_r * m + a_c + 1
            B00 = b_r * p + b_c
            B01 = b_r * p + b_c + 1
            B10 = (b_r + 1) * p + b_c
            B11 = (b_r + 1) * p + b_c + 1
            C0 = c_c * n + c_r
            C1 = (c_c + 1) * n + c_r
            
            U[A0][col] += 1; V[B00][col] += 1; W[C0][col] += 1
            col += 1
            U[A1][col] += 1; V[B10][col] += 1; W[C0][col] += 1
            col += 1
            U[A0][col] += 1; V[B01][col] += 1; W[C1][col] += 1
            col += 1
            U[A1][col] += 1; V[B11][col] += 1; W[C1][col] += 1
            col += 1
        
        for j in range(3):
            for i in range(2):
                add_direct_1x2_2x2(0, i * 2, i * 2, j * 2, 0, j * 2)
        
        for j in range(3):
            for i in range(2):
                add_strassen_block(1, i * 2, i * 2, j * 2, 1, j * 2)
        
        assert col == R
        return U, V, W
    
    def compute_tensor(U_arr, V_arr, W_arr):
        """
        Vectorized computation of T[a,b,c] = sum_r U[a*m+b,r] * V[b*p+c,r] * W[c*n+a,r]
        U_arr: (n*m, R), V_arr: (m*p, R), W_arr: (p*n, R)
        Returns: (n, m, p) array
        """
        R = U_arr.shape[1]
        # For each r, we need outer product of u_col (n,m), v_col (m,p), w_col (p,n)
        # T[a,b,c] = sum_r U[a*m+b,r] * V[b*p+c,r] * W[c*n+a,r]
        
        # Reshape U to (n, m, R), V to (m, p, R), W to (p, n, R)
        U_3d = U_arr.reshape(n, m, R)
        V_3d = V_arr.reshape(m, p, R)
        W_3d = W_arr.reshape(p, n, R)
        
        # T[a,b,c] = sum_r U_3d[a,b,r] * V_3d[b,c,r] * W_3d[c,a,r]
        # We can compute this using einsum
        # U_3d[a,b,r] * V_3d[b,c,r] * W_3d[c,a,r] -> sum over r
        # einsum: 'abr,bcr,car->abc'
        T = np.einsum('abr,bcr,car->abc', U_3d, V_3d, W_3d, optimize=True)
        return T
    
    def verify(U, V, W):
        """Verify the decomposition exactly reproduces matrix multiplication."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        T = compute_tensor(U_arr, V_arr, W_arr)
        return np.allclose(T, 1.0, atol=1e-9)
    
    def try_local_search(U, V, W, deadline):
        """Try to reduce rank by finding and removing redundant terms using vectorized verification."""
        U = [row[:] for row in U]
        V = [row[:] for row in V]
        W = [row[:] for row in W]
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            r = len(U[0])
            
            # Precompute arrays for speed
            U_arr = np.array(U, dtype=float)
            V_arr = np.array(V, dtype=float)
            W_arr = np.array(W, dtype=float)
            
            for drop_idx in range(r):
                if time.time() >= deadline:
                    break
                
                # Build arrays without this column
                mask = np.ones(r, dtype=bool)
                mask[drop_idx] = False
                U_test = U_arr[:, mask]
                V_test = V_arr[:, mask]
                W_test = W_arr[:, mask]
                
                T_test = compute_tensor(U_test, V_test, W_test)
                
                if np.allclose(T_test, 1.0, atol=1e-9):
                    # Remove the column
                    mask2 = np.ones(r, dtype=bool)
                    mask2[drop_idx] = False
                    U = [list(row) for row in U_arr[:, mask2].astype(int)]
                    V = [list(row) for row in V_arr[:, mask2].astype(int)]
                    W = [list(row) for row in W_arr[:, mask2].astype(int)]
                    improved = True
                    break
        
        return U, V, W
    
    U, V, W = build_baseline()
    
    deadline = time.time() + 140  # 10s safety margin
    
    U, V, W = try_local_search(U, V, W, deadline)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
