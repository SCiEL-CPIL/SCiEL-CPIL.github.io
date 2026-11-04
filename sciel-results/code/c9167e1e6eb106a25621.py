import numpy as np
import time

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 110.0  # Safety margin
    
    # Target tensor T[i, j, k] = 1
    # Shape (n, m, p)
    T = np.ones((n, m, p))
    
    # Reshape to matrix form for tensor contraction
    # We want U (nm x r), V (mp x r), W (pn x r) such that
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # Index mappings:
    # U index: i*m + j
    # V index: j*p + k
    # W index: k*n + i
    
    def check_solution(U, V, W):
        # Compute the reconstructed tensor
        # Rec[i, j, k] = sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c]
        # We can do this efficiently using einsum or loops
        # n=2, m=3, p=4 is small, so loops are fine.
        rec = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    # Dot product of row u_idx of U, row v_idx of V, row w_idx of W
                    # rec[i,j,k] = U[u_idx, :] . (V[v_idx, :] * W[w_idx, :])
                    rec[i, j, k] = np.dot(U[u_idx, :], V[v_idx, :] * W[w_idx, :])
        # Check if rec is all 1s
        return np.allclose(rec, T, atol=1e-9)

    def snap_to_half_integer(mat):
        return np.round(mat * 2) / 2.0

    # Try ranks from 18 up to 24
    best_rank = n * m * p
    best_U = None
    best_V = None
    best_W = None
    
    # Schoolbook solution as fallback
    rank_sb = n * m * p
    U_sb = np.zeros((n * m, rank_sb))
    V_sb = np.zeros((m * p, rank_sb))
    W_sb = np.zeros((p * n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = rank_sb

    for r in range(18, best_rank + 1):
        if time.time() > deadline:
            break
            
        # Try to find a solution of rank r
        found = False
        for attempt in range(5):
            if time.time() > deadline:
                break
            
            # Random initialization
            U = np.random.randn(n * m, r)
            V = np.random.randn(m * p, r)
            W = np.random.randn(p * n, r)
            
            # Alternating Least Squares
            for step in range(100):
                if time.time() > deadline:
                    break
                
                # Update U
                # For each (i, j), U[i*m+j, :] should satisfy:
                # sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] = 1 for all k
                # This is a linear system for U[i*m+j, :]
                # Let u_row = U[i*m+j, :]
                # We want min_c || (V[j*p+0, c] * W[0*n+i, c]) * u_row - 1 ||^2 ...
                # Actually, for fixed j, k varies.
                # The equation is:
                # sum_c u_c * (V[j*p+k, c] * W[k*n+i, c]) = 1
                # This is a system A_u u = b, where A_u[k, c] = V[j*p+k, c] * W[k*n+i, c]
                # b[k] = 1
                
                for i in range(n):
                    for j in range(m):
                        u_idx = i * m + j
                        # Build A_u (p x r)
                        A_u = np.zeros((p, r))
                        for k in range(p):
                            v_idx = j * p + k
                            w_idx = k * n + i
                            A_u[k, :] = V[v_idx, :] * W[w_idx, :]
                        # Solve A_u @ u_row = ones(p)
                        # Use least squares
                        u_row, _, _, _ = np.linalg.lstsq(A_u, np.ones(p), rcond=None)
                        U[u_idx, :] = u_row

                # Update V
                # For each (j, k), V[j*p+k, :] should satisfy:
                # sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] = 1 for all i
                # Let v_row = V[j*p+k, :]
                # A_v[i, c] = U[i*m+j, c] * W[k*n+i, c]
                # b[i] = 1
                for j in range(m):
                    for k in range(p):
                        v_idx = j * p + k
                        A_v = np.zeros((n, r))
                        for i in range(n):
                            u_idx = i * m + j
                            w_idx = k * n + i
                            A_v[i, :] = U[u_idx, :] * W[w_idx, :]
                        v_row, _, _, _ = np.linalg.lstsq(A_v, np.ones(n), rcond=None)
                        V[v_idx, :] = v_row

                # Update W
                # For each (k, i), W[k*n+i, :] should satisfy:
                # sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] = 1 for all j
                # Let w_row = W[k*n+i, :]
                # A_w[j, c] = U[i*m+j, c] * V[j*p+k, c]
                # b[j] = 1
                for k in range(p):
                    for i in range(n):
                        w_idx = k * n + i
                        A_w = np.zeros((m, r))
                        for j in range(m):
                            u_idx = i * m + j
                            v_idx = j * p + k
                            A_w[j, :] = U[u_idx, :] * V[v_idx, :]
                        w_row, _, _, _ = np.linalg.lstsq(A_w, np.ones(m), rcond=None)
                        W[w_idx, :] = w_row
            
            # Check if the current solution is close to a half-integer solution
            U_snap = snap_to_half_integer(U)
            V_snap = snap_to_half_integer(V)
            W_snap = snap_to_half_integer(W)
            
            if check_solution(U_snap, V_snap, W_snap):
                best_rank = r
                best_U = U_snap
                best_V = V_snap
                best_W = W_snap
                found = True
                break
        
        if found:
            break

    # Convert to lists of lists of numbers (int or float)
    # The interface expects integer/half-integer entries
    # We can return them as floats if they are exactly 0.5, 1.0, etc.
    
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
# EVOLVE-BLOCK-END
