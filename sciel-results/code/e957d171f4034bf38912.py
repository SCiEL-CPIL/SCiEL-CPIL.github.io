import numpy as np
import time
import scipy.optimize as opt

def construct():
    n, m, p = 3, 3, 6
    deadline = time.time() + 120  # Safety margin
    
    def verify_3x3(U, V, W, R):
        # U: 9xR, V: 9xR, W: 9xR
        # Check sum_r U[i*3+k, r] V[k*3+j, r] W[i*3+j, r] == 1
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    s = 0.0
                    for r in range(R):
                        s += U[i*3+k, r] * V[k*3+j, r] * W[i*3+j, r]
                    if abs(s - 1.0) > 1e-5:
                        return False
        return True

    def residuals(params, R):
        # params: flat array of size 27*R
        # U: 9*R, V: 9*R, W: 9*R
        U = params[0:9*R].reshape(9, R)
        V = params[9*R:18*R].reshape(9, R)
        W = params[18*R:27*R].reshape(9, R)
        
        res = np.zeros(27)
        idx = 0
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    s = 0.0
                    for r in range(R):
                        s += U[i*3+k, r] * V[k*3+j, r] * W[i*3+j, r]
                    res[idx] = s - 1.0
                    idx += 1
        return res

    def try_find_decomp(R, max_restarts=100):
        best_sol = None
        best_err = float('inf')
        
        for restart in range(max_restarts):
            if time.time() > deadline - 20:
                break
                
            # Random initialization
            # Scale: 1/sqrt(R)
            init_scale = 1.0 / np.sqrt(R)
            params = np.random.randn(27 * R) * init_scale
            
            # Optimization
            try:
                sol = opt.minimize(
                    lambda x: np.sum(residuals(x, R)**2), 
                    params, 
                    method='BFGS', 
                    options={'maxiter': 1000, 'disp': False}
                )
                if sol.fun < best_err:
                    best_err = sol.fun
                    best_sol = sol.x
                    
                    # Check if close to zero
                    if best_err < 1e-8:
                        U = best_sol[0:9*R].reshape(9, R)
                        V = best_sol[9*R:18*R].reshape(9, R)
                        W = best_sol[18*R:27*R].reshape(9, R)
                        
                        # Round to nearest half-integer
                        U_r = np.round(U * 2) / 2
                        V_r = np.round(V * 2) / 2
                        W_r = np.round(W * 2) / 2
                        
                        if verify_3x3(U_r, V_r, W_r, R):
                            return U_r, V_r, W_r
            except Exception:
                continue
        return None

    # Try ranks 19 to 23
    for R in range(19, 24):
        if time.time() > deadline - 30:
            break
        result = try_find_decomp(R)
        if result:
            U3, V3, W3 = result
            # Construct 3x3x6 decomposition
            # C = [A B1, A B2]
            # U_total: 9 x (2*R)
            # V_total: 18 x (2*R)
            # W_total: 18 x (2*R)
            
            U_total = np.zeros((9, 2*R))
            V_total = np.zeros((18, 2*R))
            W_total = np.zeros((18, 2*R))
            
            # First 3x3 block (B1, C1)
            U_total[:, :R] = U3
            V_total[:9, :R] = V3
            W_total[:9, :R] = W3
            
            # Second 3x3 block (B2, C2)
            U_total[:, R:] = U3
            V_total[9:, R:] = V3
            W_total[9:, R:] = W3
            
            # Verify full 3x3x6
            # U: 9 x Rank, V: 18 x Rank, W: 18 x Rank
            # Indices: U[i*3+k], V[k*3+j], W[j*3+i]? 
            # Task convention: W[k*n+i, r]. Here n=3, p=6.
            # Wait, the task says:
            # U: (n*m, rank) -> (9, rank)
            # V: (m*p, rank) -> (18, rank)
            # W: (p*n, rank) -> (18, rank)
            # W index is k*n + i.
            # In my 3x3x3 decomposition, W index was i*3+j (output index).
            # For 3x3x6, the output is C (3x6).
            # C = [C1, C2].
            # C1 is 3x3, C2 is 3x3.
            # W index for C1: k*n + i? No, n=3, p=6.
            # W shape is (p*n, rank) = (18, rank).
            # Index w = k*n + i?
            # Let's check the convention: "W[k*n+i,r]".
            # k is the column of B (0..5), i is the row of C (0..2).
            # So W index is k*3 + i.
            # In my block decomposition:
            # C1 corresponds to k=0,1,2.
            # C2 corresponds to k=3,4,5.
            # For C1, the output element is C[i, k] where k in 0..2.
            # My W3 index was i*3 + j where j is the column of C1 (0..2).
            # So W3[i*3+j] corresponds to C[i,j].
            # In the full W, the index for C[i,j] (j in 0..2) is j*3 + i.
            # So we need to map W3[i*3+j] to W_total[j*3+i].
            
            # Let's re-map W.
            W_total_mapped = np.zeros((18, 2*R))
            for i in range(3):
                for j in range(3):
                    # W3 index: i*3 + j
                    val = W3[i*3+j, :]
                    # W_total index: j*3 + i (for first block)
                    W_total_mapped[j*3+i, :R] = val
                    # W_total index: (j+3)*3 + i (for second block)
                    W_total_mapped[(j+3)*3+i, R:] = val
                    
            W_total = W_total_mapped
            
            # Verify full
            def verify_full(U, V, W, n, m, p):
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0
                            for c in range(U.shape[1]):
                                s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                            if abs(s - 1) > 1e-5:
                                return False
                return True

            if verify_full(U_total, V_total, W_total, n, m, p):
                return {"U": U_total.tolist(), "V": V_total.tolist(), "W": W_total.tolist()}
            else:
                # Rounding error?
                pass

    # Fallback: Schoolbook
    rank_schoolbook = n * m * p
    U = np.zeros((n * m, rank_schoolbook))
    V = np.zeros((m * p, rank_schoolbook))
    W = np.zeros((p * n, rank_schoolbook))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
