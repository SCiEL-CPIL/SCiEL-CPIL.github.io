import random
import numpy as np
import time
import sys

DIMENSIONS = [3, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s kill
    
    # Build the target tensor T of shape (n*m, m*p, p*n)
    # T[a, b, c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i for some i,j,k, else 0
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    def check(U, V, W, tol=1e-8):
        recon = np.einsum('ar,br,cr->abc', U, V, W)
        return np.max(np.abs(recon - T)) < tol
    
    def als_decompose(U, V, W, T, max_iter=500, tol=1e-10):
        """Alternating Least Squares for tensor decomposition."""
        U = U.copy()
        V = V.copy()
        W = W.copy()
        R = U.shape[1]
        prev_err = np.inf
        
        for it in range(max_iter):
            if time.time() > deadline - 10:
                break
                
            # Update U
            M = np.einsum('br,cr->bcr', V, W)  # (m*p, p*n, R)
            M_flat = M.transpose(2, 0, 1).reshape(R, m*p * p*n)
            U_new = np.zeros_like(U)
            for a in range(n*m):
                try:
                    U_new[a, :] = np.linalg.lstsq(M_flat.T, T[a].reshape(-1), rcond=None)[0]
                except:
                    U_new[a, :] = U[a, :]
            
            # Update V
            M2 = np.einsum('ar,cr->acr', U, W)  # (n*m, p*n, R)
            M2_flat = M2.transpose(2, 0, 1).reshape(R, n*m * p*n)
            V_new = np.zeros_like(V)
            for b in range(m*p):
                T_slice = T[:, b, :]
                try:
                    V_new[b, :] = np.linalg.lstsq(M2_flat.T, T_slice.reshape(-1), rcond=None)[0]
                except:
                    V_new[b, :] = V[b, :]
            
            # Update W
            M3 = np.einsum('ar,br->abr', U, V)  # (n*m, m*p, R)
            M3_flat = M3.transpose(2, 0, 1).reshape(R, n*m * m*p)
            W_new = np.zeros_like(W)
            for c in range(p*n):
                T_slice = T[:, :, c]
                try:
                    W_new[c, :] = np.linalg.lstsq(M3_flat.T, T_slice.reshape(-1), rcond=None)[0]
                except:
                    W_new[c, :] = W[c, :]
            
            U, V, W = U_new, V_new, W_new
            
            recon = np.einsum('ar,br,cr->abc', U, V, W)
            err = np.max(np.abs(recon - T))
            
            if err < tol:
                break
            if abs(prev_err - err) < 1e-14:
                break
            prev_err = err
        
        return U, V, W, err
    
    # Start with schoolbook decomposition as baseline
    best_rank = n * m * p
    best_U = np.zeros((n*m, best_rank))
    best_V = np.zeros((m*p, best_rank))
    best_W = np.zeros((p*n, best_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                best_U[i * m + j][col] = 1
                best_V[j * p + k][col] = 1
                best_W[k * n + i][col] = 1
                col += 1
    
    # Try to find lower rank decompositions
    # Based on known results, try ranks from lower values up
    # For 3x4x8, known lower bounds suggest rank could be around 40-60
    
    for target_rank in range(40, best_rank + 1):
        if time.time() > deadline - 30:
            break
        
        # Multiple random restarts for each rank
        max_restarts = 10 if target_rank < 60 else 5
        found = False
        
        for restart in range(max_restarts):
            if time.time() > deadline - 20:
                break
            
            # Initialize with random values
            np.random.seed(42 + restart * 1000 + target_rank)
            U_init = np.random.randn(n*m, target_rank)
            V_init = np.random.randn(m*p, target_rank)
            W_init = np.random.randn(p*n, target_rank)
            
            # Normalize to avoid scaling issues
            U_init /= np.linalg.norm(U_init)
            V_init /= np.linalg.norm(V_init)
            W_init /= np.linalg.norm(W_init)
            
            U, V, W, err = als_decompose(U_init, V_init, W_init, T, max_iter=300, tol=1e-10)
            
            if check(U, V, W, tol=1e-8):
                # Found a valid decomposition at this rank
                best_rank = target_rank
                best_U = U
                best_V = V
                best_W = W
                found = True
                break
        
        if found:
            break
    
    # Convert to list format for return
    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}

if __name__ == "__main__":
    start_time = time.time()
    res = construct()
    end_time = time.time()
    
    n, m, p = DIMENSIONS
    rank = len(res["U"][0])
    assert len(res["U"]) == n * m, f"U rows: {len(res['U'])} != {n*m}"
    assert len(res["V"]) == m * p, f"V rows: {len(res['V'])} != {m*p}"
    assert len(res["W"]) == p * n, f"W rows: {len(res['W'])} != {p*n}"
    
    print(f"Rank found: {rank}")
    print(f"Time: {end_time - start_time:.4f}s")
    
    # Quick verification
    import random
    random.seed(123)
    A = [[random.randint(-5, 5) for _ in range(m)] for _ in range(n)]
    B = [[random.randint(-5, 5) for _ in range(p)] for _ in range(m)]
    
    C_true = [[sum(A[i][j] * B[j][k] for j in range(m)) for k in range(p)] for i in range(n)]
    
    C_calc = [[0 for _ in range(p)] for _ in range(n)]
    for i in range(n):
        for k in range(p):
            w_idx = k * n + i
            s = 0
            for r in range(rank):
                u_val = sum(res["U"][u][r] * A[u // m][u % m] for u in range(n*m))
                v_val = sum(res["V"][v][r] * B[v // p][v % p] for v in range(m*p))
                s += res["W"][w_idx][r] * u_val * v_val
            C_calc[i][k] = s
    
    max_err = max(abs(C_true[i][k] - C_calc[i][k]) for i in range(n) for k in range(p))
    print(f"Max error: {max_err:.2e}")
    assert max_err < 1e-6, f"Decomposition error too large: {max_err}"
    print("Verification passed")
