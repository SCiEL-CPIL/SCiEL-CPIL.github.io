import time
import sys
import numpy as np
from collections import defaultdict
import scipy.optimize as opt

DIMENSIONS = [2, 4, 6]

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill

    n, m, p = DIMENSIONS  # 2, 4, 6
    nm, mp, pn = n * m, m * p, p * n  # 8, 24, 12

    # Target tensor T[a, b, c] = 1 if (a,b,c) is a valid index triple, else 0
    # a = i*m + j, b = j*p + k, c = k*n + i
    T_target = np.zeros((nm, mp, pn), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_target[a, b, c] = 1.0

    def verify(U_mat, V_mat, W_mat, tol=1e-8):
        r = U_mat.shape[1]
        # Check all 1s
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    s = np.sum(U_mat[a, :] * V_mat[b, :] * W_mat[c, :])
                    if abs(s - 1.0) > tol:
                        return False
        
        # Check all 0s
        valid_set = set()
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    valid_set.add((i*m+j, j*p+k, k*n+i))
                    
        for a in range(nm):
            for b in range(mp):
                for c in range(pn):
                    if (a,b,c) in valid_set:
                        continue
                    s = np.sum(U_mat[a, :] * V_mat[b, :] * W_mat[c, :])
                    if abs(s) > tol:
                        return False
        return True

    # Generate initial Strassen Rank 42 decomposition
    rank42 = 42
    U42 = np.zeros((nm, rank42), dtype=float)
    V42 = np.zeros((mp, rank42), dtype=float)
    W42 = np.zeros((pn, rank42), dtype=float)

    def add_strassen_block(i_start, j_start, k_start, offset):
        a00 = i_start * m + j_start
        a01 = i_start * m + j_start + 1
        a10 = (i_start + 1) * m + j_start
        a11 = (i_start + 1) * m + j_start + 1
        
        b00 = j_start * p + k_start
        b01 = j_start * p + k_start + 1
        b10 = (j_start + 1) * p + k_start
        b11 = (j_start + 1) * p + k_start + 1
        
        w00 = k_start * n + i_start
        w01 = (k_start + 1) * n + i_start
        w10 = k_start * n + i_start + 1
        w11 = (k_start + 1) * n + i_start + 1
        
        U42[a00, offset] += 1.0; U42[a11, offset] += 1.0
        V42[b00, offset] += 1.0; V42[b11, offset] += 1.0
        W42[w00, offset] += 1.0; W42[w11, offset] += 1.0

        U42[a10, offset+1] += 1.0; U42[a11, offset+1] += 1.0
        V42[b00, offset+1] += 1.0
        W42[w10, offset+1] += 1.0; W42[w11, offset+1] -= 1.0

        U42[a00, offset+2] += 1.0
        V42[b01, offset+2] += 1.0; V42[b11, offset+2] -= 1.0
        W42[w01, offset+2] += 1.0; W42[w11, offset+2] += 1.0

        U42[a11, offset+3] += 1.0
        V42[b10, offset+3] += 1.0; V42[b00, offset+3] -= 1.0
        W42[w00, offset+3] += 1.0; W42[w10, offset+3] += 1.0

        U42[a00, offset+4] += 1.0; U42[a01, offset+4] += 1.0
        V42[b11, offset+4] += 1.0
        W42[w00, offset+4] -= 1.0; W42[w01, offset+4] += 1.0

        U42[a10, offset+5] += 1.0; U42[a00, offset+5] -= 1.0
        V42[b00, offset+5] += 1.0; V42[b01, offset+5] += 1.0
        W42[w11, offset+5] += 1.0

        U42[a01, offset+6] += 1.0; U42[a11, offset+6] -= 1.0
        V42[b10, offset+6] += 1.0; V42[b11, offset+6] += 1.0
        W42[w00, offset+6] += 1.0

    offset = 0
    for i_start in range(0, n, 2):
        for j_start in range(0, m, 2):
            for k_start in range(0, p, 2):
                add_strassen_block(i_start, j_start, k_start, offset)
                offset += 7

    best_U, best_V, best_W = U42.copy(), V42.copy(), W42.copy()
    best_rank = 42

    if not verify(best_U, best_V, best_W):
        # Fallback to schoolbook rank 48
        rank0 = n * m * p
        U = np.zeros((nm, rank0), dtype=float)
        V = np.zeros((mp, rank0), dtype=float)
        W = np.zeros((pn, rank0), dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    col = i * (m * p) + j * p + k
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Optimization Strategy:
    # Try to find a decomposition of rank R < 42.
    # We use the Strassen blocks as a starting point, but we allow the optimizer to adjust values.
    # We flatten U, V, W into a single vector x.
    
    def objective(x):
        U = x[:nm*R].reshape(nm, R)
        V = x[nm*R:nm*R+mp*R].reshape(mp, R)
        W = x[nm*R+mp*R:].reshape(pn, R)
        
        # Compute tensor T = sum U_r V_r W_r
        # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
        # We can compute this efficiently using einsum or loops.
        # Given small sizes, loops are fine.
        error = 0.0
        # Only check non-zero entries of T_target to save time?
        # But we must ensure zero entries are also zero.
        # We can compute the full tensor and compare.
        T_comp = np.einsum('ar,br,cr->abc', U, V, W)
        diff = T_comp - T_target
        return np.sum(diff**2)

    # Try ranks from 24 to 41
    for R in range(24, 42):
        if time.time() > deadline - 10:
            break
            
        # Initialize x from Strassen blocks if R <= 42
        # We can't directly take a subset of columns if R < 42 easily without losing structure.
        # Better: Random initialization or scaled Strassen.
        # Let's try random initialization with small noise.
        np.random.seed(42 + R)
        x0 = np.random.randn(nm*R + mp*R + pn*R) * 0.1
        
        # Try a few restarts
        found = False
        for restart in range(3):
            if time.time() > deadline - 5:
                break
                
            # Add noise to x0 for restart
            x_init = x0 + np.random.randn(len(x0)) * 0.1
            
            # Use L-BFGS-B
            try:
                result = opt.minimize(objective, x_init, method='L-BFGS-B', 
                                      options={'maxiter': 100, 'ftol': 1e-8, 'gtol': 1e-8})
                
                if result.fun < 1e-6:
                    # Extract U, V, W
                    U_opt = result.x[:nm*R].reshape(nm, R)
                    V_opt = result.x[nm*R:nm*R+mp*R].reshape(mp, R)
                    W_opt = result.x[nm*R+mp*R:].reshape(pn, R)
                    
                    # Round to nearest 0.5
                    U_rounded = np.round(U_opt * 2) / 2
                    V_rounded = np.round(V_opt * 2) / 2
                    W_rounded = np.round(W_opt * 2) / 2
                    
                    if verify(U_rounded, V_rounded, W_rounded):
                        best_U, best_V, best_W = U_rounded, V_rounded, W_rounded
                        best_rank = R
                        found = True
                        break
            except Exception:
                pass
        
        if found:
            break

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

if __name__ == "__main__":
    result = construct()
    print("Constructed", file=sys.stderr)
