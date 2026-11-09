import time
import numpy as np
from scipy.optimize import linprog

DIMENSIONS = [4, 4, 5]
START = time.time()

def _deadline():
    return time.time() - START > 145.0

def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, c] = 1.0
                V[j * p + k, c] = 1.0
                W[k * n + i, c] = 1.0
                c += 1
    return U, V, W

def _verify(U, V, W, n, m, p, tol=1e-6):
    if U.shape != (n * m, U.shape[1]) or V.shape != (m * p, V.shape[1]):
        return False
    if W.shape != (p * n, U.shape[1]):
        return False
    R = U.shape[1]
    if R != V.shape[1] or R != W.shape[1]:
        return False
    A = np.arange(n * m, dtype=float).reshape(n, m)
    B = np.arange(m * p, dtype=float).reshape(m, p)
    Ctrue = A @ B
    Uf = U.reshape(n, m, R)
    Vf = V.reshape(m, p, R)
    Wf = W.reshape(p, n, R)
    C = np.einsum('ijr,jkr,kir->ik', Uf, Vf, Wf)
    return np.allclose(C, Ctrue, atol=tol)

def _make_2x2x2_strassen():
    U = np.zeros((4, 7), dtype=float)
    V = np.zeros((4, 7), dtype=float)
    W = np.zeros((4, 7), dtype=float)
    
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[1, 0] = 1; W[2, 0] = 1; W[3, 0] = 1
    
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = 1
    
    U[0, 2] = 1
    V[2, 2] = 1; V[3, 2] = -1
    W[0, 2] = 1; W[1, 2] = 1
    
    U[3, 3] = 1
    V[1, 3] = 1; V[0, 3] = -1
    W[2, 3] = 1; W[3, 3] = 1
    
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = 1; W[1, 4] = 1
    
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[2, 5] = 1; W[3, 5] = 1
    
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1; W[1, 6] = 1
    
    return U, V, W

def _make_4x4x4_strassen():
    n = m = p = 4
    R = 49
    U = np.zeros((n * m, R), dtype=float)
    V = np.zeros((m * p, R), dtype=float)
    W = np.zeros((p * n, R), dtype=float)
    
    inner_U, inner_V, inner_W = _make_2x2x2_strassen()
    
    top_A = [
        [(0, 0, 1), (1, 1, 1)],
        [(1, 0, 1), (1, 1, 1)],
        [(0, 0, 1)],
        [(1, 1, 1)],
        [(0, 0, 1), (0, 1, 1)],
        [(1, 0, 1), (0, 0, -1)],
        [(0, 1, 1), (1, 1, -1)],
    ]
    top_B = [
        [(0, 0, 1), (1, 1, 1)],
        [(0, 0, 1)],
        [(1, 0, 1), (1, 1, -1)],
        [(0, 1, 1), (0, 0, -1)],
        [(1, 1, 1)],
        [(0, 0, 1), (0, 1, 1)],
        [(1, 0, 1), (1, 1, 1)],
    ]
    
    top_C = [
        [0, 3, 4, 6],
        [4, 5],
        [2, 4],
        [1, 2, 3],
    ]
    
    for t in range(7):
        for tp in range(7):
            col = t * 7 + tp
            
            for (br, bc, acoef) in top_A[t]:
                for ii in range(2):
                    for jj in range(2):
                        gi = br * 2 + ii
                        gj = bc * 2 + jj
                        U[gi * m + gj, col] += acoef * inner_U[ii * 2 + jj, tp]
            
            for (bc2, bk, bcoef) in top_B[t]:
                for jj in range(2):
                    for kk in range(2):
                        gj = bc2 * 2 + jj
                        gk = bk * 2 + kk
                        V[gj * p + gk, col] += bcoef * inner_V[jj * 2 + kk, tp]
            
            for c_idx in range(4):
                if t in top_C[c_idx]:
                    cr, cc = c_idx // 2, c_idx % 2
                    for ii in range(2):
                        for kk in range(2):
                            gi = cr * 2 + ii
                            gk = cc * 2 + kk
                            W[gk * n + gi, col] += inner_W[kk * 2 + ii, tp]
    
    return U, V, W

def _strassen_444_5(n, m, p):
    U4, V4, W4 = _make_4x4x4_strassen()
    R4 = U4.shape[1]
    extra = n * m
    R = R4 + extra
    
    U = np.zeros((n * m, R), dtype=float)
    V = np.zeros((m * p, R), dtype=float)
    W = np.zeros((p * n, R), dtype=float)
    
    U[:, :R4] = U4
    V[:, :R4] = V4
    W[:, :R4] = W4
    
    k = 4
    c = R4
    for i in range(n):
        for j in range(m):
            U[i * m + j, c] = 1.0
            V[j * p + k, c] = 1.0
            W[k * n + i, c] = 1.0
            c += 1
    
    return U, V, W

def _try_reduce_rank(U, V, W, n, m, p):
    """
    Attempts to reduce the rank by 1 by solving a linear system for perturbations.
    """
    R = U.shape[1]
    if R <= 49: # Don't go below known lower bound for 4x4x4 part? 
        return None

    # Flatten tensor T
    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    # Try removing each term
    for drop_idx in range(R):
        if _deadline():
            return None
            
        # Current terms excluding drop_idx
        indices = [i for i in range(R) if i != drop_idx]
        
        # We want to find delta_U, delta_V, delta_W such that:
        # sum_{r in indices} (U_r + dU_r) (V_r + dV_r) (W_r + dW_r) = T
        # Linearization:
        # sum_{r in indices} (U_r V_r W_r + dU_r V_r W_r + U_r dV_r W_r + U_r V_r dW_r) = T
        # sum_{r in indices} U_r V_r W_r + sum_{r in indices} (dU_r V_r W_r + U_r dV_r W_r + U_r V_r dW_r) = T
        # Let S = sum_{r in indices} U_r V_r W_r
        # We need: sum_{r in indices} (dU_r V_r W_r + U_r dV_r W_r + U_r V_r dW_r) = T - S
        
        # Build A matrix for linear system
        # Variables: dU (n*m*R'), dV (m*p*R'), dW (p*n*R') where R' = R-1
        R_prime = R - 1
        n_vars = (n * m + m * p + p * n) * R_prime
        n_eqs = n * m * m * p * p * n # 4*4*4*5*5*4 = 6400? No, T is 4x4x5 -> 80 non-zeros? 
        # T has 80 non-zero entries. The equation is for ALL entries of the tensor.
        # The tensor space is 4*4*5 = 80 dimensions.
        # So we have 80 equations.
        
        A = np.zeros((80, n_vars))
        b = np.zeros(80)
        
        # Compute S = sum of existing terms
        S = np.zeros((n, m, p, n), dtype=float) # n*m*p*n is 320? No, T is (n, m, p, n) ? 
        # T indices: (i, j, k, l) where C[i,l] gets A[i,j]B[j,k].
        # So T is a 4-index tensor? No, the multiplication tensor is usually T[i, j, k] -> C[i,k].
        # My T definition in _verify:
        # C = einsum('ijr, jkr, kir -> ik', U, V, W)
        # This implies U[i,j,r], V[j,k,r], W[k,i,r].
        # The target is C[i,k] = sum_j A[i,j] B[j,k].
        # So the tensor is T[i, j, k, i_out]? No.
        # The tensor is T[i, j, k] = delta_{i_out, i} delta_{k_out, k}?
        # Let's stick to the flattening used in _verify.
        
        # Flattened target vector of size 80 (n*m*p*n? No, n*m is 16, m*p is 20, p*n is 20? No.)
        # The tensor is a map from (A, B) to C.
        # A is 16-dim, B is 20-dim, C is 20-dim.
        # The tensor T is in A^* \otimes B^* \otimes C.
        # Dimension 16 * 20 * 20 = 6400.
        # But most entries are 0.
        # T[a, b, c] = 1 if a=(i,j), b=(j,k), c=(i,k) for some i,j,k.
        # There are 4*4*5 = 80 such triples.
        
        # Let's map the 80 non-zero entries to indices 0..79.
        idx_map = {}
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    idx_map[(a, b, c)] = idx
                    idx += 1
        
        # Compute S (sum of current terms)
        S_flat = np.zeros(80)
        for r in indices:
            u_r = U[:, r].reshape(n, m)
            v_r = V[:, r].reshape(m, p)
            w_r = W[:, r].reshape(p, n)
            # Term is u_r \otimes v_r \otimes w_r
            # Flatten it
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = u_r[i, j] * v_r[j, k] * w_r[k, i]
                        if val != 0:
                            a = i * m + j
                            b = j * p + k
                            c = k * n + i
                            S_flat[idx_map[(a, b, c)]] += val
        
        # Target T_flat
        T_flat = np.zeros(80)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    T_flat[idx_map[(a, b, c)]] = 1.0
        
        b = T_flat - S_flat
        
        # Build A matrix
        # For each term r in indices:
        # Contribution to dU: dU[i,j,r] * V[j,k,r] * W[k,i,r]
        # Contribution to dV: U[i,j,r] * dV[j,k,r] * W[k,i,r]
        # Contribution to dW: U[i,j,r] * V[j,k,r] * dW[k,i,r]
        
        # Variable indexing:
        # dU: r * (n*m) + i*m + j
        # dV: R_prime * (n*m) + r * (m*p) + j*p + k
        # dW: R_prime * (n*m + m*p) + r * (p*n) + k*n + i
        
        for r_idx, r in enumerate(indices):
            u_r = U[:, r].reshape(n, m)
            v_r = V[:, r].reshape(m, p)
            w_r = W[:, r].reshape(p, n)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        a = i * m + j
                        b = j * p + k
                        c = k * n + i
                        eq_idx = idx_map[(a, b, c)]
                        
                        # dU term
                        var_u = r_idx * (n * m) + a
                        A[eq_idx, var_u] += v_r[j, k] * w_r[k, i]
                        
                        # dV term
                        var_v = R_prime * (n * m) + r_idx * (m * p) + b
                        A[eq_idx, var_v] += u_r[i, j] * w_r[k, i]
                        
                        # dW term
                        var_w = R_prime * (n * m + m * p) + r_idx * (p * n) + c
                        A[eq_idx, var_w] += u_r[i, j] * v_r[j, k]
        
        # Solve A x = b
        try:
            x, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
            # Check if solution is good
            if np.linalg.norm(A @ x - b) < 1e-5:
                # Construct new U, V, W
                U_new = U.copy()
                V_new = V.copy()
                W_new = W.copy()
                
                # Set drop_idx to 0
                U_new[:, drop_idx] = 0
                V_new[:, drop_idx] = 0
                W_new[:, drop_idx] = 0
                
                # Apply perturbations
                for r_idx, r in enumerate(indices):
                    U_new[:, r] += x[r_idx * (n * m) : (r_idx + 1) * (n * m)].reshape(n * m)
                    V_new[:, r] += x[R_prime * (n * m) + r_idx * (m * p) : R_prime * (n * m) + (r_idx + 1) * (m * p)].reshape(m * p)
                    W_new[:, r] += x[R_prime * (n * m + m * p) + r_idx * (p * n) : R_prime * (n * m + m * p) + (r_idx + 1) * (p * n)].reshape(p * n)
                
                # Verify
                if _verify(U_new, V_new, W_new, n, m, p):
                    return U_new, V_new, W_new
        except Exception:
            pass
            
    return None

def construct():
    n, m, p = DIMENSIONS
    best = None
    
    Us, Vs, Ws = _schoolbook(n, m, p)
    if _verify(Us, Vs, Ws, n, m, p):
        best = (Us, Vs, Ws, Us.shape[1])
    
    if not _deadline():
        try:
            U2, V2, W2 = _strassen_444_5(n, m, p)
            if _verify(U2, V2, W2, n, m, p):
                if best is None or U2.shape[1] < best[3]:
                    best = (U2, V2, W2, U2.shape[1])
        except Exception:
            pass
    
    if best is None:
        best = (Us, Vs, Ws, Us.shape[1])
        
    U, V, W, _ = best
    
    # Try to reduce rank further
    if not _deadline():
        try:
            U_red, V_red, W_red = _try_reduce_rank(U, V, W, n, m, p)
            if U_red is not None:
                if U_red.shape[1] < best[3]:
                    best = (U_red, V_red, W_red, U_red.shape[1])
        except Exception:
            pass

    U, V, W, _ = best
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
