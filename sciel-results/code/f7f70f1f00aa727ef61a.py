import time
import numpy as np
from fractions import Fraction
from itertools import product

DIMENSIONS = [4, 4, 7]

def _build_tensor(n, m, p):
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1.0
    return T

def _verify(U, V, W, T, tol=1e-8):
    recon = np.einsum('ur,vr,wr->uvw', U, V, W)
    return np.allclose(recon, T, atol=tol)

def _strassen_2x2x2():
    U = np.array([
        [1, 0, 1, 0, 1, 0, 1],
        [0, 0, 0, 0, 1, -1, 0],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1],
    ], dtype=np.float64)
    V = np.array([
        [1, 1, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, -1, 1, 0, 1],
    ], dtype=np.float64)
    W = np.array([
        [1, 0, 0, 0, 1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0],
    ], dtype=np.float64)
    return U, V, W

def _build_rank93_decomposition():
    n, m, p = 4, 4, 7
    R4 = 49
    R3 = 44
    R_total = R4 + R3
    
    U = np.zeros((16, R_total))
    V = np.zeros((28, R_total))
    W = np.zeros((28, R_total))
    
    Us, Vs, Ws = _strassen_2x2x2()
    for r1 in range(7):
        for r2 in range(7):
            r = r1 * 7 + r2
            for i1 in range(2):
                for i2 in range(2):
                    U[i1 * 2 + i2, r] = Us[i1, r1] * Us[i2, r2]
            for j1 in range(2):
                for j2 in range(2):
                    V[j1 * 2 + j2, r] = Vs[j1, r1] * Vs[j2, r2]
            for k1 in range(2):
                for k2 in range(2):
                    W[k1 * 2 + k2, r] = Ws[k1, r1] * Ws[k2, r2]
    
    def _build_2x2x3_block(offset_i, offset_j, offset_k):
        U_loc = np.zeros((4, 11))
        V_loc = np.zeros((6, 11))
        W_loc = np.zeros((6, 11))
        
        Us, Vs, Ws = _strassen_2x2x2()
        for r in range(7):
            for i_loc in range(2):
                for j_loc in range(2):
                    U_loc[i_loc * 2 + j_loc, r] = Us[i_loc * 2 + j_loc, r]
            for j_loc in range(2):
                for k_loc in range(2):
                    V_loc[j_loc * 3 + k_loc, r] = Vs[j_loc * 2 + k_loc, r]
            for k_loc in range(2):
                for i_loc in range(2):
                    W_loc[k_loc * 2 + i_loc, r] = Ws[k_loc * 2 + i_loc, r]
        
        U_loc[0, 7] = 1
        V_loc[0 * 3 + 2, 7] = 1
        W_loc[2 * 2 + 0, 7] = 1
        
        U_loc[1, 8] = 1
        V_loc[1 * 3 + 2, 8] = 1
        W_loc[2 * 2 + 1, 8] = 1
        
        U_loc[2, 9] = 1
        V_loc[0 * 3 + 2, 9] = 1
        W_loc[2 * 2 + 0, 9] = 1
        
        U_loc[3, 10] = 1
        V_loc[1 * 3 + 2, 10] = 1
        W_loc[2 * 2 + 1, 10] = 1
        
        U_g = np.zeros((16, 11))
        V_g = np.zeros((28, 11))
        W_g = np.zeros((28, 11))
        
        for r in range(11):
            for i_loc in range(2):
                for j_loc in range(2):
                    u_g_idx = (offset_i + i_loc) * 4 + (offset_j + j_loc)
                    U_g[u_g_idx, r] = U_loc[i_loc * 2 + j_loc, r]
            
            for j_loc in range(2):
                for k_loc in range(3):
                    v_g_idx = (offset_j + j_loc) * 7 + (offset_k + k_loc)
                    V_g[v_g_idx, r] = V_loc[j_loc * 3 + k_loc, r]
            
            for k_loc in range(3):
                for i_loc in range(2):
                    w_g_idx = (offset_k + k_loc) * 4 + (offset_i + i_loc)
                    W_g[w_g_idx, r] = W_loc[k_loc * 2 + i_loc, r]
        
        return U_g, V_g, W_g

    U1, V1, W1 = _build_2x2x3_block(0, 0, 4)
    U[:, 49:60] += U1
    V[:, 49:60] += V1
    W[:, 49:60] += W1
    
    U2, V2, W2 = _build_2x2x3_block(0, 2, 4)
    U[:, 60:71] += U2
    V[:, 60:71] += V2
    W[:, 60:71] += W2
    
    U3, V3, W3 = _build_2x2x3_block(2, 0, 4)
    U[:, 71:82] += U3
    V[:, 71:82] += V3
    W[:, 71:82] += W3
    
    U4, V4, W4 = _build_2x2x3_block(2, 2, 4)
    U[:, 82:93] += U4
    V[:, 82:93] += V4
    W[:, 82:93] += W4
    
    return U, V, W

def _snap_to_rationals(U, V, W, max_den=16):
    """
    Attempt to snap the floating point factors to exact rationals with small denominators.
    Returns None if snapping fails to produce an exact decomposition.
    """
    n, m, p = 4, 4, 7
    T = _build_tensor(n, m, p)
    
    U_frac = np.zeros_like(U, dtype=object)
    V_frac = np.zeros_like(V, dtype=object)
    W_frac = np.zeros_like(W, dtype=object)
    
    for r in range(U.shape[1]):
        for i in range(U.shape[0]):
            val = U[i, r]
            if abs(val) < 1e-9:
                U_frac[i, r] = Fraction(0, 1)
            else:
                # Try to find p/q approximating val
                best_frac = None
                best_err = 1e-8
                for q in range(1, max_den + 1):
                    p_val = round(val * q)
                    frac = Fraction(p_val, q)
                    err = abs(float(frac) - val)
                    if err < best_err:
                        best_err = err
                        best_frac = frac
                if best_err > 1e-7:
                    return None # Snapping failed
                U_frac[i, r] = best_frac
        
        for i in range(V.shape[0]):
            val = V[i, r]
            if abs(val) < 1e-9:
                V_frac[i, r] = Fraction(0, 1)
            else:
                best_frac = None
                best_err = 1e-8
                for q in range(1, max_den + 1):
                    p_val = round(val * q)
                    frac = Fraction(p_val, q)
                    err = abs(float(frac) - val)
                    if err < best_err:
                        best_err = err
                        best_frac = frac
                if best_err > 1e-7:
                    return None
                V_frac[i, r] = best_frac
        
        for i in range(W.shape[0]):
            val = W[i, r]
            if abs(val) < 1e-9:
                W_frac[i, r] = Fraction(0, 1)
            else:
                best_frac = None
                best_err = 1e-8
                for q in range(1, max_den + 1):
                    p_val = round(val * q)
                    frac = Fraction(p_val, q)
                    err = abs(float(frac) - val)
                    if err < best_err:
                        best_err = err
                        best_frac = frac
                if best_err > 1e-7:
                    return None
                W_frac[i, r] = best_frac
    
    # Verify exactness using Fraction
    recon = np.zeros((n*m, m*p, p*n), dtype=object)
    for r in range(U.shape[1]):
        for u in range(n*m):
            if U_frac[u, r] == 0: continue
            for v in range(m*p):
                if V_frac[v, r] == 0: continue
                prod_uv = U_frac[u, r] * V_frac[v, r]
                if prod_uv == 0: continue
                for w in range(p*n):
                    recon[u, v, w] += prod_uv * W_frac[w, r]
    
    for u in range(n*m):
        for v in range(m*p):
            for w in range(p*n):
                if recon[u, v, w] != T[u, v, w]:
                    return None
    
    # Convert back to float for output
    U_out = np.array([[float(x) for x in row] for row in U_frac])
    V_out = np.array([[float(x) for x in row] for row in V_frac])
    W_out = np.array([[float(x) for x in row] for row in W_frac])
    
    return U_out, V_out, W_out

def _als_reduce_rank(U0, V0, W0, T, target_rank, time_limit, deadline):
    """
    Use ALS to try to find a valid decomposition of rank target_rank.
    Starts from a random sparse initialization or a projection of U0, V0, W0.
    """
    n, m, p = 4, 4, 7
    dims = [n*m, m*p, p*n]
    R = target_rank
    
    # Try multiple random initializations
    for attempt in range(20):
        if time.time() > deadline:
            return None
        
        # Initialize
        if attempt == 0:
            # Use first R columns of the known decomposition as a warm start
            U = U0[:, :R].copy()
            V = V0[:, :R].copy()
            W = W0[:, :R].copy()
        else:
            # Random sparse initialization
            density = 0.3
            U = np.random.randn(dims[0], R)
            V = np.random.randn(dims[1], R)
            W = np.random.randn(dims[2], R)
            # Apply sparsity mask
            mask_u = np.random.rand(dims[0], R) < density
            mask_v = np.random.rand(dims[1], R) < density
            mask_w = np.random.rand(dims[2], R) < density
            U *= mask_u
            V *= mask_v
            W *= mask_w
            
            # Scale
            U /= np.linalg.norm(U, axis=0, keepdims=True) + 1e-9
            V /= np.linalg.norm(V, axis=0, keepdims=True) + 1e-9
            W /= np.linalg.norm(W, axis=0, keepdims=True) + 1e-9

        best_err = float('inf')
        converged = False
        
        for step in range(500):
            if time.time() > deadline:
                break
            
            # Update U: U = T_(1) (V ⊙ W)^+
            # T_(1) is (n*m, m*p * p*n)
            T1 = T.reshape(dims[0], -1)
            VW = np.kron(V, W) # (m*p*p*n, R)
            # Solve U * (VW)^T = T1 -> U = T1 @ (VW)^+
            # Using SVD for stability
            try:
                U = T1 @ np.linalg.pinv(VW)
            except:
                break
            
            # Update V
            T2 = T.reshape(dims[1], -1).T # (m*p, n*m * p*n)
            UW = np.kron(U, W)
            try:
                V = np.linalg.pinv(UW) @ T2.T
            except:
                break
                
            # Update W
            T3 = T.reshape(dims[2], -1).T # (p*n, n*m * m*p)
            UV = np.kron(U, V)
            try:
                W = np.linalg.pinv(UV) @ T3.T
            except:
                break
            
            # Calculate error
            recon = np.einsum('ur,vr,wr->uvw', U, V, W)
            err = np.linalg.norm(recon - T)
            if err < best_err:
                best_err = err
                U_best, V_best, W_best = U.copy(), V.copy(), W.copy()
            
            if err < 1e-6:
                converged = True
                break
        
        if converged:
            # Try to snap to rationals
            snapped = _snap_to_rationals(U_best, V_best, W_best, max_den=32)
            if snapped:
                U_s, V_s, W_s = snapped
                if _verify(U_s, V_s, W_s, T):
                    return U_s, V_s, W_s
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0 # 10s margin
    
    T = _build_tensor(n, m, p)
    
    # Start with the known Rank 93 decomposition
    U93, V93, W93 = _build_rank93_decomposition()
    
    if not _verify(U93, V93, W93, T):
        # Fallback to schoolbook if construction failed
        R = n * m * p
        U = np.zeros((n * m, R))
        V = np.zeros((m * p, R))
        W = np.zeros((p * n, R))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return {'U': [[float(x) for x in U[r]] for r in range(U.shape[0])],
                'V': [[float(x) for x in V[r]] for r in range(V.shape[0])],
                'W': [[float(x) for x in W[r]] for r in range(W.shape[0])] }

    best_U, best_V, best_W = U93, V93, W93
    best_rank = 93
    
    # Try to reduce rank using ALS
    # We know R(4,4,7) is likely around 80-90. Let's try a few targets.
    # Start from 92 down to 80.
    for target_rank in range(92, 79, -1):
        if time.time() > deadline:
            break
            
        result = _als_reduce_rank(U93, V93, W93, T, target_rank, time.time() - start_time, deadline)
        if result:
            best_U, best_V, best_W = result
            best_rank = target_rank
            # If we found a valid one, we can stop or try to go lower.
            # Given the difficulty, finding one below 93 is a win.
            # Let's try to go a bit lower if time permits.
            if time.time() < deadline - 20:
                continue
            else:
                break

    U_out = [[float(x) for x in best_U[r]] for r in range(best_U.shape[0])]
    V_out = [[float(x) for x in best_V[r]] for r in range(best_V.shape[0])]
    W_out = [[float(x) for x in best_W[r]] for r in range(best_W.shape[0])]
    
    return {'U': U_out, 'V': V_out, 'W': W_out}

if __name__ == '__main__':
    res = construct()
    print("Rank:", len(res['U'][0]))
