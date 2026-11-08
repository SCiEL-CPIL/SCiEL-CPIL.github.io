import numpy as np
from datetime import datetime, timedelta
import random
import time

DIMENSIONS = [2, 3, 9]

def construct():
    n, m, p = DIMENSIONS
    
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit
    
    # Start with the known rank-50 decomposition
    U0, V0, W0 = _rank50_baseline(n, m, p)
    best_rank = 50
    best_U, best_V, best_W = U0, V0, W0
    
    # Try to reduce rank by merging columns
    improved = _merge_columns_search(best_U, best_V, best_W, n, m, p, start_time, deadline)
    if improved is not None:
        U, V, W = improved
        if _verify_decomposition(U, V, W, n, m, p):
            new_rank = len(U[0])
            if new_rank < best_rank:
                best_rank = new_rank
                best_U, best_V, best_W = U, V, W
    
    return {"U": best_U, "V": best_V, "W": best_W}

def _merge_columns_search(U, V, W, n, m, p, start_time, deadline):
    """Try to reduce rank by merging pairs of columns that form a rank-1 sum."""
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    current_rank = U_np.shape[1]
    improved = False
    
    for iteration in range(200):
        if time.time() > deadline:
            break
        
        merged = False
        for r1 in range(current_rank):
            if time.time() > deadline:
                break
            for r2 in range(r1 + 1, current_rank):
                u1 = U_np[:, r1]
                u2 = U_np[:, r2]
                v1 = V_np[:, r1]
                v2 = V_np[:, r2]
                w1 = W_np[:, r1]
                w2 = W_np[:, r2]
                
                # Check proportionality of u vectors
                nu1 = np.linalg.norm(u1)
                nu2 = np.linalg.norm(u2)
                if nu1 < 1e-12 or nu2 < 1e-12:
                    continue
                
                # Check if u2 ≈ a*u1
                a = np.dot(u1, u2) / (nu1 * nu1)
                if np.linalg.norm(u2 - a * u1) < 1e-9 * nu2:
                    nv1 = np.linalg.norm(v1)
                    nv2 = np.linalg.norm(v2)
                    if nv1 < 1e-12 or nv2 < 1e-12:
                        continue
                    b = np.dot(v1, v2) / (nv1 * nv1)
                    if np.linalg.norm(v2 - b * v1) < 1e-9 * nv2:
                        nw1 = np.linalg.norm(w1)
                        nw2 = np.linalg.norm(w2)
                        if nw1 < 1e-12 or nw2 < 1e-12:
                            continue
                        c = np.dot(w1, w2) / (nw1 * nw1)
                        if np.linalg.norm(w2 - c * w1) < 1e-9 * nw2:
                            u_new = u1.copy()
                            v_new = v1.copy()
                            w_new = (1 + a * b * c) * w1
                            
                            U_np = np.delete(U_np, [r1, r2], axis=1)
                            V_np = np.delete(V_np, [r1, r2], axis=1)
                            W_np = np.delete(W_np, [r1, r2], axis=1)
                            U_np = np.column_stack([U_np, u_new])
                            V_np = np.column_stack([V_np, v_new])
                            W_np = np.column_stack([W_np, w_new])
                            current_rank -= 1
                            improved = True
                            merged = True
                            break
                    else:
                        pass
                
                # Check if v1 and v2 are proportional (and u not)
                nv1 = np.linalg.norm(v1)
                nv2 = np.linalg.norm(v2)
                if nv1 > 1e-12 and nv2 > 1e-12:
                    b = np.dot(v1, v2) / (nv1 * nv1)
                    if np.linalg.norm(v2 - b * v1) < 1e-9 * nv2:
                        nu1 = np.linalg.norm(u1)
                        nu2 = np.linalg.norm(u2)
                        if nu1 > 1e-12 and nu2 > 1e-12:
                            a = np.dot(u1, u2) / (nu1 * nu1)
                            if np.linalg.norm(u2 - a * u1) < 1e-9 * nu2:
                                nw1 = np.linalg.norm(w1)
                                nw2 = np.linalg.norm(w2)
                                if nw1 > 1e-12 and nw2 > 1e-12:
                                    c = np.dot(w1, w2) / (nw1 * nw1)
                                    if np.linalg.norm(w2 - c * w1) < 1e-9 * nw2:
                                        u_new = u1.copy()
                                        v_new = v1.copy()
                                        w_new = (1 + a * b * c) * w1
                                        U_np = np.delete(U_np, [r1, r2], axis=1)
                                        V_np = np.delete(V_np, [r1, r2], axis=1)
                                        W_np = np.delete(W_np, [r1, r2], axis=1)
                                        U_np = np.column_stack([U_np, u_new])
                                        V_np = np.column_stack([V_np, v_new])
                                        W_np = np.column_stack([W_np, w_new])
                                        current_rank -= 1
                                        improved = True
                                        merged = True
                                        break
                            else:
                                nw1 = np.linalg.norm(w1)
                                nw2 = np.linalg.norm(w2)
                                if nw1 > 1e-12 and nw2 > 1e-12:
                                    c = np.dot(w1, w2) / (nw1 * nw1)
                                    if np.linalg.norm(w2 - c * w1) < 1e-9 * nw2:
                                        u_new = u1 + b * c * u2
                                        v_new = v1.copy()
                                        w_new = w1.copy()
                                        U_np = np.delete(U_np, [r1, r2], axis=1)
                                        V_np = np.delete(V_np, [r1, r2], axis=1)
                                        W_np = np.delete(W_np, [r1, r2], axis=1)
                                        U_np = np.column_stack([U_np, u_new])
                                        V_np = np.column_stack([V_np, v_new])
                                        W_np = np.column_stack([W_np, w_new])
                                        current_rank -= 1
                                        improved = True
                                        merged = True
                                        break
                        nw1 = np.linalg.norm(w1)
                        nw2 = np.linalg.norm(w2)
                        if nw1 > 1e-12 and nw2 > 1e-12:
                            c = np.dot(w1, w2) / (nw1 * nw1)
                            if np.linalg.norm(w2 - c * w1) < 1e-9 * nw2:
                                u_new = u1 + b * c * u2
                                v_new = v1.copy()
                                w_new = w1.copy()
                                U_np = np.delete(U_np, [r1, r2], axis=1)
                                V_np = np.delete(V_np, [r1, r2], axis=1)
                                W_np = np.delete(W_np, [r1, r2], axis=1)
                                U_np = np.column_stack([U_np, u_new])
                                V_np = np.column_stack([V_np, v_new])
                                W_np = np.column_stack([W_np, w_new])
                                current_rank -= 1
                                improved = True
                                merged = True
                                break
                
                # Check if w1 and w2 are proportional
                nw1 = np.linalg.norm(w1)
                nw2 = np.linalg.norm(w2)
                if nw1 > 1e-12 and nw2 > 1e-12:
                    c = np.dot(w1, w2) / (nw1 * nw1)
                    if np.linalg.norm(w2 - c * w1) < 1e-9 * nw2:
                        nu1 = np.linalg.norm(u1)
                        nu2 = np.linalg.norm(u2)
                        nv1 = np.linalg.norm(v1)
                        nv2 = np.linalg.norm(v2)
                        if nu1 > 1e-12 and nu2 > 1e-12 and nv1 > 1e-12 and nv2 > 1e-12:
                            a = np.dot(u1, u2) / (nu1 * nu1)
                            if np.linalg.norm(u2 - a * u1) < 1e-9 * nu2:
                                v_new = v1 + c * a * v2
                                u_new = u1.copy()
                                w_new = w1.copy()
                                U_np = np.delete(U_np, [r1, r2], axis=1)
                                V_np = np.delete(V_np, [r1, r2], axis=1)
                                W_np = np.delete(W_np, [r1, r2], axis=1)
                                U_np = np.column_stack([U_np, u_new])
                                V_np = np.column_stack([V_np, v_new])
                                W_np = np.column_stack([W_np, w_new])
                                current_rank -= 1
                                improved = True
                                merged = True
                                break
                            else:
                                b = np.dot(v1, v2) / (nv1 * nv1)
                                if np.linalg.norm(v2 - b * v1) < 1e-9 * nv2:
                                    u_new = u1 + c * b * u2
                                    v_new = v1.copy()
                                    w_new = w1.copy()
                                    U_np = np.delete(U_np, [r1, r2], axis=1)
                                    V_np = np.delete(V_np, [r1, r2], axis=1)
                                    W_np = np.delete(W_np, [r1, r2], axis=1)
                                    U_np = np.column_stack([U_np, u_new])
                                    V_np = np.column_stack([V_np, v_new])
                                    W_np = np.column_stack([W_np, w_new])
                                    current_rank -= 1
                                    improved = True
                                    merged = True
                                    break
                        continue
                
                # General case: full rank check via SVD
                nu1 = np.linalg.norm(u1)
                nu2 = np.linalg.norm(u2)
                nv1 = np.linalg.norm(v1)
                nv2 = np.linalg.norm(v2)
                nw1 = np.linalg.norm(w1)
                nw2 = np.linalg.norm(w2)
                if min(nu1, nu2, nv1, nv2, nw1, nw2) < 1e-12:
                    continue
                
                # Check near-proportionality as a filter
                a_u = np.dot(u1, u2) / (nu1 * nu1)
                b_v = np.dot(v1, v2) / (nv1 * nv1)
                c_w = np.dot(w1, w2) / (nw1 * nw1)
                res_u = np.linalg.norm(u2 - a_u * u1) / nu2
                res_v = np.linalg.norm(v2 - b_v * v1) / nv2
                res_w = np.linalg.norm(w2 - c_w * w1) / nw2
                
                if min(res_u, res_v, res_w) < 1e-6:
                    try:
                        term1 = np.outer(U_np[:, r1], np.outer(V_np[:, r1], W_np[:, r1]).flatten())
                        term2 = np.outer(U_np[:, r2], np.outer(V_np[:, r2], W_np[:, r2]).flatten())
                        T_mat = term1 + term2
                        s = np.linalg.svd(T_mat, full_matrices=False, compute_uv=False)
                        if len(s) > 1 and s[1] < 1e-8 * s[0]:
                            U_r, s_r, Vt_r = np.linalg.svd(T_mat, full_matrices=False)
                            if s_r[0] > 1e-10:
                                u_new = U_r[:, 0] / np.sqrt(s_r[0])
                                vw_flat = Vt_r[0, :] / np.sqrt(s_r[0])
                                vw_2d = vw_flat.reshape(m * p, p * n)
                                s2 = np.linalg.svd(vw_2d, full_matrices=False, compute_uv=False)
                                if len(s2) > 1 and s2[1] < 1e-8 * s2[0]:
                                    U_r2, s_r2, Vt_r2 = np.linalg.svd(vw_2d, full_matrices=False)
                                    if s_r2[0] > 1e-10:
                                        v_new = U_r2[:, 0] / np.sqrt(s_r2[0])
                                        w_new = Vt_r2[0, :] / np.sqrt(s_r2[0])
                                        U_np = np.delete(U_np, [r1, r2], axis=1)
                                        V_np = np.delete(V_np, [r1, r2], axis=1)
                                        W_np = np.delete(W_np, [r1, r2], axis=1)
                                        U_np = np.column_stack([U_np, u_new])
                                        V_np = np.column_stack([V_np, v_new])
                                        W_np = np.column_stack([W_np, w_new])
                                        current_rank -= 1
                                        improved = True
                                        merged = True
                                        break
                    except:
                        pass
                
            if merged:
                break
        
        if not merged:
            break
    
    if improved:
        return U_np.tolist(), V_np.tolist(), W_np.tolist()
    
    return None

def _rank50_baseline(n, m, p):
    """Construct the known rank-50 decomposition using Strassen's algorithm."""
    rank = 50
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for b in range(4):
        k0 = 2 * b
        k1 = 2 * b + 1
        
        v_e = 0 * p + k0
        v_f = 0 * p + k1
        v_g = 1 * p + k0
        v_h = 1 * p + k1
        
        u_a = 0 * m + 0
        u_b = 0 * m + 1
        u_c = 1 * m + 0
        u_d = 1 * m + 1
        
        U[u_a][col] += 1; U[u_d][col] += 1
        V[v_e][col] += 1; V[v_h][col] += 1
        
        U[u_c][col+1] += 1; U[u_d][col+1] += 1
        V[v_e][col+1] += 1
        
        U[u_a][col+2] += 1
        V[v_f][col+2] += 1; V[v_h][col+2] -= 1
        
        U[u_d][col+3] += 1
        V[v_g][col+3] += 1; V[v_e][col+3] -= 1
        
        U[u_a][col+4] += 1; U[u_b][col+4] += 1
        V[v_h][col+4] += 1
        
        U[u_c][col+5] += 1; U[u_a][col+5] -= 1
        V[v_e][col+5] += 1; V[v_f][col+5] += 1
        
        U[u_b][col+6] += 1; U[u_d][col+6] -= 1
        V[v_g][col+6] += 1; V[v_h][col+6] += 1
        
        w_00 = k0 * n + 0; w_01 = k1 * n + 0
        w_10 = k0 * n + 1; w_11 = k1 * n + 1
        
        W[w_00][col] += 1; W[w_00][col+3] += 1
        W[w_00][col+4] -= 1; W[w_00][col+6] += 1
        
        W[w_01][col+2] += 1; W[w_01][col+4] += 1
        
        W[w_10][col+1] += 1; W[w_10][col+3] += 1
        
        W[w_11][col] += 1; W[w_11][col+1] -= 1
        W[w_11][col+2] += 1; W[w_11][col+5] += 1
        
        col += 7
    
    k8 = 8
    v_e = 0 * p + k8; v_g = 1 * p + k8
    u_a = 0 * m + 0; u_b = 0 * m + 1
    u_c = 1 * m + 0; u_d = 1 * m + 1
    
    U[u_a][col] += 1; V[v_e][col] += 1
    U[u_b][col+1] += 1; V[v_g][col+1] += 1
    U[u_c][col+2] += 1; V[v_e][col+2] += 1
    U[u_d][col+3] += 1; V[v_g][col+3] += 1
    
    w_08 = k8 * n + 0; w_18 = k8 * n + 1
    W[w_08][col] += 1; W[w_08][col+1] += 1
    W[w_18][col+2] += 1; W[w_18][col+3] += 1
    col += 4
    
    u_ar0 = 0 * m + 2; u_ar1 = 1 * m + 2
    for k in range(9):
        v_b = 2 * p + k
        U[u_ar0][col] += 1; V[v_b][col] += 1; W[k * n + 0][col] += 1
        U[u_ar1][col+1] += 1; V[v_b][col+1] += 1; W[k * n + 1][col+1] += 1
        col += 2
    
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition correctly computes matrix multiplication."""
    if not U or not V or not W:
        return False
    
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    rank = U_np.shape[1]
    
    if U_np.shape[0] != n * m:
        return False
    if V_np.shape[0] != m * p:
        return False
    if W_np.shape[0] != p * n:
        return False
    if U_np.shape[1] != rank or V_np.shape[1] != rank or W_np.shape[1] != rank:
        return False
    
    np.random.seed(42)
    for trial in range(3):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        
        C_computed = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                s = 0.0
                for j in range(m):
                    for r in range(rank):
                        s += A[i, j] * V_np[j * p + k, r] * W_np[k * n + i, r]
                C_computed[i, k] = s
        
        if not np.allclose(C_computed, C_expected, atol=1e-10):
            return False
    
    return True
