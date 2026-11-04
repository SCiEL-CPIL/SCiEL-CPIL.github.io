import numpy as np
import time

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 145.0

    # Build target tensor
    dim_u = n * m  # 16
    dim_v = m * p  # 28
    dim_w = p * n  # 28

    T = np.zeros((dim_u, dim_v, dim_w), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1

    schoolbook_rank = n * m * p  # 112
    best_result = None
    best_rank = schoolbook_rank

    # Strategy 1: Structured decomposition using Strassen for 4x4x4 part
    U_2, V_2, W_2 = _strassen_2x2x2()
    
    # Kronecker product for 4x4x4: rank 49
    U_444 = np.zeros((16, 49))
    V_444_std = np.zeros((16, 49))
    W_444_std = np.zeros((16, 49))
    
    for r1 in range(7):
        for r2 in range(7):
            idx = r1 * 7 + r2
            U_444[:, idx] = np.kron(U_2[:, r1], U_2[:, r2])
            V_444_std[:, idx] = np.kron(V_2[:, r1], V_2[:, r2])
            W_444_std[:, idx] = np.kron(W_2[:, r1], W_2[:, r2])
    
    V_444_full = np.zeros((28, 49))
    for j in range(4):
        for k in range(4):
            V_444_full[j*7+k, :] = V_444_std[j*4+k, :]
    
    W_444_full = np.zeros((28, 49))
    for k in range(4):
        for i in range(4):
            W_444_full[k*4+i, :] = W_444_std[k*4+i, :]
    
    recon_444 = np.einsum('ak,bk,ck->abc', U_444, V_444_full, W_444_full)
    T_444_part = np.zeros((16, 28, 28), dtype=np.float64)
    for i in range(4):
        for j in range(4):
            for k in range(4):
                a = i * 4 + j
                b = j * 7 + k
                c = k * 4 + i
                T_444_part[a, b, c] = 1
    
    strassen_valid = np.allclose(recon_444, T_444_part, atol=1e-10)
    
    # Strategy 2: ALS-based peeling for the full tensor
    if time.time() < deadline - 10.0:
        als_result = _als_peel(T, dim_u, dim_v, dim_w, deadline - 5.0, 110)
        if als_result is not None:
            r_als = len(als_result["U"][0])
            if r_als < best_rank:
                best_rank = r_als
                best_result = als_result
    
    # Strategy 3: Combine Strassen with ALS on residual
    if strassen_valid and time.time() < deadline - 8.0:
        # Residual after Strassen 4x4x4 part
        recon_so_far = np.einsum('ak,bk,ck->abc', U_444, V_444_full, W_444_full)
        R = T - recon_so_far
        
        # ALS peel the residual
        residual_result = _als_peel(R, dim_u, dim_v, dim_w, deadline - 5.0, 50)
        if residual_result is not None:
            r_res = len(residual_result["U"][0])
            U_res = np.array(residual_result["U"]).T
            V_res = np.array(residual_result["V"]).T
            W_res = np.array(residual_result["W"]).T
            
            U_combined = np.hstack([U_444, U_res])
            V_combined = np.hstack([V_444_full, V_res])
            W_combined = np.hstack([W_444_full, W_res])
            
            recon = np.einsum('ak,bk,ck->abc', U_combined, V_combined, W_combined)
            if np.allclose(recon, T, atol=1e-10):
                total_rank = 49 + r_res
                if total_rank < best_rank:
                    best_rank = total_rank
                    best_result = {
                        "U": U_combined.tolist(),
                        "V": V_combined.tolist(),
                        "W": W_combined.tolist()
                    }
    
    # Fallback: schoolbook if nothing better found
    if best_result is None or best_rank >= schoolbook_rank:
        return _schoolbook(n, m, p)
    
    return best_result


def _als_peel(T, dim_u, dim_v, dim_w, deadline, max_terms):
    """ALS-based rank-1 peeling with continuous variables. Returns dict or None."""
    R = T.copy()
    U_list = []
    V_list = []
    W_list = []
    
    for term_idx in range(max_terms):
        if time.time() > deadline:
            break
        if not np.any(R):
            break
        
        best_err = np.inf
        best_u, best_v, best_w = None, None, None
        
        # Multiple restarts with different initializations
        n_restarts = 6
        for restart in range(n_restarts):
            if time.time() > deadline:
                break
            
            # Different initialization strategies
            if restart == 0:
                cu = np.sum(R, axis=(1, 2))
                cv = np.sum(R, axis=(0, 2))
                cw = np.sum(R, axis=(0, 1))
            elif restart == 1:
                cu = np.random.randn(dim_u)
                cv = np.random.randn(dim_v)
                cw = np.random.randn(dim_w)
            elif restart == 2:
                cu = np.random.choice([-1, 0, 1], size=dim_u).astype(np.float64)
                cv = np.random.choice([-1, 0, 1], size=dim_v).astype(np.float64)
                cw = np.random.choice([-1, 0, 1], size=dim_w).astype(np.float64)
            elif restart == 3:
                # Random sparse initialization
                cu = np.zeros(dim_u)
                cv = np.zeros(dim_v)
                cw = np.zeros(dim_w)
                cu[np.random.choice(dim_u, dim_u//4, replace=False)] = 1
                cv[np.random.choice(dim_v, dim_v//4, replace=False)] = 1
                cw[np.random.choice(dim_w, dim_w//4, replace=False)] = 1
            else:
                cu = np.random.randn(dim_u)
                cv = np.random.randn(dim_v)
                cw = np.random.randn(dim_w)
            
            # ALS iterations with continuous variables
            n_iters = 20
            for it in range(n_iters):
                if time.time() > deadline:
                    break
                
                # Update U: for each a, minimize sum_{b,c} (R[a,b,c] - U[a]*V[b]*W[c])^2
                # This is: U[a] = sum_{b,c} R[a,b,c]*V[b]*W[c] / sum_{b,c} (V[b]*W[c])^2
                VW = np.outer(cv, cw)
                denom_u = np.sum(VW**2)
                if denom_u > 1e-10:
                    cu = np.einsum('abc,bc->a', R, VW) / denom_u
                else:
                    break
                
                # Update V
                UW = np.outer(cu, cw)
                denom_v = np.sum(UW**2)
                if denom_v > 1e-10:
                    cv = np.einsum('abc,ac->b', R, UW) / denom_v
                else:
                    break
                
                # Update W
                UV = np.outer(cu, cv)
                denom_w = np.sum(UV**2)
                if denom_w > 1e-10:
                    cw = np.einsum('abc,ab->c', R, UV) / denom_w
                else:
                    break
            
            # Compute error
            recon = np.einsum('a,b,c->abc', cu, cv, cw)
            err = np.linalg.norm(R - recon)
            if err < best_err:
                best_err = err
                best_u, best_v, best_w = cu.copy(), cv.copy(), cw.copy()
        
        if best_u is None or not np.isfinite(best_err):
            break
        
        R -= np.einsum('a,b,c->abc', best_u, best_v, best_w)
        U_list.append(best_u)
        V_list.append(best_v)
        W_list.append(best_w)
        
        if np.allclose(R, 0, atol=1e-10):
            break
    
    if np.allclose(R, 0, atol=1e-10) and len(U_list) > 0:
        rank = len(U_list)
        U = np.array(U_list).T
        V = np.array(V_list).T
        W = np.array(W_list).T
        
        reconstructed = np.einsum('ak,bk,ck->abc', U, V, W)
        if np.allclose(reconstructed, T, atol=1e-10):
            return {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }
    
    return None


def _greedy_peel(T, dim_u, dim_v, dim_w, deadline, max_terms):
    """Greedy rank-1 peeling with sign-based updates. Returns dict or None."""
    R = T.copy()
    U_list = []
    V_list = []
    W_list = []
    
    for term_idx in range(max_terms):
        if time.time() > deadline:
            break
        if not np.any(R):
            break
        
        best_score = 0
        best_u, best_v, best_w = None, None, None
        
        n_restarts = 8
        for restart in range(n_restarts):
            if time.time() > deadline:
                break
            
            if restart == 0:
                cu = np.sign(np.sum(R, axis=(1, 2)))
                cv = np.sign(np.sum(R, axis=(0, 2)))
                cw = np.sign(np.sum(R, axis=(0, 1)))
            elif restart == 1:
                cu = np.random.choice([-1, 1], size=dim_u).astype(np.float64)
                cv = np.random.choice([-1, 1], size=dim_v).astype(np.float64)
                cw = np.random.choice([-1, 1], size=dim_w).astype(np.float64)
            else:
                cu = np.random.choice([-1, 0, 1], size=dim_u).astype(np.float64)
                cv = np.random.choice([-1, 0, 1], size=dim_v).astype(np.float64)
                cw = np.random.choice([-1, 0, 1], size=dim_w).astype(np.float64)
            
            n_iters = 15
            for it in range(n_iters):
                if time.time() > deadline:
                    break
                
                M_u = np.einsum('abc,b,c->a', R, cv, cw)
                new_u = np.sign(M_u)
                
                M_v = np.einsum('abc,a,c->b', R, new_u, cw)
                new_v = np.sign(M_v)
                
                M_w = np.einsum('abc,a,b->c', R, new_u, new_v)
                new_w = np.sign(M_w)
                
                cu, cv, cw = new_u, new_v, new_w
            
            score = np.einsum('abc,a,b,c', R, cu, cv, cw)
            if score > best_score:
                best_score = score
                best_u, best_v, best_w = cu.copy(), cv.copy(), cw.copy()
        
        if best_score <= 0 or best_u is None:
            break
        
        R -= np.einsum('a,b,c->abc', best_u, best_v, best_w)
        U_list.append(best_u)
        V_list.append(best_v)
        W_list.append(best_w)
        
        if np.allclose(R, 0, atol=1e-10):
            break
    
    if np.allclose(R, 0, atol=1e-10) and len(U_list) > 0:
        rank = len(U_list)
        U = np.array(U_list).T
        V = np.array(V_list).T
        W = np.array(W_list).T
        
        reconstructed = np.einsum('ak,bk,ck->abc', U, V, W)
        if np.allclose(reconstructed, T, atol=1e-10):
            return {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }
    
    return None


def _strassen_2x2x2():
    """Return Strassen decomposition of 2x2x2 matrix multiplication.
    
    Returns U, V, W each of shape (4, 7) such that
    sum_k U[a,k]*V[b,k]*W[c,k] = T[a,b,c]
    where T[i*2+j, j*2+k, k*2+i] = 1
    """
    # Strassen's 7 products for 2x2 matrix multiplication
    # A = [[a11, a12], [a21, a22]], B = [[b11, b12], [b21, b22]]
    # C = A*B
    
    # Strassen products:
    # M1 = (a11+a22)(b11+b22)
    # M2 = (a21+a22)b11
    # M3 = a11(b12-b22)
    # M4 = a22(b21-b11)
    # M5 = (a11+a12)b22
    # M6 = (a21-a11)(b11+b12)
    # M7 = (a12-a22)(b21+b22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # U vectors (for A indices: a11=0, a12=1, a21=2, a22=3)
    # V vectors (for B indices: b11=0, b12=1, b21=2, b22=3)
    # W vectors (for C indices: c11=0, c12=1, c21=2, c22=3)
    
    U = np.zeros((4, 7), dtype=np.float64)
    V = np.zeros((4, 7), dtype=np.float64)
    W = np.zeros((4, 7), dtype=np.float64)
    
    # M1 = (a11+a22)(b11+b22)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1  # C11 and C22
    
    # M2 = (a21+a22)b11
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1  # C21 += M2, C22 -= M2
    
    # M3 = a11(b12-b22)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1  # C12 += M3, C22 += M3
    
    # M4 = a22(b21-b11)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1  # C11 += M4, C21 += M4
    
    # M5 = (a11+a12)b22
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1  # C11 -= M5, C12 += M5
    
    # M6 = (a21-a11)(b11+b12)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1  # C22 += M6
    
    # M7 = (a12-a22)(b21+b22)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1  # C11 += M7
    
    return U, V, W


def _schoolbook(n, m, p):
    """Return schoolbook decomposition of n x m times m x p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
