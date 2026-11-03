import time
import numpy as np
import random

DIMENSIONS = [4, 4, 7]
WALL_DEADLINE = 145.0
SAFETY_MARGIN = 5.0

def construct():
    start_time = time.time()
    n, m, p = 4, 4, 7

    def elapsed():
        return time.time() - start_time

    def check_valid(U, V, W, tol=1e-8):
        r = len(U[0])
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        # Vectorized check:
        # Sum over c of U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # We can reshape and use einsum
        # U: (n*m, r), V: (m*p, r), W: (p*n, r)
        # We want T[i,j,k] = sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c]
        
        # Reshape to (n, m, r), (m, p, r), (p, n, r)
        U3 = U_arr.reshape(n, m, r)
        V3 = V_arr.reshape(m, p, r)
        W3 = W_arr.reshape(p, n, r)
        
        # Compute tensor
        # T[i, j, k] = sum_c U3[i, j, c] V3[j, k, c] W3[k, i, c]
        # Use einsum
        T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
        
        # Target is 1 for all i, j, k
        target = np.ones((n, m, p))
        
        if not np.allclose(T, target, atol=tol):
            return False
        return True

    def build_strassen_2x2_2x2():
        # Returns U, V, W for 2x2x2 with rank 7
        U = [[0] * 7 for _ in range(4)]
        V = [[0] * 7 for _ in range(4)]
        W = [[0] * 7 for _ in range(4)]
        
        # Standard Strassen
        # c1 = (a11+a22)(b11+b22)
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        W[0][0] = 1; W[3][0] = 1
        
        # c2 = (a21+a22)b11
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[1][1] = 1; W[3][1] = -1
        
        # c3 = a11(b21-b22)
        U[0][2] = 1
        V[1][2] = 1; V[3][2] = -1
        W[2][2] = 1; W[3][2] = 1
        
        # c4 = a22(b11-b21)
        U[3][3] = 1
        V[2][3] = 1; V[0][3] = -1
        W[0][3] = 1; W[1][3] = 1
        
        # c5 = (a11+a12)b22
        U[0][4] = 1; U[1][4] = 1
        V[3][4] = 1
        W[0][4] = -1; W[2][4] = 1
        
        # c6 = (a21-a11)(b11+b12)
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        W[3][5] = 1
        
        # c7 = (a12-a22)(b21+b22)
        U[1][6] = 1; U[3][6] = -1
        V[2][6] = 1; V[3][6] = 1
        W[0][6] = 1
        
        return U, V, W

    def build_4x4x4_kron():
        # Kronecker product of two 2x2x2 Strassen decompositions
        U2, V2, W2 = build_strassen_2x2_2x2()
        r2 = 7
        r4 = r2 * r2 # 49
        
        U4 = np.zeros((16, r4))
        V4 = np.zeros((16, r4))
        W4 = np.zeros((16, r4))
        
        U2_arr = np.array(U2)
        V2_arr = np.array(V2)
        W2_arr = np.array(W2)
        
        for r1 in range(r2):
            for r2_idx in range(r2):
                r = r1 * r2 + r2_idx
                # U4[i1*4+j1, r1*7+r2_idx] += U2[i1*2+j1, r1] * U2[i1*2+j1... wait indices]
                # Actually, the Kronecker product of tensors T1 and T2:
                # T[i1,i2, j1,j2, k1,k2] = T1[i1,j1,k1] * T2[i2,j2,k2]
                # U4[(i1*2+i2)*4 + (j1*2+j2), r] = U2[i1*2+j1, r1] * U2[i2*2+j2, r2_idx]
                
                for i1 in range(2):
                    for j1 in range(2):
                        for i2 in range(2):
                            for j2 in range(2):
                                i = i1 * 2 + i2
                                j = j1 * 2 + j2
                                val = U2_arr[i1 * 2 + j1, r1] * U2_arr[i2 * 2 + j2, r2_idx]
                                if val != 0:
                                    U4[i * 4 + j, r] += val
                                
                for j1 in range(2):
                    for k1 in range(2):
                        for j2 in range(2):
                            for k2 in range(2):
                                j = j1 * 2 + j2
                                k = k1 * 2 + k2
                                val = V2_arr[j1 * 2 + k1, r1] * V2_arr[j2 * 2 + k2, r2_idx]
                                if val != 0:
                                    V4[j * 4 + k, r] += val
                                    
                for k1 in range(2):
                    for i1 in range(2):
                        for k2 in range(2):
                            for i2 in range(2):
                                k = k1 * 2 + k2
                                i = i1 * 2 + i2
                                val = W2_arr[k1 * 2 + i1, r1] * W2_arr[k2 * 2 + i2, r2_idx]
                                if val != 0:
                                    W4[k * 4 + i, r] += val
                                    
        return U4, V4, W4

    def build_4x4x3_optimized():
        # Decompose 4x4x3 into four 2x2x3 blocks
        # Each 2x2x3 block has rank 11 (7 Strassen for 2x2x2 part + 4 Schoolbook for 2x2x1 part)
        # Total rank 44
        
        r_block = 11
        r_total = 4 * r_block # 44
        
        U = np.zeros((16, r_total))
        V = np.zeros((12, r_total)) # 4 rows * 3 cols = 12
        W = np.zeros((12, r_total)) # 3 rows * 4 cols = 12
        
        # Build one 2x2x3 block
        # Indices for block:
        # U_block: (4, 11) -> U[i*2+j]
        # V_block: (6, 11) -> V[j*3+k]
        # W_block: (6, 11) -> W[k*2+i]
        
        U22 = np.zeros((4, r_block))
        V22 = np.zeros((6, r_block))
        W22 = np.zeros((6, r_block))
        
        # Part 1: 2x2x2 Strassen (k=0,1)
        U_s, V_s, W_s = build_strassen_2x2_2x2()
        U22[:, :7] = np.array(U_s)
        # V_s is (4,7) for 2x2x2. We need to embed into V22 (6,11)
        # V22[j*3+k] for k=0,1.
        # j=0: V22[0], V22[1]
        # j=1: V22[3], V22[4]
        V_s_arr = np.array(V_s)
        V22[0, :7] = V_s_arr[0]
        V22[1, :7] = V_s_arr[1]
        V22[3, :7] = V_s_arr[2]
        V22[4, :7] = V_s_arr[3]
        
        # W_s is (4,7) for 2x2x2. Embed into W22 (6,11)
        # W22[k*2+i] for k=0,1.
        # k=0: W22[0], W22[1]
        # k=1: W22[2], W22[3]
        W_s_arr = np.array(W_s)
        W22[0, :7] = W_s_arr[0]
        W22[1, :7] = W_s_arr[1]
        W22[2, :7] = W_s_arr[2]
        W22[3, :7] = W_s_arr[3]
        
        # Part 2: 2x2x1 Schoolbook (k=2)
        # Rank 4
        # U[i*2+j], V[j*3+2], W[2*2+i]
        # j=0, k=2 -> V[2]
        # j=1, k=2 -> V[5]
        # k=2, i=0 -> W[4]
        # k=2, i=1 -> W[5]
        
        # Term 1: i=0, j=0
        U22[0, 7] = 1
        V22[2, 7] = 1
        W22[4, 7] = 1
        
        # Term 2: i=0, j=1
        U22[1, 8] = 1
        V22[5, 8] = 1
        W22[4, 8] = 1
        
        # Term 3: i=1, j=0
        U22[2, 9] = 1
        V22[2, 9] = 1
        W22[5, 9] = 1
        
        # Term 4: i=1, j=1
        U22[3, 10] = 1
        V22[5, 10] = 1
        W22[5, 10] = 1
        
        # Place 4 blocks
        for bi in range(2):
            for bj in range(2):
                offset = (bi * 2 + bj) * r_block
                for i2 in range(2):
                    for j2 in range(2):
                        i = bi * 2 + i2
                        j = bj * 2 + j2
                        U[i * 4 + j, offset:offset+r_block] = U22[i2 * 2 + j2]
                        
                for j2 in range(2):
                    for k2 in range(3):
                        j = bj * 2 + j2
                        V[j * 7 + k2, offset:offset+r_block] = V22[j2 * 3 + k2]
                        
                for k2 in range(3):
                    for i2 in range(2):
                        i = bi * 2 + i2
                        W[k2 * 4 + i, offset:offset+r_block] = W22[k2 * 2 + i2]
                        
        return U, V, W

    def build_4x4x7():
        U4, V4, W4 = build_4x4x4_kron()
        U3, V3, W3 = build_4x4x3_optimized()
        
        r4 = U4.shape[1]
        r3 = U3.shape[1]
        r_total = r4 + r3
        
        U = np.zeros((16, r_total))
        V = np.zeros((28, r_total))
        W = np.zeros((28, r_total))
        
        # Copy 4x4x4 part
        U[:, :r4] = U4
        # V4 is (16, r4) for 4x4x4. We need V for 4x7.
        # V[j*7+k]. For k=0..3, it's V4[j*4+k].
        for j in range(4):
            for k in range(4):
                V[j*7+k, :r4] = V4[j*4+k]
        
        # W4 is (16, r4) for 4x4x4. W[k*4+i].
        for k in range(4):
            for i in range(4):
                W[k*4+i, :r4] = W4[k*4+i]
                
        # Copy 4x4x3 part
        U[:, r4:r4+r3] = U3
        # V3 is (12, r3) for 4x4x3. V3[j*3+k].
        # We need to map to V[j*7+k] for k=4..6.
        # Wait, the 4x4x3 part handles the last 3 columns of the output matrix (k=4,5,6).
        # In the 4x4x3 block construction, k2 ranges 0..2.
        # So V3[j*3+k2] corresponds to the global V[j*7 + (4+k2)].
        for j in range(4):
            for k2 in range(3):
                V[j*7 + 4 + k2, r4:r4+r3] = V3[j*3+k2]
                
        # W3 is (12, r3) for 4x4x3. W3[k2*4+i].
        # Global W[k*4+i] where k = 4+k2.
        for k2 in range(3):
            for i in range(4):
                W[(4+k2)*4+i, r4:r4+r3] = W3[k2*4+i]
                
        return U, V, W

    def residual_norm(U, V, W):
        r = U.shape[1]
        U3 = U.reshape(n, m, r)
        V3 = V.reshape(m, p, r)
        W3 = W.reshape(p, n, r)
        T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
        target = np.ones((n, m, p))
        return np.linalg.norm(T - target)

    def refine_decomposition(U, V, W, max_iter=50, tol=1e-10):
        # Gauss-Newton / Levenberg-Marquardt style refinement
        # Variables: x = [U.flatten(), V.flatten(), W.flatten()]
        # We want to minimize || T(x) - 1 ||^2
        
        x = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        r = U.shape[1]
        u_len = n*m*r
        v_len = m*p*r
        w_len = p*n*r
        
        for it in range(max_iter):
            if elapsed() > WALL_DEADLINE - SAFETY_MARGIN:
                break
                
            # Compute residual
            U_c = x[:u_len].reshape(n*m, r)
            V_c = x[u_len:u_len+v_len].reshape(m*p, r)
            W_c = x[u_len+v_len:].reshape(p*n, r)
            
            U3 = U_c.reshape(n, m, r)
            V3 = V_c.reshape(m, p, r)
            W3 = W_c.reshape(p, n, r)
            T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
            res = (T - np.ones((n, m, p))).flatten()
            
            if np.linalg.norm(res) < tol:
                break
            
            # Compute Jacobian
            # J has shape (n*m*p, u_len+v_len+w_len)
            # This is expensive to build fully, but n*m*p = 112.
            # u_len+v_len+w_len = 16r + 28r + 28r = 72r.
            # For r=93, dim = 6696. J is 112x6696.
            # We can compute it efficiently.
            
            J = np.zeros((n*m*p, u_len + v_len + w_len))
            
            # dT[i,j,k] / dU[i*m+j, c] = V[j*p+k, c] * W[k*n+i, c]
            # dT[i,j,k] / dV[j*p+k, c] = U[i*m+j, c] * W[k*n+i, c]
            # dT[i,j,k] / dW[k*n+i, c] = U[i*m+j, c] * V[j*p+k, c]
            
            # We can vectorize this using broadcasting or loops.
            # Given the small size, a loop over c is fine.
            for c in range(r):
                # U contribution
                # U[:, c] is (n*m,). We need to map to (n, m).
                Uc = U_c[:, c].reshape(n, m)
                Vc = V_c[:, c].reshape(m, p)
                Wc = W_c[:, c].reshape(p, n)
                
                # J[row, col]
                # For each i,j,k:
                # row = i*m*p + j*p + k
                # col_U = i*m + j (offset 0)
                # col_V = j*p + k (offset u_len)
                # col_W = k*n + i (offset u_len+v_len)
                
                # Term for U: V[j,k,c] * W[k,i,c]
                # We need to fill J[:, col_U] with this term for all k.
                # Actually, J[row, col] where row=(i,j,k) and col=(i,j) for U.
                # The value is Vc[j,k] * Wc[k,i].
                
                # Let's construct the contributions.
                # U_part: shape (n, m, p) where U_part[i,j,k] = Vc[j,k] * Wc[k,i]
                # V_part: shape (n, m, p) where V_part[i,j,k] = Uc[i,j] * Wc[k,i]
                # W_part: shape (n, m, p) where W_part[i,j,k] = Uc[i,j] * Vc[j,k]
                
                U_part = np.einsum('jk,ki->ijk', Vc, Wc)
                V_part = np.einsum('ij,ki->ijk', Uc, Wc)
                W_part = np.einsum('ij,jk->ijk', Uc, Vc)
                
                # Flatten and assign to J
                # J is (n*m*p, total_dim)
                # U_part.flatten() -> J[:, 0:u_len]
                # V_part.flatten() -> J[:, u_len:u_len+v_len]
                # W_part.flatten() -> J[:, u_len+v_len:]
                
                J[:, c*r:c*r + r] = 0 # This is wrong indexing. J columns are not grouped by c in this way simply.
                # Wait, x is flattened.
                # x[0:u_len] is U.flatten(). U.flatten() is row-major.
                # U[i*m+j, c] is at index (i*m+j)*r + c.
                # So column index for U[i*m+j, c] is (i*m+j)*r + c.
                
                # Let's re-fill J properly.
                # J[row, col]
                # row = i*m*p + j*p + k
                # col_U = (i*m+j)*r + c
                # col_V = u_len + (j*p+k)*r + c
                # col_W = u_len + v_len + (k*n+i)*r + c
                
                # This loop is too slow if done element by element.
                # But n*m*p=112, r~93.
                # 112 * 93 * 3 ~ 30k operations per iteration. 50 iterations = 1.5M ops. Fast.
                
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            row = i*m*p + j*p + k
                            val_u = Vc[j, k] * Wc[k, i]
                            val_v = Uc[i, j] * Wc[k, i]
                            val_w = Uc[i, j] * Vc[j, k]
                            
                            col_u = (i*m+j)*r + c
                            col_v = u_len + (j*p+k)*r + c
                            col_w = u_len + v_len + (k*n+i)*r + c
                            
                            J[row, col_u] = val_u
                            J[row, col_v] = val_v
                            J[row, col_w] = val_w

            # Solve J dx = -res
            # Use least squares
            try:
                dx, *_ = np.linalg.lstsq(J, -res, rcond=None)
            except Exception:
                break
                
            # Line search
            alpha = 1.0
            improved = False
            for _ in range(20):
                x_trial = x + alpha * dx
                U_t = x_trial[:u_len].reshape(n*m, r)
                V_t = x_trial[u_len:u_len+v_len].reshape(m*p, r)
                W_t = x_trial[u_len+v_len:].reshape(p*n, r)
                
                res_trial = residual_norm(U_t, V_t, W_t)
                if res_trial < np.linalg.norm(res):
                    improved = True
                    break
                alpha *= 0.5
                
            if not improved:
                break
                
            x = x + alpha * dx

        U_final = x[:u_len].reshape(n*m, r)
        V_final = x[u_len:u_len+v_len].reshape(m*p, r)
        W_final = x[u_len+v_len:].reshape(p*n, r)
        
        if residual_norm(U_final, V_final, W_final) < 1e-8:
            return U_final, V_final, W_final
        return None

    def try_reduce_rank(U, V, W, target_rank, rng):
        # Try to project to target_rank and refine
        r_current = U.shape[1]
        if r_current <= target_rank:
            return U, V, W
            
        # Random projection
        # P: (r_current, target_rank)
        # We want U_new = U @ P.T
        # To preserve rank, P should have full column rank.
        # Random Gaussian projection is good.
        
        for attempt in range(5):
            P = rng.standard_normal((r_current, target_rank))
            # Normalize columns to avoid scaling issues
            P /= np.linalg.norm(P, axis=0, keepdims=True)
            
            U_new = U @ P.T
            V_new = V @ P.T
            W_new = W @ P.T
            
            # Refine
            res = refine_decomposition(U_new, V_new, W_new, max_iter=30, tol=1e-9)
            if res is not None:
                return res[0], res[1], res[2]
                
        return None

    # Main flow
    U, V, W = build_4x4x7()
    best_rank = U.shape[1]
    best_U, best_V, best_W = U, V, W
    
    if not check_valid(best_U, best_V, best_W):
        # Fallback to schoolbook if construction failed (unlikely)
        r_school = n*m*p
        U_s = np.zeros((n*m, r_school))
        V_s = np.zeros((m*p, r_school))
        W_s = np.zeros((p*n, r_school))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    c = i*m*p + j*p + k
                    U_s[i*m+j, c] = 1
                    V_s[j*p+k, c] = 1
                    W_s[k*n+i, c] = 1
        return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}

    rng = random.Random(12345)
    
    # Attempt to reduce rank
    current_rank = best_rank
    while current_rank > 70 and elapsed() < WALL_DEADLINE - SAFETY_MARGIN:
        target = current_rank - 1
        # Try a few times to reduce by 1
        success = False
        for _ in range(3):
            if elapsed() > WALL_DEADLINE - SAFETY_MARGIN:
                break
            res = try_reduce_rank(best_U, best_V, best_W, target, rng)
            if res is not None:
                best_U, best_V, best_W = res
                current_rank = target
                success = True
                break
        
        if not success:
            # Try reducing by 2
            target2 = current_rank - 2
            if target2 > 70:
                res = try_reduce_rank(best_U, best_V, best_W, target2, rng)
                if res is not None:
                    best_U, best_V, best_W = res
                    current_rank = target2
                    success = True
            
            if not success:
                break

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
