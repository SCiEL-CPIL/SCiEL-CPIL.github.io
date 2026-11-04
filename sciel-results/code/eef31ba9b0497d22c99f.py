import numpy as np
import time

def construct():
    # Dimensions
    n, m, p = 2, 3, 9
    
    # Target Tensor T[a,b,c] for M_{2,3,9}
    # a = i*m + j
    # b = j*p + k
    # c = k*n + i
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0

    # Schoolbook fallback
    rank_schoolbook = n * m * p
    U_sb = np.zeros((n*m, rank_schoolbook))
    V_sb = np.zeros((m*p, rank_schoolbook))
    W_sb = np.zeros((p*n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
                
    def to_list(mat):
        return [[float(x) for x in row] for row in mat]

    best_rank = rank_schoolbook
    best_U, best_V, best_W = U_sb, V_sb, W_sb

    deadline = time.time() + 135

    # Strategy: Decompose M_{2,3,3} first, then expand to M_{2,3,9}
    # M_{2,3,3} has n=2, m=3, p=3
    n_s, m_s, p_s = 2, 3, 3
    T_s = np.zeros((n_s*m_s, m_s*p_s, p_s*n_s))
    for i in range(n_s):
        for j in range(m_s):
            for k in range(p_s):
                T_s[i*m_s+j, j*p_s+k, k*n_s+i] = 1.0

    def als_step_small(U, V, W, T, r):
        nm, mp, pn = n_s*m_s, m_s*p_s, p_s*n_s
        # Update U
        for a in range(nm):
            M_a = (V[:, None, :] * W[None, :, :]).reshape(mp * pn, r)
            target = T[a, :, :].flatten()
            try:
                U[a, :], _, _, _ = np.linalg.lstsq(M_a.T, target, rcond=None)
            except:
                pass
        # Update V
        for b in range(mp):
            M_b = (U[:, None, :] * W[None, :, :]).reshape(nm * pn, r)
            target = T[:, b, :].reshape(nm, pn).flatten()
            try:
                V[b, :], _, _, _ = np.linalg.lstsq(M_b.T, target, rcond=None)
            except:
                pass
        # Update W
        for c in range(pn):
            M_c = (U[:, None, :] * V[None, :, :]).reshape(nm * mp, r)
            target = T[:, :, c].reshape(nm, mp).flatten()
            try:
                W[c, :], _, _, _ = np.linalg.lstsq(M_c.T, target, rcond=None)
            except:
                pass
        return U, V, W

    def compute_error_small(U, V, W, T):
        T_approx = np.einsum('ar,br,cr->abc', U, V, W)
        return np.sum((T_approx - T)**2)

    # Find minimal rank for M_{2,3,3}
    best_rank_s = n_s * m_s * p_s
    best_U_s, best_V_s, best_W_s = None, None, None
    
    # Try ranks from low to high
    # Known upper bound is 15, schoolbook is 18.
    for r in range(10, 19):
        if time.time() > deadline - 20:
            break
        if r >= best_rank_s:
            continue
            
        success = False
        for trial in range(20):
            if time.time() > deadline - 20:
                break
            np.random.seed(42 + r * 100 + trial)
            U = np.random.randn(n_s*m_s, r)
            V = np.random.randn(m_s*p_s, r)
            W = np.random.randn(p_s*n_s, r)
            
            # Normalize
            for c in range(r):
                norm = np.sqrt(np.sum(U[:, c]**2) + np.sum(V[:, c]**2) + np.sum(W[:, c]**2))
                if norm > 1e-10:
                    U[:, c] /= norm
                    V[:, c] /= norm
                    W[:, c] /= norm
                    
            prev_error = compute_error_small(U, V, W, T_s)
            
            for it in range(100):
                if time.time() > deadline - 20:
                    break
                U, V, W = als_step_small(U, V, W, T_s, r)
                error = compute_error_small(U, V, W, T_s)
                if error < 1e-8:
                    success = True
                    break
                if error > 1e6:
                    break
                if abs(error - prev_error) < 1e-12:
                    if error < 1e-6:
                        success = True
                    break
                prev_error = error
                
            if success:
                best_rank_s = r
                best_U_s, best_V_s, best_W_s = U, V, W
                break
                
        if best_rank_s == r:
            break

    # Construct M_{2,3,9} from M_{2,3,3}
    if best_U_s is not None:
        rank_candidate = 3 * best_rank_s
        if rank_candidate < best_rank:
            # U is repeated 3 times horizontally
            U_new = np.hstack([best_U_s, best_U_s, best_U_s])
            
            # V is block diagonal
            # V_s is (m_s * p_s, r) = (9, r)
            # V_new is (m * p, 3*r) = (27, 3*r)
            # Indices in V_new: row = j*p + k
            # Block 0 (k=0,1,2): uses V_s rows corresponding to k=0,1,2
            # V_s row index is j*p_s + k.
            # We need to map V_s to V_new.
            # V_new[0:9, 0:r] = V_s
            # V_new[9:18, r:2r] = V_s
            # V_new[18:27, 2r:3r] = V_s
            
            V_new = np.zeros((m*p, 3*best_rank_s))
            for block in range(3):
                # Rows for this block in V_new
                # j ranges 0..2, k ranges block*3 .. block*3+2
                # Row index in V_new: j*9 + (block*3 + k_local)
                # Row index in V_s: j*3 + k_local
                # We can just copy V_s into the block
                # The rows of V_s are indexed by j*3+k.
                # The rows of V_new are indexed by j*9+k.
                # For a fixed block, the k indices are offset.
                # But the structure of V_s depends on j and k.
                # V_s[j*3+k, c] is the coefficient for B_{j,k}.
                # In V_new, B_{j, block*3+k} corresponds to V_new[j*9 + block*3 + k, block*r + c].
                # So V_new[j*9 + block*3 + k, block*r + c] = V_s[j*3 + k, c].
                
                # Let's just fill it
                for j in range(m_s):
                    for k in range(p_s):
                        row_new = j * p + (block * p_s + k)
                        row_s = j * p_s + k
                        V_new[row_new, block*best_rank_s : (block+1)*best_rank_s] = V_s[row_s, :]
                        
            # W is block diagonal
            # W_s is (p_s * n_s, r) = (9, r)
            # W_new is (p * n, 3*r) = (18, 3*r)
            # Indices in W_new: row = k*n + i
            # Block 0 (k=0,1,2): uses W_s rows corresponding to k=0,1,2
            # W_new[0:6, 0:r] = W_s
            # W_new[6:12, r:2r] = W_s
            # W_new[12:18, 2r:3r] = W_s
            
            W_new = np.zeros((p*n, 3*best_rank_s))
            for block in range(3):
                for k in range(p_s):
                    for i in range(n_s):
                        row_new = (block * p_s + k) * n + i
                        row_s = k * n_s + i
                        W_new[row_new, block*best_rank_s : (block+1)*best_rank_s] = W_s[row_s, :]
                        
            # Verify
            T_approx = np.einsum('ar,br,cr->abc', U_new, V_new, W_new)
            error = np.sum((T_approx - T)**2)
            if error < 1e-6:
                best_rank = rank_candidate
                best_U, best_V, best_W = U_new, V_new, W_new

    # If we found a candidate, try to polish it or reduce rank further using ALS on the full tensor
    # But given time constraints, we might just return the block construction.
    # The block construction is exact if the small one is exact.
    
    U_list = to_list(best_U)
    V_list = to_list(best_V)
    W_list = to_list(best_W)
    
    return {"U": U_list, "V": V_list, "W": W_list}
