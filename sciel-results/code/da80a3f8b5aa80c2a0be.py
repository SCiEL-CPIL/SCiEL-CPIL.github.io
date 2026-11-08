import time
import numpy as np

DIMENSIONS = [3, 4, 6]

def _build_known_66():
    n, m, p = DIMENSIONS
    rank = 66
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    current_rank = 0

    def add_rank1(u_idx, u_vals, v_idx, v_vals, w_idx, w_vals):
        nonlocal current_rank
        if current_rank >= rank:
            return
        for idx, val in zip(u_idx, u_vals):
            U[idx][current_rank] += val
        for idx, val in zip(v_idx, v_vals):
            V[idx][current_rank] += val
        for idx, val in zip(w_idx, w_vals):
            W[idx][current_rank] += val
        current_rank += 1

    def add_1_4_2(i_off, j_off, k_off):
        for j in range(4):
            for k in range(2):
                u_idx = (i_off) * m + (j_off + j)
                v_idx = (j_off + j) * p + (k_off + k)
                w_idx = (k_off + k) * n + i_off
                add_rank1([u_idx], [1], [v_idx], [1], [w_idx], [1])

    def add_2_2_2(i_off, j_off, k_off):
        def u(i, j): return (i_off + i) * m + (j_off + j)
        def v(j, k): return (j_off + j) * p + (k_off + k)
        def w(i, k): return (k_off + k) * n + (i_off + i)
        add_rank1([u(0,0), u(1,1)], [1, 1], [v(0,0), v(1,1)], [1, 1], [w(0,0), w(1,1)], [1, 1])
        add_rank1([u(1,0), u(1,1)], [1, 1], [v(0,0)], [1], [w(1,0), w(1,1)], [1, -1])
        add_rank1([u(0,0)], [1], [v(0,1), v(1,1)], [1, -1], [w(0,1), w(1,1)], [1, 1])
        add_rank1([u(1,1)], [1], [v(1,0), v(0,0)], [1, -1], [w(0,0), w(1,0)], [1, 1])
        add_rank1([u(0,0), u(0,1)], [1, 1], [v(1,1)], [1], [w(0,0), w(0,1)], [-1, 1])
        add_rank1([u(1,0), u(0,0)], [1, -1], [v(0,0), v(0,1)], [1, 1], [w(1,1)], [1])
        add_rank1([u(0,1), u(1,1)], [1, -1], [v(1,0), v(1,1)], [1, 1], [w(0,0)], [1])

    def add_2_4_2(i_off, j_off, k_off):
        add_2_2_2(i_off, j_off, k_off)
        add_2_2_2(i_off, j_off + 2, k_off)

    for block_k in range(3):
        k_off = block_k * 2
        add_1_4_2(0, 0, k_off)
        add_2_4_2(1, 0, k_off)

    return {"U": U, "V": V, "W": W}


def _validate(U, V, W, n, m, p):
    Ua = np.asarray(U, dtype=float)
    Va = np.asarray(V, dtype=float)
    Wa = np.asarray(W, dtype=float)
    R = Ua.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for t in range(R):
                    s += Ua[i * m + j, t] * Va[j * p + k, t] * Wa[k * n + i, t]
                if abs(s - 1.0) > 1e-6:
                    return False
    return True


def _round_to_half(x):
    return np.round(2.0 * x) / 2.0


def _try_svd_compression(U, V, W, n, m, p, deadline):
    """Compress using SVD on the tensor's mode-1 unfolding."""
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    R0 = Ua.shape[1]
    
    # Build the full tensor T[i,j,k] = sum_t U[i*m+j,t] * V[j*p+k,t] * W[k*n+i,t]
    T = np.zeros((n, m, p))
    for t in range(R0):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, j, k] += Ua[i * m + j, t] * Va[j * p + k, t] * Wa[k * n + i, t]
    
    # Mode-1 unfolding: shape (n, m*p)
    T1 = T.reshape(n, m * p)
    U1, s1, V1 = np.linalg.svd(T1, full_matrices=False)
    
    # Mode-2 unfolding: shape (m, n*p)
    T2 = T.transpose(1, 0, 2).reshape(m, n * p)
    U2, s2, V2 = np.linalg.svd(T2, full_matrices=False)
    
    # Mode-3 unfolding: shape (p, n*m)
    T3 = T.transpose(2, 0, 1).reshape(p, n * m)
    U3, s3, V3 = np.linalg.svd(T3, full_matrices=False)
    
    # Try different rank truncations
    best_result = None
    for target_rank in [R0 - 1, R0 - 2, R0 - 3, R0 - 4, R0 - 5, R0 - 6, R0 - 8, R0 - 10, R0 - 12, R0 - 15]:
        if target_rank < 1:
            break
        if time.time() > deadline - 10:
            break
        
        # Use the smallest singular value threshold to determine effective rank
        # Try to find a rank that works
        try:
            # For each mode, determine the rank needed
            r1 = min(target_rank, len(s1))
            r2 = min(target_rank, len(s2))
            r3 = min(target_rank, len(s3))
            
            # Reconstruct T with truncated SVD
            T1_trunc = U1[:, :r1] * s1[:r1] * V1[:r1, :]
            T2_trunc = U2[:, :r2] * s2[:r2] * V2[:r2, :]
            T3_trunc = U3[:, :r3] * s3[:r3] * V3[:r3, :]
            
            # Check if the truncated tensors match
            T1_check = T1_trunc.reshape(n, m, p)
            T2_check = T2_trunc.reshape(m, n, p).transpose(1, 0, 2)
            T3_check = T3_trunc.reshape(p, n, m).transpose(1, 2, 0)
            
            if (np.allclose(T1_check, T, atol=1e-8) and 
                np.allclose(T2_check, T, atol=1e-8) and 
                np.allclose(T3_check, T, atol=1e-8)):
                # Now we need to find U_new, V_new, W_new with rank = target_rank
                # such that sum_t U_new[i*m+j,t] * V_new[j*p+k,t] * W_new[k*n+i,t] = T[i,j,k]
                
                # Use the mode-1 SVD components
                # U1[:, :r1] gives us the left singular vectors
                # We can set U_new[i*m+j, t] = U1[i, t] * something
                # This is complex; let's try a different approach
                
                # Actually, let's just try to solve for one factor given the other two
                # Fix U and V, solve for W
                if target_rank < R0:
                    # We want to reduce from R0 to target_rank
                    # Keep the first target_rank columns of U and V
                    U_keep = Ua[:, :target_rank]
                    V_keep = Va[:, :target_rank]
                    
                    # Solve for W_keep
                    n_vars = target_rank * p * n
                    n_eqs = n * m * p
                    A = np.zeros((n_eqs, n_vars))
                    b = np.ones(n_eqs)
                    
                    eq_idx = 0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                for t in range(target_rank):
                                    coeff = U_keep[i * m + j, t] * V_keep[j * p + k, t]
                                    var_idx = t * (p * n) + (k * n + i)
                                    A[eq_idx, var_idx] += coeff
                                eq_idx += 1
                    
                    x, residuals, rank_a, sv = np.linalg.lstsq(A, b, rcond=None)
                    
                    if len(residuals) > 0 and residuals[0] < 1e-8:
                        W_keep = np.zeros((p * n, target_rank))
                        for t in range(target_rank):
                            for k in range(p):
                                for i in range(n):
                                    W_keep[k * n + i, t] = x[t * (p * n) + (k * n + i)]
                        
                        # Round to half-integers
                        W_rounded = _round_to_half(W_keep)
                        
                        # Validate
                        if _validate(U_keep.tolist(), V_keep.tolist(), W_rounded.tolist(), n, m, p):
                            if best_result is None or target_rank < best_result[0]:
                                best_result = (target_rank, U_keep.tolist(), V_keep.tolist(), W_rounded.tolist())
                            return best_result
        except Exception:
            continue
    
    return best_result


def _try_column_dropping(U, V, W, n, m, p, deadline):
    """Try dropping columns one by one and re-solving."""
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    R0 = Ua.shape[1]
    
    current_U = Ua.copy()
    current_V = Va.copy()
    current_W = Wa.copy()
    current_R = R0
    best_rank = R0
    best_result = None
    
    # Try dropping from the end
    for drop_count in range(1, min(15, R0)):
        if time.time() > deadline - 15:
            break
        
        target_R = R0 - drop_count
        if target_R < 1:
            break
        
        # Keep the first target_R columns
        U_keep = current_U[:, :target_R]
        V_keep = current_V[:, :target_R]
        
        # Solve for W
        n_vars = target_R * p * n
        n_eqs = n * m * p
        A = np.zeros((n_eqs, n_vars))
        b = np.ones(n_eqs)
        
        eq_idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for t in range(target_R):
                        coeff = U_keep[i * m + j, t] * V_keep[j * p + k, t]
                        var_idx = t * (p * n) + (k * n + i)
                        A[eq_idx, var_idx] += coeff
                    eq_idx += 1
        
        try:
            x, residuals, rank_a, sv = np.linalg.lstsq(A, b, rcond=None)
            
            if len(residuals) > 0 and residuals[0] < 1e-8:
                W_keep = np.zeros((p * n, target_R))
                for t in range(target_R):
                    for k in range(p):
                        for i in range(n):
                            W_keep[k * n + i, t] = x[t * (p * n) + (k * n + i)]
                
                W_rounded = _round_to_half(W_keep)
                
                if _validate(U_keep.tolist(), V_keep.tolist(), W_rounded.tolist(), n, m, p):
                    best_rank = target_R
                    best_result = (target_R, U_keep.tolist(), V_keep.tolist(), W_rounded.tolist())
                    # Continue trying to drop more
                else:
                    # Rounding broke it, try without rounding
                    if _validate(U_keep.tolist(), V_keep.tolist(), W_keep.tolist(), n, m, p):
                        # Use the unrounded version but note it may not have half-integer entries
                        # The task says "integer/half-integer entries", so we need rounding
                        # If rounding fails, we can't use this
                        pass
                    break
            else:
                break
        except Exception:
            break
    
    return best_result


def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 115.0
    
    known = _build_known_66()
    U, V, W = known["U"], known["V"], known["W"]
    
    if not _validate(U, V, W, n, m, p):
        return known
    
    best_rank = 66
    best_U, best_V, best_W = U, V, W
    
    # Try SVD-based compression
    if time.time() < deadline - 30:
        result = _try_svd_compression(U, V, W, n, m, p, deadline)
        if result is not None:
            rank, U_new, V_new, W_new = result
            if rank < best_rank:
                best_rank = rank
                best_U, best_V, best_W = U_new, V_new, W_new
    
    # Try column dropping
    if time.time() < deadline - 20:
        result = _try_column_dropping(best_U, best_V, best_W, n, m, p, deadline)
        if result is not None:
            rank, U_new, V_new, W_new = result
            if rank < best_rank:
                best_rank = rank
                best_U, best_V, best_W = U_new, V_new, W_new
    
    return {"U": best_U, "V": best_V, "W": best_W}


if __name__ == "__main__":
    res = construct()
    print("rank:", len(res["U"][0]))
