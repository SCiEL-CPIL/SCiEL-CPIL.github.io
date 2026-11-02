import time
import numpy as np
import random

DIMENSIONS = [3, 3, 5]
DEADLINE = 150.0
SAFETY_MARGIN = 10.0

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS  # 3, 3, 5
    
    def _remaining():
        return DEADLINE - SAFETY_MARGIN - (time.time() - start_time)
    
    def _verify(U, V, W, rank):
        """Verify exact decomposition. Returns True if valid."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    def _make_schoolbook():
        """Return schoolbook decomposition of rank n*m*p = 45."""
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1.0
                    V[j*p+k, col] = 1.0
                    W[k*n+i, col] = 1.0
                    col += 1
        return U, V, W, rank

    def _try_merge(U, V, W, rank, r1, r2):
        """
        Try to merge two terms into one.
        Returns (new_U, new_V, new_W, new_rank) or None.
        """
        u1 = U[:, r1]
        u2 = U[:, r2]
        v1 = V[:, r1]
        v2 = V[:, r2]
        w1 = W[:, r1]
        w2 = W[:, r2]
        
        # Target tensor M
        M = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    M[i,j,k] = u1[i*m+j]*v1[j*p+k]*w1[k*n+i] + u2[i*m+j]*v2[j*p+k]*w2[k*n+i]
        
        # Try to find u_new, v_new, w_new such that
        # u_new[i*m+j] * v_new[j*p+k] * w_new[k*n+i] = M[i,j,k]
        
        # Heuristic: Try w_new = w1 + w2, w1 - w2, etc.
        candidates = [w1 + w2, w1 - w2, w2 - w1, w1, w2]
        
        for w_new in candidates:
            if np.all(np.abs(w_new) < 1e-12):
                continue
            
            # Check if M / w_new is rank 1 in the appropriate sense
            # For fixed i, A_i(j,k) = M[i,j,k] / w_new[k*n+i] must be rank 1
            valid = True
            u_new = np.zeros(n*m)
            v_new = np.zeros(m*p)
            
            # Determine v_new from i=0
            if np.any(np.abs(w_new[0::n]) > 1e-12):
                ref_i = 0
            elif np.any(np.abs(w_new[1::n]) > 1e-12):
                ref_i = 1
            elif np.any(np.abs(w_new[2::n]) > 1e-12):
                ref_i = 2
            else:
                continue
            
            for j in range(m):
                for k in range(p):
                    if abs(w_new[k*n+ref_i]) > 1e-12:
                        v_new[j*p+k] = M[ref_i, j, k] / w_new[k*n+ref_i]
            
            # Determine u_new
            for i in range(n):
                for j in range(m):
                    vals = []
                    for k in range(p):
                        if abs(w_new[k*n+i]) > 1e-12 and abs(v_new[j*p+k]) > 1e-12:
                            vals.append(M[i,j,k] / (w_new[k*n+i] * v_new[j*p+k]))
                    if not vals:
                        u_new[i*m+j] = 0.0
                    else:
                        if max(vals) - min(vals) > 1e-9:
                            valid = False
                            break
                        u_new[i*m+j] = vals[0]
                if not valid:
                    break
            
            if not valid:
                continue
            
            # Verify
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        if abs(u_new[i*m+j]*v_new[j*p+k]*w_new[k*n+i] - M[i,j,k]) > 1e-9:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                new_rank = rank - 1
                new_U = np.zeros((n*m, new_rank))
                new_V = np.zeros((m*p, new_rank))
                new_W = np.zeros((p*n, new_rank))
                
                col_map = [r for r in range(rank) if r != r1 and r != r2]
                for new_col, old_col in enumerate(col_map):
                    new_U[:, new_col] = U[:, old_col]
                    new_V[:, new_col] = V[:, old_col]
                    new_W[:, new_col] = W[:, old_col]
                
                new_U[:, new_rank-1] = u_new
                new_V[:, new_rank-1] = v_new
                new_W[:, new_rank-1] = w_new
                
                return new_U, new_V, new_W, new_rank
        
        return None

    U, V, W, rank = _make_schoolbook()
    best_U, best_V, best_W, best_rank = U.copy(), V.copy(), W.copy(), rank
    
    # Try to reduce rank
    for _ in range(100): # Limit iterations
        if _remaining() < 5.0:
            break
        
        merged = False
        for r1 in range(rank):
            if _remaining() < 5.0:
                break
            for r2 in range(r1+1, rank):
                if _remaining() < 5.0:
                    break
                
                res = _try_merge(U, V, W, rank, r1, r2)
                if res:
                    U, V, W, rank = res
                    if rank < best_rank:
                        best_U, best_V, best_W, best_rank = U.copy(), V.copy(), W.copy(), rank
                    merged = True
                    break
            if merged:
                break
        
        if not merged:
            break

    if _verify(best_U, best_V, best_W, best_rank):
        return {
            "U": best_U.tolist(),
            "V": best_V.tolist(),
            "W": best_W.tolist()
        }
    else:
        U, V, W, rank = _make_schoolbook()
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
