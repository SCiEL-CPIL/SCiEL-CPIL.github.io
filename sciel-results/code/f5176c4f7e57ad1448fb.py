import numpy as np
import time

DIMENSIONS = [3, 5, 7]

def construct():
    n, m, p = DIMENSIONS
    # Target tensor T of shape (n*m, m*p, p*n) implicitly
    # We check validity by ensuring:
    # 1. For all i, j, k: sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] == 1
    # 2. For all inconsistent indices, sum == 0
    
    schoolbook_rank = n * m * p
    best_rank = schoolbook_rank
    best_U = None
    best_V = None
    best_W = None
    
    start_time = time.time()
    deadline = 120.0

    def is_valid(U, V, W):
        # U: (n*m, r), V: (m*p, r), W: (p*n, r)
        r = U.shape[1]
        # Check consistent indices
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    # Sum over c
                    s = np.sum(U[u_idx, :] * V[v_idx, :] * W[w_idx, :])
                    if abs(s - 1) > 1e-9:
                        return False
        # Check inconsistent indices (sample)
        # To be fast, we check a random subset of inconsistent indices
        # An index is inconsistent if the j in U != j in V
        # U index i*m+j, V index j'*p+k. Inconsistent if j != j'
        # Also W index k*n+i must match i from U and k from V.
        # If j != j', it's automatically inconsistent for the product to be non-zero in schoolbook,
        # but in a mixed decomposition, we need the sum to be 0.
        
        # We can check all inconsistent pairs? 
        # Number of triples (u_idx, v_idx, w_idx) is nm * mp * pn = 15 * 35 * 21 = 11025.
        # For each, we check if it's consistent.
        # Consistent if:
        # u_idx = i*m+j, v_idx = j*p+k, w_idx = k*n+i
        # So j from u must equal j from v.
        # i from u must equal i from w (w is k*n+i).
        # k from v must equal k from w.
        
        for u_idx in range(n * m):
            i_u = u_idx // m
            j_u = u_idx % m
            for v_idx in range(m * p):
                j_v = v_idx // p
                k_v = v_idx % p
                if j_u != j_v:
                    # Inconsistent j
                    # We need sum_c U[u_idx, c] V[v_idx, c] W[w_idx, c] == 0 for all w_idx?
                    # No, W is fixed in the sum.
                    # The condition is that the tensor entry T[u_idx, v_idx, w_idx] is 0.
                    # T[u_idx, v_idx, w_idx] = sum_c U[u_idx, c] V[v_idx, c] W[w_idx, c]
                    # This must be 0 for all w_idx if (u_idx, v_idx) is inconsistent.
                    # Actually, if j_u != j_v, then for ANY w_idx, the entry should be 0.
                    # Let's check one w_idx to save time? No, must be 0 for all.
                    # But if it's 0 for all w_idx, it's 0 for any.
                    # We can check w_idx = 0.
                    # Wait, if sum_c U[u,c] V[v,c] W[w,c] = 0 for all w,
                    # then the vector (sum_c U[u,c] V[v,c] W[:,c]) is 0.
                    # So we just need to check if this vector is 0.
                    vec = np.sum(U[u_idx, :, None] * V[v_idx, :, None] * W, axis=1)
                    if np.any(np.abs(vec) > 1e-9):
                        return False
                else:
                    # j matches. i from u is i_u, k from v is k_v.
                    # Consistent w_idx must be k_v * n + i_u.
                    # For other w_idx, it must be 0.
                    w_consistent = k_v * n + i_u
                    # Check consistent entry
                    s = np.sum(U[u_idx, :] * V[v_idx, :] * W[w_consistent, :])
                    if abs(s - 1) > 1e-9:
                        return False
                    # Check inconsistent w_idx for this u, v
                    # There are n*p - 1 such w_idx.
                    # We can check if the vector is zero except at w_consistent.
                    vec = np.sum(U[u_idx, :, None] * V[v_idx, :, None] * W, axis=1)
                    vec[w_consistent] = 0
                    if np.any(np.abs(vec) > 1e-9):
                        return False
        return True

    # Try to find a better rank
    # We'll try a few random restarts with low rank
    for r in range(schoolbook_rank - 1, 50, -5):
        if time.time() > deadline - 10:
            break
        
        for attempt in range(10):
            if time.time() > deadline - 5:
                break
            
            # Initialize with random half-integers
            U = np.random.randint(-2, 3, size=(n*m, r)) / 2.0
            V = np.random.randint(-2, 3, size=(m*p, r)) / 2.0
            W = np.random.randint(-2, 3, size=(p*n, r)) / 2.0
            
            # Simple refinement: gradient descent?
            # Too slow.
            # Just check if valid.
            if is_valid(U, V, W):
                best_rank = r
                best_U = U
                best_V = V
                best_W = W
                break
        if best_rank < schoolbook_rank:
            break

    if best_rank < schoolbook_rank:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    else:
        # Return schoolbook
        U = [[0] * schoolbook_rank for _ in range(n * m)]
        V = [[0] * schoolbook_rank for _ in range(m * p)]
        W = [[0] * schoolbook_rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
