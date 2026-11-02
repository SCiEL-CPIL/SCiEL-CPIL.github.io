import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 120  # Safety margin for 120s limit

    n, m, p = 2, 3, 9

    # Build the rank-50 decomposition
    rank = 50
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0
    u_a, u_b, u_c, u_d = 0, 1, 3, 4

    for b in range(4):
        k0 = 2 * b
        k1 = 2 * b + 1
        v_e = k0
        v_f = k1
        v_g = p + k0
        v_h = p + k1

        U[u_a][col] += 1
        U[u_d][col] += 1
        V[v_e][col] += 1
        V[v_h][col] += 1

        U[u_c][col+1] += 1
        U[u_d][col+1] += 1
        V[v_e][col+1] += 1

        U[u_a][col+2] += 1
        V[v_f][col+2] += 1
        V[v_h][col+2] -= 1

        U[u_d][col+3] += 1
        V[v_g][col+3] += 1
        V[v_e][col+3] -= 1

        U[u_a][col+4] += 1
        U[u_b][col+4] += 1
        V[v_h][col+4] += 1

        U[u_c][col+5] += 1
        U[u_a][col+5] -= 1
        V[v_e][col+5] += 1
        V[v_f][col+5] += 1

        U[u_b][col+6] += 1
        U[u_d][col+6] -= 1
        V[v_g][col+6] += 1
        V[v_h][col+6] += 1

        w_00 = k0 * n + 0
        w_01 = k1 * n + 0
        w_10 = k0 * n + 1
        w_11 = k1 * n + 1

        W[w_00][col] += 1
        W[w_00][col+3] += 1
        W[w_00][col+4] -= 1
        W[w_00][col+6] += 1

        W[w_01][col+2] += 1
        W[w_01][col+4] += 1

        W[w_10][col+1] += 1
        W[w_10][col+3] += 1

        W[w_11][col] += 1
        W[w_11][col+1] -= 1
        W[w_11][col+2] += 1
        W[w_11][col+5] += 1

        col += 7

    k8 = 8
    v_e = k8
    v_g = p + k8

    U[u_a][col] += 1
    V[v_e][col] += 1
    U[u_b][col+1] += 1
    V[v_g][col+1] += 1
    U[u_c][col+2] += 1
    V[v_e][col+2] += 1
    U[u_d][col+3] += 1
    V[v_g][col+3] += 1

    w_08 = k8 * n + 0
    w_18 = k8 * n + 1

    W[w_08][col] += 1
    W[w_08][col+1] += 1
    W[w_18][col+2] += 1
    W[w_18][col+3] += 1

    col += 4

    u_ar0, u_ar1 = 2, 5
    for k in range(9):
        v_b = 2 * p + k
        U[u_ar0][col] += 1
        V[v_b][col] += 1
        W[k * n + 0][col] += 1
        U[u_ar1][col+1] += 1
        V[v_b][col+1] += 1
        W[k * n + 1][col+1] += 1
        col += 2

    # Convert to numpy
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)

    # Build the target tensor T
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n*m):
        ai = i // m
        am = i % m
        for j in range(m*p):
            bj = j // p
            bm = j % p
            if am == bj:
                k = bm * n + ai
                T[i, j, k] = 1.0

    # Efficient verification function
    def verify(U_new, V_new, W_new):
        """Verify decomposition using efficient matrix operations"""
        # For each rank-1 term r, compute the contribution
        # T[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
        # We can compute this as: for each i,j, the vector over k is
        # sum_r U[i,r] * V[j,r] * W[:,r]
        # Which is (U[i,:] * V[j,:]) @ W.T
        # So T[i,j,:] = (U[i,:] * V[j,:]) @ W.T
        
        # Residual computation
        # T[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
        # Let M[i,j,r] = U[i,r] * V[j,r]
        # Then T[i,j,k] = sum_r M[i,j,r] * W[k,r] = (M[i,j,:] @ W[k,:])
        
        # More efficiently: T[i,j,:] = (U[i,:] * V[j,:]) @ W.T
        # But we need to check all i,j,k
        
        # Alternative: compute the full tensor and compare
        # T_computed[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
        # This is a 3-way sum, can be done via einsum
        T_computed = np.einsum('ir,jr,kr->ijk', U_new, V_new, W_new)
        residual = np.sum((T_computed - T) ** 2)
        return residual

    # Try to find linear dependencies and reduce rank
    # Form the matrix where each column is a flattened rank-1 tensor
    # The flattened tensor has size n*m*m*p*p*n = 6*27*18 = 2916
    mat = np.zeros((n*m*m*p*p*n, rank))
    for c in range(rank):
        # Flatten u_c ⊗ v_c ⊗ w_c
        for i in range(n*m):
            for j in range(m*p):
                for k in range(p*n):
                    mat[(i*m*p + j)*p*n + k, c] = U_np[i,c] * V_np[j,c] * W_np[k,c]

    # Compute SVD to find the rank
    try:
        _, s, Vt = np.linalg.svd(mat, full_matrices=False)
        # Find the rank (number of significant singular values)
        tol = s[0] * 50 * 1e-10
        num_rank = np.sum(s > tol)
        
        if num_rank < rank:
            # There's a linear dependency
            # Try to drop columns iteratively
            U_cur = U_np.copy()
            V_cur = V_np.copy()
            W_cur = W_np.copy()
            dropped = set()
            
            # Try to drop one column at a time
            for _ in range(rank - num_rank):
                if time.time() > deadline - 5:
                    break
                    
                # Recompute SVD on remaining columns
                remaining = [c for c in range(rank) if c not in dropped]
                if len(remaining) <= num_rank:
                    break
                    
                mat_cur = mat[:, remaining]
                try:
                    _, s_cur, Vt_cur = np.linalg.svd(mat_cur, full_matrices=False)
                    tol_cur = s_cur[0] * len(remaining) * 1e-10
                    num_rank_cur = np.sum(s_cur > tol_cur)
                    
                    if num_rank_cur < len(remaining):
                        null_vec = Vt_cur[-1]
                        # Find the column with the largest absolute value in null_vec
                        drop_idx = np.argmax(np.abs(null_vec))
                        drop_col = remaining[drop_idx]
                        
                        # Remove that column
                        U_cur = np.delete(U_cur, drop_col, axis=1)
                        V_cur = np.delete(V_cur, drop_col, axis=1)
                        W_cur = np.delete(W_cur, drop_col, axis=1)
                        dropped.add(drop_col)
                    else:
                        break
                except:
                    break
            
            # Verify the reduced decomposition
            if len(dropped) > 0:
                residual = verify(U_cur, V_cur, W_cur)
                if residual < 1e-8:
                    return {
                        "U": U_cur.tolist(),
                        "V": V_cur.tolist(),
                        "W": W_cur.tolist()
                    }
    except:
        pass

    # Fall back to rank-50
    return {"U": U, "V": V, "W": W}
