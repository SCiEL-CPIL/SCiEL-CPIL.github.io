import numpy as np
import time
# ---------------------------------------------------------------------------
# Deadline guard
# ---------------------------------------------------------------------------
_EVAL_DEADLINE = 150.0          # wall-clock seconds (hard limit)
_START = time.time()
TIME_BUDGET = _EVAL_DEADLINE - (time.time() - _START) - 5.0   # leave 5 s margin

def _time_left():
    return _EVAL_DEADLINE - 5.0 - time.time()

# ---------------------------------------------------------------------------
# Build the multiplication tensor T in [r,i,j] layout
#   r = k*n + i  (n*p values),  i is the row of A,  j = row of B / col of B
# ---------------------------------------------------------------------------
def _build_tensor(n, m, p):
    r_dim = p * n
    i_dim = n
    j_dim = m * p
    T = np.zeros((r_dim, i_dim, j_dim))
    for k in range(p):
        for i in range(n):
            for j in range(m):
                r = k * n + i
                jj = j * p + k
                T[r, i, jj] = 1.0
    return T

# ---------------------------------------------------------------------------
# Known rank-50 construction (Strassen blocks + direct rank-1)
# ---------------------------------------------------------------------------
def _construct_rank50(n=2, m=4, p=7):
    rank = 50
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]

    col = [0]

    def add_strassen(i0, j0, k0):
        c = col[0]
        # M1
        U[i0*m + j0][c] += 1
        U[(i0+1)*m + (j0+1)][c] += 1
        V[j0*p + k0][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        W[k0*n + i0][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1
        c += 1
        # M2
        U[(i0+1)*m + j0][c] += 1
        U[(i0+1)*m + (j0+1)][c] += 1
        V[j0*p + k0][c] += 1
        W[k0*n + (i0+1)][c] += 1
        W[(k0+1)*n + (i0+1)][c] -= 1
        c += 1
        # M3
        U[i0*m + j0][c] += 1
        V[j0*p + (k0+1)][c] += 1
        V[(j0+1)*p + (k0+1)][c] -= 1
        W[(k0+1)*n + i0][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1
        c += 1
        # M4
        U[(i0+1)*m + (j0+1)][c] += 1
        V[(j0+1)*p + k0][c] += 1
        V[j0*p + k0][c] -= 1
        W[k0*n + i0][c] += 1
        W[k0*n + (i0+1)][c] += 1
        c += 1
        # M5
        U[i0*m + j0][c] += 1
        U[i0*m + (j0+1)][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        W[k0*n + i0][c] -= 1
        W[(k0+1)*n + i0][c] += 1
        c += 1
        # M6
        U[(i0+1)*m + j0][c] += 1
        U[i0*m + j0][c] -= 1
        V[j0*p + k0][c] += 1
        V[j0*p + (k0+1)][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1
        c += 1
        # M7
        U[i0*m + (j0+1)][c] += 1
        U[(i0+1)*m + (j0+1)][c] -= 1
        V[(j0+1)*p + k0][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        W[k0*n + i0][c] += 1
        c += 1
        col[0] = c

    # 2x4x4 block: four 2x2x2 Strassen blocks
    add_strassen(0, 0, 0)
    add_strassen(0, 0, 2)
    add_strassen(0, 2, 0)
    add_strassen(0, 2, 2)
    # 2x4x2 block: two 2x2x2 Strassen blocks
    add_strassen(0, 0, 4)
    add_strassen(0, 2, 4)
    
    # 2x4x1 block: 8 direct terms
    for i in range(n):
        for j in range(m):
            U[i*m + j][col[0]] = 1.0
            V[j*p + 6][col[0]] = 1.0
            W[6*n + i][col[0]] = 1.0
            col[0] += 1

    assert col[0] == rank, f"Rank mismatch: used {col[0]}, expected {rank}"
    return {"U": U, "V": V, "W": W}

# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------
def _verify(U, V, W, n, m, p, tol=1e-10):
    r_dim = p * n
    i_dim = n
    j_dim = m * p
    T = _build_tensor(n, m, p)
    T_mat = T.reshape(r_dim, i_dim * j_dim)
    rank = len(W[0])
    Ur = np.array(U).T   # (rank, i_dim)
    Vr = np.array(V).T   # (rank, j_dim)
    Wr = np.array(W).T   # (rank, r_dim)
    cand = np.zeros_like(T_mat)
    for t in range(rank):
        wt = Wr[t, :]          # (r_dim,)
        ut = Ur[t, :]          # (n,)
        vt = Vr[t, :]          # (j_dim,)
        outer = ut[:, None] * vt[None, :]   # (n, j_dim)
        cand += np.outer(wt, outer.ravel())
    max_err = np.max(np.abs(cand - T_mat))
    return max_err < tol

# ---------------------------------------------------------------------------
# Rank reduction search
# ---------------------------------------------------------------------------
def _try_rank_reduction(base_sol, n, m, p, target_rank, time_budget):
    r_dim = p * n
    i_dim = n
    j_dim = m * p
    
    U_base = np.array(base_sol["U"])  # (i_dim, rank)
    V_base = np.array(base_sol["V"])  # (j_dim, rank)
    W_base = np.array(base_sol["W"])  # (r_dim, rank)
    
    T = _build_tensor(n, m, p)
    T_flat = T.reshape(-1)
    
    rank = len(W_base[0])
    
    if rank <= target_rank:
        return base_sol

    if rank - target_rank != 1:
        return None

    # Pre-compute tensor vectors
    terms = []
    for t in range(rank):
        u = U_base[:, t]
        v = V_base[:, t]
        w = W_base[:, t]
        term_vec = np.outer(w, np.outer(u, v).flatten())
        terms.append(term_vec)
    
    terms = np.array(terms) # (rank, total_dim)
    total_dim = r_dim * i_dim * j_dim
    
    for drop_idx in range(rank):
        if _time_left() < 5:
            break
        
        keep_indices = [i for i in range(rank) if i != drop_idx]
        A = terms[keep_indices, :].T # (total_dim, rank-1)
        
        try:
            c, residuals, rank_A, s = np.linalg.lstsq(A, T_flat, rcond=None)
        except:
            continue
            
        valid = True
        for val in c:
            two_val = 2 * val
            if abs(two_val - round(two_val)) > 1e-6:
                valid = False
                break
        
        if valid:
            U_new = np.zeros((i_dim, rank-1))
            V_new = np.zeros((j_dim, rank-1))
            W_new = np.zeros((r_dim, rank-1))
            
            for new_idx, old_idx in enumerate(keep_indices):
                U_new[:, new_idx] = U_base[:, old_idx]
                V_new[:, new_idx] = V_base[:, old_idx]
                W_new[:, new_idx] = W_base[:, old_idx]
                
            for new_idx in range(rank-1):
                U_new[:, new_idx] *= c[new_idx]
                
            cand = np.zeros(total_dim)
            for t in range(rank-1):
                u = U_new[:, t]
                v = V_new[:, t]
                w = W_new[:, t]
                cand += np.outer(w, np.outer(u, v).flatten())
                
            if np.max(np.abs(cand - T_flat)) < 1e-6:
                U_list = [[0.0] * (rank-1) for _ in range(i_dim)]
                V_list = [[0.0] * (rank-1) for _ in range(j_dim)]
                W_list = [[0.0] * (rank-1) for _ in range(r_dim)]
                for i in range(i_dim):
                    for t in range(rank-1):
                        U_list[i][t] = U_new[i, t]
                for j in range(j_dim):
                    for t in range(rank-1):
                        V_list[j][t] = V_new[j, t]
                for r in range(r_dim):
                    for t in range(rank-1):
                        W_list[r][t] = W_new[r, t]
                        
                return {"U": U_list, "V": V_list, "W": W_list}
                
    return None

# ---------------------------------------------------------------------------
# Main construct
# ---------------------------------------------------------------------------
def construct():
    n, m, p = 2, 4, 7

    fallback = _construct_rank50(n, m, p)

    if _time_left() > 10:
        result = _try_rank_reduction(fallback, n, m, p, 49, min(_time_left(), 40.0))
        if result is not None:
            if _verify(result["U"], result["V"], result["W"], n, m, p):
                return result

    return fallback
