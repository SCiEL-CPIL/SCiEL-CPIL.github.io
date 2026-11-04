import time
import sys
import numpy as np

DIMENSIONS = [2, 4, 6]

def construct():
    start_time = time.time()
    deadline = 145.0

    n, m, p = DIMENSIONS  # 2, 4, 6
    nm, mp, pn = n * m, m * p, p * n  # 8, 24, 12

    def verify(U_mat, V_mat, W_mat, tol=1e-8):
        r = U_mat.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    s = 0.0
                    for r_idx in range(r):
                        s += U_mat[a, r_idx] * V_mat[b, r_idx] * W_mat[c, r_idx]
                    if abs(s - 1.0) > tol:
                        return False
        
        valid_set = set()
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    valid_set.add((i*m+j, j*p+k, k*n+i))
                    
        for a in range(nm):
            for b in range(mp):
                for c in range(pn):
                    if (a,b,c) in valid_set:
                        continue
                    s = 0.0
                    for r_idx in range(r):
                        s += U_mat[a, r_idx] * V_mat[b, r_idx] * W_mat[c, r_idx]
                    if abs(s) > tol:
                        return False
        return True

    # 1. Construct Rank 42 (Strassen blocks)
    rank42 = 42
    U42 = np.zeros((nm, rank42), dtype=float)
    V42 = np.zeros((mp, rank42), dtype=float)
    W42 = np.zeros((pn, rank42), dtype=float)

    def add_strassen_block(i_start, j_start, k_start, offset):
        a00 = i_start * m + j_start
        a01 = i_start * m + j_start + 1
        a10 = (i_start + 1) * m + j_start
        a11 = (i_start + 1) * m + j_start + 1
        
        b00 = j_start * p + k_start
        b01 = j_start * p + k_start + 1
        b10 = (j_start + 1) * p + k_start
        b11 = (j_start + 1) * p + k_start + 1
        
        w00 = k_start * n + i_start
        w01 = (k_start + 1) * n + i_start
        w10 = k_start * n + i_start + 1
        w11 = (k_start + 1) * n + i_start + 1
        
        U42[a00, offset] += 1.0; U42[a11, offset] += 1.0
        V42[b00, offset] += 1.0; V42[b11, offset] += 1.0
        W42[w00, offset] += 1.0; W42[w11, offset] += 1.0

        U42[a10, offset+1] += 1.0; U42[a11, offset+1] += 1.0
        V42[b00, offset+1] += 1.0
        W42[w10, offset+1] += 1.0; W42[w11, offset+1] -= 1.0

        U42[a00, offset+2] += 1.0
        V42[b01, offset+2] += 1.0; V42[b11, offset+2] -= 1.0
        W42[w01, offset+2] += 1.0; W42[w11, offset+2] += 1.0

        U42[a11, offset+3] += 1.0
        V42[b10, offset+3] += 1.0; V42[b00, offset+3] -= 1.0
        W42[w00, offset+3] += 1.0; W42[w10, offset+3] += 1.0

        U42[a00, offset+4] += 1.0; U42[a01, offset+4] += 1.0
        V42[b11, offset+4] += 1.0
        W42[w00, offset+4] -= 1.0; W42[w01, offset+4] += 1.0

        U42[a10, offset+5] += 1.0; U42[a00, offset+5] -= 1.0
        V42[b00, offset+5] += 1.0; V42[b01, offset+5] += 1.0
        W42[w11, offset+5] += 1.0

        U42[a01, offset+6] += 1.0; U42[a11, offset+6] -= 1.0
        V42[b10, offset+6] += 1.0; V42[b11, offset+6] += 1.0
        W42[w00, offset+6] += 1.0

    offset = 0
    for i_start in range(0, n, 2):
        for j_start in range(0, m, 2):
            for k_start in range(0, p, 2):
                add_strassen_block(i_start, j_start, k_start, offset)
                offset += 7

    if verify(U42, V42, W42):
        # Attempt to reduce rank from 42
        best_U, best_V, best_W = U42.copy(), V42.copy(), W42.copy()
        best_rank = 42
        
        changed = True
        while changed:
            changed = False
            current_rank = best_U.shape[1]
            if current_rank <= 24:
                break
            
            merged = False
            cols_to_keep = []
            used = [False] * current_rank
            
            for i in range(current_rank):
                if used[i]:
                    continue
                
                merge_found = False
                for j in range(i + 1, current_rank):
                    if used[j]:
                        continue
                    
                    nz_u_i = np.nonzero(best_U[:, i])[0]
                    nz_u_j = np.nonzero(best_U[:, j])[0]
                    if len(nz_u_i) == 0 or len(nz_u_j) == 0:
                        if len(nz_u_i) != len(nz_u_j):
                            continue
                        if len(nz_u_i) > 0:
                            continue
                    else:
                        if not np.array_equal(nz_u_i, nz_u_j):
                            continue
                        ratios_u = best_U[nz_u_i, i] / best_U[nz_u_i, j]
                        if np.std(ratios_u) > 1e-10:
                            continue
                        s = ratios_u[0]
                    
                    nz_v_i = np.nonzero(best_V[:, i])[0]
                    nz_v_j = np.nonzero(best_V[:, j])[0]
                    if len(nz_v_i) == 0 or len(nz_v_j) == 0:
                        if len(nz_v_i) != len(nz_v_j):
                            continue
                        if len(nz_v_i) > 0:
                            continue
                    else:
                        if not np.array_equal(nz_v_i, nz_v_j):
                            continue
                        ratios_v = best_V[nz_v_i, i] / best_V[nz_v_i, j]
                        if np.std(ratios_v) > 1e-10:
                            continue
                        t = ratios_v[0]
                    
                    scale = 1.0 / (s * t)
                    best_W[:, i] += scale * best_W[:, j]
                    used[j] = True
                    merge_found = True
                    changed = True
                    break
                
                if not used[i]:
                    cols_to_keep.append(i)
            
            if merge_found:
                new_rank = len(cols_to_keep)
                best_U = best_U[:, cols_to_keep]
                best_V = best_V[:, cols_to_keep]
                best_W = best_W[:, cols_to_keep]
                
                if not verify(best_U, best_V, best_W):
                    best_U, best_V, best_W = U42.copy(), V42.copy(), W42.copy()
                    break
                else:
                    best_rank = new_rank
                    print(f"Merged to rank {best_rank}", file=sys.stderr)
            else:
                break
        
        if best_rank < 42 and verify(best_U, best_V, best_W):
            return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    else:
        rank0 = n * m * p
        U = np.zeros((nm, rank0), dtype=float)
        V = np.zeros((mp, rank0), dtype=float)
        W = np.zeros((pn, rank0), dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    col = i * (m * p) + j * p + k
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    result = construct()
    print("Constructed", file=sys.stderr)
