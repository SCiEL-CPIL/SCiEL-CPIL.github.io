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

    # Try rank 12: two 2x2x3 blocks, each rank 6
    # Split m=4 into two blocks of 2: j in {0,1} and j in {2,3}
    # For each block, we have A (2x2) * B (2x3) -> C (2x3)
    
    rank12 = 12
    U12 = np.zeros((nm, rank12), dtype=float)
    V12 = np.zeros((mp, rank12), dtype=float)
    W12 = np.zeros((pn, rank12), dtype=float)
    
    # 2x2x3 rank-6 decomposition
    # Using a known pattern:
    # M1 = (A00+A11)(B00+B10) -> C00 += 1, C10 += 1, C02 -= 1, C12 -= 1
    # M2 = (A00+A11)B20 -> C02 += 1, C12 += 1
    # M3 = A00(B01-B11) -> C01 += 1, C11 -= 1
    # M4 = A11(B11-B01) -> C01 -= 1, C11 += 1
    # M5 = (A10-A00)(B00+B10) -> C10 += 1, C00 -= 1, C12 += 1, C02 -= 1
    # M6 = (A10-A00)B20 -> C12 += 1, C02 -= 1
    # Wait, this doesn't seem right. Let me use a verified approach.
    
    # Actually, let me just use the Strassen rank-42 which is known to work,
    # then try to reduce via linear algebra.
    
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
        
        U42[a00, offset] += 1; U42[a11, offset] += 1
        V42[b00, offset] += 1; V42[b11, offset] += 1
        W42[w00, offset] += 1; W42[w11, offset] += 1

        U42[a10, offset+1] += 1; U42[a11, offset+1] += 1
        V42[b00, offset+1] += 1
        W42[w10, offset+1] += 1; W42[w11, offset+1] -= 1

        U42[a00, offset+2] += 1
        V42[b01, offset+2] += 1; V42[b11, offset+2] -= 1
        W42[w01, offset+2] += 1; W42[w11, offset+2] += 1

        U42[a11, offset+3] += 1
        V42[b10, offset+3] += 1; V42[b00, offset+3] -= 1
        W42[w00, offset+3] += 1; W42[w10, offset+3] += 1

        U42[a00, offset+4] += 1; U42[a01, offset+4] += 1
        V42[b11, offset+4] += 1
        W42[w00, offset+4] -= 1; W42[w01, offset+4] += 1

        U42[a10, offset+5] += 1; U42[a00, offset+5] -= 1
        V42[b00, offset+5] += 1; V42[b01, offset+5] += 1
        W42[w11, offset+5] += 1

        U42[a01, offset+6] += 1; U42[a11, offset+6] -= 1
        V42[b10, offset+6] += 1; V42[b11, offset+6] += 1
        W42[w00, offset+6] += 1

    offset = 0
    for i_start in range(0, n, 2):
        for j_start in range(0, m, 2):
            for k_start in range(0, p, 2):
                add_strassen_block(i_start, j_start, k_start, offset)
                offset += 7

    best_U, best_V, best_W = U42.copy(), V42.copy(), W42.copy()
    best_rank = 42

    if not verify(best_U, best_V, best_W):
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

    # Try to reduce rank by finding linear dependencies in the tensor decomposition
    # Each term is U[:,r] ⊗ V[:,r] ⊗ W[:,r]
    # Flatten and check rank
    
    def flatten_term(U_col, V_col, W_col):
        return np.outer(np.outer(U_col, V_col), W_col).flatten()
    
    terms = [flatten_term(best_U[:, r], best_V[:, r], best_W[:, r]) for r in range(best_rank)]
    terms_matrix = np.column_stack(terms)
    
    # Find rank via SVD
    try:
        _, s_vals, _ = np.linalg.svd(terms_matrix, full_matrices=False)
        tol = 1e-10
        significant = np.sum(s_vals > tol)
    except:
        significant = best_rank
    
    # If there are dependencies, try to eliminate columns
    # Use QR to find independent set
    Q, R = np.linalg.qr(terms_matrix)
    diag_R = np.abs(np.diag(R))
    independent = diag_R > 1e-8
    n_ind = np.sum(independent)
    
    if n_ind < best_rank:
        # Keep only independent columns - but this may not preserve the decomposition
        # We need a more careful approach: express dependent terms as combos
        # For now, just note the rank reduction potential
        pass
    
    # Try pairwise merging of proportional columns
    def try_merge(U_mat, V_mat, W_mat, tol=1e-10):
        r = U_mat.shape[1]
        keep = [True] * r
        merged = False
        
        for i in range(r):
            if not keep[i]:
                continue
            for j in range(i+1, r):
                if not keep[j]:
                    continue
                # Check if U_i ∝ U_j and V_i ∝ V_j
                nz_u_i = np.nonzero(np.abs(U_mat[:, i]) > tol)[0]
                nz_u_j = np.nonzero(np.abs(U_mat[:, j]) > tol)[0]
                if len(nz_u_i) == 0 or len(nz_u_j) == 0:
                    continue
                if not np.array_equal(nz_u_i, nz_u_j):
                    continue
                ratios_u = U_mat[nz_u_i, i] / U_mat[nz_u_j, j]
                if np.std(ratios_u) > tol:
                    continue
                su = ratios_u[0]
                if abs(su) < tol:
                    continue
                
                nz_v_i = np.nonzero(np.abs(V_mat[:, i]) > tol)[0]
                nz_v_j = np.nonzero(np.abs(V_mat[:, j]) > tol)[0]
                if len(nz_v_i) == 0 or len(nz_v_j) == 0:
                    continue
                if not np.array_equal(nz_v_i, nz_v_j):
                    continue
                ratios_v = V_mat[nz_v_i, i] / V_mat[nz_v_j, j]
                if np.std(ratios_v) > tol:
                    continue
                sv = ratios_v[0]
                if abs(sv) < tol:
                    continue
                
                # U_j = (1/su) U_i, V_j = (1/sv) V_i
                # term_j = U_j⊗V_j⊗W_j = (1/(su*sv)) U_i⊗V_i⊗W_j
                # So add (1/(su*sv)) * W_j to W_i, remove j
                scale = 1.0 / (su * sv)
                W_mat[:, i] = W_mat[:, i] + scale * W_mat[:, j]
                keep[j] = False
                merged = True
        
        if not merged:
            return U_mat, V_mat, W_mat, r
        kept = [i for i in range(r) if keep[i]]
        return U_mat[:, kept], V_mat[:, kept], W_mat[:, kept], len(kept)
    
    changed = True
    while changed and time.time() < deadline - 5:
        new_U, new_V, new_W, new_rank = try_merge(best_U, best_V, best_W)
        if new_rank < best_rank:
            if verify(new_U, new_V, new_W):
                best_U, best_V, best_W = new_U, new_V, new_W
                best_rank = new_rank
                changed = True
            else:
                changed = False
        else:
            changed = False

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

if __name__ == "__main__":
    result = construct()
    print("Constructed", file=sys.stderr)
