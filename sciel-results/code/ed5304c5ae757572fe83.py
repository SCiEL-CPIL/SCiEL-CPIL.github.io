"""Matrix multiplication 2x4x8 via Strassen block decomposition with dependency reduction."""
import time
import numpy as np

DIMENSIONS = [2, 4, 8]

def construct():
    deadline = time.time() + 110
    n, m, p = DIMENSIONS
    # Strassen 2x2x2 blocks
    n_blocks_i = 1
    n_blocks_j = 2
    n_blocks_k = 4
    total_blocks = n_blocks_i * n_blocks_j * n_blocks_k
    rank = total_blocks * 7  # 56
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for bi in range(n_blocks_i):
        for bj in range(n_blocks_j):
            for bk in range(n_blocks_k):
                i0 = 2 * bi
                i1 = 2 * bi + 1
                j0 = 2 * bj
                j1 = 2 * bj + 1
                k0 = 2 * bk
                k1 = 2 * bk + 1
                
                u_a = i0 * m + j0
                u_b = i0 * m + j1
                u_c = i1 * m + j0
                u_d = i1 * m + j1
                
                v_e = j0 * p + k0
                v_f = j0 * p + k1
                v_g = j1 * p + k0
                v_h = j1 * p + k1
                
                w_c00 = k0 * n + i0
                w_c01 = k1 * n + i0
                w_c10 = k0 * n + i1
                w_c11 = k1 * n + i1
                
                # M1 = (a+d)(e+h)
                U[u_a][col] += 1; U[u_d][col] += 1
                V[v_e][col] += 1; V[v_h][col] += 1
                W[w_c00][col] += 1; W[w_c11][col] += 1
                col += 1
                
                # M2 = (c+d)e
                U[u_c][col] += 1; U[u_d][col] += 1
                V[v_e][col] += 1
                W[w_c10][col] += 1; W[w_c11][col] -= 1
                col += 1
                
                # M3 = a(f-h)
                U[u_a][col] += 1
                V[v_f][col] += 1; V[v_h][col] -= 1
                W[w_c01][col] += 1; W[w_c11][col] += 1
                col += 1
                
                # M4 = d(g-e)
                U[u_d][col] += 1
                V[v_g][col] += 1; V[v_e][col] -= 1
                W[w_c00][col] += 1; W[w_c10][col] += 1
                col += 1
                
                # M5 = (a+b)h
                U[u_a][col] += 1; U[u_b][col] += 1
                V[v_h][col] += 1
                W[w_c00][col] -= 1; W[w_c01][col] += 1
                col += 1
                
                # M6 = (c-a)(e+f)
                U[u_c][col] += 1; U[u_a][col] -= 1
                V[v_e][col] += 1; V[v_f][col] += 1
                W[w_c11][col] += 1
                col += 1
                
                # M7 = (b-d)(g+h)
                U[u_b][col] += 1; U[u_d][col] -= 1
                V[v_g][col] += 1; V[v_h][col] += 1
                W[w_c00][col] += 1
                col += 1
    
    # Now try to reduce rank by finding tensor linear dependencies
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # For tensor rank reduction: find alpha such that sum_c alpha_c * U_c ⊗ V_c ⊗ W_c = 0
    # Sample random (i,j,k) to build linear system in alpha
    rng = np.random.RandomState(42)
    n_samples = 200
    
    M = np.zeros((n_samples, rank))
    for s in range(n_samples):
        i = rng.randint(n)
        j = rng.randint(m)
        k = rng.randint(p)
        for c in range(rank):
            M[s, c] = U_np[i, c] * V_np[j, c] * W_np[k, c]
    
    try:
        U_svd, s, Vt = np.linalg.svd(M, full_matrices=False)
        tol = 1e-10
        r = np.sum(s > tol)
        
        if r < rank:
            # Greedy elimination: try to remove each column if redundant
            keep = list(range(rank))
            
            for c in range(rank):
                if c not in keep:
                    continue
                other = [d for d in keep if d != c]
                if not other:
                    break
                
                # Check if column c is in span of other kept columns
                n_check = 100
                A_check = np.zeros((n_check, len(other)))
                b_check = np.zeros(n_check)
                
                for s in range(n_check):
                    i = rng.randint(n)
                    j = rng.randint(m)
                    k = rng.randint(p)
                    for idx, d in enumerate(other):
                        A_check[s, idx] = U_np[i, d] * V_np[j, d] * W_np[k, d]
                    b_check[s] = U_np[i, c] * V_np[j, c] * W_np[k, c]
                
                try:
                    alpha, _, rank_check, _ = np.linalg.lstsq(A_check, b_check, rcond=1e-10)
                    residual = np.linalg.norm(A_check @ alpha - b_check)
                    norm_b = np.linalg.norm(b_check)
                    if norm_b > 1e-12 and residual / norm_b < 1e-8:
                        keep.remove(c)
                except:
                    pass
            
            r_final = len(keep)
            if r_final < rank:
                U_reduced = U_np[:, keep].tolist()
                V_reduced = V_np[:, keep].tolist()
                W_reduced = W_np[:, keep].tolist()
                
                # Verify with multiple random tests
                valid = True
                for _ in range(5):
                    A_test = np.random.randn(n, m)
                    B_test = np.random.randn(m, p)
                    C_expected = A_test @ B_test
                    
                    C_computed = np.zeros((p, n))
                    for c in range(r_final):
                        C_computed += U_reduced[:, c] * V_reduced[:, c] * W_reduced[:, c].reshape(n, p)
                    
                    if not np.allclose(C_computed, C_expected, atol=1e-9):
                        valid = False
                        break
                
                if valid:
                    return {"U": U_reduced, "V": V_reduced, "W": W_reduced}
    except Exception:
        pass
    
    return {"U": U, "V": V, "W": W}
