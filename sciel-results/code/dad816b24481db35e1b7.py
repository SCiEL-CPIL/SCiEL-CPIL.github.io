"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START

def _construct_rank_55():
    """
    Hardcoded rank-55 decomposition for 2x3x10 matrix multiplication.
    Based on 5 blocks of 2 columns, each using 11 ranks.
    """
    n, m, p = DIMENSIONS
    rank = 55
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # Iterate over 5 blocks of 2 columns
    for t in range(5):
        k0 = 2 * t
        k1 = 2 * t + 1
        
        # M1: (A00 + A11) * (B0,k0 + B1,k1)
        # U: A00(1), A11(1)
        # V: B0,k0(1), B1,k1(1)
        # W: C0,k0(1), C1,k1(1)
        U[0 * 3 + 0][col] += 1
        U[1 * 3 + 1][col] += 1
        V[0 * 10 + k0][col] += 1
        V[1 * 10 + k1][col] += 1
        W[k0 * 2 + 0][col] += 1
        W[k1 * 2 + 1][col] += 1
        col += 1
        
        # M2: (A10 + A11) * B0,k0
        # U: A10(1), A11(1)
        # V: B0,k0(1)
        # W: C1,k0(1), C1,k1(-1)
        U[1 * 3 + 0][col] += 1
        U[1 * 3 + 1][col] += 1
        V[0 * 10 + k0][col] += 1
        W[k0 * 2 + 1][col] += 1
        W[k1 * 2 + 1][col] -= 1
        col += 1
        
        # M3: A00 * (B0,k1 - B1,k1)
        # U: A00(1)
        # V: B0,k1(1), B1,k1(-1)
        # W: C0,k1(1), C1,k1(1)
        U[0 * 3 + 0][col] += 1
        V[0 * 10 + k1][col] += 1
        V[1 * 10 + k1][col] -= 1
        W[k1 * 2 + 0][col] += 1
        W[k1 * 2 + 1][col] += 1
        col += 1
        
        # M4: A11 * (B1,k0 - B0,k0)
        # U: A11(1)
        # V: B1,k0(1), B0,k0(-1)
        # W: C0,k0(1), C1,k0(1)
        U[1 * 3 + 1][col] += 1
        V[1 * 10 + k0][col] += 1
        V[0 * 10 + k0][col] -= 1
        W[k0 * 2 + 0][col] += 1
        W[k0 * 2 + 1][col] += 1
        col += 1
        
        # M5: (A00 + A01) * B1,k1
        # U: A00(1), A01(1)
        # V: B1,k1(1)
        # W: C0,k0(-1), C0,k1(1)
        U[0 * 3 + 0][col] += 1
        U[0 * 3 + 1][col] += 1
        V[1 * 10 + k1][col] += 1
        W[k0 * 2 + 0][col] -= 1
        W[k1 * 2 + 0][col] += 1
        col += 1
        
        # M6: (A10 - A00) * (B0,k0 + B0,k1)
        # U: A10(1), A00(-1)
        # V: B0,k0(1), B0,k1(1)
        # W: C1,k1(1)
        U[1 * 3 + 0][col] += 1
        U[0 * 3 + 0][col] -= 1
        V[0 * 10 + k0][col] += 1
        V[0 * 10 + k1][col] += 1
        W[k1 * 2 + 1][col] += 1
        col += 1
        
        # M7: (A01 - A11) * (B1,k0 + B1,k1)
        # U: A01(1), A11(-1)
        # V: B1,k0(1), B1,k1(1)
        # W: C0,k0(1)
        U[0 * 3 + 1][col] += 1
        U[1 * 3 + 1][col] -= 1
        V[1 * 10 + k0][col] += 1
        V[1 * 10 + k1][col] += 1
        W[k0 * 2 + 0][col] += 1
        col += 1
        
        # W00: A02 * B2,k0
        # U: A02(1)
        # V: B2,k0(1)
        # W: C0,k0(1)
        U[0 * 3 + 2][col] += 1
        V[2 * 10 + k0][col] += 1
        W[k0 * 2 + 0][col] += 1
        col += 1
        
        # W01: A02 * B2,k1
        # U: A02(1)
        # V: B2,k1(1)
        # W: C0,k1(1)
        U[0 * 3 + 2][col] += 1
        V[2 * 10 + k1][col] += 1
        W[k1 * 2 + 0][col] += 1
        col += 1
        
        # W10: A12 * B2,k0
        # U: A12(1)
        # V: B2,k0(1)
        # W: C1,k0(1)
        U[1 * 3 + 2][col] += 1
        V[2 * 10 + k0][col] += 1
        W[k0 * 2 + 1][col] += 1
        col += 1
        
        # W11: A12 * B2,k1
        # U: A12(1)
        # V: B2,k1(1)
        # W: C1,k1(1)
        U[1 * 3 + 2][col] += 1
        V[2 * 10 + k1][col] += 1
        W[k1 * 2 + 1][col] += 1
        col += 1
        
    return np.array(U, dtype=float), np.array(V, dtype=float), np.array(W, dtype=float)

def _verify_decomposition(U, V, W):
    """
    Verify that U, V, W exactly decompose the matrix multiplication tensor.
    Returns True if valid.
    """
    n, m, p = DIMENSIONS
    # Compute the tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # We check against the known multiplication tensor: T[i,j,k] = 1 if valid, else 0?
    # Actually, the multiplication tensor M[i,j,k] is 1 if the indices correspond to a valid term, else 0.
    # Specifically, M[i,j,k] = 1 for all i,j,k? No.
    # The tensor for matrix multiplication C = A B is:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # The tensor T has entries T[i,j,k] = 1.
    # Wait, the decomposition is:
    # C[i,k] = sum_j A[i,j] B[j,k] = sum_r (sum_j U[i*m+j, r] A[i,j]) (sum_k V[j*p+k, r] B[j,k]) W[k*n+i, r] ?
    # No, the standard decomposition is:
    # C[i,k] = sum_r U[i*m+j, r] ... this is not right.
    
    # Let's use the definition from the prompt:
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # The tensor T is of size n x m x p.
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # This must equal 1 for all i,j,k?
    # No, the matrix multiplication tensor T_mm[i,j,k] is 1 for all i,j,k?
    # Let's check: C[i,k] = sum_j A[i,j] B[j,k].
    # If we set T[i,j,k] = 1, then sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] should be 1?
    # Let's test with naive decomposition.
    # Naive: U[i*m+j, idx] = 1, V[j*p+k, idx] = 1, W[k*n+i, idx] = 1.
    # Then sum_r ... = 1 for the specific r corresponding to (i,j,k).
    # For other r, it is 0.
    # So T[i,j,k] = 1 for all i,j,k.
    
    # So we need to check if for all i,j,k:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    
    # This is a trilinear form.
    # Let's compute it.
    # U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
    # We can reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
    
    U_reshaped = U.reshape(n, m, -1)
    V_reshaped = V.reshape(m, p, -1)
    W_reshaped = W.reshape(p, n, -1)
    
    # T[i,j,k] = sum_r U_reshaped[i,j,r] * V_reshaped[j,k,r] * W_reshaped[k,i,r]
    # We can compute this as:
    # For each r, the contribution is U_reshaped[:,:,r] * V_reshaped[:,:,r] * W_reshaped[:,:,r] (element-wise product)
    # But indices are different.
    # T[i,j,k] = sum_r U_reshaped[i,j,r] * V_reshaped[j,k,r] * W_reshaped[k,i,r]
    
    # Let's just loop for verification, it's small (2*3*10 = 60 terms).
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(U.shape[1]):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(s - 1.0) > 1e-7:
                    return False
    return True

def _drop_column(U, V, W, col, keep_cols):
    """Try to drop column `col` by expressing W[:,col] as a combination of
    W[:,keep_cols]. Returns (new_U, new_V, new_W) or None."""
    Wc = W[:, col]
    Wk = W[:, keep_cols]
    # solve Wk @ a = Wc
    try:
        a, res, rank, sv = np.linalg.lstsq(Wk, Wc, rcond=None)
    except Exception:
        return None
    # check it's an exact solution
    if not np.allclose(Wk @ a, Wc, atol=1e-7, rtol=1e-7):
        return None
    new_U = np.zeros((U.shape[0], len(keep_cols)))
    new_V = np.zeros((V.shape[0], len(keep_cols)))
    new_W = np.zeros((W.shape[0], len(keep_cols)))
    for li, kc in enumerate(keep_cols):
        new_U[:, li] = U[:, kc] + a[li] * U[:, col]
        new_V[:, li] = V[:, kc] + a[li] * V[:, col]
        new_W[:, li] = W[:, kc] + a[li] * W[:, col]
    return new_U, new_V, new_W

def _local_improve(U, V, W, time_budget):
    """Attempt to reduce rank via null-space column elimination."""
    start = time.time()
    cur_U, cur_V, cur_W = U.copy(), V.copy(), W.copy()
    cur_cols = list(range(cur_U.shape[1]))
    improved = True
    while improved:
        improved = False
        if time.time() - start > time_budget:
            break
        # find the most "redundant" column: smallest norm or check each
        # For efficiency, iterate columns and attempt drop when its W is
        # in span of others. Use a heuristic ordering.
        for col in list(cur_cols):
            if time.time() - start > time_budget:
                break
            keep_cols = [c for c in cur_cols if c != col]
            if not keep_cols:
                continue
            res = _drop_column(cur_U, cur_V, cur_W, col, keep_cols)
            if res is not None:
                cur_U, cur_V, cur_W = res
                cur_cols = list(range(cur_U.shape[1]))
                improved = True
                break
    return cur_U, cur_V, cur_W, len(cur_cols)

def _to_int_matrices(U, V, W):
    """Convert numpy float matrices (possibly with small floats) to int lists.
    Rounds to nearest integer; assumes entries are integer/half-integer."""
    U = np.rint(U).astype(int)
    V = np.rint(V).astype(int)
    W = np.rint(W).astype(int)
    return U.tolist(), V.tolist(), W.tolist()

def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    DEADLINE = 130.0  # safety margin under 150s kill

    # Step 1: Construct the rank-55 decomposition
    U, V, W = _construct_rank_55()
    rank = U.shape[1]
    
    # Verify the decomposition
    if not _verify_decomposition(U, V, W):
        # Fallback to naive if verification fails (should not happen)
        r = n * m * p
        U = np.zeros((n * m, r))
        V = np.zeros((m * p, r))
        W = np.zeros((p * n, r))
        for j in range(m):
            for i in range(n):
                for k in range(p):
                    c = i * m + j
                    d = j * p + k
                    e = k * n + i
                    idx = j * (n * p) + i * p + k
                    U[c, idx] = 1.0
                    V[d, idx] = 1.0
                    W[e, idx] = 1.0
        rank = r

    best = (rank, U.copy(), V.copy(), W.copy())

    # Step 2: Local improvement
    if time.time() < DEADLINE:
        time_budget = max(1.0, DEADLINE - time.time() - 5.0)
        try:
            nU, nV, nW, new_rank = _local_improve(U, V, W, time_budget)
            if new_rank < best[0]:
                best = (new_rank, nU, nV, nW)
        except Exception:
            pass

    r, U, V, W = best
    U, V, W = _to_int_matrices(U, V, W)
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
