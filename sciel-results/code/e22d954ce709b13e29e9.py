"""
Tensor decomposition for 4x4 @ 4x5 matrix multiplication.
Structural approach: build the exact multiplication tensor, then use
rank-revealing SVD + ALS tensor completion to find a low-rank decomposition,
with half-integer rounding and exact verification.
"""
import numpy as np
import time
import itertools

DIMENSIONS = [4, 4, 5]


def _build_full_tensor(n, m, p):
    """Build the full multiplication tensor T[i,j,k] = 1 (the identity tensor).
    Shape: (n, m, p) where T[i,j,k] corresponds to output (i,k) from input (i,j) * (j,k).
    Actually the multiplication tensor has T[i,j,k] = 1 for all i,j,k (it's the identity).
    Wait - the multiplication tensor is: for each (i,j,k), coefficient is 1.
    """
    # The matrix multiplication tensor T of shape (n, m, p) has T[i,j,k] = 1 for all i,j,k
    # because C[i,k] = sum_j A[i,j] B[j,k], each term has coefficient 1.
    T = np.ones((n, m, p), dtype=np.float64)
    return T


def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i,j,k."""
    rank = len(U[0]) if U else 0
    if rank == 0:
        return False
    Ua = np.array(U, dtype=np.float64)
    Va = np.array(V, dtype=np.float64)
    Wa = np.array(W, dtype=np.float64)
    Ur = Ua.reshape(n, m, rank)
    Vr = Va.reshape(m, p, rank)
    Wr = Wa.reshape(p, n, rank)
    T = np.einsum('ijr,jkr,kir->ijk', Ur, Vr, Wr)
    return np.allclose(T, 1.0, atol=tol)


def _is_half_integer(M, tol=1e-9):
    """Check all entries are multiples of 0.5."""
    for row in M:
        for v in row:
            if abs(v * 2 - round(v * 2)) > tol:
                return False
    return True


def _round_to_half(M):
    """Round matrix entries to nearest half-integer."""
    return [[float(round(v * 2) * 0.5) for v in row] for row in M]


def _als_decompose(T, n, m, p, target_rank, max_iter=200, deadline=None, seed=42):
    """
    ALS decomposition of 3-way tensor T[i,j,k] = sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r]
    T is shape (n, m, p). We want rank-r factorization.
    """
    np.random.seed(seed)
    
    # Initialize
    U = np.random.randn(n * m, target_rank) * 0.5
    V = np.random.randn(m * p, target_rank) * 0.5
    W = np.random.randn(p * n, target_rank) * 0.5
    
    # Normalize to avoid scale issues
    for r in range(target_rank):
        norm = max(np.max(np.abs(U[:, r])), np.max(np.abs(V[:, r])), np.max(np.abs(W[:, r])), 1e-10)
        U[:, r] /= norm
        V[:, r] /= norm
        W[:, r] /= norm
        # Redistribute: make W carry the scale
        W[:, r] *= norm * norm
    
    best_err = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline - 15:
            break
        
        # Update U: for each (i,j), U[i*m+j, :] 
        # T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
        # For fixed (i,j): T[i,j,:] = sum_r U[i*m+j,r] * (V[j*p+k,r] * W[k*n+i,r]) for all k
        # This is: for each k, T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
        # Let M_ik = V[j*p+k,r] * W[k*n+i,r] (shape p x rank)
        # Then T[i,j,:] (shape p) = M_ik @ U[i*m+j,:] (shape rank)
        # So U[i*m+j,:] = pinv(M_ik) @ T[i,j,:]
        
        U_new = U.copy()
        for i in range(n):
            for j in range(m):
                # Build M: shape (p, rank)
                M = np.zeros((p, target_rank))
                for k in range(p):
                    M[k, :] = V[j * p + k, :] * W[k * n + i, :]
                # Solve: M @ U[i*m+j,:] = T[i,j,:]
                try:
                    U_new[i * m + j, :] = np.linalg.lstsq(M, T[i, j, :], rcond=None)[0]
                except:
                    pass
        
        # Update V: for each (j,k)
        V_new = V.copy()
        for j in range(m):
            for k in range(p):
                M = np.zeros((n, target_rank))
                for i in range(n):
                    M[i, :] = U[i * m + j, :] * W[k * n + i, :]
                try:
                    V_new[j * p + k, :] = np.linalg.lstsq(M, T[:, j, k], rcond=None)[0]
                except:
                    pass
        
        # Update W: for each (k,i)
        W_new = W.copy()
        for k in range(p):
            for i in range(n):
                M = np.zeros((m, target_rank))
                for j in range(m):
                    M[j, :] = U[i * m + j, :] * V[j * p + k, :]
                try:
                    W_new[k * n + i, :] = np.linalg.lstsq(M, T[i, j, k] if False else T[i, :, k], rcond=None)[0]
                except:
                    pass
        
        U, V, W = U_new, V_new, W_new
        
        # Compute error
        Ur = U.reshape(n, m, target_rank)
        Vr = V.reshape(m, p, target_rank)
        Wr = W.reshape(p, n, target_rank)
        T_hat = np.einsum('ijr,jkr,kir->ijk', Ur, Vr, Wr)
        err = np.max(np.abs(T_hat - T))
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if best_err < 1e-10:
            break
        
        # Early stop if not improving much
        if it > 50 and best_err > 0.5:
            break
    
    return best_U, best_V, best_W, best_err


def _try_complete_rank(T, n, m, p, start_rank, deadline):
    """Try to find exact decomposition at various ranks starting from start_rank going down."""
    best_result = None
    best_rank = start_rank + 1
    
    for target_rank in range(start_rank, max(start_rank - 15, 20), -1):
        if deadline is not None and time.time() > deadline - 20:
            break
        
        found = False
        # Try multiple seeds
        for seed in range(5):
            if deadline is not None and time.time() > deadline - 15:
                break
            try:
                U, V, W, err = _als_decompose(T, n, m, p, target_rank, max_iter=150, 
                                               deadline=deadline, seed=seed + 42)
                if err < 1e-8:
                    # Try rounding to half-integers
                    U_list = [[float(x) for x in row] for row in U]
                    V_list = [[float(x) for x in row] for row in V]
                    W_list = [[float(x) for x in row] for row in W]
                    
                    # Check if already half-integer
                    if _is_half_integer(U_list) and _is_half_integer(V_list) and _is_half_integer(W_list):
                        if _verify(U_list, V_list, W_list, n, m, p):
                            best_result = {'U': U_list, 'V': V_list, 'W': W_list}
                            best_rank = target_rank
                            found = True
                            break
                    
                    # Try rounding
                    U_r = _round_to_half(U_list)
                    V_r = _round_to_half(V_list)
                    W_r = _round_to_half(W_list)
                    if _verify(U_r, V_r, W_r, n, m, p):
                        best_result = {'U': U_r, 'V': V_r, 'W': W_r}
                        best_rank = target_rank
                        found = True
                        break
            except:
                pass
        
        if found:
            break
    
    return best_result, best_rank


def _build_strassen_kronecker(n, m, p):
    """
    Build a Strassen-based Kronecker product decomposition.
    For 4x4x4: use 2x2 Strassen (rank 7) Kronecker 2x2 Strassen (rank 7) = rank 49.
    For 4x4x5: extend with schoolbook for the 5th column.
    """
    # 2x2 Strassen terms
    strassen_2x2 = [
        # (U, V, W) as dicts mapping (row,col) -> coeff
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
        ({(0,0):1}, {(1,0):1, (1,1):-1}, {(0,1):1, (1,1):1}),
        ({(1,1):1}, {(0,1):1, (0,0):-1}, {(0,0):1, (1,0):1}),
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),
    ]
    
    # Build 4x4x4 using Kronecker of two 2x2 Strassen
    rank_4x4x4 = 49
    U4 = np.zeros((16, rank_4x4x4))
    V4 = np.zeros((16, rank_4x4x4))
    W4 = np.zeros((16, rank_4x4x4))
    
    for a, (Ua, Va, Wa) in enumerate(strassen_2x2):
        for b, (Ub, Vb, Wb) in enumerate(strassen_2x2):
            idx = a * 7 + b
            for (i1, j1), ca in Ua.items():
                for (i2, j2), cb in Ub.items():
                    i = i1 * 2 + i2
                    j = j1 * 2 + j2
                    U4[i * 4 + j, idx] += ca * cb
            for (j1, k1), ca in Va.items():
                for (j2, k2), cb in Vb.items():
                    j = j1 * 2 + j2
                    k = k1 * 2 + k2
                    V4[j * 4 + k, idx] += ca * cb
            for (k1, i1), ca in Wa.items():
                for (k2, i2), cb in Wb.items():
                    k = k1 * 2 + k2
                    i = i1 * 2 + i2
                    W4[k * 4 + i, idx] += ca * cb
    
    # Extend to 4x4x5: add schoolbook for column 4
    rank_total = rank_4x4x4 + 4
    U = np.zeros((n * m, rank_total))
    V = np.zeros((m * p, rank_total))
    W = np.zeros((p * n, rank_total))
    
    # Copy 4x4x4 part
    U[:, :rank_4x4x4] = U4
    for j in range(m):
        for k in range(4):
            V[j * p + k, :rank_4x4x4] = V4[j * 4 + k, :]
    for k in range(4):
        for i in range(n):
            W[k * n + i, :rank_4x4x4] = W4[k * 4 + i, :]
    
    # Schoolbook for 5th column (k=4)
    for j in range(m):
        r = rank_4x4x4 + j
        for i in range(n):
            U[i * m + j, r] = 1
        V[j * p + 4, r] = 1
        for i in range(n):
            W[4 * n + i, r] = 1
    
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    
    return {'U': U_list, 'V': V_list, 'W': W_list}, rank_total


def _build_schoolbook(n, m, p):
    """Schoolbook: rank = n*m*p = 80."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return {'U': [[float(x) for x in row] for row in U],
            'V': [[float(x) for x in row] for row in V],
            'W': [[float(x) for x in row] for row in W]}, rank


def _try_reduce_by_combination(U, V, W, n, m, p, deadline):
    """Try to reduce rank by finding linear combinations of terms that cancel."""
    rank = len(U[0])
    Ua = np.array(U, dtype=np.float64)
    Va = np.array(V, dtype=np.float64)
    Wa = np.array(W, dtype=np.float64)
    
    best_rank = rank
    best_U, best_V, best_W = Ua.copy(), Va.copy(), Wa.copy()
    
    # Try removing individual terms
    for attempt in range(3):
        if deadline is not None and time.time() > deadline - 10:
            break
        reduced = False
        for r in range(best_rank):
            if deadline is not None and time.time() > deadline - 10:
                break
            mask = [x for x in range(best_rank) if x != r]
            if len(mask) == 0:
                continue
            U_test = best_U[:, mask]
            V_test = best_V[:, mask]
            W_test = best_W[:, mask]
            Ur = U_test.reshape(n, m, len(mask))
            Vr = V_test.reshape(m, p, len(mask))
            Wr = W_test.reshape(p, n, len(mask))
            T = np.einsum('ijr,jkr,kir->ijk', Ur, Vr, Wr)
            if np.allclose(T, 1.0, atol=1e-9):
                best_U, best_V, best_W = U_test, V_test, W_test
                best_rank = len(mask)
                reduced = True
                break
        if not reduced:
            break
    
    if best_rank < rank:
        U_list = [[float(x) for x in row] for row in best_U]
        V_list = [[float(x) for x in row] for row in best_V]
        W_list = [[float(x) for x in row] for row in best_W]
        return {'U': U_list, 'V': V_list, 'W': W_list}, best_rank
    
    return None, rank


def construct():
    """Find a rank-minimal exact tensor decomposition for 4x4 @ 4x5 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    
    # Step 1: Build Strassen-Kronecker baseline (rank 53)
    base_result, base_rank = _build_strassen_kronecker(n, m, p)
    
    # Verify
    if not _verify(base_result['U'], base_result['V'], base_result['W'], n, m, p):
        # Fallback to schoolbook
        base_result, base_rank = _build_schoolbook(n, m, p)
    
    best_result = base_result
    best_rank = base_rank
    
    # Step 2: Try to reduce by removing redundant terms
    if time.time() < deadline - 30:
        reduced, red_rank = _try_reduce_by_combination(
            base_result['U'], base_result['V'], base_result['W'], n, m, p, deadline)
        if reduced is not None and red_rank < best_rank:
            best_result = reduced
            best_rank = red_rank
    
    # Step 3: Try ALS-based rank completion for lower ranks
    T = _build_full_tensor(n, m, p)
    
    # Try ranks from best_rank-1 down to 40
    if time.time() < deadline - 20:
        for target_rank in range(best_rank - 1, max(best_rank - 12, 35), -1):
            if time.time() > deadline - 20:
                break
            found = False
            for seed in range(8):
                if time.time() > deadline - 15:
                    break
                try:
                    U, V, W, err = _als_decompose(T, n, m, p, target_rank, max_iter=100,
                                                   deadline=deadline, seed=seed * 100 + 7)
                    if err < 1e-8:
                        U_list = [[float(x) for x in row] for row in U]
                        V_list = [[float(x) for x in row] for row in V]
                        W_list = [[float(x) for x in row] for row in W]
                        
                        # Try direct verification
                        if _verify(U_list, V_list, W_list, n, m, p):
                            best_result = {'U': U_list, 'V': V_list, 'W': W_list}
                            best_rank = target_rank
                            found = True
                            break
                        
                        # Try half-integer rounding
                        U_r = _round_to_half(U_list)
                        V_r = _round_to_half(V_list)
                        W_r = _round_to_half(W_list)
                        if _verify(U_r, V_r, W_r, n, m, p):
                            best_result = {'U': U_r, 'V': V_r, 'W': W_r}
                            best_rank = target_rank
                            found = True
                            break
                except:
                    pass
            if found:
                break
    
    # Step 4: If still at 53, try a different structural approach
    # Use the fact that 4x4x5 can be seen as 4x4x4 + 4x4x1
    # The 4x4x4 part has rank 49 (Strassen), but maybe we can do better
    # by exploiting the specific structure
    
    # Final verification
    if not _verify(best_result['U'], best_result['V'], best_result['W'], n, m, p):
        # Ultimate fallback
        sb_result, sb_rank = _build_schoolbook(n, m, p)
        if _verify(sb_result['U'], sb_result['V'], sb_result['W'], n, m, p):
            best_result = sb_result
            best_rank = sb_rank
    
    return best_result
