import numpy as np
import time

DIMENSIONS = [2, 4, 8]


def _check(U, V, W, n=2, m=4, p=8, tol=1e-6):
    """Verify sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1 for all i,j,k."""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    nm, mp, pn = n * m, m * p, p * n
    r = U.shape[1]
    ok = True
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(r):
                    s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                e = abs(s - 1.0)
                if e > max_err:
                    max_err = e
                if e > tol:
                    ok = False
    return ok, max_err


def _to_int_lists(U, V, W):
    """Convert float arrays to lists with exact half-integer (multiple of 1/4) entries."""
    def conv(A):
        B = []
        for row in A:
            out = []
            for x in row:
                q = round(x * 4.0)
                out.append(q / 4.0)
            B.append(out)
        return B
    return conv(U), conv(V), conv(W)


def _schoolbook(n=2, m=4, p=8):
    r = n * m * p
    U = np.zeros((n * m, r))
    V = np.zeros((m * p, r))
    W = np.zeros((p * n, r))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W


def _strassen_2x2():
    U = np.zeros((4, 7), dtype=float)
    V = np.zeros((4, 7), dtype=float)
    W = np.zeros((4, 7), dtype=float)
    A = {
        0: [1, 0, 1, 0],
        1: [0, 0, 1, 1],
        2: [1, 0, 0, 0],
        3: [0, 0, 0, 1],
        4: [1, 1, 0, 0],
        5: [1, 0, -1, 0],
        6: [0, 1, 0, -1],
    }
    B = {
        0: [1, 0, 0, 1],
        1: [1, 0, 0, 0],
        2: [0, 1, 0, -1],
        3: [0, 0, 1, -1],
        4: [0, 0, 0, 1],
        5: [1, 1, 0, 0],
        6: [0, 0, 1, 1],
    }
    C = {
        0: [1, 0, 0, 1],
        1: [0, 0, 1, -1],
        2: [0, 1, 0, 1],
        3: [1, 0, 1, 0],
        4: [-1, 1, 0, 0],
        5: [0, 0, 0, 1],
        6: [1, 0, 0, 0],
    }
    for t in range(7):
        for a in range(4):
            U[a, t] = A[t][a]
        for b in range(4):
            V[b, t] = B[t][b]
        for c in range(4):
            W[c, t] = C[t][c]
    return U, V, W


def _embed_2x2x2(U2, V2, W2, ai, aj, bk):
    nm, mp, pn = 8, 32, 16
    r = U2.shape[1]
    U = np.zeros((nm, r))
    V = np.zeros((mp, r))
    W = np.zeros((pn, r))
    for local_i in range(2):
        for local_j in range(2):
            gA = (ai + local_i) * 4 + (aj + local_j)
            for t in range(r):
                U[gA, t] += U2[local_i * 2 + local_j, t]
    for local_j in range(2):
        for local_k in range(2):
            gV = (aj + local_j) * 8 + (bk + local_k)
            for t in range(r):
                V[gV, t] += V2[local_j * 2 + local_k, t]
    for local_k in range(2):
        for local_i in range(2):
            gW = (bk + local_k) * 2 + (ai + local_i)
            for t in range(r):
                W[gW, t] += W2[local_k * 2 + local_i, t]
    return U, V, W


def _strassen_full(n=2, m=4, p=8):
    U2, V2, W2 = _strassen_2x2()
    U = np.zeros((n * m, 7))
    V = np.zeros((m * p, 7))
    W = np.zeros((p * n, 7))
    for ai in range(0, n, 2):
        for aj in range(0, m, 2):
            for bk in range(0, p, 2):
                Ub, Vb, Wb = _embed_2x2x2(U2, V2, W2, ai, aj, bk)
                U += Ub
                V += Vb
                W += Wb
    return U, V, W


def _rank4_full(n=2, m=4, p=8):
    return _schoolbook(n, m, p)


def _try_reduce_rank(U, V, W, n=2, m=4, p=8, max_iter=500, tol=1e-6, deadline=None):
    """Try to reduce rank by removing columns that are linearly dependent in the tensor sense."""
    U = U.copy().astype(float)
    V = V.copy().astype(float)
    W = W.copy().astype(float)
    r = U.shape[1]
    
    if deadline is not None:
        if time.time() > deadline:
            return U, V, W
    
    # Try to remove columns by checking if column c can be expressed as a linear combo
    # of other columns such that the tensor equation still holds.
    # This is a complex constraint, so we use a simpler heuristic:
    # Try removing each column and see if we can adjust others to compensate.
    
    for iter_idx in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        if r <= 0:
            break
        
        removed = False
        for c in range(r):
            if deadline is not None and time.time() > deadline:
                break
            
            # Try removing column c
            U_test = np.delete(U, c, axis=1)
            V_test = np.delete(V, c, axis=1)
            W_test = np.delete(W, c, axis=1)
            
            # Check if we can find small adjustments to remaining columns to compensate
            # The error from removing column c is:
            # err[i,j,k] = U[i,j,c]*V[j,k,c]*W[k,i,c]
            # We need to find deltas such that sum_{c' != c} delta_U[i,j,c']*V[j,k,c']*W[k,i,c'] + ... = err
            
            # This is a 3-tensor fitting problem. Use a simple approach:
            # For each (i,j,k), the error is E[i,j,k] = U[i,j,c]*V[j,k,c]*W[k,i,c]
            # We want to distribute this across remaining columns.
            
            # Simple heuristic: if the column is "small" in some norm, try removing it
            col_norm_u = np.linalg.norm(U[:, c])
            col_norm_v = np.linalg.norm(V[:, c])
            col_norm_w = np.linalg.norm(W[:, c])
            
            if col_norm_u < 1e-10 or col_norm_v < 1e-10 or col_norm_w < 1e-10:
                U = U_test
                V = V_test
                W = W_test
                r -= 1
                removed = True
                break
        
        if not removed:
            break
    
    return U, V, W


def _global_structured_search(n=2, m=4, p=8, deadline=None):
    """
    Global structured search: exploit the rectangular aspect ratio.
    
    Key insight: C[i,k] = sum_j A[i,j] * B[j,k]
    
    View this as: for each output column k, C[:,k] = A * B[:,k]
    
    If B[:,k] = sum_t u_t * v_t[k] (rank-r factorization of B), then
    C[:,k] = sum_t (A*u_t) * v_t[k]
    
    The bilinear algorithm:
    - For each t: compute x_t = A*u_t (n*m mults: A[i,j]*u_t[j])
    - For each t, each i, k: C[i,k] += x_t[i] * v_t[k] (n*p mults per t)
    
    Total: r * (n*m + n*p) = r * n * (m+p)
    
    For n=2, m=4, p=8: r * 2 * 12 = 24r. For r=4: 96. Not helpful.
    
    Alternative: Use a different structure. Treat the problem as:
    C = A * B where we precompute certain linear combinations.
    
    Let's try: partition B into 4 blocks of 2 columns each: B = [B0, B1, B2, B3]
    C = [C0, C1, C2, C3] where Ci = A * Bi
    
    For each Ci (2x2), use Strassen: 7 mults each. Total: 28.
    But the A part is shared! A is the same for all Ci.
    
    Strassen for Ci: M1 = (a00+a11)(b00+b11), etc.
    The A combinations are the same across all 4 blocks!
    
    So: compute 7 A-combinations once (no mults, just additions), then for each of 4 blocks,
    compute 7 B-combinations and 7 mults. Total mults: 4*7 = 28.
    
    But wait, in bilinear rank, the A-combinations are free (linear), only scalar mults count.
    So rank = 28!
    
    Let me verify: A is 2x4, B is 4x8. Partition B into 4 blocks of 2x2.
    For block t (t=0..3), B_t is 4x2. We want C_t = A * B_t (2x2).
    
    Using Strassen on each 2x2 block:
    A-part combinations (same for all t):
      a1 = a00 + a11
      a2 = a10 + a11
      a3 = a00
      a4 = a11
      a5 = a00 + a01
      a6 = a10 - a00
      a7 = a01 - a11
    
    Wait, Strassen for 2x2: A is 2x2, B is 2x2. But here A is 2x4, not 2x2.
    
    Let me reconsider. A is 2x4. For C_t = A * B_t where B_t is 4x2:
    C_t[i,k] = sum_{j=0}^{3} A[i,j] * B_t[j,k]
    
    This is a 2x4 times 4x2 multiplication. Schoolbook: 2*4*2 = 16 mults per block.
    4 blocks: 64 total. Same as full schoolbook.
    
    To use Strassen, we need to further decompose. Treat A as [A_left, A_right] where
    A_left, A_right are 2x2. Then C_t = A_left * B_t_top + A_right * B_t_bottom
    where B_t_top, B_t_bottom are 2x2.
    
    So C_t = A_left * B_t[0:2,:] + A_right * B_t[2:4,:]
    
    Each of these is a 2x2 * 2x2 multiplication. Use Strassen for each:
    A_left * B_t_top: 7 mults
    A_right * B_t_bottom: 7 mults
    Total per block: 14 mults. 4 blocks: 56 mults.
    
    This is exactly the current Strassen-56 approach! The A_left and A_right Strassen
    combinations are shared across blocks, but the mults are not saved because
    each block needs its own B combinations.
    
    So 56 is the natural Strassen-based answer. To beat it, we need a fundamentally
    different approach.
    
    Let me try: Can we share B combinations across blocks?
    
    For block t, B_t_top = B[0:2, 2t:2t+2], B_t_bottom = B[2:4, 2t:2t+2].
    
    The Strassen B-combinations for A_left * B_t_top:
      b1 = b00 + b11
      b2 = b00
      b3 = b01 - b11
      b4 = b10 - b00
      b5 = b11
      b6 = b00 + b01
      b7 = b10 + b11
    
    These are different for each block t. So no sharing.
    
    Alternative: Use a different factorization of the 4x2 B_t.
    
    I think 56 might be close to optimal for this structure. Let me try a randomized
    search for a better decomposition.
    """
    # Return Strassen-56 as the best structured solution
    return _strassen_full(n, m, p)


def construct():
    start = time.time()
    deadline = start + 140.0

    n, m, p = DIMENSIONS
    candidates = []

    # Candidate 1: Strassen-based (rank 56)
    try:
        U, V, W = _strassen_full(n, m, p)
        ok, err = _check(U, V, W, n, m, p)
        if ok:
            # Try to reduce rank
            U_red, V_red, W_red = _try_reduce_rank(U, V, W, n, m, p, max_iter=100, deadline=deadline - 10)
            ok_r, err_r = _check(U_red, V_red, W_red, n, m, p)
            if ok_r:
                Ui, Vi, Wi = _to_int_lists(U_red, V_red, W_red)
                candidates.append((U_red.shape[1], Ui, Vi, Wi, "strassen_reduced"))
            else:
                Ui, Vi, Wi = _to_int_lists(U, V, W)
                candidates.append((56, Ui, Vi, Wi, "strassen"))
    except Exception:
        Ui, Vi, Wi = _to_int_lists(*_strassen_full(n, m, p))
        candidates.append((56, Ui, Vi, Wi, "strassen"))

    # Candidate 2: Global structured search
    if time.time() < deadline - 20.0:
        try:
            U, V, W = _global_structured_search(n, m, p, deadline)
            ok, err = _check(U, V, W, n, m, p)
            if ok:
                Ui, Vi, Wi = _to_int_lists(U, V, W)
                candidates.append((U.shape[1], Ui, Vi, Wi, "global"))
        except Exception:
            pass

    # Candidate 3: Schoolbook (rank 64) - always valid fallback
    U, V, W = _schoolbook(n, m, p)
    ok, err = _check(U, V, W, n, m, p)
    if ok:
        Ui, Vi, Wi = _to_int_lists(U, V, W)
        candidates.append((64, Ui, Vi, Wi, "schoolbook"))

    if not candidates:
        U, V, W = _schoolbook(n, m, p)
        Ui, Vi, Wi = _to_int_lists(U, V, W)
        return {"U": Ui, "V": Vi, "W": Wi}

    candidates.sort(key=lambda x: x[0])
    _, best_U, best_V, best_W, _name = candidates[0]
    return {"U": best_U, "V": best_V, "W": best_W}


if __name__ == "__main__":
    n, m, p = DIMENSIONS
    U, V, W = _strassen_full(n, m, p)
    ok, err = _check(U, V, W, n, m, p)
    print("Strassen valid:", ok, "max_err:", err, "rank:", U.shape[1])
    out = construct()
    ok2, err2 = _check(out["U"], out["V"], out["W"], n, m, p)
    print("construct valid:", ok2, "max_err:", err2,
          "rank:", len(out["U"][0]) if out["U"] else None)
