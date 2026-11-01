import numpy as np
import time

DIMENSIONS = [2, 4, 4]

def _build_tensor():
    n, m, p = DIMENSIONS
    # Tensor T[i,j,k] where i in [0,n*m), j in [0,m*p), k in [0,p*n)
    # T corresponds to matmul: C[k*n+i] = sum_j A[i*m+j] * B[j*p+k]
    # So T[i, j, k] = 1 if the triple (i,j,k) corresponds to A[row,col]*B[row2,col2] with col==row2 and output at (row,col2)
    # i = row*n + ... wait, let me use the convention from the code.
    # U has shape (n*m, R), V has shape (m*p, R), W has shape (p*n, R)
    # The decomposition: T[i,j,k] = sum_r U[i,r]*V[j,r]*W[k,r]
    # For matmul: C[i_out, k_out] = sum_j A[i_out, j] * B[j, k_out]
    # In flattened form: C[k_out*n + i_out] = sum_j A[i_out*m + j] * B[j*p + k_out]
    # So T[i_out*m + j, j*p + k_out, k_out*n + i_out] = 1
    T = np.zeros((n*m, m*p, p*n))
    for i_out in range(n):
        for j in range(m):
            for k_out in range(p):
                T[i_out*m + j, j*p + k_out, k_out*n + i_out] = 1.0
    return T

def _verify(U, V, W, T, tol=1e-10):
    # Check T[i,j,k] = sum_r U[i,r]*V[j,r]*W[k,r]
    # Do it efficiently: for each r, outer product
    R = U.shape[1]
    Tc = np.zeros_like(T)
    for r in range(R):
        Tc += np.einsum('i,j,k->ijk', U[:,r], V[:,r], W[:,r])
    return np.allclose(Tc, T, atol=tol)

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = DIMENSIONS
    T = _build_tensor()
    
    # Start with the rank-28 Strassen construction
    R = 28
    U = np.zeros((n * m, R), dtype=int)
    V = np.zeros((m * p, R), dtype=int)
    W = np.zeros((p * n, R), dtype=int)

    S = [
        ([[(0,1),(3,1)], [(0,1),(3,1)], [(0,1),(3,1)]],),
        ([[(2,1),(3,1)], [(0,1)], [(2,1),(3,-1)]],),
        ([[(0,1)], [(1,1),(3,-1)], [(1,1),(3,1)]],),
        ([[(3,1)], [(2,1),(0,-1)], [(0,1),(2,1)]],),
        ([[(0,1),(1,1)], [(3,1)], [(0,-1),(1,1)]],),
        ([[(2,1),(0,-1)], [(0,1),(1,1)], [(3,1)]],),
        ([[(1,1),(3,-1)], [(2,1),(3,1)], [(0,1)]],),
    ]

    def mA(li, off):
        r, c = li // 2, li % 2
        return r * m + (off * 2 + c)

    def mB(li, ro, co):
        r, c = li // 2, li % 2
        return (ro * 2 + r) * p + (co * 2 + c)

    def mC(li, co):
        r, c = li // 2, li % 2
        return (co * 2 + c) * n + r

    blocks = [
        (0, 0, 0, 0, 0),
        (1, 1, 0, 0, 7),
        (0, 0, 1, 1, 14),
        (1, 1, 1, 1, 21),
    ]

    for a_off, b_ro, b_co, c_co, base in blocks:
        if time.time() > deadline:
            break
        for j, (parts,) in enumerate(S):
            ci = base + j
            u_entries, v_entries, w_entries = parts
            for li, cf in u_entries:
                U[mA(li, a_off), ci] += cf
            for li, cf in v_entries:
                V[mB(li, b_ro, b_co), ci] += cf
            for li, cf in w_entries:
                W[mC(li, c_co), ci] += cf

    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = R

    # Try to reduce rank by zeroing out columns where possible
    # A column r can be removed if its contribution is zero, i.e., U[:,r], V[:,r], W[:,r] are all zero
    # More generally, we can try to find linear dependencies among the rank-1 tensors
    
    # Attempt: for each column, check if we can express it as a combination of others
    # This is complex. Instead, try a greedy approach: for each column r,
    # check if removing it (setting to 0) and adjusting others can maintain the tensor.
    
    # Simpler: try to find if any column is redundant by checking if the tensor
    # can be decomposed with that column removed. This requires solving a linear system.
    
    # Let's try a different approach: use the structure of the problem.
    # The output C is 2x4. Let's see if we can find a rank < 28 decomposition
    # by exploiting the 2-row structure.
    
    # For now, return the rank-28 solution as a safe baseline.
    # The search for lower rank is complex and may not succeed within time.
    
    if time.time() > deadline:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    # Verify the construction
    if not _verify(best_U, best_V, best_W, T):
        # Fallback to schoolbook rank 32
        R = 32
        U = np.zeros((n*m, R), dtype=int)
        V = np.zeros((m*p, R), dtype=int)
        W = np.zeros((p*n, R), dtype=int)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    r = i*m*p + j*p + k
                    U[i*m+j, r] = 1
                    V[j*p+k, r] = 1
                    W[k*n+i, r] = 1
        best_U, best_V, best_W = U, V, W
        best_rank = R
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

if __name__ == "__main__":
    res = construct()
    n, m, p = DIMENSIONS
    U = np.array(res["U"], dtype=float)
    V = np.array(res["V"], dtype=float)
    W = np.array(res["W"], dtype=float)
    A = np.arange(n * m).reshape(n, m) + 1.0
    B = np.arange(m * p).reshape(m, p) + 2.0
    C = A @ B
    Cf = np.zeros((n, p))
    for ii in range(n):
        for kk in range(p):
            acc = 0.0
            for cc in range(U.shape[1]):
                for jj in range(m):
                    acc += U[ii * m + jj, cc] * V[jj * p + kk, cc] * W[kk * n + ii, cc]
            Cf[ii, kk] = acc
    assert np.allclose(C, Cf), (C, Cf)
    print("OK rank", U.shape[1])
