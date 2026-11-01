"""Rank-minimal 4x4 matrix multiplication via numerical low-rank factorization.

Strategy:
  - Build the 64x64 matrix G where G[a,b] = 1 if the (i,j,k) triple corresponding
    to a and b are the same, else 0.  More precisely, G is the "tensor Gram matrix"
    such that rank(G) = rank of the matricized tensor.
  - Actually, the correct approach: the matrix multiplication tensor T has entries
    T[i,j,k] = delta_{j,k} (for C[i,k] = sum_j A[i,j] B[j,k]).
    We matricize T into a 16x16 matrix M where M[i*4+k, j*4+j2] ... no.
    
  Let me use the standard approach: the tensor T in R^{4x4x4} with T[i,j,k] = 1 if j==k... 
  No, T[i,j,k] = delta_{jk} is wrong. The multiplication tensor has T[i,j,k] = 1 
  for all i,j,k? No.
  
  The matrix multiplication tensor T satisfies:
    C[i,k] = sum_j A[i,j] B[j,k]
  So T[i,j,k] = 1 for all i,j,k? No, T is a 3-tensor where the bilinear form is
    sum_{i,j,k} T[i,j,k] * A[i,j] * B[j,k] * C_out[i,k]
  
  Actually the standard formulation: the multiplication tensor has
    T[i, j, k] = 1 if the "output" index (i,k) matches, i.e., T is defined by
    T[i, j, k] = 1 for all valid i,j,k in the sense that the product A[i,j]*B[j,k]
    contributes to C[i,k].
    
  The rank of the tensor is the minimum number of rank-1 tensors needed.
  
  A practical approach: 
    1. Form the 64x64 matrix G where G[alpha, beta] = sum_gamma T[alpha,gamma]*T[beta,gamma]
       after appropriate matricization.
    
  Let me use a cleaner formulation:
    - Matricize T as a 16 x 64 matrix: M[row, col] where row = i*4+k (output index),
      col = j*4+j2 ... this is getting confused.
    
  The cleanest: T is in R^{(p*n) x (n*m) x (m*p)} = R^{16 x 16 x 16}.
  Matricize to M of shape (16, 256) where M[k*4+i, j*4+j2*4+... ] 
  
  Actually, let me just directly construct the 16x256 matricization:
  M[row, col] = T[row_index, col_index]
  where row = k*4 + i (the output C[i,k] index, p*n = 16)
  and col = j*4 + j2 ... no, the other two dimensions are n*m=16 and m*p=16.
  
  So M is 16 x 256: M[k*4+i, (j*4+j2)*16 + ... ]
  
  Let me just be explicit:
  T is indexed by (i, j, k) for the "A index" (i,j), "B index" (j,k), "C index" (k,i).
  Wait, the standard is: T in R^{A_dim x B_dim x C_dim} = R^{16 x 16 x 16}.
  T[a, b, c] = 1 if the bilinear term A_a * B_b * C_c appears, which happens when
  a = i*4+j, b = j*4+k, c = k*4+i for some i,j,k. So T[i*4+j, j*4+k, k*4+i] = 1.
  
  Matricize as M of shape (16, 256): M[c, a*16+b] = T[a,b,c].
  Then rank(T) = rank(M).
  
  Use SVD on M, keep top-r components, round to nearest half-integer, verify.
"""
import time
import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START

def _build_matricization():
    """Build the 16x256 matricization of the 4x4x4 matrix multiplication tensor."""
    n = 4
    # T[a, b, c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i for some i,j,k
    # Matricize: M[c, a*p*n + b] = T[a, b, c]  -> shape (p*n, n*m * m*p) = (16, 256)
    M = np.zeros((16, 256))
    for i in range(n):
        for j in range(n):
            for k in range(n):
                a = i * n + j   # index into A (n*m = 16)
                b = j * n + k   # index into B (m*p = 16)
                c = k * n + i   # index into C (p*n = 16)
                M[c, a * 16 + b] = 1.0
    return M


def _round_and_verify(U, V, W, tol=1e-6):
    """Round U, V, W to nearest half-integers and verify the decomposition."""
    n = 4
    U_r = np.round(U * 2) / 2
    V_r = np.round(V * 2) / 2
    W_r = np.round(W * 2) / 2
    rank = U_r.shape[1]
    U4 = U_r.reshape(n, n, rank)
    V4 = V_r.reshape(n, n, rank)
    W4 = W_r.reshape(n, n, rank)
    recon = np.zeros((n, n, n))
    for i in range(n):
        for j in range(n):
            for k in range(n):
                recon[i, j, k] = np.sum(U4[i, j, :] * V4[j, k, :] * W4[k, i, :])
    target = np.ones((n, n, n))
    return np.max(np.abs(recon - target)) < tol, U_r, V_r, W_r


def _try_rank(r, M, deadline):
    """Try to find a rank-r decomposition by SVD truncation + rounding."""
    if time.time() > deadline:
        return None
    # SVD of M (16 x 256)
    U_svd, s_vals, Vt = np.linalg.svd(M, full_matrices=False)
    # Truncate to rank r
    if r >= len(s_vals):
        r_eff = len(s_vals)
    else:
        r_eff = r
    U_trunc = U_svd[:, :r_eff] * np.sqrt(s_vals[:r_eff])  # 16 x r_eff
    V_trunc = (Vt[:r_eff, :].T * np.sqrt(s_vals[:r_eff]))  # 256 x r_eff
    
    # Reshape V_trunc back: V_trunc[a*16+b, r] -> U_factor[a, r] * V_factor[b, r]
    # But we need U_factor (16 x r), V_factor (16 x r), W_factor (16 x r)
    # M[c, a*16+b] = sum_r U_mat[c, r] * V_mat[a*16+b, r]
    # We need: sum_r U4[i,j,r] * V4[j,k,r] * W4[k,i,r] = 1 for all i,j,k
    # M[c, a*16+b] = T[a,b,c] = 1 iff a=i*4+j, b=j*4+k, c=k*4+i
    # So M[c, a*16+b] = W4[k,i,r] * U4[i,j,r] * V4[j,k,r] summed over r
    # But in the SVD: M[c, a*16+b] = sum_r U_svd[c,r] * Vt[r, a*16+b]
    # So we need to factor Vt[r, a*16+b] = U4[i,j,r] * V4[j,k,r] where a=i*4+j, b=j*4+k
    # This is an outer product structure in the (a,b) index space.
    
    # The issue: SVD gives us M = U_svd @ diag(s) @ Vt, but Vt[r, a*16+b] is not
    # necessarily a product of a function of a and a function of b.
    # We need a *tensor* decomposition, not just a matrix factorization.
    
    # Alternative: use the fact that rank(T) <= rank(M), and try to find
    # the tensor decomposition directly.
    
    # Let me try a different approach: use the known structure.
    # The tensor T has T[i*4+j, j*4+k, k*4+i] = 1.
    # A rank-1 tensor is U[i,j] * V[j,k] * W[k,i] for all i,j,k.
    # So we need: sum_r U_r[i,j] * V_r[j,k] * W_r[k,i] = 1 for all i,j,k.
    
    # This is a 3-way tensor decomposition. Let me use a simple iterative approach:
    # Initialize with random factors, then use ALS (alternating least squares).
    
    if time.time() > deadline:
        return None
    
    np.random.seed(42)
    best_err = np.inf
    best_factors = None
    
    for attempt in range(3):
        if time.time() > deadline:
            break
        # Initialize
        Uf = np.random.randn(16, r_eff) * 0.5
        Vf = np.random.randn(16, r_eff) * 0.5
        Wf = np.random.randn(16, r_eff) * 0.5
        
        # Scale to roughly match the tensor magnitude
        Uf *= 1.0
        Vf *= 1.0
        Wf *= 1.0
        
        for iteration in range(50):
            if time.time() > deadline:
                break
            # Update U: for each (i,j), solve least squares
            for i in range(4):
                for j in range(4):
                    # Target: sum_k T[i,j,k] * ... = 1 for all k where j==k... 
                    # Actually T[i*4+j, j*4+k, k*4+i] = 1 for all k
                    # So for fixed (i,j), we need: sum_r Uf[i*4+j, r] * Vf[j*4+k, r] * Wf[k*4+i, r] = 1 for all k
                    # Let's form a matrix: for each k, the row is Vf[j*4+k, :] * Wf[k*4+i, :] (element-wise)
                    # And we want Uf[i*4+j, :] @ (that matrix).T = ones(4)
                    A_mat = np.zeros((4, r_eff))
                    for k in range(4):
                        A_mat[k, :] = Vf[j * 4 + k, :] * Wf[k * 4 + i, :]
                    target = np.ones(4)
                    # Solve Uf[i*4+j, :] @ A_mat.T = target
                    # i.e., A_mat @ Uf[i*4+j, :].T = target
                    sol, _, _, _ = np.linalg.lstsq(A_mat, target, rcond=None)
                    Uf[i * 4 + j, :] = sol
            
            # Update V
            for j in range(4):
                for k in range(4):
                    A_mat = np.zeros((4, r_eff))
                    for i in range(4):
                        A_mat[i, :] = Uf[i * 4 + j, :] * Wf[k * 4 + i, :]
                    target = np.ones(4)
                    sol, _, _, _ = np.linalg.lstsq(A_mat, target, rcond=None)
                    Vf[j * 4 + k, :] = sol
            
            # Update W
            for k in range(4):
                for i in range(4):
                    A_mat = np.zeros((4, r_eff))
                    for j in range(4):
                        A_mat[j, :] = Uf[i * 4 + j, :] * Vf[j * 4 + k, :]
                    target = np.ones(4)
                    sol, _, _, _ = np.linalg.lstsq(A_mat, target, rcond=None)
                    Wf[k * 4 + i, :] = sol
            
            # Compute error
            recon = np.zeros((4, 4, 4))
            for i in range(4):
                for j in range(4):
                    for k in range(4):
                        recon[i, j, k] = np.sum(Uf[i * 4 + j, :] * Vf[j * 4 + k, :] * Wf[k * 4 + i, :])
            err = np.max(np.abs(recon - 1.0))
            if err < best_err:
                best_err = err
                best_factors = (Uf.copy(), Vf.copy(), Wf.copy())
            
            if best_err < 1e-8:
                break
        
        if best_err < 1e-8:
            break
    
    if best_factors is None:
        return None
    
    Uf, Vf, Wf = best_factors
    ok, Ur, Vr, Wr = _round_and_verify(Uf, Vf, Wf)
    if ok:
        return {"U": Ur.tolist(), "V": Vr.tolist(), "W": Wr.tolist()}
    return None


def _schoolbook_64():
    """Return the rank-64 schoolbook decomposition."""
    n = 4
    rank = 64
    U = np.zeros((16, rank))
    V = np.zeros((16, rank))
    W = np.zeros((16, rank))
    col = 0
    for i in range(n):
        for j in range(n):
            for k in range(n):
                U[i * n + j, col] = 1.0
                V[j * n + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}


def _strassen_49():
    """Construct the rank-49 recursive Strassen decomposition for 4x4 MM."""
    n, m, p = 4, 4, 4
    rank = 49
    
    U49 = np.zeros((16, rank))
    V49 = np.zeros((16, rank))
    W49 = np.zeros((16, rank))
    
    a_super = [
        [[1, 0], [0, 1]],
        [[0, 0], [1, 1]],
        [[1, 0], [0, 0]],
        [[0, 0], [0, 1]],
        [[1, 1], [0, 0]],
        [[-1, 0], [1, 0]],
        [[0, 1], [0, -1]],
    ]
    b_super = [
        [[1, 0], [0, 1]],
        [[1, 0], [0, 0]],
        [[0, 0], [1, -1]],
        [[-1, 1], [0, 0]],
        [[0, 0], [0, 1]],
        [[1, 1], [0, 0]],
        [[0, 0], [1, 1]],
    ]
    c_super = [
        [[1, 0, 0, 1, -1, 0, 1],
         [0, 0, 1, 0, 1, 0, 0]],
        [[0, 1, 0, 1, 0, 0, 0],
         [1, -1, 1, 0, 0, 1, 0]]
    ]
    
    for q in range(7):
        for s in range(7):
            col = q * 7 + s
            for br in range(2):
                for bc in range(2):
                    for i2 in range(2):
                        for j2 in range(2):
                            i = 2 * br + i2
                            j = 2 * bc + j2
                            coeff = a_super[q][br][bc] * a_super[s][i2][j2]
                            U49[i * 4 + j, col] += coeff
            for kr in range(2):
                for kc in range(2):
                    for j2 in range(2):
                        for k2 in range(2):
                            j = 2 * kr + j2
                            k = 2 * kc + k2
                            coeff = b_super[q][kr][kc] * b_super[s][j2][k2]
                            V49[j * 4 + k, col] += coeff
            for i_blk in range(2):
                for k_blk in range(2):
                    for i2 in range(2):
                        for k2 in range(2):
                            i = 2 * i_blk + i2
                            k = 2 * k_blk + k2
                            coeff = c_super[i_blk][k_blk][q] * c_super[i2][k2][s]
                            W49[k * 4 + i, col] += coeff
    
    return {"U": U49.tolist(), "V": V49.tolist(), "W": W49.tolist()}


def _block_rank37():
    """Block-partition rank-37 decomposition."""
    n = 4
    total_rank = 37
    U = np.zeros((16, total_rank))
    V = np.zeros((16, total_rank))
    W = np.zeros((16, total_rank))
    col = 0
    
    # 3x3 * 3x3 schoolbook: rank 9
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U[i * 4 + j, col] = 1.0
                V[j * 4 + k, col] = 1.0
                W[k * 4 + i, col] = 1.0
                col += 1
    
    # 3x3 * 3x1: rank 9
    for i in range(3):
        for j in range(3):
            U[i * 4 + j, col] = 1.0
            V[j * 4 + 3, col] = 1.0
            W[3 * 4 + i, col] = 1.0
            col += 1
    
    # 3x1 * 1x1: rank 3
    for i in range(3):
        U[i * 4 + 3, col] = 1.0
        V[3 * 4 + 3, col] = 1.0
        W[3 * 4 + i, col] = 1.0
        col += 1
    
    # 1x3 * 3x3: rank 9
    for k in range(3):
        for j in range(3):
            U[3 * 4 + j, col] = 1.0
            V[j * 4 + k, col] = 1.0
            W[k * 4 + 3, col] = 1.0
            col += 1
    
    # 1x1 * 1x3: rank 3
    for k in range(3):
        U[3 * 4 + 3, col] = 1.0
        V[3 * 4 + k, col] = 1.0
        W[k * 4 + 3, col] = 1.0
        col += 1
    
    # 1x3 * 3x1: rank 3
    for j in range(3):
        U[3 * 4 + j, col] = 1.0
        V[j * 4 + 3, col] = 1.0
        W[3 * 4 + 3, col] = 1.0
        col += 1
    
    # 1x1 * 1x1: rank 1
    U[3 * 4 + 3, col] = 1.0
    V[3 * 4 + 3, col] = 1.0
    W[3 * 4 + 3, col] = 1.0
    col += 1
    
    assert col == total_rank
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}


def _verify(U, V, W, tol=1e-6):
    """Verify the decomposition."""
    n = 4
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    rank = U.shape[1]
    U4 = U.reshape(n, n, rank)
    V4 = V.reshape(n, n, rank)
    W4 = W.reshape(n, n, rank)
    recon = np.zeros((n, n, n))
    for i in range(n):
        for j in range(n):
            for k in range(n):
                recon[i, j, k] = np.sum(U4[i, j, :] * V4[j, k, :] * W4[k, i, :])
    target = np.ones((n, n, n))
    return np.max(np.abs(recon - target)) < tol


def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin
    
    # Strategy 1: Try numerical ALS-based tensor decomposition for various ranks
    # The known rank of 4x4 MM is at most 49 (Strassen). Try to find lower.
    M = _build_matricization()
    
    # Try ranks from 32 up to 48 (known upper bounds)
    for r in [32, 33, 34, 35, 36, 37, 38, 39, 40, 42, 44, 45, 46, 48]:
        if time.time() > deadline - 30:
            break
        try:
            result = _try_rank(r, M, deadline - 30)
            if result is not None:
                return result
        except Exception:
            continue
    
    # Strategy 2: Block partition rank 37
    if time.time() < deadline - 10:
        try:
            sol37 = _block_rank37()
            if _verify(sol37['U'], sol37['V'], sol37['W']):
                return sol37
        except Exception:
            pass
    
    # Strategy 3: Strassen rank 49
    if time.time() < deadline - 5:
        try:
            sol49 = _strassen_49()
            if _verify(sol49['U'], sol49['V'], sol49['W']):
                return sol49
        except Exception:
            pass
    
    # Fallback: schoolbook rank 64
    return _schoolbook_64()

# EVOLVE-BLOCK-END
