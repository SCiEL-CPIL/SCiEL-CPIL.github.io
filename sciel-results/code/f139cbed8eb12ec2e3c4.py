import numpy as np
import time
from itertools import combinations

DIMENSIONS = [2, 4, 4]

def _build_tensor():
    n, m, p = DIMENSIONS
    T = np.zeros((n*m, m*p, p*n))
    for i_out in range(n):
        for j in range(m):
            for k_out in range(p):
                T[i_out*m + j, j*p + k_out, k_out*n + i_out] = 1.0
    return T

def _verify(U, V, W, T, tol=1e-10):
    R = U.shape[1]
    Tc = np.zeros_like(T)
    for r in range(R):
        Tc += np.einsum('i,j,k->ijk', U[:,r], V[:,r], W[:,r])
    return np.allclose(Tc, T, atol=tol)

def _verify_fast(U, V, W, T, tol=1e-10):
    """Faster verification using matrix operations."""
    R = U.shape[1]
    # Tc[i,j,k] = sum_r U[i,r]*V[j,r]*W[k,r]
    # For each i, Tc[i,:,:] = sum_r U[i,r] * (V[:,r] * W[:,r])
    # V[:,r] * W[:,r] is element-wise product, shape (m*p, p*n)
    # Actually, let's just do it directly but more efficiently
    Tc = np.zeros_like(T)
    for r in range(R):
        Tc += np.outer(U[:,r], np.outer(V[:,r], W[:,r]).reshape(V.shape[0], -1))
    # Wait, that's not right. Let me think again.
    # Tc[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
    # For fixed r: outer(U[:,r], V[:,r]) gives matrix M_r where M_r[i,j] = U[i,r]*V[j,r]
    # Then Tc[i,j,k] = sum_r M_r[i,j] * W[k,r]
    # So Tc = sum_r np.einsum('ij,k->ijk', M_r, W[:,r])
    Tc = np.zeros_like(T)
    for r in range(R):
        M_r = np.outer(U[:,r], V[:,r])
        Tc += M_r[:, :, np.newaxis] * W[:, r][np.newaxis, np.newaxis, :]
    return np.allclose(Tc, T, atol=tol)

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = DIMENSIONS
    T = _build_tensor()
    
    # Build the rank-28 Strassen construction
    R = 28
    U = np.zeros((n * m, R), dtype=float)
    V = np.zeros((m * p, R), dtype=float)
    W = np.zeros((p * n, R), dtype=float)

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

    # Verify the construction
    if not _verify(best_U, best_V, best_W, T):
        # Fallback to schoolbook rank 32
        R = 32
        U = np.zeros((n*m, R), dtype=float)
        V = np.zeros((m*p, R), dtype=float)
        W = np.zeros((p*n, R), dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    r = i*m*p + j*p + k
                    U[i*m+j, r] = 1
                    V[j*p+k, r] = 1
                    W[k*n+i, r] = 1
        best_U, best_V, best_W = U, V, W
        best_rank = R

    # Try to find rank-27 by checking if any column is redundant
    # A column r is redundant if T can be written as sum of the other 27 rank-1 terms
    # with adjusted coefficients. This means the 28 rank-1 tensors are linearly dependent.
    
    # Check linear dependence of the 28 rank-1 tensors
    # Flatten each rank-1 tensor to a vector of length 8*16*8 = 1024
    # Then check if the 28 vectors are linearly dependent
    
    if time.time() < deadline:
        dim_a, dim_b, dim_c = n*m, m*p, p*n
        # Create matrix where each column is a flattened rank-1 tensor
        M = np.zeros((dim_a * dim_b * dim_c, R))
        for r in range(R):
            T_r = np.einsum('i,j,k->ijk', U[:,r], V[:,r], W[:,r])
            M[:, r] = T_r.flatten()
        
        # Check rank of M
        rank_M = np.linalg.matrix_rank(M, tol=1e-10)
        
        if rank_M < R:
            # There's linear dependence, so we might be able to reduce rank
            # Find the null space
            # If null space is non-trivial, we can adjust coefficients
            # But we need the adjusted decomposition to still be valid
            # (i.e., the new rank-1 terms must still be rank-1)
            # This is non-trivial, so let's just try removing each column
            for r_remove in range(R):
                if time.time() > deadline:
                    break
                keep = [r for r in range(R) if r != r_remove]
                U_keep = best_U[:, keep]
                V_keep = best_V[:, keep]
                W_keep = best_W[:, keep]
                
                Tc = np.zeros_like(T)
                for idx in range(len(keep)):
                    Tc += np.einsum('i,j,k->ijk', U_keep[:,idx], V_keep[:,idx], W_keep[:,idx])
                
                if np.allclose(Tc, T, atol=1e-10):
                    best_U, best_V, best_W = U_keep, V_keep, W_keep
                    best_rank = 27
                    break
    
    # Try rank-26 by removing two columns
    if best_rank > 26 and time.time() < deadline:
        R0 = best_rank
        for r1 in range(R0):
            for r2 in range(r1+1, R0):
                if time.time() > deadline:
                    break
                keep = [r for r in range(R0) if r != r1 and r != r2]
                U_keep = best_U[:, keep]
                V_keep = best_V[:, keep]
                W_keep = best_W[:, keep]
                
                Tc = np.zeros_like(T)
                for idx in range(len(keep)):
                    Tc += np.einsum('i,j,k->ijk', U_keep[:,idx], V_keep[:,idx], W_keep[:,idx])
                
                if np.allclose(Tc, T, atol=1e-10):
                    best_U, best_V, best_W = U_keep, V_keep, W_keep
                    best_rank = 26
                    break
            if best_rank == 26:
                break

    # Try a more aggressive search: use the tensor splitting idea
    # Partition m=4 into {0,1} and {2,3}
    # For each partition, solve the 2x2x4 sub-problem optimally
    # The 2x2x4 multiplication: A(2x2) * B(2x4) = C(2x4)
    # Tensor: T[i,j,k] = A[i,j]*B[j,k], i in {0,1}, j in {0,1}, k in {0,1,2,3}
    # This is equivalent to two 2x2x2 problems sharing A
    # Rank is at most 14 (7+7), but might be less
    
    # For now, let's try to verify if rank 27 is achievable by a different construction
    # Try: use Strassen for the 2x2x2 blocks but try to merge some terms
    
    if time.time() > deadline:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

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
