"""Matrix multiplication 2x4x6: rank-42 Strassen-block construction, cleaned and verified."""

import time
import numpy as np

DIMENSIONS = [2, 4, 6]

def construct():
    t0 = time.time()
    deadline = 145.0
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n

    rank = 42
    U = np.zeros((nm, rank), dtype=float)
    V = np.zeros((mp, rank), dtype=float)
    W = np.zeros((pn, rank), dtype=float)

    # Strassen 7-term pattern for 2x2x2 blocks
    # Term offsets and their (u_indices, v_indices, w_indices, signs)
    # Local 2x2x2: A has 4 entries (a00,a01,a10,a11), B has 4 (b00,b01,b10,b11), C has 4 (c00,c01,c10,c11)
    # C = A*B, so C(i,k) = sum_j A(i,j)B(j,k)
    # Strassen:
    # M1 = (A00+A11)(B00+B11) -> C00+=M1, C11+=M1
    # M2 = (A10+A11)B00       -> C10+=M2, C11-=M2
    # M3 = A00(B01-B11)       -> C01+=M3, C11+=M3
    # M4 = A11(B10-B00)       -> C00+=M4, C10+=M4
    # M5 = (A00+A01)B11       -> C00-=M5, C01+=M5
    # M6 = (A10-A00)(B00+B01) -> C11+=M6
    # M7 = (A01-A11)(B10+B11) -> C00+=M7

    # For each 2x2x2 block at (i_start, j_start, k_start), 7 terms
    def add_block(i0, j0, k0, offset):
        # A indices: global row i*m + j
        a00 = i0 * m + j0
        a01 = i0 * m + j0 + 1
        a10 = (i0+1) * m + j0
        a11 = (i0+1) * m + j0 + 1

        # B indices: global row j*p + k
        b00 = j0 * p + k0
        b01 = j0 * p + k0 + 1
        b10 = (j0+1) * p + k0
        b11 = (j0+1) * p + k0 + 1

        # W indices: W[k*n + i, r]
        w00 = k0 * n + i0
        w01 = (k0+1) * n + i0
        w10 = k0 * n + i0 + 1
        w11 = (k0+1) * n + i0 + 1

        # M1
        U[a00, offset] = 1.0; U[a11, offset] = 1.0
        V[b00, offset] = 1.0; V[b11, offset] = 1.0
        W[w00, offset] = 1.0; W[w11, offset] = 1.0

        # M2
        U[a10, offset+1] = 1.0; U[a11, offset+1] = 1.0
        V[b00, offset+1] = 1.0
        W[w10, offset+1] = 1.0; W[w11, offset+1] = -1.0

        # M3
        U[a00, offset+2] = 1.0
        V[b01, offset+2] = 1.0; V[b11, offset+2] = -1.0
        W[w01, offset+2] = 1.0; W[w11, offset+2] = 1.0

        # M4
        U[a11, offset+3] = 1.0
        V[b10, offset+3] = 1.0; V[b00, offset+3] = -1.0
        W[w00, offset+3] = 1.0; W[w10, offset+3] = 1.0

        # M5
        U[a00, offset+4] = 1.0; U[a01, offset+4] = 1.0
        V[b11, offset+4] = 1.0
        W[w00, offset+4] = -1.0; W[w01, offset+4] = 1.0

        # M6
        U[a10, offset+5] = 1.0; U[a00, offset+5] = -1.0
        V[b00, offset+5] = 1.0; V[b01, offset+5] = 1.0
        W[w11, offset+5] = 1.0

        # M7
        U[a01, offset+6] = 1.0; U[a11, offset+6] = -1.0
        V[b10, offset+6] = 1.0; V[b11, offset+6] = 1.0
        W[w00, offset+6] = 1.0

    # n=2, m=4, p=6
    # 2x2x2 blocks: i_start in {0}, j_start in {0,2}, k_start in {0,2,4}
    # That's 1 * 2 * 3 = 6 blocks, each with 7 terms = 42
    offset = 0
    for i0 in range(0, n, 2):
        for j0 in range(0, m, 2):
            for k0 in range(0, p, 2):
                add_block(i0, j0, k0, offset)
                offset += 7

    def verify(U_mat, V_mat, W_mat, tol=1e-8):
        r = U_mat.shape[1]
        # Check all (a,b,c) where the tensor should be 1 or 0
        # Tensor T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
        # T should be 1 when a=i*m+j, b=j*p+k, c=k*n+i for some i,j,k
        # and 0 otherwise
        # Build the expected tensor as a sparse set
        ones = set()
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    ones.add((i*m+j, j*p+k, k*n+i))
        
        # For efficiency, compute T as tensor contraction
        # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
        # We can compute this as: for each r, outer product
        # But with 42 terms and 8*24*12 = 2304 entries, it's fine to check directly
        
        for a in range(nm):
            i_a = a // m
            j_a = a % m
            for b in range(mp):
                j_b = b // p
                k_b = b % p
                for c in range(pn):
                    k_c = c // n
                    i_c = c % n
                    expected = 1.0 if (i_a == i_c and j_a == j_b and k_b == k_c) else 0.0
                    s = 0.0
                    for r in range(r):
                        s += U_mat[a, r] * V_mat[b, r] * W_mat[c, r]
                    if abs(s - expected) > tol:
                        return False
        return True

    if not verify(U, V, W):
        # Fallback: schoolbook rank 48
        rank0 = n * m * p
        U0 = np.zeros((nm, rank0), dtype=float)
        V0 = np.zeros((mp, rank0), dtype=float)
        W0 = np.zeros((pn, rank0), dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    col = i * (m * p) + j * p + k
                    U0[i*m+j, col] = 1.0
                    V0[j*p+k, col] = 1.0
                    W0[k*n+i, col] = 1.0
        return {"U": U0.tolist(), "V": V0.tolist(), "W": W0.tolist()}

    # Try to reduce rank by removing redundant columns
    # Check if any column is linearly dependent on others in a way that preserves the tensor
    # Simple approach: try removing each column and see if we can adjust others
    # This is expensive, so use a heuristic: look for zero rows/cols patterns
    
    # Actually, the Strassen construction for 2x2x2 is known to be rank-optimal (rank 7).
    # The Kronecker product with the 1x2x1 structure gives rank 42, which is likely optimal
    # for this dimension combination.
    
    # Let's try a quick check: can we merge any two columns?
    # Two columns r1, r2 can be merged if there exists alpha such that
    # U[:,r1] = alpha*U[:,r2], V[:,r1] = alpha*V[:,r2], W[:,r1] = alpha*W[:,r2]
    # (i.e., they are scalar multiples)
    
    best_rank = rank
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Check for proportional columns
    for r1 in range(rank):
        if time.time() > deadline - 5:
            break
        for r2 in range(r1+1, rank):
            # Check if columns are proportional
            u1 = U[:, r1]; u2 = U[:, r2]
            v1 = V[:, r1]; v2 = V[:, r2]
            w1 = W[:, r1]; w2 = W[:, r2]
            
            # Find non-zero entries in u1
            nz1 = np.nonzero(u1)[0]
            if len(nz1) == 0:
                continue
            
            # Check if u2 is proportional to u1
            alpha = u2[nz1[0]] / u1[nz1[0]]
            if not np.allclose(u1 * alpha, u2, atol=1e-10):
                continue
            if not np.allclose(v1 * alpha, v2, atol=1e-10):
                continue
            if not np.allclose(w1 * alpha, w2, atol=1e-10):
                continue
            
            # Columns are proportional: merge
            U[:, r1] = U[:, r1] + U[:, r2]
            V[:, r1] = V[:, r1] + V[:, r2]
            W[:, r1] = W[:, r1] + W[:, r2]
            U[:, r2] = 0
            V[:, r2] = 0
            W[:, r2] = 0
            break
        if time.time() > deadline - 5:
            break
    
    # Remove zero columns
    keep = []
    for r in range(rank):
        if np.any(U[:, r] != 0) or np.any(V[:, r] != 0) or np.any(W[:, r] != 0):
            keep.append(r)
    
    if len(keep) < best_rank:
        U_new = U[:, keep]
        V_new = V[:, keep]
        W_new = W[:, keep]
        if verify(U_new, V_new, W_new):
            best_U, best_V, best_W = U_new, V_new, W_new
            best_rank = len(keep)
    
    if not verify(best_U, best_V, best_W):
        # Fallback to original
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
