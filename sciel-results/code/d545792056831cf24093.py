import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 4, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 130  # safety margin

    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n  # 8, 24, 12

    # Build the rank-42 Strassen decomposition
    rank0 = 42
    U = np.zeros((nm, rank0), dtype=float)
    V = np.zeros((mp, rank0), dtype=float)
    W = np.zeros((pn, rank0), dtype=float)

    def add_strassen_block(i_start, j_start, k_start, offset):
        a00 = (i_start + 0) * m + (j_start + 0)
        a01 = (i_start + 0) * m + (j_start + 1)
        a10 = (i_start + 1) * m + (j_start + 0)
        a11 = (i_start + 1) * m + (j_start + 1)

        b00 = (j_start + 0) * p + (k_start + 0)
        b01 = (j_start + 0) * p + (k_start + 1)
        b10 = (j_start + 1) * p + (k_start + 0)
        b11 = (j_start + 1) * p + (k_start + 1)

        w00 = (k_start + 0) * n + (i_start + 0)
        w01 = (k_start + 1) * n + (i_start + 0)
        w10 = (k_start + 0) * n + (i_start + 1)
        w11 = (k_start + 1) * n + (i_start + 1)

        # M1: (A00+A11)(B00+B11)
        U[a00, offset] += 1.0; U[a11, offset] += 1.0
        V[b00, offset] += 1.0; V[b11, offset] += 1.0
        W[w00, offset] += 1.0; W[w11, offset] += 1.0

        # M2: (A10+A11)B00
        U[a10, offset+1] += 1.0; U[a11, offset+1] += 1.0
        V[b00, offset+1] += 1.0
        W[w10, offset+1] += 1.0; W[w11, offset+1] -= 1.0

        # M3: A00(B01-B11)
        U[a00, offset+2] += 1.0
        V[b01, offset+2] += 1.0; V[b11, offset+2] -= 1.0
        W[w01, offset+2] += 1.0; W[w11, offset+2] += 1.0

        # M4: A11(B10-B00)
        U[a11, offset+3] += 1.0
        V[b10, offset+3] += 1.0; V[b00, offset+3] -= 1.0
        W[w00, offset+3] += 1.0; W[w10, offset+3] += 1.0

        # M5: (A00+A01)B11
        U[a00, offset+4] += 1.0; U[a01, offset+4] += 1.0
        V[b11, offset+4] += 1.0
        W[w00, offset+4] -= 1.0; W[w01, offset+4] += 1.0

        # M6: (A10-A00)(B00+B01)
        U[a10, offset+5] += 1.0; U[a00, offset+5] -= 1.0
        V[b00, offset+5] += 1.0; V[b01, offset+5] += 1.0
        W[w11, offset+5] += 1.0

        # M7: (A01-A11)(B10+B11)
        U[a01, offset+6] += 1.0; U[a11, offset+6] -= 1.0
        V[b10, offset+6] += 1.0; V[b11, offset+6] += 1.0
        W[w00, offset+6] += 1.0

    # Six 2x2x2 Strassen blocks: 2 (i-blocks) x 1 (j-block) x 3 (k-blocks)
    # Actually n=2, m=4, p=6: i in {0,1}, j in {0,1,2,3}, k in {0,1,2,3,4,5}
    # Split j into 2 blocks of 2, k into 3 blocks of 2
    for j_blk in range(2):
        for k_blk in range(3):
            add_strassen_block(0, j_blk * 2, k_blk * 2, (j_blk * 3 + k_blk) * 7)

    # Now find linear dependencies among the 42 terms
    # Flatten each rank-1 term into a vector of length nm*mp*pn = 2304
    total_dim = nm * mp * pn
    terms_mat = np.zeros((total_dim, rank0), dtype=float)
    for r in range(rank0):
        terms_mat[:, r] = np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r]).reshape(total_dim)

    # Use QR with column pivoting to find the numerical rank
    # This is more efficient than the greedy approach
    from scipy.linalg import qr
    Q, R, piv = qr(terms_mat, mode='economic', pivoting=True)

    # Determine numerical rank from R
    diag_R = np.abs(np.diag(R))
    if len(diag_R) > 0:
        tol = diag_R[0] * max(total_dim, rank0) * np.finfo(float).eps * 10
        num_rank = np.sum(diag_R > tol)
    else:
        num_rank = rank0

    # If we found dependencies, extract the independent columns
    if num_rank < rank0:
        # The first num_rank columns of terms_mat after pivoting form a basis
        # piv gives the column permutation
        selected_cols = piv[:num_rank]

        U_new = U[:, selected_cols].copy()
        V_new = V[:, selected_cols].copy()
        W_new = W[:, selected_cols].copy()
        new_rank = num_rank
    else:
        U_new = U.copy()
        V_new = V.copy()
        W_new = W.copy()
        new_rank = rank0

    # Verify the decomposition
    def verify_decomp(U_mat, V_mat, W_mat, tol=1e-8):
        for i in range(n):
            for j in range(m):
                a = i * m + j
                for k in range(p):
                    b = j * p + k
                    c = k * n + i
                    # sum_r U[a,r] * V[b,r] * W[c,r] should equal 1
                    s = np.dot(U_mat[a, :], V_mat[b, :] * W_mat[c, :])
                    if abs(s - 1.0) > tol:
                        return False
        return True

    if verify_decomp(U_new, V_new, W_new):
        best_rank = new_rank
        best_U = U_new
        best_V = V_new
        best_W = W_new
    else:
        # Fall back to original rank-42
        best_rank = rank0
        best_U = U
        best_V = V
        best_W = W

    # Try to further reduce: check if we can do better with a direct approach
    # Build the full matmul tensor and try to find its tensor rank via unfolding
    if time.time() < deadline - 20:
        # The matmul tensor T[a,b,c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i
        # There are n*m*p = 48 non-zero entries
        # The tensor rank is at most 48 (schoolbook)
        # We already have 42 or less via Strassen
        
        # Try to see if we can find rank < best_rank by checking
        # if any of the remaining terms are redundant
        # For now, accept the QR result
        
        pass

    # Final verification
    if not verify_decomp(best_U, best_V, best_W):
        # Absolute fallback: schoolbook rank 48
        best_rank = n * m * p
        best_U = np.zeros((nm, best_rank), dtype=float)
        best_V = np.zeros((mp, best_rank), dtype=float)
        best_W = np.zeros((pn, best_rank), dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    r = i * (m * p) + j * p + k
                    best_U[i * m + j, r] = 1
                    best_V[j * p + k, r] = 1
                    best_W[k * n + i, r] = 1

    # Time check
    if time.time() > deadline - 5:
        pass  # Return immediately

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
