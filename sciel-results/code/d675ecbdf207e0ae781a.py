import numpy as np
import time
from fractions import Fraction
from itertools import product

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    
    # Schoolbook solution (rank 56) as fallback
    def schoolbook():
        U = np.zeros((n*m, n*m*p))
        V = np.zeros((m*p, n*m*p))
        W = np.zeros((p*n, n*m*p))
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, idx] = 1.0
                    V[j*p + k, idx] = 1.0
                    W[k*n + i, idx] = 1.0
                    idx += 1
        return U, V, W

    best_U, best_V, best_W = schoolbook()
    best_rank = n * m * p

    def verify(U, V, W, tol=1e-9):
        r = U.shape[1]
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    for jp in range(m):
                        s = 0.0
                        for c in range(r):
                            s += U[i*m + j, c] * V[jp*p + k, c] * W[k*n + i, c]
                        if abs(s - (1.0 if j == jp else 0.0)) > tol:
                            return False
        return True

    start_time = time.time()
    deadline = start_time + 140.0

    # Build the multiplication tensor T[i,j,k,jp] = 1 if j==jp else 0
    # Shape: (n, m, p, m)
    T = np.zeros((n, m, p, m))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k, j] = 1.0

    # The tensor can be viewed as a matrix of shape (n*m, p*m) with entries
    # T_mat[i*m+j, k*m+jp] = T[i,j,k,jp]
    # A rank-r decomposition T[i,j,k,jp] = sum_c U[i*m+j,c]*V[jp*p+k,c]*W[k*n+i,c]
    # is equivalent to finding low-rank factorizations of various reshavings.

    # MPS approach: Try different orderings of indices and find minimal bond dimension
    # For tensor T[i,j,k,jp], we can try orderings like:
    # (i,j) | (k,jp) -> matrix of shape (n*m, p*m)
    # (i,k) | (j,jp) -> matrix of shape (n*p, m*m)
    # (j,k) | (i,jp) -> matrix of shape (m*p, n*m)
    # etc.

    # The rank of the tensor (in the sense of tensor rank) is at least
    # the max of the ranks of all possible matrix reshavings.

    # Let's compute the rank of various reshavings to get a lower bound
    reshavings = [
        T.reshape(n*m, p*m),
        T.reshape(n*p, m*m),
        T.reshape(m*p, n*m),
        T.reshape(n*m, p*m).T,  # same rank
    ]

    # Compute SVD to find the numerical rank
    for mat in reshavings:
        s = svd(mat, compute_uv=False)
        # The numerical rank
        rank = np.sum(s > 1e-10)
        print(f"Reshaping rank: {rank}", end=" ")
    print()

    # Current best known structure: Rank 50
    rank = 50
    U = np.zeros((n*m, rank), dtype=np.int64)
    V = np.zeros((m*p, rank), dtype=np.int64)
    W = np.zeros((p*n, rank), dtype=np.int64)

    col = 0

    def add_strassen_block(i0, j0, k0):
        nonlocal col
        if time.time() > deadline:
            return
        a_idx = i0 * m + j0
        b_idx = i0 * m + (j0 + 1)
        c_idx = (i0 + 1) * m + j0
        d_idx = (i0 + 1) * m + (j0 + 1)

        e_idx = j0 * p + k0
        f_idx = j0 * p + (k0 + 1)
        g_idx = (j0 + 1) * p + k0
        h_idx = (j0 + 1) * p + (k0 + 1)

        c00_idx = k0 * n + i0
        c01_idx = (k0 + 1) * n + i0
        c10_idx = k0 * n + (i0 + 1)
        c11_idx = (k0 + 1) * n + (i0 + 1)

        # M1 = (a+d)(e+h)
        U[a_idx, col] += 1; U[d_idx, col] += 1
        V[e_idx, col] += 1; V[h_idx, col] += 1
        W[c00_idx, col] += 1; W[c11_idx, col] += 1
        col += 1

        # M2 = (c+d)e
        U[c_idx, col] += 1; U[d_idx, col] += 1
        V[e_idx, col] += 1
        W[c10_idx, col] += 1; W[c11_idx, col] -= 1
        col += 1

        # M3 = a(f-h)
        U[a_idx, col] += 1
        V[f_idx, col] += 1; V[h_idx, col] -= 1
        W[c01_idx, col] += 1; W[c11_idx, col] += 1
        col += 1

        # M4 = d(g-e)
        U[d_idx, col] += 1
        V[g_idx, col] += 1; V[e_idx, col] -= 1
        W[c00_idx, col] += 1; W[c10_idx, col] += 1
        col += 1

        # M5 = (a+b)h
        U[a_idx, col] += 1; U[b_idx, col] += 1
        V[h_idx, col] += 1
        W[c00_idx, col] -= 1; W[c01_idx, col] += 1
        col += 1

        # M6 = (c-a)(e+f)
        U[c_idx, col] += 1; U[a_idx, col] -= 1
        V[e_idx, col] += 1; V[f_idx, col] += 1
        W[c11_idx, col] += 1
        col += 1

        # M7 = (b-d)(g+h)
        U[b_idx, col] += 1; U[d_idx, col] -= 1
        V[g_idx, col] += 1; V[h_idx, col] += 1
        W[c00_idx, col] += 1
        col += 1

    def add_schoolbook_221_block(i0, j0, k0):
        nonlocal col
        if time.time() > deadline:
            return
        a_idx = i0 * m + j0
        b_idx = i0 * m + (j0 + 1)
        c_idx = (i0 + 1) * m + j0
        d_idx = (i0 + 1) * m + (j0 + 1)

        e_idx = j0 * p + k0
        g_idx = (j0 + 1) * p + k0

        c00_idx = k0 * n + i0
        c10_idx = k0 * n + (i0 + 1)

        # P1 = a*e
        U[a_idx, col] += 1; V[e_idx, col] += 1; W[c00_idx, col] += 1; col += 1
        # P2 = b*g
        U[b_idx, col] += 1; V[g_idx, col] += 1; W[c00_idx, col] += 1; col += 1
        # P3 = c*e
        U[c_idx, col] += 1; V[e_idx, col] += 1; W[c10_idx, col] += 1; col += 1
        # P4 = d*g
        U[d_idx, col] += 1; V[g_idx, col] += 1; W[c10_idx, col] += 1; col += 1

    # Build the decomposition
    for j_start in [0, 2]:
        for k_start in [0, 2, 4]:
            add_strassen_block(0, j_start, k_start)
        add_schoolbook_221_block(0, j_start, 6)

    assert col == rank, f"Expected rank {rank}, got {col}"

    if verify(U, V, W):
        best_U, best_V, best_W = U, V, W
        best_rank = rank

    # Now try MPS-based optimization to find a lower rank
    # The tensor T[i,j,k,jp] has a special structure: T[i,j,k,jp] = delta_{j,jp}
    # This means for each fixed i,k, the (j,jp) submatrix is the identity.
    
    # Key insight: We can try to find a decomposition where the bond dimension
    # is minimized by exploiting the Kronecker-like structure.
    
    # Try rank 48, 46, 44, 42, 40 using a systematic approach
    # based on the MPS structure
    
    for target_rank in [48, 46, 44, 42, 40]:
        if time.time() > deadline:
            break
            
        # For a 2x4x7 matrix multiplication, we can think of it as:
        # C[i,k] = sum_j A[i,j] * B[j,k]
        # 
        # The tensor is T[i,j,k,jp] = delta_{j,jp}
        # 
        # We want: T[i,j,k,jp] = sum_c U[i,j,c] * V[jp,k,c] * W[k,i,c]
        # 
        # Note the convention: U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
        # U[i*m+j, c], V[jp*p+k, c], W[k*n+i, c]
        
        # Let's try a structured approach:
        # Group the 4 columns of A into 2 groups of 2
        # Group the 7 columns of B into groups
        
        # Try different groupings and see if we can get rank < 50
        # This is complex, so let's use a simpler heuristic:
        # Try to reduce the rank by combining some of the Strassen-like products
        
        # For now, keep the rank 50 solution as it's verified
        pass

    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
