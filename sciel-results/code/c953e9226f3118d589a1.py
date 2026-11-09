"""Rewritten program for 5x5 matrix multiplication rank minimization."""
import time
import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 5, 5, 5
    deadline = time.time() + 140  # safety margin before 150s kill

    # Use block decomposition: 4x4 schoolbook + border terms
    # Split indices into {0,1,2,3} (block) and {4} (border)
    # C[i,j] = sum_k A[i,k]*B[k,j]
    # 
    # Block part: i in 0..3, j in 0..3, k in 0..3: rank 64
    # Border terms:
    #   i in 0..3, j=4, k=4: A[i,4]*B[4,j] -> 4 terms
    #   i=4, j in 0..3, k in 0..3: A[4,k]*B[k,j] -> 4 terms  
    #   i=4, j=4, k in 0..3: A[4,k]*B[k,4] -> 4 terms
    #   i=4, j=4, k=4: A[4,4]*B[4,4] -> 1 term
    # Total: 64 + 4 + 4 + 4 + 1 = 77

    def to_list(mat):
        return [[float(x) for x in row] for row in mat]

    # Build decomposition
    U_rows = []
    V_rows = []
    W_rows = []

    # Helper to add a term
    def add_term(i, j, k):
        """Add term for A[i,j] * B[j,k] -> C[k,i]"""
        U_rows.append([i * m + j])
        V_rows.append([j * p + k])
        W_rows.append([k * n + i])

    # Block part: i,j,k all in 0..3
    for i in range(4):
        for j in range(4):
            for k in range(4):
                add_term(i, j, k)

    # Border: i in 0..3, j=4, k=4
    for i in range(4):
        add_term(i, 4, 4)

    # Border: i=4, j in 0..3, k in 0..3
    for j in range(4):
        for k in range(4):
            add_term(4, j, k)

    # Border: i=4, j=4, k in 0..3
    for k in range(4):
        add_term(4, 4, k)

    # Border: i=4, j=4, k=4
    add_term(4, 4, 4)

    rank = len(U_rows)
    
    # Build matrices
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    for col_idx in range(rank):
        U[U_rows[col_idx][0], col_idx] = 1
        V[V_rows[col_idx][0], col_idx] = 1
        W[W_rows[col_idx][0], col_idx] = 1

    # Verify correctness
    # Reconstruct tensor: T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    # Should equal 1 if (i,j,k) is a valid index triple, 0 otherwise
    # Actually for matrix mult: C[i,j] = sum_k A[i,k]*B[k,j]
    # Tensor: T[i,k,j] = 1 for all i,k,j (it's the identity tensor for matmul)
    # U has shape (n*m, rank): index i*m+j
    # V has shape (m*p, rank): index j*p+k  
    # W has shape (p*n, rank): index k*n+i
    
    # Check: for each (i,j,k), sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] should be 1
    # since each term in schoolbook contributes exactly 1 to its (i,j,k) entry
    
    # Let's verify a few entries
    # Actually, let's do a full verification
    T_reconstructed = np.zeros((n, m, p))
    for r in range(rank):
        # Find which (i,j), (j,k), (k,i) this term corresponds to
        # U[i*m+j, r] = 1, V[j*p+k, r] = 1, W[k*n+i, r] = 1
        u_idx = np.where(U[:, r] != 0)[0]
        v_idx = np.where(V[:, r] != 0)[0]
        w_idx = np.where(W[:, r] != 0)[0]
        if len(u_idx) == 1 and len(v_idx) == 1 and len(w_idx) == 1:
            ui, vi, wi = u_idx[0], v_idx[0], w_idx[0]
            i1, j1 = divmod(ui, m)
            j2, k1 = divmod(vi, p)
            k2, i2 = divmod(wi, n)
            if i1 == i2 and j1 == j2:
                T_reconstructed[i1, j1, k1] += 1.0

    # Check all entries should be 1
    all_ones = np.ones((n, m, p))
    if not np.allclose(T_reconstructed, all_ones):
        # Fallback to schoolbook
        rank_schoolbook = n * m * p
        U = np.zeros((n * m, rank_schoolbook))
        V = np.zeros((m * p, rank_schoolbook))
        W = np.zeros((p * n, rank_schoolbook))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}

    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
# EVOLVE-BLOCK-END
