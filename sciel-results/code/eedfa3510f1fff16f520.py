import numpy as np
import time
from itertools import combinations

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin
    
    # Build the rank-28 solution
    rank = 28
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0

    def add_strassen(i1, i2, j1, j2, k1, k2):
        nonlocal col
        # M1 = (A[i1,j1]+A[i2,j2]) * (B[j1,k1]+B[j2,k2])
        U[i1*m+j1, col] = 1
        U[i2*m+j2, col] = 1
        V[j1*p+k1, col] = 1
        V[j2*p+k2, col] = 1
        W[k1*n+i1, col] = 1
        W[k2*n+i2, col] = 1
        col += 1

        # M2 = (A[i2,j1]+A[i2,j2]) * B[j1,k1]
        U[i2*m+j1, col] = 1
        U[i2*m+j2, col] = 1
        V[j1*p+k1, col] = 1
        W[k1*n+i2, col] = 1
        W[k2*n+i2, col] = -1
        col += 1

        # M3 = A[i1,j1] * (B[j1,k2]-B[j2,k2])
        U[i1*m+j1, col] = 1
        V[j1*p+k2, col] = 1
        V[j2*p+k2, col] = -1
        W[k2*n+i1, col] = 1
        W[k2*n+i2, col] = 1
        col += 1

        # M4 = A[i2,j2] * (B[j2,k1]-B[j1,k1])
        U[i2*m+j2, col] = 1
        V[j2*p+k1, col] = 1
        V[j1*p+k1, col] = -1
        W[k1*n+i1, col] = 1
        W[k1*n+i2, col] = 1
        col += 1

        # M5 = (A[i1,j1]+A[i1,j2]) * B[j2,k2]
        U[i1*m+j1, col] = 1
        U[i1*m+j2, col] = 1
        V[j2*p+k2, col] = 1
        W[k1*n+i1, col] = -1
        W[k2*n+i1, col] = 1
        col += 1

        # M6 = (A[i2,j1]-A[i1,j1]) * (B[j1,k1]+B[j1,k2])
        U[i2*m+j1, col] = 1
        U[i1*m+j1, col] = -1
        V[j1*p+k1, col] = 1
        V[j1*p+k2, col] = 1
        W[k2*n+i2, col] = 1
        col += 1

        # M7 = (A[i1,j2]-A[i2,j2]) * (B[j2,k1]+B[j2,k2])
        U[i1*m+j2, col] = 1
        U[i2*m+j2, col] = -1
        V[j2*p+k1, col] = 1
        V[j2*p+k2, col] = 1
        W[k1*n+i1, col] = 1
        col += 1

    def add_schoolbook(i_list, j_list, k_list):
        nonlocal col
        for i in i_list:
            for j in j_list:
                for k in k_list:
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1

    add_strassen(0, 1, 0, 1, 0, 1)
    add_strassen(0, 1, 0, 1, 2, 3)
    add_schoolbook([0, 1], [0, 1], [4])
    add_schoolbook([0, 1], [2], [0, 1])
    add_schoolbook([0, 1], [2], [2, 3])
    add_schoolbook([0, 1], [2], [4])

    assert col == rank

    def verify(U_arr, V_arr, W_arr, r):
        tensor = np.zeros((n, m, p))
        for c in range(r):
            uc = U_arr[:, c].reshape(n, m)
            vc = V_arr[:, c].reshape(m, p)
            wc = W_arr[:, c].reshape(p, n)
            tensor += np.einsum('ij,jk,ki->ijk', uc, vc, wc)
        return np.allclose(tensor, 1.0, atol=1e-9)

    if not verify(U, V, W, rank):
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Try to reduce rank
    # Check if we can find a rank-27 solution
    # by trying to eliminate one term and adjusting others
    
    # For each term to remove, check if the remaining 27 terms
    # can be adjusted to still decompose the tensor
    # This is a linear system problem
    
    # The tensor equation is:
    # Σ_c U_c ⊗ V_c ⊗ W_c = T
    # where T is the all-ones tensor
    
    # If we remove term r, we need:
    # Σ_{c≠r} U_c ⊗ V_c ⊗ W_c = T - U_r ⊗ V_r ⊗ W_r
    
    # This is a linear system in the coefficients, but since we're
    # using fixed U_c, V_c, W_c, we just need to check if the
    # remaining terms can sum to the target.
    
    # Actually, since the U_c, V_c, W_c are fixed (binary), we can't
    # adjust them. So we can only remove a term if the remaining
    # terms already sum to T.
    
    # This means: T - U_r ⊗ V_r ⊗ W_r must be decomposable by the
    # remaining 27 terms.
    
    # Let's check this for each r.
    
    target = np.ones((n, m, p))
    
    for r in range(rank):
        if time.time() > deadline - 5:
            break
        
        # Compute what the remaining terms give
        remaining = np.zeros((n, m, p))
        for c in range(rank):
            if c == r:
                continue
            uc = U[:, c].reshape(n, m)
            vc = V[:, c].reshape(m, p)
            wc = W[:, c].reshape(p, n)
            remaining += np.einsum('ij,jk,ki->ijk', uc, vc, wc)
        
        # Check if remaining equals target
        if np.allclose(remaining, target, atol=1e-9):
            # We can remove term r
            new_rank = rank - 1
            U_new = np.delete(U, r, axis=1)
            V_new = np.delete(V, r, axis=1)
            W_new = np.delete(W, r, axis=1)
            if verify(U_new, V_new, W_new, new_rank):
                return {"U": U_new.tolist(), "V": V_new.tolist(), "W": W_new.tolist()}
    
    # If no single term can be removed, try removing 2 terms
    # and see if we can find a replacement
    # This is more complex, so let's skip for now
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
