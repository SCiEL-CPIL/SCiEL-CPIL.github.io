import numpy as np
import time
from fractions import Fraction
import itertools

_START_TIME = time.time()
TIME_LIMIT = 145

def _time_remaining():
    return TIME_LIMIT - (time.time() - _START_TIME)

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition is correct."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    if U.shape != (n * m, rank):
        return False
    if V.shape != (m * p, rank):
        return False
    if W.shape != (p * n, rank):
        return False
    
    i_idx, j_idx, k_idx = np.meshgrid(
        np.arange(n), np.arange(m), np.arange(p), indexing='ij'
    )
    i_flat = i_idx.ravel()
    j_flat = j_idx.ravel()
    k_flat = k_idx.ravel()
    
    U_idx = i_flat * m + j_flat
    V_idx = j_flat * p + k_flat
    W_idx = k_flat * n + i_flat
    
    prod = U[U_idx, :] * V[V_idx, :] * W[W_idx, :]
    sums = prod.sum(axis=1)
    
    if not np.allclose(sums, 1.0, atol=1e-9):
        return False
    return True

def construct():
    n, m, p = 4, 4, 7
    total_rank = 93
    
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    current_rank = 0
    
    # Strassen 2x2x2 terms
    strassen_2x2 = [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (0,1): 0, (1,0): 0, (1,1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(0,0): 0, (0,1): 0, (1,0): 1, (1,1): -1}),
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,0): 0, (0,1): 1, (1,0): 0, (1,1): 1}),
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 0, (1,0): 1, (1,1): 0}),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1, (1,0): 0, (1,1): 0}),
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(0,0): 0, (0,1): 0, (1,0): 0, (1,1): 1}),
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1, (0,1): 0, (1,0): 0, (1,1): 0}),
    ]
    
    # Strassen 4x4x4: 7 outer terms
    strassen_4x4_outer = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}),
        ({2: 1, 3: 1}, {0: 1}),
        ({0: 1}, {1: 1, 3: -1}),
        ({3: 1}, {2: 1, 0: -1}),
        ({0: 1, 1: 1}, {3: 1}),
        ({2: 1, 0: -1}, {0: 1, 1: 1}),
        ({1: 1, 3: -1}, {2: 1, 3: 1}),
    ]
    
    c_blocks_4x4 = [
        {(0,0): 1.0, (1,1): 1.0},
        {(1,0): 1.0, (1,1): -1.0},
        {(0,1): 1.0, (1,1): 1.0},
        {(0,0): 1.0, (1,0): 1.0},
        {(0,0): -1.0, (0,1): 1.0},
        {(1,1): 1.0},
        {(0,0): 1.0},
    ]
    
    # Part 1: 4x4x4 Strassen (rank 49)
    for m_idx in range(7):
        x_def, y_def = strassen_4x4_outer[m_idx]
        c_block = c_blocks_4x4[m_idx]
        for t_idx in range(7):
            r = current_rank
            current_rank += 1
            x_sub, y_sub, c_sub = strassen_2x2[t_idx]
            
            for (li, lj), cx in x_sub.items():
                for b, cb in x_def.items():
                    bi, bj = b // 2, b % 2
                    i = 2 * bi + li
                    j = 2 * bj + lj
                    U[i * m + j, r] += cx * cb
            
            for (li, lj), cy in y_sub.items():
                for b, cb in y_def.items():
                    bi, bj = b // 2, b % 2
                    i = 2 * bi + li
                    j = 2 * bj + lj
                    V[i * p + j, r] += cy * cb
            
            for (li, lj), cc in c_sub.items():
                if cc == 0:
                    continue
                for (br, bc), cb_c in c_block.items():
                    if cb_c == 0:
                        continue
                    i_final = 2 * br + li
                    k_final = 2 * bc + lj
                    W[k_final * n + i_final, r] += cc * cb_c
    
    # Part 2: 4x4x3 as 4 * (2x2x3), each of rank 11
    def add_2x2x3(ar, ac, br, cr):
        nonlocal current_rank
        
        # Strassen 2x2x2 for columns 4,5
        for t_idx in range(7):
            r = current_rank
            current_rank += 1
            x_sub, y_sub, c_sub = strassen_2x2[t_idx]
            
            for (li, lj), cx in x_sub.items():
                i = 2 * ar + li
                j = 2 * ac + lj
                U[i * m + j, r] += cx
            
            for (li, lj), cy in y_sub.items():
                i = 2 * br + li
                j = 4 + lj
                V[i * p + j, r] += cy
            
            for (li, lj), cc in c_sub.items():
                if cc == 0:
                    continue
                i_final = 2 * cr + li
                k_final = 4 + lj
                W[k_final * n + i_final, r] += cc
        
        # Schoolbook for column 6
        r = current_rank
        current_rank += 1
        U[(2 * ar + 0) * m + (2 * ac + 0), r] = 1.0
        V[(2 * br + 0) * p + 6, r] = 1.0
        W[6 * n + (2 * cr + 0), r] = 1.0
        
        r = current_rank
        current_rank += 1
        U[(2 * ar + 0) * m + (2 * ac + 1), r] = 1.0
        V[(2 * br + 1) * p + 6, r] = 1.0
        W[6 * n + (2 * cr + 0), r] = 1.0
        
        r = current_rank
        current_rank += 1
        U[(2 * ar + 1) * m + (2 * ac + 0), r] = 1.0
        V[(2 * br + 0) * p + 6, r] = 1.0
        W[6 * n + (2 * cr + 1), r] = 1.0
        
        r = current_rank
        current_rank += 1
        U[(2 * ar + 1) * m + (2 * ac + 1), r] = 1.0
        V[(2 * br + 1) * p + 6, r] = 1.0
        W[6 * n + (2 * cr + 1), r] = 1.0

    add_2x2x3(0, 0, 0, 0)
    add_2x2x3(0, 1, 1, 0)
    add_2x2x3(1, 0, 0, 1)
    add_2x2x3(1, 1, 1, 1)
    
    assert current_rank == 93, f"Expected 93, got {current_rank}"
    
    # Now attempt to reduce rank by finding linear dependencies among the rank-93 terms
    # We look for a linear combination of the 93 terms that sums to zero
    # This would allow us to eliminate one term, reducing rank to 92
    
    U_np = U[:, :current_rank]
    V_np = V[:, :current_rank]
    W_np = W[:, :current_rank]
    
    # Build the tensor T as a 3D array
    # T[i,j,k] = sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
    T = np.zeros((n, m, p))
    for t in range(current_rank):
        # For each term t, add U[:,t] * V[:,t] * W[:,t] to T
        for i in range(n):
            for j in range(m):
                u_val = U_np[i*m + j, t]
                if u_val == 0:
                    continue
                for k in range(p):
                    v_val = V_np[j*p + k, t]
                    if v_val == 0:
                        continue
                    w_val = W_np[k*n + i, t]
                    if w_val == 0:
                        continue
                    T[i,j,k] += u_val * v_val * w_val
    
    # The tensor should be the identity-like tensor for matrix multiplication
    # Now we want to find if there's a linear dependency among the 93 terms
    # Each term t contributes a rank-1 tensor U_t ⊗ V_t ⊗ W_t
    # We want to find coefficients c_0, ..., c_92 such that sum_t c_t * T_t = 0
    # where T_t is the rank-1 tensor for term t
    
    # This is equivalent to finding a non-trivial solution to:
    # For all (i,j,k): sum_t c_t * U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] = 0
    
    # We can set up a linear system. The number of equations is n*m*p = 4*4*7 = 112
    # The number of unknowns is 93 (the coefficients c_t)
    # Since 112 > 93, the system is overdetermined, and generically only the trivial solution exists
    
    # However, there might be structure that allows a non-trivial solution
    # Let's try to find one using least squares
    
    # Build the constraint matrix A where A[(i,j,k), t] = U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
    A = np.zeros((n*m*p, current_rank))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx = i * (m*p) + j * p + k
                A[idx, :] = U_np[i*m + j, :] * V_np[j*p + k, :] * W_np[k*n + i, :]
    
    # Find the null space of A
    try:
        _, s, vh = np.linalg.svd(A, full_matrices=False)
        # The null space is spanned by the rows of vh corresponding to zero singular values
        tol = 1e-10
        rank_A = np.sum(s > tol)
        if rank_A < current_rank:
            # There's a non-trivial null space
            null_space = vh[rank_A:, :]
            print(f"Found null space of dimension {current_rank - rank_A}")
            # Use the first null vector
            c = null_space[0]
            # Check if this gives a valid reduction
            # We can eliminate one term by expressing it as a combination of others
            # For simplicity, let's just verify the decomposition still works
            # and return the original rank-93 decomposition
        else:
            print(f"No null space found, rank of A is {rank_A}")
    except Exception as e:
        print(f"Error in SVD: {e}")
    
    # For now, return the rank-93 decomposition
    # A more sophisticated approach would be to use the null space to reduce rank
    # but that requires careful handling to maintain exactness
    
    U_out = U[:, :current_rank].tolist()
    V_out = V[:, :current_rank].tolist()
    W_out = W[:, :current_rank].tolist()
    
    if not _verify(U_out, V_out, W_out, n, m, p):
        raise ValueError("Decomposition verification failed")
    
    return {"U": U_out, "V": V_out, "W": W_out}
