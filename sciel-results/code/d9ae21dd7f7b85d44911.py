import numpy as np
import random
import time
from itertools import combinations

DIMENSIONS = [2, 3, 10]

def _verify_fast(U, V, W, n, m, p, tol=1e-9):
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    rank = U.shape[1]
    
    U_mat = U.reshape(n, m, rank)
    V_mat = V.reshape(m, p, rank)
    W_mat = W.reshape(p, n, rank)
    
    for i in range(n):
        for k in range(p):
            Uw = U_mat[i] * W_mat[k, i]
            M_ik = Uw @ V_mat[:, k].T
            if not np.allclose(M_ik, np.eye(m), atol=tol):
                return False
    return True

def _is_valid_entries(U, V, W):
    for matrix in [U, V, W]:
        for row in matrix:
            for val in row:
                if abs(val * 2 - round(val * 2)) > 1e-9:
                    return False
    return True

def _find_rank10_block(deadline):
    """
    Finds a rank-10 decomposition for 2x3 * 3x2 matrix multiplication.
    Returns U, V, W as numpy arrays of shape (6, 10), (6, 10), (4, 10) or None.
    """
    n, m, p = 2, 3, 2
    rank = 10
    
    # Target tensor T[i, j, k] = 1 if i,j,k is a valid index for multiplication
    # Actually, we want T to represent the identity in the bilinear form.
    # The condition is sum_r U[i,j,r] V[j,k,r] W[k,i,r] == delta_{i,i} delta_{j,j} delta_{k,k}?
    # No, the tensor for matrix multiplication is T[i,j,k] where the product is C[i,k] = sum_j A[i,j] B[j,k].
    # The tensor T has indices i (row of A), j (col of A / row of B), k (col of B).
    # T[i,j,k] is 1 if the term A[i,j]B[j,k] contributes to C[i,k].
    # In the decomposition, we need:
    # sum_r U[i,j,r] V[j,k,r] W[k,i,r] = 1 if (i,j,k) is a "valid" triple?
    # No, the standard tensor T_MM has T[i,j,k] = 1 for all i,j,k?
    # Let's check the verification logic.
    # M_ik = sum_j sum_r U[i,j,r] W[k,i,r] V[j,k,r]
    # We want M_ik = I.
    # So sum_r U[i,j,r] W[k,i,r] V[j,k,r] = delta_{i,i} delta_{j,j}... wait.
    # M_ik is a m x m matrix. M_ik[j, j'] should be delta_{j,j'}?
    # No, M_ik is the contribution to the (i,k) block of the result?
    # Let's look at _verify_fast.
    # U_mat[i] is (m, rank). W_mat[k,i] is (m, rank).
    # Uw = U_mat[i] * W_mat[k,i] -> (m, rank).
    # M_ik = Uw @ V_mat[:, k].T -> (m, m).
    # We want M_ik == I_m.
    # So for each i, k: sum_j sum_r U[i,j,r] W[k,i,r] V[j,k,r] = delta_{j, j'}?
    # Wait, V_mat[:, k] is (m, rank).
    # M_ik[j, j'] = sum_r U[i,j,r] W[k,i,r] V[j',k,r].
    # We want this to be 1 if j==j' and 0 otherwise?
    # No, the matrix multiplication tensor is defined such that
    # C[i,k] = sum_j A[i,j] B[j,k].
    # The bilinear form is <U, A> <V, B> <W, C>.
    # The tensor T satisfies T[i,j,k] = 1.
    # Then sum_{i,j,k} T[i,j,k] A[i,j] B[j,k] C[k,i] = sum_{i,j,k} A[i,j] B[j,k] C[k,i].
    # This equals sum_i,k (sum_j A[i,j] B[j,k]) C[k,i] = sum_i,k C[i,k] C[k,i] = ||C||_F^2.
    # This is not 1.
    
    # The verification in the code checks:
    # M_ik = sum_r U[i,:,r] W[k,i,r] V[:,k,r]^T ?
    # U_mat[i] is shape (m, rank). Indices (j, r).
    # W_mat[k,i] is shape (m, rank). Indices (j, r)?
    # W is (p*n, rank). Reshaped to (p, n, rank).
    # W_mat[k, i] is (m, rank). Indices (j, r).
    # Uw = U_mat[i] * W_mat[k,i] is (m, rank).
    # V_mat[:, k] is (m, rank). Indices (j, r).
    # M_ik = Uw @ V_mat[:, k].T is (m, m).
    # M_ik[j, j'] = sum_r U[i,j,r] W[k,i,j,r] V[j',k,r].
    # We want M_ik == I.
    # This implies sum_r U[i,j,r] W[k,i,j,r] V[j',k,r] = delta_{j,j'}.
    # This is the condition for the tensor to decompose the matrix multiplication.
    
    # So for the 2x3x2 block:
    # n=2, m=3, p=2.
    # We want U (6, 10), V (6, 10), W (4, 10).
    # U[i,j,r], V[j,k,r], W[k,i,r].
    # Condition: sum_r U[i,j,r] W[k,i,r] V[j',k,r] = delta_{j,j'}.
    
    target = np.zeros((n, m, p, m)) # M_ik is m x m
    for i in range(n):
        for k in range(p):
            target[i, :, k, :] = np.eye(m)
            
    # Flatten target to vector of length n*p*m*m = 2*2*3*3 = 36.
    # But the equations are coupled.
    # We can just use the matrix form in the solver.
    
    best_err = float('inf')
    best_U = None
    best_V = None
    best_W = None
    
    for seed in range(20):
        if time.time() > deadline:
            break
        random.seed(seed)
        np.random.seed(seed)
        
        U = np.random.randn(n*m, rank)
        V = np.random.randn(m*p, rank)
        W = np.random.randn(p*n, rank)
        
        # Scale to avoid overflow
        U /= np.linalg.norm(U)
        V /= np.linalg.norm(V)
        W /= np.linalg.norm(W)
        
        for iter in range(100):
            if time.time() > deadline:
                break
            
            # Solve for U
            # M_ik[j, j'] = sum_r U[i,j,r] W[k,i,j,r] V[j',k,r]
            # For fixed V, W, this is linear in U.
            # Equations for each i, k, j, j':
            # sum_r U[i,j,r] (W[k,i,j,r] * V[j',k,r]) = target[i,j,k,j']
            # Let's construct the linear system for U.
            # Unknowns: U_flat (n*m*rank)
            # Eq for (i, k, j, j'):
            # sum_r U[i,j,r] * C_{i,k,j,j',r} = T_{i,k,j,j'}
            # where C_{i,k,j,j',r} = W[k,i,j,r] * V[j',k,r]
            
            A_U = np.zeros((n*p*m*m, n*m*rank))
            b_U = np.zeros(n*p*m*m)
            eq_idx = 0
            for i in range(n):
                for k in range(p):
                    for j in range(m):
                        for jp in range(m):
                            for r in range(rank):
                                A_U[eq_idx, (i*m+j)*rank + r] = W[k*n+i, j*rank+r] * V[j*p+k, r]
                            b_U[eq_idx] = 1.0 if j == jp else 0.0
                            eq_idx += 1
                            
            U_sol, _, _, _ = np.linalg.lstsq(A_U, b_U, rcond=None)
            U = U_sol.reshape(n*m, rank)
            
            # Solve for V
            A_V = np.zeros((n*p*m*m, m*p*rank))
            b_V = np.zeros(n*p*m*m)
            eq_idx = 0
            for i in range(n):
                for k in range(p):
                    for j in range(m):
                        for jp in range(m):
                            for r in range(rank):
                                A_V[eq_idx, (j*p+k)*rank + r] = U[i*m+j, r] * W[k*n+i, j*rank+r]
                            b_V[eq_idx] = 1.0 if j == jp else 0.0
                            eq_idx += 1
                            
            V_sol, _, _, _ = np.linalg.lstsq(A_V, b_V, rcond=None)
            V = V_sol.reshape(m*p, rank)
            
            # Solve for W
            A_W = np.zeros((n*p*m*m, p*n*rank))
            b_W = np.zeros(n*p*m*m)
            eq_idx = 0
            for i in range(n):
                for k in range(p):
                    for j in range(m):
                        for jp in range(m):
                            for r in range(rank):
                                A_W[eq_idx, (k*n+i)*rank + r] = U[i*m+j, r] * V[j*p+k, r]
                            b_W[eq_idx] = 1.0 if j == jp else 0.0
                            eq_idx += 1
                            
            W_sol, _, _, _ = np.linalg.lstsq(A_W, b_W, rcond=None)
            W = W_sol.reshape(p*n, rank)
            
            # Check error
            err = 0
            for i in range(n):
                for k in range(p):
                    for j in range(m):
                        for jp in range(m):
                            val = sum(U[i*m+j, r] * W[k*n+i, j*rank+r] * V[jp*p+k, r] for r in range(rank))
                            target_val = 1.0 if j == jp else 0.0
                            err += (val - target_val)**2
                            
            if err < best_err:
                best_err = err
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()
                
            if err < 1e-12:
                break
                
        if best_err < 1e-12:
            # Try to snap to half-integers
            U_snap = np.round(best_U * 2) / 2
            V_snap = np.round(best_V * 2) / 2
            W_snap = np.round(best_W * 2) / 2
            
            # Verify snapped version
            if _verify_fast(U_snap, V_snap, W_snap, n, m, p):
                return U_snap, V_snap, W_snap
            # If snapping fails, maybe the solution is not half-integer.
            # But for this problem, it should be.
            # Try integer snap
            U_int = np.round(best_U)
            V_int = np.round(best_V)
            W_int = np.round(best_W)
            if _verify_fast(U_int, V_int, W_int, n, m, p):
                return U_int, V_int, W_int

    return None

def _build_strassen55(n, m, p):
    rank = 55
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0
    for i in range(n):
        for k in range(p):
            u_idx = i * m + 0
            v_idx = 0 * p + k
            w_idx = k * n + i
            U[u_idx][col] = 1
            V[v_idx][col] = 1
            W[w_idx][col] = 1
            col += 1

    for b in range(5):
        k0 = 2 * b
        k1 = 2 * b + 1

        u_11 = 0 * m + 1
        u_12 = 0 * m + 2
        u_21 = 1 * m + 1
        u_22 = 1 * m + 2

        v_11 = 1 * p + k0
        v_12 = 1 * p + k1
        v_21 = 2 * p + k0
        v_22 = 2 * p + k1

        w_00 = k0 * n + 0
        w_01 = k1 * n + 0
        w_10 = k0 * n + 1
        w_11 = k1 * n + 1

        U[u_11][col] += 1
        U[u_22][col] += 1
        V[v_11][col] += 1
        V[v_22][col] += 1
        W[w_00][col] += 1
        W[w_11][col] += 1
        col += 1

        U[u_21][col] += 1
        U[u_22][col] += 1
        V[v_11][col] += 1
        W[w_10][col] += 1
        W[w_11][col] += -1
        col += 1

        U[u_11][col] += 1
        V[v_12][col] += 1
        V[v_22][col] += -1
        W[w_01][col] += 1
        W[w_11][col] += 1
        col += 1

        U[u_22][col] += 1
        V[v_21][col] += 1
        V[v_11][col] += -1
        W[w_00][col] += 1
        W[w_10][col] += 1
        col += 1

        U[u_11][col] += 1
        U[u_12][col] += 1
        V[v_22][col] += 1
        W[w_00][col] += -1
        W[w_01][col] += 1
        col += 1

        U[u_21][col] += 1
        U[u_11][col] += -1
        V[v_11][col] += 1
        V[v_12][col] += 1
        W[w_11][col] += 1
        col += 1

        U[u_12][col] += 1
        U[u_22][col] += -1
        V[v_21][col] += 1
        V[v_22][col] += 1
        W[w_00][col] += 1
        col += 1

    assert col == rank, f"Expected {rank} columns, got {col}"
    return U, V, W

def _build_schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140

    U_best, V_best, W_best = _build_strassen55(n, m, p)
    best_rank = 55

    if not (_verify_fast(U_best, V_best, W_best, n, m, p) and _is_valid_entries(U_best, V_best, W_best)):
        U_best, V_best, W_best = _build_schoolbook(n, m, p)
        best_rank = n * m * p

    # Try to find rank 50 decomposition
    if time.time() < deadline - 20:
        block_res = _find_rank10_block(deadline)
        if block_res is not None:
            U_b, V_b, W_b = block_res
            # Assemble full matrices
            rank_full = 50
            U_full = np.zeros((n*m, rank_full))
            V_full = np.zeros((m*p, rank_full))
            W_full = np.zeros((p*n, rank_full))
            
            for b in range(5):
                offset = b * 10
                U_full[:, offset:offset+10] = U_b
                for j in range(m):
                    for kb in range(2):
                        k = 2*b + kb
                        V_full[j*p+k, offset:offset+10] = V_b[j*2+kb, :]
                for i in range(n):
                    for kb in range(2):
                        k = 2*b + kb
                        W_full[k*n+i, offset:offset+10] = W_b[kb*n+i, :]
                        
            if _verify_fast(U_full, V_full, W_full, n, m, p) and _is_valid_entries(U_full.tolist(), V_full.tolist(), W_full.tolist()):
                U_best, V_best, W_best = U_full.tolist(), V_full.tolist(), W_full.tolist()
                best_rank = 50
                return {"U": U_best, "V": V_best, "W": W_best}

    # Fallback to rank 55
    return {"U": U_best, "V": V_best, "W": W_best}
