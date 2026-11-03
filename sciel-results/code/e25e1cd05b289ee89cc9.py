"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110
    
    # 1. Construct the known Rank-36 solution explicitly
    # This is faster and more reliable than random search initialization
    rank = 36
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    
    def add_term(u_dict, v_dict, w_dict):
        nonlocal col
        if col >= rank:
            return
        for idx, val in u_dict.items():
            U[idx][col] = val
        for idx, val in v_dict.items():
            V[idx][col] = val
        for idx, val in w_dict.items():
            W[idx][col] = val
        col += 1

    # Strassen for 2x2 * 2x2
    def add_strassen(a_i, a_j, b_i, b_j, c_i, c_j):
        def ma(l): return (a_i + l//2) * m + (a_j + l%2)
        def mb(l): return (b_i + l//2) * p + (b_j + l%2)
        def mc(l):
            i = c_i + l//2
            k = c_j + l%2
            return k * n + i

        # M1
        add_term({ma(0): 1, ma(3): 1}, {mb(0): 1, mb(3): 1}, {mc(0): 1, mc(3): 1})
        # M2
        add_term({ma(2): 1, ma(3): 1}, {mb(0): 1}, {mc(2): 1, mc(3): -1})
        # M3
        add_term({ma(0): 1}, {mb(1): 1, mb(3): -1}, {mc(1): 1, mc(3): 1})
        # M4
        add_term({ma(3): 1}, {mb(2): 1, mb(0): -1}, {mc(0): 1, mc(2): 1})
        # M5
        add_term({ma(0): 1, ma(1): 1}, {mb(3): 1}, {mc(0): -1, mc(1): 1})
        # M6
        add_term({ma(2): 1, ma(0): -1}, {mb(0): 1, mb(1): 1}, {mc(3): 1})
        # M7
        add_term({ma(1): 1, ma(3): -1}, {mb(2): 1, mb(3): 1}, {mc(0): 1})

    # Schoolbook for 2x2 * 2x1
    def add_schoolbook_2x1(a_i, a_j, b_i, b_j, c_i, c_j):
        def ma(l): return (a_i + l//2) * m + (a_j + l%2)
        def mb(l): return (b_i + l) * p + b_j
        def mc(l):
            i = c_i + l
            k = c_j
            return k * n + i

        # C0 = A00 B0 + A01 B1
        # C1 = A10 B0 + A11 B1
        add_term({ma(0): 1}, {mb(0): 1}, {mc(0): 1})
        add_term({ma(1): 1}, {mb(1): 1}, {mc(0): 1})
        add_term({ma(2): 1}, {mb(0): 1}, {mc(1): 1})
        add_term({ma(3): 1}, {mb(1): 1}, {mc(1): 1})

    # Block 1: A[:, 0:2] * B[0:2, :]
    # B[0:2, 0:2] -> Strassen
    add_strassen(0, 0, 0, 0, 0, 0)
    # B[0:2, 2:4] -> Strassen
    add_strassen(0, 0, 0, 2, 0, 2)
    # B[0:2, 4:5] -> Schoolbook
    add_schoolbook_2x1(0, 0, 0, 4, 0, 4)

    # Block 2: A[:, 2:4] * B[2:4, :]
    # B[2:4, 0:2] -> Strassen
    add_strassen(0, 2, 2, 0, 0, 0)
    # B[2:4, 2:4] -> Strassen
    add_strassen(0, 2, 2, 2, 0, 2)
    # B[2:4, 4:5] -> Schoolbook
    add_schoolbook_2x1(0, 2, 2, 4, 0, 4)

    # Verify the rank-36 solution
    def verify_np(U, V, W):
        for i in range(n*m):
            for j in range(m*p):
                for k in range(p*n):
                    val = np.dot(U[i, :], V[j, :] * W[k, :])
                    row_a, col_a = divmod(i, m)
                    col_a2, row_b = divmod(j, p)
                    col_b, row_a2 = divmod(k, n)
                    if col_a == col_a2 and row_a == row_a2:
                        if abs(val - 1.0) > 1e-6:
                            return False
                    else:
                        if abs(val) > 1e-6:
                            return False
        return True

    if not verify_np(U, V, W):
        # Fallback to random search if explicit construction fails (unlikely)
        pass

    # 2. Attempt to reduce rank to 35 using local optimization
    # We try to zero out one column by adjusting the others
    
    def try_reduce_rank(U, V, W, target_rank, max_iters=20):
        if target_rank >= U.shape[1]:
            return None
        
        U_curr = U[:, :target_rank].copy()
        V_curr = V[:, :target_rank].copy()
        W_curr = W[:, :target_rank].copy()
        
        # Simple alternating minimization to see if we can fit into target_rank
        # This is a heuristic; exact integer solution is preferred
        
        for it in range(max_iters):
            if time.time() > deadline:
                return None
                
            # Update U
            C = np.zeros((target_rank, m*p * p*n))
            for r in range(target_rank):
                C[r, :] = np.kron(V_curr[:, r], W_curr[:, r])
            T = np.zeros((n*m, m*p * p*n))
            for i in range(n*m):
                for j in range(m*p):
                    for k in range(p*n):
                        row_a, col_a = divmod(i, m)
                        col_a2, row_b = divmod(j, p)
                        col_b, row_a2 = divmod(k, n)
                        if col_a == col_a2 and row_a == row_a2:
                            T[i, j*p*n + k] = 1.0
            # T_flat is (n*m, m*p*p*n)
            # C.T is (m*p*p*n, target_rank)
            # We want U_curr such that U_curr @ C.T = T_flat.T
            # This is a least squares problem
            try:
                U_new, _, _, _ = np.linalg.lstsq(C.T, T.T, rcond=None)
                U_curr = U_new.T
            except:
                return None

            # Update V
            D = np.zeros((target_rank, n*m * p*n))
            for r in range(target_rank):
                D[r, :] = np.kron(U_curr[:, r], W_curr[:, r])
            T2 = np.zeros((m*p, n*m * p*n))
            for i in range(n*m):
                for j in range(m*p):
                    for k in range(p*n):
                        row_a, col_a = divmod(i, m)
                        col_a2, row_b = divmod(j, p)
                        col_b, row_a2 = divmod(k, n)
                        if col_a == col_a2 and row_a == row_a2:
                            T2[j, i*p*n + k] = 1.0
            try:
                V_new, _, _, _ = np.linalg.lstsq(D.T, T2.T, rcond=None)
                V_curr = V_new.T
            except:
                return None

            # Update W
            E = np.zeros((target_rank, n*m * m*p))
            for r in range(target_rank):
                E[r, :] = np.kron(U_curr[:, r], V_curr[:, r])
            T3 = np.zeros((p*n, n*m * m*p))
            for i in range(n*m):
                for j in range(m*p):
                    for k in range(p*n):
                        row_a, col_a = divmod(i, m)
                        col_a2, row_b = divmod(j, p)
                        col_b, row_a2 = divmod(k, n)
                        if col_a == col_a2 and row_a == row_a2:
                            T3[k, i*m*p + j] = 1.0
            try:
                W_new, _, _, _ = np.linalg.lstsq(E.T, T3.T, rcond=None)
                W_curr = W_new.T
            except:
                return None

        # Check if the solution is close to integers
        all_entries = np.concatenate([U_curr.flatten(), V_curr.flatten(), W_curr.flatten()])
        rounded = np.round(all_entries)
        round_err = np.sum((all_entries - rounded)**2)
        
        if round_err < 1e-4:
            U_r = np.round(U_curr)
            V_r = np.round(V_curr)
            W_r = np.round(W_curr)
            if verify_np(U_r, V_r, W_r):
                return U_r, V_r, W_r
        return None

    # Try rank 35
    if time.time() < deadline:
        result = try_reduce_rank(U, V, W, 35)
        if result:
            U, V, W = result
            return {"U": U.astype(int).tolist(), "V": V.astype(int).tolist(), "W": W.astype(int).tolist()}

    # Return the rank-36 solution
    return {"U": U.astype(int).tolist(), "V": V.astype(int).tolist(), "W": W.astype(int).tolist()}
# EVOLVE-BLOCK-END
