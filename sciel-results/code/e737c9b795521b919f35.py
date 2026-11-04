import time
import numpy as np
from itertools import combinations

def construct():
    n, m, p = 2, 4, 4
    nm, mp, pn = n * m, m * p, p * n  # 8, 16, 8

    start_time = time.time()
    deadline = start_time + 140.0

    # Build the multiplication tensor T of shape (nm, mp, pn)
    T = np.zeros((nm, mp, pn), dtype=float)
    for i in range(n):
        for l in range(m):
            for k in range(p):
                a = i * m + l
                b = l * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    def verify_decomp(U_mat, V_mat, W_mat, rank_val):
        S = np.zeros((nm, mp, pn), dtype=float)
        for r in range(rank_val):
            S += np.einsum('a,b,c->abc', U_mat[:, r], V_mat[:, r], W_mat[:, r])
        return np.max(np.abs(S - T))

    def to_list(mat):
        result = []
        for row in mat:
            row_list = []
            for val in row:
                rounded = round(val * 2) / 2.0
                if abs(rounded - val) < 1e-10:
                    row_list.append(rounded)
                else:
                    row_list.append(val)
            result.append(row_list)
        return result

    # Known rank-28 solution via Strassen on 2x2 blocks
    rank = 28
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]

    def add_block(col_offset, X_rows, Y_rows, Z_rows):
        U[X_rows[0]][col_offset + 0] += 1
        U[X_rows[3]][col_offset + 0] += 1
        V[Y_rows[0]][col_offset + 0] += 1
        V[Y_rows[3]][col_offset + 0] += 1
        W[Z_rows[0]][col_offset + 0] += 1
        W[Z_rows[3]][col_offset + 0] += 1
        U[X_rows[2]][col_offset + 1] += 1
        U[X_rows[3]][col_offset + 1] += 1
        V[Y_rows[0]][col_offset + 1] += 1
        W[Z_rows[2]][col_offset + 1] += 1
        W[Z_rows[3]][col_offset + 1] -= 1
        U[X_rows[0]][col_offset + 2] += 1
        V[Y_rows[1]][col_offset + 2] += 1
        V[Y_rows[3]][col_offset + 2] -= 1
        W[Z_rows[1]][col_offset + 2] += 1
        W[Z_rows[3]][col_offset + 2] += 1
        U[X_rows[3]][col_offset + 3] += 1
        V[Y_rows[2]][col_offset + 3] += 1
        V[Y_rows[0]][col_offset + 3] -= 1
        W[Z_rows[0]][col_offset + 3] += 1
        W[Z_rows[2]][col_offset + 3] += 1
        U[X_rows[0]][col_offset + 4] += 1
        U[X_rows[1]][col_offset + 4] += 1
        V[Y_rows[3]][col_offset + 4] += 1
        W[Z_rows[0]][col_offset + 4] -= 1
        W[Z_rows[1]][col_offset + 4] += 1
        U[X_rows[2]][col_offset + 5] += 1
        U[X_rows[0]][col_offset + 5] -= 1
        V[Y_rows[0]][col_offset + 5] += 1
        V[Y_rows[1]][col_offset + 5] += 1
        W[Z_rows[3]][col_offset + 5] += 1
        U[X_rows[1]][col_offset + 6] += 1
        U[X_rows[3]][col_offset + 6] -= 1
        V[Y_rows[2]][col_offset + 6] += 1
        V[Y_rows[3]][col_offset + 6] += 1
        W[Z_rows[0]][col_offset + 6] += 1

    # Block 1: A_L B_TL -> C_LL
    X1 = [0, 1, 4, 5]
    Y1 = [0, 1, 4, 5]
    Z1 = [0, 2, 1, 3]
    add_block(0, X1, Y1, Z1)
    # Block 2: A_R B_BL -> C_LR
    X2 = [2, 3, 6, 7]
    Y2 = [8, 9, 12, 13]
    Z2 = [0, 2, 1, 3]
    add_block(7, X2, Y2, Z2)
    # Block 3: A_L B_TR -> C_RL
    X3 = [0, 1, 4, 5]
    Y3 = [2, 3, 6, 7]
    Z3 = [4, 6, 5, 7]
    add_block(14, X3, Y3, Z3)
    # Block 4: A_R B_BR -> C_RR
    X4 = [2, 3, 6, 7]
    Y4 = [10, 11, 14, 15]
    Z4 = [4, 6, 5, 7]
    add_block(21, X4, Y4, Z4)

    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)

    # Verify rank-28 solution
    max_err = verify_decomp(U_arr, V_arr, W_arr, rank)
    if max_err > 1e-10:
        return {"U": to_list(U_arr), "V": to_list(V_arr), "W": to_list(W_arr)}

    best_rank = rank
    best_U = U_arr.copy()
    best_V = V_arr.copy()
    best_W = W_arr.copy()

    # Larsen-Raz style: compute flattening ranks to get lower bounds
    # Flatten T along each mode to get matrix representations
    T_flat_ab = T.reshape(nm * mp, pn)  # flatten (a,b) vs c
    T_flat_ac = T.reshape(nm, mp * pn)  # flatten a vs (b,c)
    T_flat_bc = T.reshape(nm, pn * mp)  # flatten a vs (c,b) - same as ac but permuted
    
    rank_ab = np.linalg.matrix_rank(T_flat_ab, tol=1e-10)
    rank_ac = np.linalg.matrix_rank(T_flat_ac, tol=1e-10)
    rank_bc = np.linalg.matrix_rank(T_flat_bc, tol=1e-10)
    
    lower_bound = max(rank_ab, rank_ac, rank_bc)
    
    # Try to find rank-27 solution by structured perturbation
    # Strategy: take rank-28 solution, try to express one term as linear combo of others
    # with small perturbation, then optimize
    
    def try_rank_reduction(U_cur, V_cur, W_cur, target_rank):
        """Try to reduce rank by expressing terms as linear combinations"""
        if target_rank >= U_cur.shape[1]:
            return None
        
        r = U_cur.shape[1]
        # Try removing each column and checking if we can adjust others
        for col_to_remove in range(r):
            remaining = [c for c in range(r) if c != col_to_remove]
            U_rem = U_cur[:, remaining]
            V_rem = V_cur[:, remaining]
            W_rem = W_cur[:, remaining]
            
            # Check if the removed term can be approximated by remaining
            T_removed = np.einsum('a,b,c->abc', U_cur[:, col_to_remove], 
                                  V_cur[:, col_to_remove], W_cur[:, col_to_remove])
            T_remaining = np.zeros_like(T_removed)
            for idx in range(len(remaining)):
                T_remaining += np.einsum('a,b,c->abc', U_rem[:, idx], V_rem[:, idx], W_rem[:, idx])
            
            # The error is T_removed since we're removing it
            # We need to see if we can scale/adjust remaining terms to compensate
            # This is a nonlinear problem, so use a simple approach:
            # Check if T_removed is in the span of {U_rem[:,i] ⊗ V_rem[:,i] ⊗ W_rem[:,i]}
            
            err = np.max(np.abs(T_removed))
            if err < 1e-10:
                # Already zero, can remove
                return (U_rem, V_rem, W_rem)
        
        return None

    # More aggressive: try all pairs of columns to merge
    def try_merge_columns(U_cur, V_cur, W_cur):
        """Try to merge two columns by finding if their sum/difference is redundant"""
        r = U_cur.shape[1]
        for i in range(r):
            for j in range(i + 1, r):
                if time.time() > deadline - 30:
                    return None
                # Try: can we replace columns i and j with a single column?
                # This would require finding a single (u,v,w) such that
                # u⊗v⊗w = u_i⊗v_i⊗w_i + u_j⊗v_j⊗w_j
                # Generally not possible, but check if one is scalar multiple of other
                u_i, v_i, w_i = U_cur[:, i], V_cur[:, i], W_cur[:, i]
                u_j, v_j, w_j = U_cur[:, j], V_cur[:, j], W_cur[:, j]
                
                # Check if they are proportional
                # u_i = alpha * u_j, v_i = alpha * v_j, w_i = (1/alpha) * w_j
                # This is a rank-1 condition on the outer product structure
                
                # Simple check: if u_i and u_j are proportional
                if np.linalg.norm(u_i) > 1e-10 and np.linalg.norm(u_j) > 1e-10:
                    alpha_u = np.dot(u_i, u_j) / np.dot(u_j, u_j)
                    if np.linalg.norm(u_i - alpha_u * u_j) < 1e-8:
                        # Check v
                        if np.linalg.norm(v_i - alpha_u * v_j) < 1e-8:
                            # Check w: need w_i = (1/alpha_u) * w_j
                            if np.linalg.norm(w_i - (1.0/alpha_u) * w_j) < 1e-8:
                                # They can be merged
                                remaining = [c for c in range(r) if c != i and c != j]
                                # Use the sum as the merged column
                                U_new = U_cur[:, remaining].copy()
                                V_new = V_cur[:, remaining].copy()
                                W_new = W_cur[:, remaining].copy()
                                
                                # Add merged column
                                U_merged = u_i + u_j
                                V_merged = v_i + v_j
                                W_merged = w_i + w_j
                                
                                U_final = np.column_stack([U_new, U_merged]) if len(remaining) > 0 else U_merged.reshape(-1, 1)
                                V_final = np.column_stack([V_new, V_merged]) if len(remaining) > 0 else V_merged.reshape(-1, 1)
                                W_final = np.column_stack([W_new, W_merged]) if len(remaining) > 0 else W_merged.reshape(-1, 1)
                                
                                err = verify_decomp(U_final, V_final, W_final, U_final.shape[1])
                                if err < 1e-10:
                                    return (U_final, V_final, W_final)
        
        return None

    # Try direct rank reduction
    result = try_rank_reduction(U_arr, V_arr, W_arr, 27)
    if result is not None:
        best_U, best_V, best_W = result
        best_rank = 27
    
    # Try merging columns
    if best_rank > 27:
        result = try_merge_columns(best_U, best_V, best_W)
        if result is not None:
            best_U, best_V, best_W = result
            best_rank = result[0].shape[1]
    
    # Randomized search: perturb the solution and try to project to lower rank
    if best_rank > 27 and time.time() < deadline - 40:
        np.random.seed(42)
        for trial in range(50):
            if time.time() > deadline - 30:
                break
            
            # Create a random perturbation
            eps = 0.01
            U_pert = U_arr + eps * np.random.randn(nm, 28)
            V_pert = V_arr + eps * np.random.randn(mp, 28)
            W_pert = W_arr + eps * np.random.randn(pn, 28)
            
            # Try to find a rank-27 approximation by SVD-like approach
            # Build the tensor from perturbed solution
            T_pert = np.zeros((nm, mp, pn), dtype=float)
            for r in range(28):
                T_pert += np.einsum('a,b,c->abc', U_pert[:, r], V_pert[:, r], W_pert[:, r])
            
            # Check if T_pert is close to T
            if np.max(np.abs(T_pert - T)) < 0.1:
                # Try to find rank-27 decomposition of T_pert
                # This is hard, so just check if we can remove one term
                for col in range(28):
                    U_rem = np.delete(U_pert, col, axis=1)
                    V_rem = np.delete(V_pert, col, axis=1)
                    W_rem = np.delete(W_pert, col, axis=1)
                    
                    T_rem = np.zeros((nm, mp, pn), dtype=float)
                    for r in range(27):
                        T_rem += np.einsum('a,b,c->abc', U_rem[:, r], V_rem[:, r], W_rem[:, r])
                    
                    if np.max(np.abs(T_rem - T)) < 1e-8:
                        best_U, best_V, best_W = U_rem, V_rem, W_rem
                        best_rank = 27
                        break
                if best_rank == 27:
                    break

    # Final verification and cleanup
    final_err = verify_decomp(best_U, best_V, best_W, best_rank)
    if final_err > 1e-10:
        best_U = U_arr
        best_V = V_arr
        best_W = W_arr
        best_rank = 28

    U_final = to_list(best_U)
    V_final = to_list(best_V)
    W_final = to_list(best_W)

    return {"U": U_final, "V": V_final, "W": W_final}
