import time
import numpy as np
from itertools import combinations

DIMENSIONS = [2, 4, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 110

    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n  # 8, 24, 12

    # Build the rank-42 Strassen decomposition
    def build_strassen_42():
        rank = 42
        U = np.zeros((nm, rank), dtype=float)
        V = np.zeros((mp, rank), dtype=float)
        W = np.zeros((pn, rank), dtype=float)

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

            U[a00, offset] += 1.0
            U[a11, offset] += 1.0
            V[b00, offset] += 1.0
            V[b11, offset] += 1.0
            W[w00, offset] += 1.0
            W[w11, offset] += 1.0

            U[a10, offset + 1] += 1.0
            U[a11, offset + 1] += 1.0
            V[b00, offset + 1] += 1.0
            W[w10, offset + 1] += 1.0
            W[w11, offset + 1] -= 1.0

            U[a00, offset + 2] += 1.0
            V[b01, offset + 2] += 1.0
            V[b11, offset + 2] -= 1.0
            W[w01, offset + 2] += 1.0
            W[w11, offset + 2] += 1.0

            U[a11, offset + 3] += 1.0
            V[b10, offset + 3] += 1.0
            V[b00, offset + 3] -= 1.0
            W[w00, offset + 3] += 1.0
            W[w10, offset + 3] += 1.0

            U[a00, offset + 4] += 1.0
            U[a01, offset + 4] += 1.0
            V[b11, offset + 4] += 1.0
            W[w00, offset + 4] -= 1.0
            W[w01, offset + 4] += 1.0

            U[a10, offset + 5] += 1.0
            U[a00, offset + 5] -= 1.0
            V[b00, offset + 5] += 1.0
            V[b01, offset + 5] += 1.0
            W[w11, offset + 5] += 1.0

            U[a01, offset + 6] += 1.0
            U[a11, offset + 6] -= 1.0
            V[b10, offset + 6] += 1.0
            V[b11, offset + 6] += 1.0
            W[w00, offset + 6] += 1.0

        offset = 0
        for i_start in range(0, n, 2):
            for j_start in range(0, m, 2):
                for k_start in range(0, p, 2):
                    add_strassen_block(i_start, j_start, k_start, offset)
                    offset += 7

        return U, V, W, rank

    def verify_decomp(U_mat, V_mat, W_mat, rank_val, tol=1e-8):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    s = 0.0
                    for r in range(rank_val):
                        s += U_mat[a, r] * V_mat[b, r] * W_mat[c, r]
                    if abs(s - 1.0) > tol:
                        return False
        for a in range(nm):
            for b in range(mp):
                for c in range(pn):
                    i = a // m
                    j = a % m
                    j_b = b // p
                    k = b % p
                    i_w = c % n
                    k_w = c // n
                    if i == i_w and j == j_b and k == k_w:
                        continue
                    s = 0.0
                    for r in range(rank_val):
                        s += U_mat[a, r] * V_mat[b, r] * W_mat[c, r]
                    if abs(s) > tol:
                        return False
        return True

    U42, V42, W42, r42 = build_strassen_42()
    best_U, best_V, best_W, best_rank = U42, V42, W42, r42

    # Strategy: Support pruning - enumerate sparse rank-40 and rank-41 decompositions
    # derived from the Strassen rank-42 pattern, and solve for coefficients.
    
    # First, check if any single column can be removed (rank 41)
    # For each candidate column to remove, check if the remaining columns
    # can represent T_full with half-integer coefficients.
    
    T_full = np.zeros((nm, mp, pn), dtype=float)
    for r in range(r42):
        T_full += np.einsum('a,b,c->abc', U42[:, r], V42[:, r], W42[:, r])
    
    T_flat = T_full.flatten()
    vec_dim = nm * mp * pn
    
    # Precompute the 42 rank-1 tensor vectors
    T_vecs = np.zeros((r42, vec_dim), dtype=float)
    for r in range(r42):
        T_vecs[r, :] = np.einsum('a,b,c->abc', U42[:, r], V42[:, r], W42[:, r]).flatten()

    # Try rank 41: remove one column and solve for coefficients
    for remove_idx in range(r42):
        if time.time() > deadline - 30:
            break
        
        mask = [i for i in range(r42) if i != remove_idx]
        T_sub = T_vecs[mask, :]
        
        # Solve least squares: T_sub^T * T_sub * x = T_sub^T * T_flat
        try:
            A = T_sub.T @ T_sub
            b = T_sub.T @ T_flat
            x, residuals, rank_ls, sv = np.linalg.lstsq(A, b, rcond=1e-10)
            
            # Check reconstruction error
            T_recon = (T_sub @ x).reshape(nm, mp, pn)
            error = np.max(np.abs(T_recon - T_full))
            
            if error < 1e-8:
                # Check if coefficients are close to half-integers
                valid = True
                new_U = U42[:, mask].copy()
                new_V = V42[:, mask].copy()
                new_W = W42[:, mask].copy()
                
                for i in range(len(mask)):
                    coeff = x[i]
                    nearest = np.round(coeff * 2) / 2
                    if abs(coeff - nearest) > 1e-6:
                        valid = False
                        break
                    new_U[:, i] *= nearest
                    new_V[:, i] *= nearest
                    new_W[:, i] *= nearest
                
                if valid:
                    # Verify the decomposition
                    if verify_decomp(new_U, new_V, new_W, len(mask)):
                        best_U, best_V, best_W, best_rank = new_U, new_V, new_W, len(mask)
                        break
        except:
            continue

    # Try rank 40: remove two columns and solve for coefficients
    if best_rank == r42 and time.time() < deadline - 20:
        for remove1 in range(r42):
            if time.time() > deadline - 10:
                break
            for remove2 in range(remove1 + 1, r42):
                if time.time() > deadline - 10:
                    break
                
                mask = [i for i in range(r42) if i != remove1 and i != remove2]
                T_sub = T_vecs[mask, :]
                
                # Solve least squares
                try:
                    A = T_sub.T @ T_sub
                    b = T_sub.T @ T_flat
                    x, residuals, rank_ls, sv = np.linalg.lstsq(A, b, rcond=1e-10)
                    
                    # Check reconstruction error
                    T_recon = (T_sub @ x).reshape(nm, mp, pn)
                    error = np.max(np.abs(T_recon - T_full))
                    
                    if error < 1e-8:
                        # Check if coefficients are close to half-integers
                        valid = True
                        new_U = U42[:, mask].copy()
                        new_V = V42[:, mask].copy()
                        new_W = W42[:, mask].copy()
                        
                        for i in range(len(mask)):
                            coeff = x[i]
                            nearest = np.round(coeff * 2) / 2
                            if abs(coeff - nearest) > 1e-6:
                                valid = False
                                break
                            new_U[:, i] *= nearest
                            new_V[:, i] *= nearest
                            new_W[:, i] *= nearest
                        
                        if valid:
                            # Verify the decomposition
                            if verify_decomp(new_U, new_V, new_W, len(mask)):
                                best_U, best_V, best_W, best_rank = new_U, new_V, new_W, len(mask)
                                break
                except:
                    continue
            if best_rank < r42:
                break

    # If we found a better decomposition, use it
    if best_rank < r42:
        U_list = best_U.tolist()
        V_list = best_V.tolist()
        W_list = best_W.tolist()
    else:
        U_list = U42.tolist()
        V_list = V42.tolist()
        W_list = W42.tolist()

    return {"U": U_list, "V": V_list, "W": W_list}
