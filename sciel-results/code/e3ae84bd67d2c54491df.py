import time
import numpy as np

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145  # Safety margin for 150s limit
    
    U_dim = n * m  # 10
    V_dim = m * p  # 25
    W_dim = p * n  # 10
    
    # Build target tensor
    T = np.zeros((U_dim, V_dim, W_dim))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                T[u_idx, v_idx, w_idx] = 1.0

    def validate(U, V, W, atol=1e-6):
        prod = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        return np.allclose(prod, T, atol=atol)
    
    def snap_to_half(arr):
        return np.round(arr * 2) / 2
    
    def snap_to_quarter(arr):
        return np.round(arr * 4) / 4
        
    def try_snap_validate(U, V, W):
        # Try half-integers first (most likely for Strassen-based)
        U_s = snap_to_half(U)
        V_s = snap_to_half(V)
        W_s = snap_to_half(W)
        if validate(U_s, V_s, W_s):
            return True, U_s, V_s, W_s
        # Try integers
        U_i = np.round(U)
        V_i = np.round(V)
        W_i = np.round(W)
        if validate(U_i, V_i, W_i):
            return True, U_i, V_i, W_i
        # Try quarters
        U_q = snap_to_quarter(U)
        V_q = snap_to_quarter(V)
        W_q = snap_to_quarter(W)
        if validate(U_q, V_q, W_q):
            return True, U_q, V_q, W_q
        return False, None, None, None

    def build_rank46():
        rank = 46
        U = np.zeros((U_dim, rank))
        V = np.zeros((V_dim, rank))
        W = np.zeros((W_dim, rank))
        
        def add_strassen(j_off, k_off, col_start):
            c = col_start
            # P1 = (A00 + A11)(B00 + B11)
            U[0 * m + j_off][c] += 1
            U[1 * m + (j_off + 1)][c] += 1
            V[j_off * p + k_off][c] += 1
            V[(j_off + 1) * p + (k_off + 1)][c] += 1
            W[k_off * n + 0][c] += 1
            W[(k_off + 1) * n + 1][c] += 1
            c += 1
            
            # P2 = (A10 + A11) B00
            U[1 * m + j_off][c] += 1
            U[1 * m + (j_off + 1)][c] += 1
            V[j_off * p + k_off][c] += 1
            W[k_off * n + 1][c] += 1
            W[(k_off + 1) * n + 1][c] -= 1
            c += 1
            
            # P3 = A00 (B01 - B11)
            U[0 * m + j_off][c] += 1
            V[j_off * p + (k_off + 1)][c] += 1
            V[(j_off + 1) * p + (k_off + 1)][c] -= 1
            W[(k_off + 1) * n + 0][c] += 1
            W[(k_off + 1) * n + 1][c] += 1
            c += 1
            
            # P4 = A11 (B10 - B00)
            U[1 * m + (j_off + 1)][c] += 1
            V[(j_off + 1) * p + k_off][c] += 1
            V[j_off * p + k_off][c] -= 1
            W[k_off * n + 0][c] += 1
            W[k_off + 1 * n + 1][c] += 1
            c += 1
            
            # P5 = (A00 + A01) B11
            U[0 * m + j_off][c] += 1
            U[0 * m + (j_off + 1)][c] += 1
            V[(j_off + 1) * p + (k_off + 1)][c] += 1
            W[k_off * n + 0][c] -= 1
            W[(k_off + 1) * n + 0][c] += 1
            c += 1
            
            # P6 = (A10 - A00)(B00 + B01)
            U[1 * m + j_off][c] += 1
            U[0 * m + j_off][c] -= 1
            V[j_off * p + k_off][c] += 1
            V[j_off * p + (k_off + 1)][c] += 1
            W[(k_off + 1) * n + 1][c] += 1
            c += 1
            
            # P7 = (A01 - A11)(B10 + B11)
            U[0 * m + (j_off + 1)][c] += 1
            U[1 * m + (j_off + 1)][c] -= 1
            V[(j_off + 1) * p + k_off][c] += 1
            V[(j_off + 1) * p + (k_off + 1)][c] += 1
            W[k_off * n + 0][c] += 1

        add_strassen(0, 0, 0)
        add_strassen(0, 2, 7)
        add_strassen(2, 0, 14)
        add_strassen(2, 2, 21)
        
        col = 28
        for i in range(2):
            for k in range(4):
                U[i * m + 4][col] = 1
                V[4 * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                if time.time() > deadline - 10: return U, V, W
                
        for i in range(2):
            for j in range(4):
                U[i * m + j][col] = 1
                V[j * p + 4][col] = 1
                W[4 * n + i][col] = 1
                col += 1
                if time.time() > deadline - 10: return U, V, W
                
        for i in range(2):
            U[i * m + 4][col] = 1
            V[4 * p + 4][col] = 1
            W[4 * n + i][col] = 1
            col += 1
            if time.time() > deadline - 10: return U, V, W
            
        return U, V, W

    # Build and verify rank-46
    U46, V46, W46 = build_rank46()
    if not validate(U46, V46, W46):
        # Fallback to standard rank 50 if something went wrong
        rank = n * m * p
        U = np.zeros((U_dim, rank))
        V = np.zeros((V_dim, rank))
        W = np.zeros((W_dim, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {
            "U": [[float(x) for x in row] for row in U],
            "V": [[float(x) for x in row] for row in V],
            "W": [[float(x) for x in row] for row in W]
        }

    # Strategy: Systematic Term Removal via Linear Solving
    # We treat the 46 terms as a basis. We want to find a sparse linear combination
    # that equals T. We know 1+1+...+1 works.
    # We try to set some coefficients to 0 and solve for the rest.
    
    # Flatten the rank-1 tensors
    terms = []
    for c in range(46):
        u = U46[:, c]
        v = V46[:, c]
        w = W46[:, c]
        t_flat = np.einsum('a,b,c->abc', u, v, w).flatten()
        terms.append(t_flat)
    
    M = np.column_stack(terms) # Shape (2500, 46)
    T_flat = T.flatten()
    
    best_rank = 46
    best_U, best_V, best_W = U46, V46, W46
    
    # Helper to check if a solution is valid and snap it
    def check_solution(coeffs, kept_indices):
        # coeffs is a vector of length len(kept_indices)
        # Reconstruct U, V, W
        r = len(kept_indices)
        U_new = np.zeros((U_dim, r))
        V_new = np.zeros((V_dim, r))
        W_new = np.zeros((W_dim, r))
        for i, idx in enumerate(kept_indices):
            U_new[:, i] = U46[:, idx] * coeffs[i]
            V_new[:, i] = V46[:, idx] * coeffs[i]
            W_new[:, i] = W46[:, idx] * coeffs[i]
            
        # Check validity
        if not validate(U_new, V_new, W_new, atol=1e-5):
            return False, None, None, None
            
        # Try to snap
        valid, U_s, V_s, W_s = try_snap_validate(U_new, V_new, W_new)
        if valid:
            return True, U_s, V_s, W_s, r
        return False, None, None, None

    # Try removing 1 term
    for c_remove in range(46):
        if time.time() > deadline - 20:
            break
            
        kept = [i for i in range(46) if i != c_remove]
        M_sub = M[:, kept]
        
        # Solve M_sub * c = T_flat
        # Use lstsq for robustness
        try:
            c_sol, residuals, rank, s = np.linalg.lstsq(M_sub, T_flat, rcond=None)
        except np.linalg.LinAlgError:
            continue
            
        # Check if residual is small
        if residuals[0] > 1e-4:
            continue
            
        # Check sparsity of solution
        non_zeros = np.count_nonzero(c_sol, atol=1e-6)
        if non_zeros < best_rank:
            valid, U_s, V_s, W_s, r = check_solution(c_sol, kept)
            if valid:
                best_rank = r
                best_U, best_V, best_W = U_s, V_s, W_s
                if best_rank == 45: # Can't get much lower with 1 removal
                    break

    # Try removing 2 terms
    if best_rank > 45 and time.time() < deadline - 10:
        for c1 in range(46):
            if time.time() > deadline - 5:
                break
            for c2 in range(c1 + 1, 46):
                if time.time() > deadline - 5:
                    break
                kept = [i for i in range(46) if i != c1 and i != c2]
                M_sub = M[:, kept]
                
                try:
                    c_sol, residuals, rank, s = np.linalg.lstsq(M_sub, T_flat, rcond=None)
                except np.linalg.LinAlgError:
                    continue
                    
                if residuals[0] > 1e-4:
                    continue
                    
                non_zeros = np.count_nonzero(c_sol, atol=1e-6)
                if non_zeros < best_rank:
                    valid, U_s, V_s, W_s, r = check_solution(c_sol, kept)
                    if valid:
                        best_rank = r
                        best_U, best_V, best_W = U_s, V_s, W_s
                        if best_rank == 44:
                            break
            if best_rank == 44:
                break

    return {
        "U": [[float(x) for x in row] for row in best_U],
        "V": [[float(x) for x in row] for row in best_V],
        "W": [[float(x) for x in row] for row in best_W]
    }
# EVOLVE-BLOCK-END
