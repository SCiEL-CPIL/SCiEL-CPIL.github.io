import numpy as np
import time

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit

    # Build the matrix multiplication tensor T[a, b, c]
    # a = i*m + j, b = j*p + k, c = k*n + i
    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    # First, construct the rank-28 Strassen decomposition as a baseline
    rank28 = 28
    U28 = np.zeros((n * m, rank28), dtype=int)
    V28 = np.zeros((m * p, rank28), dtype=int)
    W28 = np.zeros((p * n, rank28), dtype=int)

    strassen_terms = [
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
        ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]),
        ([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)]),
        ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)]),
        ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)]),
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)]),
    ]

    def map_u_a(side, local_idx):
        if side == 'L':
            return [0, 1, 4, 5][local_idx]
        else:
            return [2, 3, 6, 7][local_idx]

    def map_v_b(block, local_idx):
        r = local_idx // 2
        c = local_idx % 2
        if block == 'TL':
            return r * 4 + c
        elif block == 'TR':
            return r * 4 + (2 + c)
        elif block == 'BL':
            return (2 + r) * 4 + c
        elif block == 'BR':
            return (2 + r) * 4 + (2 + c)

    def map_w_c(side, local_idx):
        r = local_idx // 2
        c = local_idx % 2
        if side == 'L':
            return c * 2 + r
        else:
            return (2 + c) * 2 + r

    col = 0
    # Apply Strassen to 4 block products
    block_configs = [
        ('L', 'TL', 'L'),  # A_L * B_TL -> C_L
        ('R', 'BL', 'L'),  # A_R * B_BL -> C_L
        ('L', 'TR', 'R'),  # A_L * B_TR -> C_R
        ('R', 'BR', 'R'),  # A_R * B_BR -> C_R
    ]

    for a_side, b_block, c_side in block_configs:
        for u_l, v_l, w_l in strassen_terms:
            for idx, coeff in u_l:
                U28[map_u_a(a_side, idx), col] += coeff
            for idx, coeff in v_l:
                V28[map_v_b(b_block, idx), col] += coeff
            for idx, coeff in w_l:
                W28[map_w_c(c_side, idx), col] += coeff
            col += 1

    # Verify rank-28 solution
    def verify_decomp(U, V, W, tol=1e-10):
        """Check if U, V, W exactly decompose the tensor T."""
        rank = U.shape[1]
        for a in range(n * m):
            for b in range(m * p):
                for c in range(p * n):
                    val = 0.0
                    for r in range(rank):
                        val += U[a, r] * V[b, r] * W[c, r]
                    if abs(val - T[a, b, c]) > tol:
                        return False
        return True

    # Check if rank 28 works
    if not verify_decomp(U28, V28, W28):
        # Should not happen, but fall back to naive if something is wrong
        rank = 32
        U = np.zeros((n * m, rank), dtype=int)
        V = np.zeros((m * p, rank), dtype=int)
        W = np.zeros((p * n, rank), dtype=int)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx = i * m * p + j * p + k
                    U[i * m + j, idx] = 1
                    V[j * p + k, idx] = 1
                    W[k * n + i, idx] = 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Try to find a rank-27 decomposition by fixing U and V and solving for W
    # For each term to eliminate, try fixing U and V from the rank-28 solution
    # (with that term removed) and solve the linear system for W.
    
    def solve_for_W(U_fixed, V_fixed):
        """Given U (8x27) and V (16x27), solve for W (8x27) such that
        T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r] for all a,b,c."""
        # For each c, we have 128 equations in 27 unknowns W[c,0..26]
        # M_c[a*16+b, r] = U[a,r] * V[b,r]
        # t_c[a*16+b] = T[a,b,c]
        
        W_solution = np.zeros((p * n, 27))
        for c in range(p * n):
            # Build M_c
            M_c = np.zeros((n * m * m * p, 27))
            for a in range(n * m):
                for b in range(m * p):
                    for r in range(27):
                        M_c[a * (m * p) + b, r] = U_fixed[a, r] * V_fixed[b, r]
            t_c = np.zeros(n * m * m * p)
            for a in range(n * m):
                for b in range(m * p):
                    t_c[a * (m * p) + b] = T[a, b, c]
            
            # Solve the least squares problem
            try:
                w_c, residuals, rank_c, sv = np.linalg.lstsq(M_c, t_c, rcond=None)
                W_solution[c, :] = w_c
            except:
                return None
        
        return W_solution

    # Try each possible term to eliminate
    np.random.seed(42)
    found = False
    best_rank = 28
    best_U, best_V, best_W = U28.copy(), V28.copy(), W28.copy()
    
    for elim in range(28):
        if time.time() > deadline - 5:
            break
        
        # Keep all terms except elim
        keep = [i for i in range(28) if i != elim]
        
        U_fixed = U28[:, keep].astype(float)
        V_fixed = V28[:, keep].astype(float)
        
        # Solve for W
        W_solution = solve_for_W(U_fixed, V_fixed)
        
        if W_solution is not None:
            # Check if the solution is exact
            if verify_decomp(U_fixed, V_fixed, W_solution):
                # Check if W has integer/half-integer entries
                W_rounded = np.round(W_solution * 2) / 2
                if verify_decomp(U_fixed, V_fixed, W_rounded):
                    found = True
                    best_U, best_V, best_W = U_fixed, V_fixed, W_rounded
                    best_rank = 27
                    break
    
    # If we didn't find a solution with the exact U and V from the rank-28 solution,
    # try with small perturbations
    if not found and time.time() < deadline - 5:
        for elim in range(28):
            if time.time() > deadline - 5:
                break
            
            keep = [i for i in range(28) if i != elim]
            
            for attempt in range(5):
                if time.time() > deadline - 5:
                    break
                
                if attempt == 0:
                    U_fixed = U28[:, keep].astype(float)
                    V_fixed = V28[:, keep].astype(float)
                else:
                    U_fixed = U28[:, keep].astype(float) + np.random.randn(n * m, 27) * 0.1
                    V_fixed = V28[:, keep].astype(float) + np.random.randn(m * p, 27) * 0.1
                
                W_solution = solve_for_W(U_fixed, V_fixed)
                
                if W_solution is not None:
                    if verify_decomp(U_fixed, V_fixed, W_solution):
                        W_rounded = np.round(W_solution * 2) / 2
                        if verify_decomp(U_fixed, V_fixed, W_rounded):
                            found = True
                            best_U, best_V, best_W = U_fixed, V_fixed, W_rounded
                            best_rank = 27
                            break
            
            if found:
                break
    
    # Return the best solution found
    if best_rank == 27:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    else:
        return {"U": U28.tolist(), "V": V28.tolist(), "W": W28.tolist()}

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    res = construct()
    n, m, p = DIMENSIONS
    U = np.array(res["U"], dtype=float)
    V = np.array(res["V"], dtype=float)
    W = np.array(res["W"], dtype=float)
    A = np.arange(n * m).reshape(n, m) + 1.0
    B = np.arange(m * p).reshape(m, p) + 2.0
    C = A @ B
    Cf = np.zeros((n, p))
    for ii in range(n):
        for kk in range(p):
            acc = 0.0
            for cc in range(U.shape[1]):
                for jj in range(m):
                    acc += U[ii * m + jj, cc] * V[jj * p + kk, cc] * W[kk * n + ii, cc]
            Cf[ii, kk] = acc
    assert np.allclose(C, Cf), (C, Cf)
    print("OK rank", U.shape[1])
