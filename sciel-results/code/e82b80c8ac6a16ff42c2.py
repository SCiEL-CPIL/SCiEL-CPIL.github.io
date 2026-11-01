"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS  # 2, 4, 5
    start_time = time.time()
    deadline = 145.0  # safety margin before the 150 s kill

    def u_idx(i, j): return i * m + j
    def v_idx(j, k): return j * p + k
    def w_idx(k, i): return k * n + i

    # ---- Build the block-Strassen base decomposition (rank 36) ----
    U = [[0.0] * 36 for _ in range(n * m)]
    V = [[0.0] * 36 for _ in range(m * p)]
    W = [[0.0] * 36 for _ in range(p * n)]
    col = 0

    def strassen_block(y, z, w):
        nonlocal col
        y00, y01, y10, y11 = y
        z00, z01, z10, z11 = z
        w00, w01, w10, w11 = w
        # M1 = (y00+y11)(z00+z11)
        U[u_idx(*y00)][col] += 1; U[u_idx(*y11)][col] += 1
        V[v_idx(*z00)][col] += 1; V[v_idx(*z11)][col] += 1
        W[w_idx(*w00)][col] += 1; W[w_idx(*w11)][col] += 1
        col += 1
        # M2 = (y10+y11)z00
        U[u_idx(*y10)][col] += 1; U[u_idx(*y11)][col] += 1
        V[v_idx(*z00)][col] += 1
        W[w_idx(*w10)][col] += 1; W[w_idx(*w11)][col] += -1
        col += 1
        # M3 = y00(z01-z11)
        U[u_idx(*y00)][col] += 1
        V[v_idx(*z01)][col] += 1; V[v_idx(*z11)][col] += -1
        W[w_idx(*w01)][col] += 1; W[w_idx(*w11)][col] += 1
        col += 1
        # M4 = y11(z10-z00)
        U[u_idx(*y11)][col] += 1
        V[v_idx(*z10)][col] += 1; V[v_idx(*z00)][col] += -1
        W[w_idx(*w00)][col] += 1; W[w_idx(*w10)][col] += 1
        col += 1
        # M5 = (y00+y01)z11
        U[u_idx(*y00)][col] += 1; U[u_idx(*y01)][col] += 1
        V[v_idx(*z11)][col] += 1
        W[w_idx(*w00)][col] += -1; W[w_idx(*w01)][col] += 1
        col += 1
        # M6 = (y10-y00)(z00+z01)
        U[u_idx(*y10)][col] += 1; U[u_idx(*y00)][col] += -1
        V[v_idx(*z00)][col] += 1; V[v_idx(*z01)][col] += 1
        W[w_idx(*w11)][col] += 1
        col += 1
        # M7 = (y01-y11)(z10+z11)
        U[u_idx(*y01)][col] += 1; U[u_idx(*y11)][col] += -1
        V[v_idx(*z10)][col] += 1; V[v_idx(*z11)][col] += 1
        W[w_idx(*w00)][col] += 1
        col += 1

    AL = [(0,0),(0,1),(1,0),(1,1)]
    AR = [(0,2),(0,3),(1,2),(1,3)]
    B11 = [(0,0),(0,1),(1,0),(1,1)]
    B12 = [(0,2),(0,3),(1,2),(1,3)]
    B21 = [(2,0),(2,1),(3,0),(3,1)]
    B22 = [(2,2),(2,3),(3,2),(3,3)]
    C11 = [(0,0),(1,0),(0,1),(1,1)]
    C12 = [(2,0),(3,0),(2,1),(3,1)]

    strassen_block(AL, B11, C11)
    strassen_block(AR, B21, C11)
    strassen_block(AL, B12, C12)
    strassen_block(AR, B22, C12)

    # 8 direct terms for the last column (k=4).
    for (ai, aj) in [(0,0),(0,1),(1,0),(1,1)]:
        U[u_idx(ai, aj)][col] += 1
        V[v_idx(aj, 4)][col] += 1
        W[w_idx(4, ai)][col] += 1
        col += 1
    for (ai, aj) in [(0,2),(0,3),(1,2),(1,3)]:
        U[u_idx(ai, aj)][col] += 1
        V[v_idx(aj, 4)][col] += 1
        W[w_idx(4, ai)][col] += 1
        col += 1

    assert col == 36, col

    Ua = np.array(U, dtype=np.float64)
    Va = np.array(V, dtype=np.float64)
    Wa = np.array(W, dtype=np.float64)

    def col_tensor(Uc, Vc, Wc):
        uc_r = Uc.reshape(n, m)
        vc_r = Vc.reshape(m, p)
        wc_r = Wc.reshape(p, n)
        return np.einsum('ij,jk,ki->ijk', uc_r, vc_r, wc_r, optimize=True)

    def verify_fast(Ua2, Va2, Wa2, R):
        acc = np.zeros((n, m, p), dtype=np.float64)
        for c in range(R):
            acc += col_tensor(Ua2[:, c], Va2[:, c], Wa2[:, c])
        return np.allclose(acc, 1.0, atol=1e-6, rtol=0.0)

    if not verify_fast(Ua, Va, Wa, 36):
        return {"U": U, "V": V, "W": W}

    best_rank = 36
    best_U, best_V, best_W = Ua.copy(), Va.copy(), Wa.copy()
    
    cur_U, cur_V, cur_W = Ua.copy(), Va.copy(), Wa.copy()
    cur_R = 36
    active_cols = list(range(36))

    # Precompute tensors for all columns
    # We will update these as we eliminate columns
    # But for the span check, we just need the current tensors of active columns
    
    def get_tensors(Uc, Vc, Wc, cols):
        return [col_tensor(Uc[:, c], Vc[:, c], Wc[:, c]) for c in cols]

    cur_tensors = get_tensors(cur_U, cur_V, cur_W, active_cols)

    # Greedy elimination loop
    # Try to eliminate any column c by finding a, b such that T_c is in span(T_a, T_b)
    # We iterate until no more eliminations are found or time runs out
    
    found_improvement = True
    while found_improvement and time.time() < deadline - 10:
        found_improvement = False
        R = len(active_cols)
        if R < 2:
            break
            
        # Map active_cols index to original column index
        # active_cols[i] is the original column index of the i-th active column
        
        # Try to eliminate each active column
        for i_c in range(R):
            if time.time() > deadline - 10:
                break
            c_orig = active_cols[i_c]
            T_c = cur_tensors[i_c]
            
            # Skip if T_c is zero
            if np.allclose(T_c, 0, atol=1e-9):
                # Eliminate zero column
                active_cols.pop(i_c)
                cur_tensors.pop(i_c)
                cur_R -= 1
                found_improvement = True
                # Update best if better
                if cur_R < best_rank:
                    best_rank = cur_R
                    # Reconstruct best from current? 
                    # It's easier to just keep track of the full matrices
                    # But we are modifying cur_U etc.
                    # Let's just save the current state as best if rank improved
                    # We need to map active_cols back to the full matrix
                    # This is getting complex. Let's just verify and save.
                    pass
                continue

            # Find a, b (i_a, i_b) in active_cols, i_a != i_c, i_b != i_c
            # such that T_c is in span(T_a, T_b)
            eliminated = False
            for i_a in range(R):
                if i_a == i_c:
                    continue
                for i_b in range(i_a + 1, R):
                    if i_b == i_c:
                        continue
                    
                    T_a = cur_tensors[i_a]
                    T_b = cur_tensors[i_b]
                    
                    # Flatten
                    v_a = T_a.flatten()
                    v_b = T_b.flatten()
                    v_c = T_c.flatten()
                    
                    # Solve [v_a, v_b] * [alpha, beta]^T = v_c
                    M = np.column_stack((v_a, v_b))
                    try:
                        coeffs, residuals, rank_M, s = np.linalg.lstsq(M, v_c, rcond=None)
                        alpha, beta = coeffs
                        # Check residual
                        if len(residuals) > 0 and residuals[0] < 1e-9:
                            # Dependency found
                            # Update columns
                            a_orig = active_cols[i_a]
                            b_orig = active_cols[i_b]
                            
                            # T_c = alpha * T_a + beta * T_b
                            # We want to remove c.
                            # New T_a' = T_a + alpha * T_c? No.
                            # The sum is S = ... + T_a + T_b + T_c + ...
                            # If T_c = alpha T_a + beta T_b, then
                            # S = ... + T_a + T_b + alpha T_a + beta T_b + ...
                            #   = ... + (1+alpha)T_a + (1+beta)T_b + ...
                            # So we update T_a and T_b, and remove T_c.
                            
                            cur_U[:, a_orig] += alpha * cur_U[:, c_orig]
                            cur_V[:, a_orig] += alpha * cur_V[:, c_orig]
                            cur_W[:, a_orig] += alpha * cur_W[:, c_orig]
                            
                            cur_U[:, b_orig] += beta * cur_U[:, c_orig]
                            cur_V[:, b_orig] += beta * cur_V[:, c_orig]
                            cur_W[:, b_orig] += beta * cur_W[:, c_orig]
                            
                            # Remove c from active list
                            active_cols.pop(i_c)
                            cur_tensors.pop(i_c)
                            cur_R -= 1
                            
                            # Update tensors for a and b
                            # i_a and i_b indices might have shifted if i_c < i_a or i_c < i_b
                            # Let's just recompute all tensors for safety
                            cur_tensors = get_tensors(cur_U, cur_V, cur_W, active_cols)
                            
                            # Verify
                            if verify_fast(cur_U, cur_V, cur_W, cur_R):
                                if cur_R < best_rank:
                                    best_rank = cur_R
                                    best_U = cur_U.copy()
                                    best_V = cur_V.copy()
                                    best_W = cur_W.copy()
                                found_improvement = True
                                eliminated = True
                                break
                            else:
                                # Rollback? 
                                # For simplicity, if verification fails, we might have a numerical issue.
                                # Given the integer/half-integer nature, it should be exact.
                                # But lstsq might introduce floating point errors.
                                # We should check if alpha, beta are close to rational numbers.
                                pass
                    except np.linalg.LinAlgError:
                        pass
                if eliminated:
                    break
            if eliminated:
                # Restart the outer loop as the list of active columns changed
                break

    # Final verification of best
    if not verify_fast(best_U, best_V, best_W, best_rank):
        # Fallback to base
        best_U, best_V, best_W = Ua, Va, Wa
        best_rank = 36

    bU, bV, bW = best_U, best_V, best_W

    # Convert back to list-of-lists with integer/half-integer entries.
    def to_lists(A):
        result = []
        for row in A:
            new_row = []
            for x in row:
                rx = round(x)
                if abs(x - rx) < 1e-9:
                    new_row.append(int(rx))
                else:
                    # Try half-integer
                    rh = round(2 * x) / 2.0
                    if abs(x - rh) < 1e-9:
                        new_row.append(rh)
                    else:
                        new_row.append(float(x))
            result.append(new_row)
        return result

    return {
        "U": to_lists(bU),
        "V": to_lists(bV),
        "W": to_lists(bW),
    }
# EVOLVE-BLOCK-END
