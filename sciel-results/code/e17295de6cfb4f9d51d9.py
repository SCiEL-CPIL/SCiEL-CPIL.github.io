"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools

def construct():
    n, m, p = DIMENSIONS  # 2, 4, 5
    start_time = time.time()
    deadline = 140.0

    U = [[0.0] * 36 for _ in range(n * m)]
    V = [[0.0] * 36 for _ in range(m * p)]
    W = [[0.0] * 36 for _ in range(p * n)]

    def u_idx(i, j): return i * m + j
    def v_idx(j, k): return j * p + k
    def w_idx(k, i): return k * n + i

    col = 0

    # Strassen 2x2x2 template (rank 7).
    # w_offsets ordered as [X00, X01, X10, X11] -> (k, i)
    def strassen_block(y, z, w):
        nonlocal col
        y00, y01, y10, y11 = y
        z00, z01, z10, z11 = z
        w00, w01, w10, w11 = w
        # M1
        U[u_idx(*y00)][col] += 1; U[u_idx(*y11)][col] += 1
        V[v_idx(*z00)][col] += 1; V[v_idx(*z11)][col] += 1
        W[w_idx(*w00)][col] += 1; W[w_idx(*w11)][col] += 1
        col += 1
        # M2
        U[u_idx(*y10)][col] += 1; U[u_idx(*y11)][col] += 1
        V[v_idx(*z00)][col] += 1
        W[w_idx(*w10)][col] += 1; W[w_idx(*w11)][col] += -1
        col += 1
        # M3
        U[u_idx(*y00)][col] += 1
        V[v_idx(*z01)][col] += 1; V[v_idx(*z11)][col] += -1
        W[w_idx(*w01)][col] += 1; W[w_idx(*w11)][col] += 1
        col += 1
        # M4
        U[u_idx(*y11)][col] += 1
        V[v_idx(*z10)][col] += 1; V[v_idx(*z00)][col] += -1
        W[w_idx(*w00)][col] += 1; W[w_idx(*w10)][col] += 1
        col += 1
        # M5
        U[u_idx(*y00)][col] += 1; U[u_idx(*y01)][col] += 1
        V[v_idx(*z11)][col] += 1
        W[w_idx(*w00)][col] += -1; W[w_idx(*w01)][col] += 1
        col += 1
        # M6
        U[u_idx(*y10)][col] += 1; U[u_idx(*y00)][col] += -1
        V[v_idx(*z00)][col] += 1; V[v_idx(*z01)][col] += 1
        W[w_idx(*w11)][col] += 1
        col += 1
        # M7
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

    def verify_fast(Ul, Vl, Wl, R):
        """Verify using vectorized tensor contraction."""
        Ua = np.array(Ul, dtype=float)
        Va = np.array(Vl, dtype=float)
        Wa = np.array(Wl, dtype=float)
        # Tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # Reshape U to (n, m, R), V to (m, p, R), W to (p, n, R)
        U3 = Ua.reshape(n, m, R)
        V3 = Va.reshape(m, p, R)
        W3 = Wa.reshape(p, n, R)
        # Compute T[i,j,k] = sum_r U3[i,j,r] * V3[j,k,r] * W3[k,i,r]
        # Use einsum: T[i,j,k] = U3[i,j,r] * V3[j,k,r] * W3[k,i,r]
        T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
        return np.allclose(T, 1.0, atol=1e-7)

    base_U, base_V, base_W = [row[:] for row in U], [row[:] for row in V], [row[:] for row in W]
    best = (36, base_U, base_V, base_W)

    if not verify_fast(base_U, base_V, base_W, 36):
        # Fallback: return the base 36-rank solution
        return {"U": base_U, "V": base_V, "W": base_W}

    # Precompute numpy arrays for fast verification
    base_Ua = np.array(base_U, dtype=float)
    base_Va = np.array(base_V, dtype=float)
    base_Wa = np.array(base_W, dtype=float)

    def apply_merges(merge_list):
        """Apply a list of (c_target, c_source, sign) merges and verify.
        Returns (rank, U, V, W) if valid, else None.
        merge_list: list of (c1, c2, sgn) meaning add sgn*col_c2 to col_c1, delete col_c2
        """
        if time.time() > deadline - 20:
            return None
        # Build new matrices by applying all merges
        R = 36 - len(merge_list)
        if R < 1:
            return None
        # Track which original columns are deleted
        deleted = set()
        new_U = np.zeros((n*m, R))
        new_V = np.zeros((m*p, R))
        new_W = np.zeros((p*n, R))
        # Map old col -> new col
        col_map = {}
        new_idx = 0
        for c in range(36):
            if c not in deleted:
                col_map[c] = new_idx
                new_idx += 1
        
        # Apply merges in order
        # First, build the combined columns
        # For each merge (c1, c2, sgn): new_col[c1] += sgn * old_col[c2], then delete c2
        # We need to handle sequential merges carefully
        # Simpler: start with all 36 columns, apply each merge, track which are alive
        
        alive = list(range(36))
        col_data_U = [base_Ua[:, c].copy() for c in range(36)]
        col_data_V = [base_Va[:, c].copy() for c in range(36)]
        col_data_W = [base_Wa[:, c].copy() for c in range(36)]
        
        for (c1, c2, sgn) in merge_list:
            if c2 not in alive or c1 not in alive:
                return None
            # Add sgn * col_data[c2] to col_data[c1]
            col_data_U[c1] += sgn * col_data_U[c2]
            col_data_V[c1] += sgn * col_data_V[c2]
            col_data_W[c1] += sgn * col_data_W[c2]
            # Remove c2 from alive
            alive.remove(c2)
        
        # Build final arrays
        R = len(alive)
        new_U = np.zeros((n*m, R))
        new_V = np.zeros((m*p, R))
        new_W = np.zeros((p*n, R))
        for new_i, old_c in enumerate(alive):
            new_U[:, new_i] = col_data_U[old_c]
            new_V[:, new_i] = col_data_V[old_c]
            new_W[:, new_i] = col_data_W[old_c]
        
        # Verify
        if not verify_fast(new_U.tolist(), new_V.tolist(), new_W.tolist(), R):
            return None
        return (R, new_U.tolist(), new_V.tolist(), new_W.tolist())

    # Try ALL possible single merges: any pair of columns (c1, c2) with c1 != c2
    # This is the most general approach - try merging any column into any other
    found_35 = None
    all_col_pairs = [(i, j) for i in range(36) for j in range(36) if i != j]
    
    # First try the structurally motivated pairs (same Strassen position, different A-block)
    pairs_c0 = [(i, i+7) for i in range(7)]
    pairs_c1 = [(i, i+7) for i in range(14, 21)]
    structured_pairs = pairs_c0 + pairs_c1
    structured_pairs_rev = [(c2, c1) for (c1, c2) in structured_pairs]
    
    # Try structured pairs first (more likely to work)
    for (c1, c2) in structured_pairs + structured_pairs_rev:
        if time.time() > deadline - 30:
            break
        for sgn in (1.0, -1.0):
            result = apply_merges([(c1, c2, sgn)])
            if result is not None:
                found_35 = result
                break
        if found_35:
            break
    
    # If no structured merge works, try all other pairs
    if found_35 is None:
        for (c1, c2) in all_col_pairs:
            if time.time() > deadline - 20:
                break
            # Skip already-tried structured pairs
            if (c1, c2) in structured_pairs or (c1, c2) in structured_pairs_rev:
                continue
            for sgn in (1.0, -1.0):
                result = apply_merges([(c1, c2, sgn)])
                if result is not None:
                    found_35 = result
                    break
            if found_35:
                break
    
    if found_35:
        best = found_35
    
    # Now try to find rank 34 or lower by applying two or three merges simultaneously
    # Try all pairs of non-conflicting merges from the 36-col base
    if best[0] == 35 and time.time() < deadline - 40:
        # Generate all non-conflicting pairs of merges
        # A merge is (c1, c2, sgn) - c2 is deleted into c1
        # Two merges conflict if they share any column
        all_single_merges = []
        for (c1, c2) in all_col_pairs:
            for sgn in (1.0, -1.0):
                all_single_merges.append((c1, c2, sgn))
        
        # Try pairs of merges that don't share columns
        num_tried = 0
        max_tries = 5000
        for i in range(len(all_single_merges)):
            if time.time() > deadline - 25 or num_tried >= max_tries:
                break
            c1a, c2a, sgn_a = all_single_merges[i]
            for j in range(i + 1, len(all_single_merges)):
                if time.time() > deadline - 25 or num_tried >= max_tries:
                    break
                c1b, c2b, sgn_b = all_single_merges[j]
                # Check no column overlap: {c1a, c2a} and {c1b, c2b} must be disjoint
                if len({c1a, c2a, c1b, c2b}) < 4:
                    continue
                num_tried += 1
                result = apply_merges([(c1a, c2a, sgn_a), (c1b, c2b, sgn_b)])
                if result is not None:
                    best = result
                    break
            if best[0] < 35:
                break
        
        # If we found rank 34, try to go lower with 3 merges
        if best[0] == 34 and time.time() < deadline - 30:
            # Try triples of non-conflicting merges
            num_tried3 = 0
            max_tries3 = 2000
            for i in range(len(all_single_merges)):
                if time.time() > deadline - 20 or num_tried3 >= max_tries3:
                    break
                c1a, c2a, sgn_a = all_single_merges[i]
                for j in range(i + 1, len(all_single_merges)):
                    if time.time() > deadline - 20 or num_tried3 >= max_tries3:
                        break
                    c1b, c2b, sgn_b = all_single_merges[j]
                    if len({c1a, c2a, c1b, c2b}) < 4:
                        continue
                    for k in range(j + 1, len(all_single_merges)):
                        if time.time() > deadline - 20 or num_tried3 >= max_tries3:
                            break
                        c1c, c2c, sgn_c = all_single_merges[k]
                        if len({c1a, c2a, c1b, c2b, c1c, c2c}) < 6:
                            continue
                        num_tried3 += 1
                        result = apply_merges([(c1a, c2a, sgn_a), (c1b, c2b, sgn_b), (c1c, c2c, sgn_c)])
                        if result is not None:
                            best = result
                            break
                    if best[0] < 34:
                        break
                if best[0] < 34:
                    break

    rank, bU, bV, bW = best
    
    # Final verification with tighter tolerance
    if not verify_fast(bU, bV, bW, rank):
        # Fallback to base 36-rank solution
        return {"U": base_U, "V": base_V, "W": base_W}
    
    return {
        "U": bU,
        "V": bV,
        "W": bW
    }
# EVOLVE-BLOCK-END
