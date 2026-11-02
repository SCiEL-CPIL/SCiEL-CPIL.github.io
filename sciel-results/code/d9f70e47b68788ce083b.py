"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools
import random

def construct():
    n, m, p = DIMENSIONS  # 2, 4, 5
    start_time = time.time()
    deadline = 140.0  # Safety margin for 150s evaluator limit

    # Precompute tensor entries for fast verification
    # Tensor T[i,j,k] = 1 if we need U[i,j]·V[j,k]·W[k,i] = 1
    # Actually, the tensor is: T[i,j,k] = 1 for all (i,j,k) since it's matrix mult
    # The condition is: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all (i,j,k)

    def verify_fast(Ua, Va, Wa, R):
        """Vectorized verification using numpy."""
        # Ua: (n*m, R), Va: (m*p, R), Wa: (p*n, R)
        # For each (i,j,k): sum_r Ua[i*m+j, r] * Va[j*p+k, r] * Wa[k*n+i, r] == 1
        # Build the full tensor contraction
        # T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
        # We can do this by iterating over r and accumulating outer products
        # But that's O(R * n*m*p) which is fine for R<=36, n*m*p=40
        
        # Alternative: for each r, compute outer product of U[:,r], V[:,r], W[:,r]
        # and add to a tensor, then check all entries equal 1
        T = np.zeros((n, m, p), dtype=float)
        for r in range(R):
            u = Ua[:, r].reshape(n, m)
            v = Va[:, r].reshape(m, p)
            w = Wa[:, r].reshape(p, n)
            # T[i,j,k] += u[i,j] * v[j,k] * w[k,i]
            # Use einsum
            T += np.einsum('ij,jk,ki->ijk', u, v, w)
        return np.allclose(T, 1.0, atol=1e-7)

    def verify(Ul, Vl, Wl, R):
        Ua = np.array(Ul, dtype=float)
        Va = np.array(Vl, dtype=float)
        Wa = np.array(Wl, dtype=float)
        return verify_fast(Ua, Va, Wa, R)

    U = [[0.0] * 36 for _ in range(n * m)]
    V = [[0.0] * 36 for _ in range(m * p)]
    W = [[0.0] * 36 for _ in range(p * n)]

    def u_idx(i, j): return i * m + j
    def v_idx(j, k): return j * p + k
    def w_idx(k, i): return k * n + i

    col = 0

    # Strassen 2x2x2 template (rank 7).
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
        V[v_idx(*z10)][col] += 1; V[v_idx(*z00)][col] -= 1
        W[w_idx(*w00)][col] += 1; W[w_idx(*w10)][col] += 1
        col += 1
        # M5
        U[u_idx(*y00)][col] += 1; U[u_idx(*y01)][col] += 1
        V[v_idx(*z11)][col] += 1
        W[w_idx(*w00)][col] -= 1; W[w_idx(*w01)][col] += 1
        col += 1
        # M6
        U[u_idx(*y10)][col] += 1; U[u_idx(*y00)][col] -= 1
        V[v_idx(*z00)][col] += 1; V[v_idx(*z01)][col] += 1
        W[w_idx(*w11)][col] += 1
        col += 1
        # M7
        U[u_idx(*y01)][col] += 1; U[u_idx(*y11)][col] -= 1
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

    base_U, base_V, base_W = [row[:] for row in U], [row[:] for row in V], [row[:] for row in W]
    best = (36, base_U, base_V, base_W)

    if verify(base_U, base_V, base_W, 36):
        def apply_merge(U_src, V_src, W_src, c_target, c_source, sgn, R):
            """Merge column c_source into c_target with sign sgn, return (U,V,W) of rank R."""
            U2 = [row[:] for row in U_src]
            V2 = [row[:] for row in V_src]
            W2 = [row[:] for row in W_src]
            for a in range(n*m):
                U2[a][c_target] += sgn * U_src[a][c_source]
            for a in range(m*p):
                V2[a][c_target] += sgn * V_src[a][c_source]
            for a in range(p*n):
                W2[a][c_target] += sgn * W_src[a][c_source]
            # Remove column c_source (shift columns after it left)
            if c_source < c_target:
                for a in range(n*m):
                    U2[a] = U2[a][:c_source] + U2[a][c_source+1:]
                for a in range(m*p):
                    V2[a] = V2[a][:c_source] + V2[a][c_source+1:]
                for a in range(p*n):
                    W2[a] = W2[a][:c_source] + W2[a][c_source+1:]
            else:
                for a in range(n*m):
                    U2[a] = U2[a][:c_source] + U2[a][c_source+1:]
                for a in range(m*p):
                    V2[a] = V2[a][:c_source] + V2[a][c_source+1:]
                for a in range(p*n):
                    W2[a] = W2[a][:c_source] + W2[a][c_source+1:]
            return U2, V2, W2

        # Phase 1: Find all single-merge pairs that give rank 35
        l_r_pairs = [(0,7),(1,8),(2,9),(3,10),(4,11),(5,12),(6,13),
                     (14,21),(15,22),(16,23),(17,24),(18,25),(19,26),(20,27)]
        other_pairs = []
        for c1 in range(36):
            for c2 in range(c1+1, 36):
                if (c1, c2) not in l_r_pairs:
                    other_pairs.append((c1, c2))

        valid_merges = []
        all_pairs = l_r_pairs + other_pairs

        for (c1, c2) in all_pairs:
            if time.time() > deadline - 50:
                break
            for sgn in (1.0, -1.0):
                U35, V35, W35 = apply_merge(base_U, base_V, base_W, c1, c2, sgn, 36)
                if verify(U35, V35, W35, 35):
                    valid_merges.append((c1, c2, sgn, U35, V35, W35))
                    best = (35, U35, V35, W35)

        # Phase 2: Try sequential merges from valid rank-35 results to get rank 34
        if valid_merges and time.time() < deadline - 40:
            for (c1, c2, sgn, U35, V35, W35) in valid_merges:
                if time.time() > deadline - 35:
                    break
                for nc1 in range(35):
                    if time.time() > deadline - 30:
                        break
                    for nc2 in range(nc1+1, 35):
                        if time.time() > deadline - 30:
                            break
                        for sgn2 in (1.0, -1.0):
                            U34, V34, W34 = apply_merge(U35, V35, W35, nc1, nc2, sgn2, 35)
                            if verify(U34, V34, W34, 34):
                                best = (34, U34, V34, W34)
                                break
                        if best[0] < 35:
                            break
                    if best[0] < 35:
                        break
                if best[0] < 35:
                    break

        # Phase 3: Use linear algebra to find dependencies
        # Build the constraint matrix and find null space
        if time.time() < deadline - 20:
            current_rank = best[0]
            if current_rank <= 35:
                cur_U = np.array(best[1], dtype=float)
                cur_V = np.array(best[2], dtype=float)
                cur_W = np.array(best[3], dtype=float)
                
                # Try to find a column that can be expressed as a linear combination
                # of others by checking if removing it and adjusting works
                # This is equivalent to checking if the column is in the span of others
                # in the tensor space
                
                # Alternative: try random linear combinations
                rng = random.Random(123)
                for attempt in range(1000):
                    if time.time() > deadline - 15:
                        break
                    c1 = rng.randint(0, current_rank - 1)
                    c2 = rng.randint(0, current_rank - 1)
                    if c1 == c2:
                        continue
                    # Try various signs and coefficients
                    for sgn in (1.0, -1.0):
                        U_new, V_new, W_new = apply_merge(
                            best[1], best[2], best[3], c1, c2, sgn, current_rank)
                        if verify(U_new, V_new, W_new, current_rank - 1):
                            best = (current_rank - 1, U_new, V_new, W_new)
                            current_rank -= 1
                            break

    rank, bU, bV, bW = best
    return {
        "U": bU,
        "V": bV,
        "W": bW
    }
# EVOLVE-BLOCK-END
