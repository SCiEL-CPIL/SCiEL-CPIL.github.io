import time
import sys
from fractions import Fraction

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    deadline = t0 + 130.0

    def _elapsed_ok():
        return (time.time() - t0) < deadline

    n, m, p = 2, 3, 8
    nm, mp, pn = n * m, m * p, p * n

    # ---- Schoolbook fallback (rank 48) ----
    def _schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for a in range(m):
                for k in range(p):
                    U[i * m + a][col] = 1
                    V[a * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    # ---- Exact validation using Fractions ----
    def _validate(candidate):
        U = candidate["U"]
        V = candidate["V"]
        W = candidate["W"]
        rank = len(U[0])
        for i in range(n):
            for a in range(m):
                for k in range(p):
                    acc = Fraction(0)
                    u_row = U[i * m + a]
                    v_row = V[a * p + k]
                    w_row = W[k * n + i]
                    for c in range(rank):
                        if u_row[c] == 0 or v_row[c] == 0 or w_row[c] == 0:
                            continue
                        acc += Fraction(u_row[c]) * Fraction(v_row[c]) * Fraction(w_row[c])
                    if acc != 1:
                        return False
        return True

    # ---- Strassen 2x2x2 decomposition ----
    S_U = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    S_V = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    S_W = [
        [1, 0, 0, 1],
        [0, 1, 0, -1],
        [0, 0, 1, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ]

    def _verify_strassen():
        for i in range(2):
            for a in range(2):
                for k in range(2):
                    acc = 0
                    for r in range(7):
                        u_val = S_U[r][i * 2 + a]
                        v_val = S_V[r][a * 2 + k]
                        w_val = S_W[r][k * 2 + i]
                        acc += u_val * v_val * w_val
                    if acc != 1:
                        return False
        return True

    if not _verify_strassen():
        return _schoolbook()

    # ---- Build the rank-44 base: Strassen for 2x2x8 block + schoolbook for 2x1x8 ----
    def _build_base():
        rank = 44
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        col = 0

        # Part 1: A1 (2x2) * B1 (2x8) using Strassen on 4 blocks of (2x2)*(2x2)
        for block_idx in range(4):
            k0 = block_idx * 2
            for r in range(7):
                for li in range(2):
                    for la in range(2):
                        u_val = S_U[r][li * 2 + la]
                        if u_val != 0:
                            U[li * m + la][col] = u_val
                for la in range(2):
                    for lk in range(2):
                        v_val = S_V[r][la * 2 + lk]
                        if v_val != 0:
                            V[la * p + (k0 + lk)][col] = v_val
                for lk in range(2):
                    for li in range(2):
                        w_val = S_W[r][lk * 2 + li]
                        if w_val != 0:
                            W[(k0 + lk) * n + li][col] = w_val
                col += 1

        # Part 2: A2 (2x1) * B2 (1x8) schoolbook
        for i in range(2):
            for k in range(8):
                U[i * m + 2][col] = 1
                V[2 * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1

        return {"U": U, "V": V, "W": W}

    base = _build_base()
    if not _validate(base):
        return _schoolbook()

    # ---- Search for rank reduction in the 2x1x8 sub-block ----
    # The sub-block has U rows [i*3+2 for i in 0,1], V rows [2*8+k for k in 0..7],
    # W rows [k*2+i for k in 0..7, i in 0,1].
    # We seek a factorization of rank r < 16 where:
    #   sum_c u_i[c] * v_k[c] * w_{k,i}[c] = 1  for all i, k
    # Try rank 8 with u_i = (1,1,...,1) and solve for v, w.
    # This gives a system of 16 equations in 16 unknowns (v_k, w_{k,i}).

    def _search_subblock():
        """Search for a compressed factorization of the 2x1x8 sub-block."""
        if not _elapsed_ok():
            return None

        # Try rank r = 8: u_i = all-ones vector of length r
        # Then v_k * w_{k,i} = 1 for each (k, i)
        # This requires w_{k,0} * w_{k,1} = 1 / (v_k)^2 ... 
        # Actually: for each k, we need v_k * w_{k,0} = 1 AND v_k * w_{k,1} = 1
        # So w_{k,0} = w_{k,1} = 1/v_k for all k.
        # But then w_{k,0} = w_{k,1} for all k, meaning the W rows for i=0 and i=1
        # are identical in the sub-block portion. This is valid!
        # So rank 8 works: v_k = 1, w_{k,i} = 1 for all k, i.
        # Wait, that's just the schoolbook with u_i = 1, v_k = 1, w = 1.
        # That's rank 8 (shared across k), not rank 16.
        
        # Let me reconsider. In the base, each (i,k) pair has its own column.
        # For rank 8, we can have 8 columns, one per k, with:
        #   u_i[c_k] = 1 for both i=0 and i=1 (same value)
        #   v_k[c_k] = 1
        #   w_{k,i}[c_k] = 1 for both i=0 and i=1
        # Then for each (i,k): sum over c of u_i[c]*v_k[c]*w_{k,i}[c] = 1*1*1 = 1. ✓
        # This works! Rank 8 for the sub-block.
        
        # But wait - can we do even better? Try rank 4.
        # With 4 columns, we need to cover 8 k-values.
        # If two k-values share a column, say k1 and k2 both use column c:
        #   v_{k1}[c] * w_{k1,i}[c] = 1 and v_{k2}[c] * w_{k2,i}[c] = 1
        # This is fine as long as the other contributions are zero.
        # But w_{k1,i} and w_{k2,i} are different rows, so they can have different values.
        # Actually this always works with rank = p = 8 (one column per k).
        # Can we do rank < 8?
        # With rank r, we have r columns. Each column c has values u_i[c], v_k[c], w_{k,i}[c].
        # For the equation to hold, for each (i,k): sum_c u_i[c]*v_k[c]*w_{k,i}[c] = 1.
        # If r < 8, some k-values must share columns.
        # Let's say k1 and k2 both have nonzero v in column c.
        # Then for (i,k1): u_i[c]*v_{k1}[c]*w_{k1,i}[c] contributes.
        # For (i,k2): u_i[c]*v_{k2}[c]*w_{k2,i}[c] contributes.
        # These are independent equations, so it should still work.
        # The constraint is that u_i is the same for all k.
        # Let u_0 = (u_0[0],...,u_0[r-1]), u_1 = (u_1[0],...,u_1[r-1]).
        # For each k and each i: sum_c u_i[c]*v_k[c]*w_{k,i}[c] = 1.
        # This is 2*8 = 16 equations in 2*8 + 8 + 8 = 32 unknowns (u, v, w).
        # So rank 4 should be feasible (16 eqs, 32 unknowns).
        
        # Let me try a simple construction for rank 4:
        # u_0 = (1, 1, 1, 1), u_1 = (1, 1, 1, 1)
        # v_k = (1, 0, 0, 0) for k=0, (0,1,0,0) for k=1, etc. -- but that's rank 8.
        
        # Alternative: use a Hadamard-like structure.
        # Let's just verify rank 8 works and use it.
        
        # Actually, let me think more carefully. The minimum rank of the 2x1x8 tensor
        # (which is just a matrix of 1s, shape 2x8) as a tensor T[i,k] = 1.
        # The tensor rank of a matrix M (as a 3-tensor with one index of size 1)
        # equals the matrix rank of M. The 2x8 all-ones matrix has rank 1.
        # So the tensor rank is 1!
        # T[i,k] = u_i * v_k * w_k  where w_k is a scalar.
        # With u_0 = 1, u_1 = 1, v_k = 1, w_k = 1: T[i,k] = 1*1*1 = 1. ✓
        # So rank 1 suffices for the sub-block!
        
        # But wait, in our tensor decomposition framework:
        # T[i,a,k] = sum_c U[i,a,c] * V[a,k,c] * W[k,i,c]
        # For a=2: T[i,2,k] = sum_c U[i,2,c] * V[2,k,c] * W[k,i,c] = 1
        # With one column c=0: U[i,2,0]=1, V[2,k,0]=1, W[k,i,0]=1 for all i,k.
        # Then sum = 1*1*1 = 1. ✓
        # So rank 1 works for the sub-block!
        
        # Total rank: 28 (Strassen) + 1 (sub-block) = 29.
        
        if not _elapsed_ok():
            return None

        rank = 29
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        col = 0

        # Part 1: Strassen for 2x2x8 (28 terms)
        for block_idx in range(4):
            k0 = block_idx * 2
            for r in range(7):
                for li in range(2):
                    for la in range(2):
                        u_val = S_U[r][li * 2 + la]
                        if u_val != 0:
                            U[li * m + la][col] = u_val
                for la in range(2):
                    for lk in range(2):
                        v_val = S_V[r][la * 2 + lk]
                        if v_val != 0:
                            V[la * p + (k0 + lk)][col] = v_val
                for lk in range(2):
                    for li in range(2):
                        w_val = S_W[r][lk * 2 + li]
                        if w_val != 0:
                            W[(k0 + lk) * n + li][col] = w_val
                col += 1

        # Part 2: Rank-1 for 2x1x8 sub-block
        # U[i,2] = 1, V[2,k] = 1, W[k,i] = 1 for all i,k
        for i in range(2):
            U[i * m + 2][col] = 1
        for k in range(8):
            V[2 * p + k][col] = 1
        for k in range(8):
            for i in range(2):
                W[k * n + i][col] = 1
        col += 1

        candidate = {"U": U, "V": V, "W": W}
        if _validate(candidate):
            return candidate
        return None

    # Try the rank-29 candidate
    result = _search_subblock()
    if result is not None:
        return result

    # Fallback to rank 44
    return base

# EVOLVE-BLOCK-END
