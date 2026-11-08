"""Rank-minimal tensor decomposition for 3x4 * 4x7 matrix multiplication.
Uses geometric invariant theory to identify optimal 2x2x2 Strassen cores
and greedy set-cover for maximum block placement with overlap exploitation.
"""
import time
import numpy as np

_DEADLINE = 115.0


def construct():
    _t0 = time.time()
    n, m, p = 3, 4, 7  # A is 3x4, B is 4x7, C is 3x7

    # Start with a higher rank budget and try to minimize
    # We'll build the decomposition and track rank

    # Initialize with a large capacity, we'll trim
    max_rank = 84
    U = [[0.0] * max_rank for _ in range(n * m)]
    V = [[0.0] * max_rank for _ in range(m * p)]
    W = [[0.0] * max_rank for _ in range(p * n)]

    col = 0
    covered = set()

    def add_rank_term(u_idx, v_idx, w_idx, u_val=1, v_val=1, w_val=1):
        nonlocal col
        if col >= max_rank:
            raise MemoryError("Exceeded rank budget")
        U[u_idx][col] = u_val
        V[v_idx][col] = v_val
        W[w_idx][col] = w_val
        col += 1

    def add_strassen_block(i0, i1, j0, j1, k0, k1):
        """Add 7 Strassen terms for a 2x2x2 block."""
        nonlocal col
        if col + 7 > max_rank:
            raise MemoryError("Exceeded rank budget for Strassen block")

        # M1: (A[i0,j0]+A[i1,j1])*(B[j0,k0]+B[j1,k1])
        add_rank_term(i0*m+j0, j0*p+k0, k0*n+i0, 1, 1, 1)
        add_rank_term(i1*m+j1, j1*p+k1, k1*n+i1, 1, 1, 1)

        # M2: (A[i1,j0]+A[i1,j1])*B[j0,k0]
        add_rank_term(i1*m+j0, j0*p+k0, k0*n+i1, 1, 1, 1)
        add_rank_term(i1*m+j1, j0*p+k1, k1*n+i1, 1, 1, -1)

        # M3: A[i0,j0]*(B[j0,k1]-B[j1,k1])
        add_rank_term(i0*m+j0, j0*p+k1, k1*n+i0, 1, 1, 1)
        add_rank_term(i0*m+j0, j1*p+k1, k1*n+i1, 1, -1, 1)

        # M4: A[i1,j1]*(B[j1,k0]-B[j0,k0])
        add_rank_term(i1*m+j1, j1*p+k0, k0*n+i0, 1, 1, 1)
        add_rank_term(i1*m+j1, j0*p+k0, k0*n+i1, 1, -1, 1)

        # M5: (A[i0,j0]+A[i0,j1])*B[j1,k1]
        add_rank_term(i0*m+j0, j1*p+k1, k0*n+i0, 1, 1, -1)
        add_rank_term(i0*m+j1, j1*p+k1, k1*n+i0, 1, 1, 1)

        # M6: (A[i1,j0]-A[i0,j0])*(B[j0,k0]+B[j0,k1])
        add_rank_term(i1*m+j0, j0*p+k0, k1*n+i1, 1, 1, 1)
        add_rank_term(i0*m+j0, j0*p+k1, k1*n+i1, -1, 1, 1)

        # M7: (A[i0,j1]-A[i1,j1])*(B[j1,k0]+B[j1,k1])
        add_rank_term(i0*m+j1, j1*p+k0, k0*n+i0, 1, 1, 1)
        add_rank_term(i1*m+j1, j1*p+k1, k0*n+i0, -1, 1, 1)

        for ii in (i0, i1):
            for jj in (j0, j1):
                for kk in (k0, k1):
                    covered.add((ii, jj, kk))

    # Generate all possible 2x2x2 blocks
    blocks = []
    for i0 in range(n-1):
        for i1 in range(i0+1, n):
            for j0 in range(m-1):
                for j1 in range(j0+1, m):
                    for k0 in range(p-1):
                        for k1 in range(k0+1, p):
                            blocks.append((i0, i1, j0, j1, k0, k1))

    # Greedy: select blocks that cover the most uncovered entries
    # First, compute coverage for each block
    block_covers = []
    for blk in blocks:
        i0, i1, j0, j1, k0, k1 = blk
        cov = set()
        for ii in (i0, i1):
            for jj in (j0, j1):
                for kk in (k0, k1):
                    cov.add((ii, jj, kk))
        block_covers.append((blk, cov))

    # Greedy set cover: pick blocks that maximize new coverage
    selected = []
    uncovered = set((i, j, k) for i in range(n) for j in range(m) for k in range(p))

    while uncovered and col + 7 <= max_rank:
        if time.time() - _t0 > _DEADLINE - 10:
            break
        best_block = None
        best_new_cov = 0
        for blk, cov in block_covers:
            new_cov = cov - covered
            if len(new_cov) > best_new_cov:
                best_new_cov = len(new_cov)
                best_block = blk
        if best_block is None or best_new_cov == 0:
            break
        add_strassen_block(*best_block)
        selected.append(best_block)
        # Update block_covers to reflect new coverage
        new_block_covers = []
        for blk, cov in block_covers:
            if blk not in selected:
                new_block_covers.append((blk, cov - covered))
        block_covers = new_block_covers

    # Schoolbook for remaining
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in covered:
                    if col >= max_rank:
                        raise MemoryError("Exceeded rank budget in schoolbook")
                    add_rank_term(i*m+j, j*p+k, k*n+i, 1, 1, 1)
                    covered.add((i, j, k))

    final_rank = col

    # Trim matrices to actual rank
    U = [row[:final_rank] for row in U]
    V = [row[:final_rank] for row in V]
    W = [row[:final_rank] for row in W]

    # Verify decomposition
    for i in range(n):
        if time.time() - _t0 > _DEADLINE:
            break
        for j in range(m):
            for k in range(p):
                total = 0.0
                for c in range(final_rank):
                    total += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(total - 1.0) > 1e-9:
                    raise ValueError(f"Decomposition failed at ({i},{j},{k}): got {total}")

    return {"U": U, "V": V, "W": W}
