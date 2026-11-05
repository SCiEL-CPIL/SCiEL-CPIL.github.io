"""Deterministic multi-Strassen rank-81 decomposition for 3x4*4x7 matrix multiplication."""
import time

DIMENSIONS = [3, 4, 7]
_DEADLINE = 140.0  # safety margin before 150s kill


def construct():
    _t0 = time.time()
    n, m, p = DIMENSIONS  # n=3, m=4, p=7

    # We will use 2 full Strassen blocks (7 terms each) and 1 partial Strassen block
    # that saves 1 term, plus schoolbook for the rest.
    # Total rank = 7 + 7 + 6 + schoolbook_terms
    # Schoolbook terms = total - 2*8 - 7 = 84 - 16 - 7 = 61
    # Total rank = 7 + 7 + 6 + 61 = 81

    rank = 81
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0

    def add_strassen_block_full(col_start, i0, j0, k0):
        """Add 7 Strassen rank-1 terms for a 2x2x2 block starting at (i0, j0, k0)."""
        i1 = i0 + 1
        j1 = j0 + 1
        k1 = k0 + 1

        # M1: (A[i0,j0]+A[i1,j1])*(B[j0,k0]+B[j1,k1]) -> C[i0,k0]+, C[i1,k1]+=
        U[i0 * m + j0][col_start] = 1
        U[i1 * m + j1][col_start] = 1
        V[j0 * p + k0][col_start] = 1
        V[j1 * p + k1][col_start] = 1
        W[k0 * n + i0][col_start] = 1
        W[k1 * n + i1][col_start] = 1

        # M2: (A[i1,j0]+A[i1,j1])*B[j0,k0] -> C[i1,k0]+, C[i1,k1]-
        U[i1 * m + j0][col_start + 1] = 1
        U[i1 * m + j1][col_start + 1] = 1
        V[j0 * p + k0][col_start + 1] = 1
        W[k0 * n + i1][col_start + 1] = 1
        W[k1 * n + i1][col_start + 1] = -1

        # M3: A[i0,j0]*(B[j0,k1]-B[j1,k1]) -> C[i0,k1]+, C[i1,k1]+=
        U[i0 * m + j0][col_start + 2] = 1
        V[j0 * p + k1][col_start + 2] = 1
        V[j1 * p + k1][col_start + 2] = -1
        W[k1 * n + i0][col_start + 2] = 1
        W[k1 * n + i1][col_start + 2] = 1

        # M4: A[i1,j1]*(B[j1,k0]-B[j0,k0]) -> C[i0,k0]+, C[i1,k0]+=
        U[i1 * m + j1][col_start + 3] = 1
        V[j1 * p + k0][col_start + 3] = 1
        V[j0 * p + k0][col_start + 3] = -1
        W[k0 * n + i0][col_start + 3] = 1
        W[k0 * n + i1][col_start + 3] = 1

        # M5: (A[i0,j0]+A[i0,j1])*B[j1,k1] -> C[i0,k0]-, C[i0,k1]+=
        U[i0 * m + j0][col_start + 4] = 1
        U[i0 * m + j1][col_start + 4] = 1
        V[j1 * p + k1][col_start + 4] = 1
        W[k0 * n + i0][col_start + 4] = -1
        W[k1 * n + i0][col_start + 4] = 1

        # M6: (A[i1,j0]-A[i0,j0])*(B[j0,k0]+B[j0,k1]) -> C[i1,k1]+=
        U[i1 * m + j0][col_start + 5] = 1
        U[i0 * m + j0][col_start + 5] = -1
        V[j0 * p + k0][col_start + 5] = 1
        V[j0 * p + k1][col_start + 5] = 1
        W[k1 * n + i1][col_start + 5] = 1

        # M7: (A[i0,j1]-A[i1,j1])*(B[j1,k0]+B[j1,k1]) -> C[i0,k0]+=
        U[i0 * m + j1][col_start + 6] = 1
        U[i1 * m + j1][col_start + 6] = -1
        V[j1 * p + k0][col_start + 6] = 1
        V[j1 * p + k1][col_start + 6] = 1
        W[k0 * n + i0][col_start + 6] = 1

        return col_start + 7

    def add_strassen_block_partial(col_start, i0, j0, k0):
        """
        Add 6 Strassen rank-1 terms for a 2x2x2 block, merging one term into schoolbook.
        We handle C[i0,k0] via schoolbook (4 terms: (i0,j0,k0),(i0,j1,k0),(i1,j0,k0),(i1,j1,k0))
        and use 6 Strassen terms for the remaining 4 C entries.
        
        Actually, let's think more carefully. In a 2x2x2 block, the 8 C entries are:
        C[i0,k0], C[i0,k1], C[i1,k0], C[i1,k1]
        
        If we use schoolbook for C[i0,k0], that's 4 rank terms.
        The remaining 4 C entries can be done with 6 Strassen-like terms? No, that's not right.
        
        Let me reconsider. The standard Strassen uses 7 terms for 8 C entries.
        If we use schoolbook for 1 C entry (4 terms) and Strassen for the other 3... 
        that doesn't work cleanly.
        
        Alternative: Use a 2x2x2 Strassen but merge one of the 7 terms with a schoolbook term
        that shares the same U and V pattern. This is hard to do cleanly.
        
        Let me try a different approach: use a 3x2x2 or 2x3x2 block structure.
        """
        pass

    # Block 1: i∈{0,1}, j∈{0,1}, k∈{0,1}
    col = add_strassen_block_full(col, 0, 0, 0)

    # Block 2: i∈{1,2}, j∈{2,3}, k∈{5,6}
    col = add_strassen_block_full(col, 1, 2, 5)

    # For the third block, let's try a different approach.
    # Consider the subproblem for i∈{0,1}, j∈{2,3}, k∈{0,1} (a 2x2x2 block)
    # and i∈{0,1}, j∈{2,3}, k∈{2,3} (another 2x2x2 block)
    # But we only have 2 full Strassen blocks.
    
    # Let's try: use a 2x2x2 Strassen for i∈{0,1}, j∈{2,3}, k∈{0,1}
    # That would be a third full Strassen block, giving rank = 21 + schoolbook
    # Schoolbook terms = 84 - 3*8 = 60
    # Total rank = 21 + 60 = 81
    # 
    # But wait, the blocks must have disjoint (i,j,k) index sets.
    # Block 1: i∈{0,1}, j∈{0,1}, k∈{0,1}
    # Block 2: i∈{1,2}, j∈{2,3}, k∈{5,6}
    # Block 3: i∈{0,1}, j∈{2,3}, k∈{0,1}
    #
    # Check disjointness:
    # Block 1 uses i∈{0,1}, j∈{0,1}, k∈{0,1}
    # Block 2 uses i∈{1,2}, j∈{2,3}, k∈{5,6}
    # Block 3 uses i∈{0,1}, j∈{2,3}, k∈{0,1}
    #
    # Block 1 and Block 3: i overlap at {0,1}, j disjoint ({0,1} vs {2,3}), k overlap at {0,1}
    # Since j is disjoint, the (i,j,k) tuples are disjoint. Good.
    # Block 2 and Block 3: i overlap at {1}, j overlap at {2,3}, k disjoint ({5,6} vs {0,1})
    # Since k is disjoint, the (i,j,k) tuples are disjoint. Good.
    # Block 1 and Block 2: already verified in the original.
    
    # Block 3: i∈{0,1}, j∈{2,3}, k∈{0,1}
    col = add_strassen_block_full(col, 0, 2, 0)

    # Schoolbook for all remaining (i,j,k)
    covered = set()
    for i0 in range(n):
        for j0 in range(m):
            for k0 in range(p):
                # Check if in Block 1
                if i0 in (0, 1) and j0 in (0, 1) and k0 in (0, 1):
                    covered.add((i0, j0, k0))
                    continue
                # Check if in Block 2
                if i0 in (1, 2) and j0 in (2, 3) and k0 in (5, 6):
                    covered.add((i0, j0, k0))
                    continue
                # Check if in Block 3
                if i0 in (0, 1) and j0 in (2, 3) and k0 in (0, 1):
                    covered.add((i0, j0, k0))
                    continue
                U[i0 * m + j0][col] = 1
                V[j0 * p + k0][col] = 1
                W[k0 * n + i0][col] = 1
                col += 1

    assert col == rank, f"Expected rank {rank}, got {col}"

    # Verify the decomposition
    for i in range(n):
        if time.time() - _t0 > _DEADLINE:
            break
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                total = 0
                for c in range(rank):
                    total += U[u_idx][c] * V[v_idx][c] * W[w_idx][c]
                if total != 1:
                    raise ValueError(f"Decomposition failed at ({i},{j},{k}): got {total}, expected 1")

    return {"U": U, "V": V, "W": W}
