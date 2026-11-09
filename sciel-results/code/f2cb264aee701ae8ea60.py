"""Matrix multiplication 4,5,5 tensor decomposition."""
import time
from fractions import Fraction

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115

    n, m, p = DIMENSIONS  # 4, 5, 5

    # Strassen's 7-term decomposition for 2x2x2
    strassen_terms = [
        ([1, 0, 1, 0], [1, 0, 0, 1], [1, 0, 0, 1]),
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]),
        ([1, 0, 0, 0], [0, 1, -1, 0], [0, 1, 0, 1]),
        ([0, 0, 0, 1], [0, 0, 1, -1], [1, 0, 1, 0]),
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 0, 0, 1]),
        ([0, 0, 1, -1], [1, 1, 0, 0], [0, 0, 0, 1]),
        ([0, 1, -1, 0], [0, 0, 1, 1], [1, 0, 0, 0]),
    ]

    def build_decomposition(blocks):
        """Build decomposition given a set of non-overlapping 2x2x2 blocks."""
        # Determine which (i,j,k) are covered by blocks
        covered = set()
        for (i0, j0, k0) in blocks:
            for di in range(2):
                for dj in range(2):
                    for dk in range(2):
                        covered.add((i0 + di, j0 + dj, k0 + dk))

        # Count schoolbook terms (not in blocks)
        num_schoolbook = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if (i, j, k) not in covered:
                        num_schoolbook += 1

        num_strassen = len(blocks) * 7
        rank = num_schoolbook + num_strassen

        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]

        col = 0
        # Schoolbook terms for uncovered (i,j,k)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if (i, j, k) not in covered:
                        U[i * m + j][col] = 1
                        V[j * p + k][col] = 1
                        W[k * n + i][col] = 1
                        col += 1

        # Strassen terms for each block
        for (i0, j0, k0) in blocks:
            for (u_c, v_c, w_c) in strassen_terms:
                a_idx = [(i0, j0), (i0, j0 + 1), (i0 + 1, j0), (i0 + 1, j0 + 1)]
                b_idx = [(j0, k0), (j0, k0 + 1), (j0 + 1, k0), (j0 + 1, k0 + 1)]
                c_idx = [(k0, i0), (k0, i0 + 1), (k0 + 1, i0), (k0 + 1, i0 + 1)]
                for la, cf in enumerate(u_c):
                    if cf != 0:
                        U[a_idx[la][0] * m + a_idx[la][1]][col] += cf
                for lb, cf in enumerate(v_c):
                    if cf != 0:
                        V[b_idx[lb][0] * p + b_idx[lb][1]][col] += cf
                for lc, cf in enumerate(w_c):
                    if cf != 0:
                        W[c_idx[lc][0] * n + c_idx[lc][1]][col] += cf
                col += 1

        assert col == rank, f"Column mismatch: {col} != {rank}"
        return {"U": U, "V": V, "W": W}

    def verify(U, V, W):
        """Verify the decomposition exactly."""
        rank = len(U[0]) if U else 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = Fraction(0)
                    for c in range(rank):
                        s += Fraction(U[i * m + j][c]) * Fraction(V[j * p + k][c]) * Fraction(W[k * n + i][c])
                    if s != 1:
                        return False
        return True

    # Find all possible non-overlapping 2x2x2 blocks
    # i0 in {0,2}, j0 in {0,2,3} (but j0+1 < 5 so j0 in {0,1,2,3}), k0 in {0,1,2,3}
    # But for non-overlap, we need blocks to not share any (i,j,k)
    # Simple approach: enumerate all valid block positions and find max independent set
    
    # All possible block positions
    all_blocks = []
    for i0 in range(0, n - 1, 2):
        for j0 in range(0, m - 1):
            for k0 in range(0, p - 1):
                all_blocks.append((i0, j0, k0))

    # Check which pairs of blocks overlap
    def blocks_overlap(b1, b2):
        i0a, j0a, k0a = b1
        i0b, j0b, k0b = b2
        # Overlap if there exists (i,j,k) in both
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    i, j, k = i0a + di, j0a + dj, k0a + dk
                    if (i0b <= i <= i0b + 1) and (j0b <= j <= j0b + 1) and (k0b <= k <= k0b + 1):
                        return True
        return False

    # Greedy: pick blocks that don't overlap
    # Try to maximize number of blocks
    # For n=4, m=5, p=5: i0 can be 0 or 2 (2 choices)
    # j0 can be 0,1,2,3 (4 choices) but blocks at j0 and j0+1 overlap
    # So for m=5, we can have at most 2 non-overlapping blocks in j dimension: e.g., j0=0 and j0=2, or j0=1 and j0=3
    # Similarly for p=5: k0=0 and k0=2, or k0=1 and k0=3
    
    # Maximum independent set: for each (i0, j0, k0), pick non-overlapping ones
    # i0=0 and i0=2 don't overlap (since i range is 0-1 and 2-3)
    # j0=0 (covers 0-1) and j0=2 (covers 2-3) don't overlap; j0=1 (covers 1-2) and j0=3 (covers 3-4) don't overlap
    # But j0=0 and j0=1 overlap at j=1
    
    # Best strategy: 
    # i: use both i0=0 and i0=2 (2 blocks in i)
    # j: use j0=0 and j0=2 (or j0=1 and j0=3) -> 2 blocks in j
    # k: use k0=0 and k0=2 (or k0=1 and k0=3) -> 2 blocks in k
    # Total: 2*2*2 = 8 blocks, saving 8 rank -> rank 92
    
    # Let's verify: blocks at (0,0,0), (0,0,2), (0,2,0), (0,2,2), (2,0,0), (2,0,2), (2,2,0), (2,2,2)
    # Check (0,0,0) covers i=0,1 j=0,1 k=0,1
    # (0,0,2) covers i=0,1 j=0,1 k=2,3
    # (0,2,0) covers i=0,1 j=2,3 k=0,1
    # (0,2,2) covers i=0,1 j=2,3 k=2,3
    # (2,0,0) covers i=2,3 j=0,1 k=0,1
    # (2,0,2) covers i=2,3 j=0,1 k=2,3
    # (2,2,0) covers i=2,3 j=2,3 k=0,1
    # (2,2,2) covers i=2,3 j=2,3 k=2,3
    # All disjoint! And they cover i=0-3, j=0-3, k=0-3
    # Uncovered: j=4 (all i,k) and k=4 (i, j=0-3)
    # j=4: 4*5=20 positions, minus those already covered by k... 
    # Actually uncovered (i,j,k): j=4 (20) + k=4,j<4 (4*4=16) - (j=4,k=4: 4) = 20+16-4=32
    # Wait: total 100, covered = 8*8=64, uncovered=36
    # Let me recount: j=4: i=0-3, k=0-4 -> 20. k=4,j=0-3: i=0-3,j=0-3 -> 16. But (j=4,k=4) counted twice: 4.
    # So uncovered = 20+16-4 = 32. Plus... wait 64+32=96≠100. Let me recheck.
    # Covered positions: for each block, 8 positions. 8 blocks * 8 = 64.
    # Total positions: 4*5*5=100. Uncovered: 36.
    # j=4: (i,4,k) for i=0..3,k=0..4: 20
    # k=4: (i,j,4) for i=0..3,j=0..4: 20
    # Both j=4 AND k=4: (i,4,4) for i=0..3: 4
    # Uncovered = 20 + 20 - 4 = 36. Yes!
    
    blocks_optimal = [(0,0,0), (0,0,2), (0,2,0), (0,2,2), (2,0,0), (2,0,2), (2,2,0), (2,2,2)]
    
    best_result = None
    best_rank = n * m * p + 1

    # Try the optimal 8-block decomposition
    try:
        candidate = build_decomposition(blocks_optimal)
        if verify(candidate["U"], candidate["V"], candidate["W"]):
            best_result = candidate
            best_rank = len(candidate["U"][0])
    except Exception:
        pass

    # If that fails, try fewer blocks (greedy)
    if best_result is None:
        # Try 4 blocks: (0,0,0),(0,2,0),(2,0,0),(2,2,0)
        blocks_4 = [(0,0,0), (0,2,0), (2,0,0), (2,2,0)]
        try:
            candidate = build_decomposition(blocks_4)
            if verify(candidate["U"], candidate["V"], candidate["W"]):
                best_result = candidate
                best_rank = len(candidate["U"][0])
        except Exception:
            pass

    # Fallback to schoolbook
    if best_result is None:
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        best_result = {"U": U, "V": V, "W": W}

    if time.time() > deadline:
        return best_result

    return best_result
# EVOLVE-BLOCK-END
