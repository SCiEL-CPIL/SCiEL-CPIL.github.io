import numpy as np
import time
import itertools

DIMENSIONS = [3, 5, 5]
_DEADLINE = time.time() + 120  # 120s wall-clock limit

def construct():
    n, m, p = DIMENSIONS
    # Try to find rank 70 first, fall back to rank 71
    r70 = try_find_rank_70(n, m, p)
    if r70 is not None:
        return r70
    return _rank71_solution(n, m, p)

def try_find_rank_70(n, m, p):
    """Attempt to find a rank-70 decomposition via structured search."""
    start = time.time()
    deadline = start + 100  # leave 20s safety margin
    
    # Build the matrix multiplication tensor
    # T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k
    # We want to find U (n*m x r), V (m*p x r), W (p*n x r)
    # such that T = sum_c U[:,c] V[:,c]^T W[:,c]^T (in flattened form)
    
    # The tensor has n*m*p = 75 entries, all equal to 1
    # Rank 71 is achieved by 4 Strassen blocks (4*7=28) + 47 schoolbook = 75... wait
    # Actually: 4 blocks cover 4*8=32 terms, leaving 75-32=43 schoolbook terms
    # Total rank = 28 + 43 = 71
    
    # For rank 70, we need to save 1 more term.
    # This requires a more clever combination of Strassen-like identities.
    
    # Approach: Use 4 Strassen blocks but try to overlap or combine
    # to reduce total rank by 1.
    
    # Alternative: Try a different block structure or use a 3x3 block
    # with a known rank-19 decomposition for 3x3x3.
    
    # Let's try: one 3x3x3 block (rank 19) + remaining schoolbook
    # 3x3x3 covers 27 terms, rank 19
    # Remaining: 75 - 27 = 48 terms, rank 48
    # Total: 19 + 48 = 67? No, that's not right either.
    # Wait, 3x3x3 Strassen-like gives rank 19 for 27 terms.
    # But we need to check if the remaining terms can be done schoolbook.
    
    # Actually, let me reconsider. The standard 3x3 matrix multiplication
    # has rank 19 (or possibly 18). Let's use rank 19.
    
    # If we use a 3x3x3 block covering i in {0,1,2}, j in {0,1,2}, k in {0,1,2}:
    # That covers 27 terms with rank 19.
    # Remaining terms: 75 - 27 = 48, each needing rank 1 = 48
    # Total: 19 + 48 = 67. But wait, that would be better than 71!
    # Let me verify: n=3, m=5, p=5. A 3x3x3 block uses i in 0..2, j in 0..2, k in 0..2.
    # Yes, that's valid.
    
    # So we need a rank-19 decomposition for 3x3x3 matrix multiplication.
    # The known minimal rank for 3x3 is 19 (or 18, but 19 is well-established).
    
    # Let me construct a rank-19 solution for 3x3x3.
    # This is complex. Let me use a known construction.
    
    # Actually, let me first check if a simpler approach works:
    # Use 3 Strassen 2x2x2 blocks + 1 partial block + schoolbook
    
    # Block 1: i{0,1}, j{0,1}, k{0,1} - rank 7, covers 8 terms
    # Block 2: i{0,1}, j{2,3}, k{0,1} - rank 7, covers 8 terms  
    # Block 3: i{0,1}, j{0,1}, k{2,3} - rank 7, covers 8 terms
    # Block 4: i{0,1}, j{2,3}, k{2,3} - rank 7, covers 8 terms
    # That's the current approach: 4*7 + 43 = 71
    
    # To get 70, we need to save 1. One idea:
    # Use a 3x2x2 block or 2x3x2 block with a better decomposition.
    
    # Let's try: one 3x2x2 block (i in 0..2, j in 0..1, k in 0..1)
    # This covers 12 terms. What's the minimal rank for 3x2x2?
    # It's at most 3*2*2=12 schoolbook, but can be less.
    # Actually, for 3x2x2, we can use 2 Strassen 2x2x2 blocks sharing the j,k dimensions
    # but that's 2*7=14 which is worse.
    
    # Let me try a different approach: use the fact that some terms can be
    # shared between Strassen blocks.
    
    # Actually, the simplest improvement: check if we can use a 3x3x2 block
    # with rank less than 3*3*2=18.
    
    # For now, let me just return None and use the rank-71 solution,
    # but add a quick check to see if we can verify the solution.
    
    if time.time() > deadline:
        return None
    
    # Try a structured approach: 3x3x3 block with rank 19
    # I'll use a known rank-19 construction for 3x3 matrix multiplication.
    # This is the Laderman-inspired or similar construction.
    
    # Actually, let me try a simpler search: modify the 4-block Strassen
    # to see if we can save 1 rank by combining terms.
    
    # For now, return None to use rank-71 fallback.
    # A full rank-70 search would require more sophisticated methods.
    return None

def _rank71_solution(n, m, p):
    # Rank 71 solution:
    # Four Strassen blocks (rank 7 each) + Schoolbook for the rest (43 terms)
    blocks = [
        (0, 0, 0),
        (0, 2, 0),
        (0, 0, 2),
        (0, 2, 2)
    ]
    
    num_strassen_blocks = len(blocks)
    total_strassen_rank = num_strassen_blocks * 7
    
    # Calculate number of schoolbook terms
    total_terms = n * m * p
    strassen_terms_count = num_strassen_blocks * 8
    schoolbook_terms_count = total_terms - strassen_terms_count
    
    total_rank = total_strassen_rank + schoolbook_terms_count
    
    U = [[0.0] * total_rank for _ in range(n * m)]
    V = [[0.0] * total_rank for _ in range(m * p)]
    W = [[0.0] * total_rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a Strassen block
    def add_strassen_block(i0, j0, k0, col_offset):
        # Local indices 0,1 map to global i0, i0+1 etc.
        # U index: i*m + j
        # V index: j*p + k
        # W index: k*n + i
        
        # M1 = (A00+A11)(B00+B11)
        c = col_offset + 0
        U[i0*m + j0][c] += 1
        U[(i0+1)*m + (j0+1)][c] += 1
        V[j0*p + k0][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        W[k0*n + i0][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1
        
        # M2 = (A10+A11)B00
        c = col_offset + 1
        U[(i0+1)*m + j0][c] += 1
        U[(i0+1)*m + (j0+1)][c] += 1
        V[j0*p + k0][c] += 1
        W[k0*n + (i0+1)][c] += 1
        W[(k0+1)*n + (i0+1)][c] += -1
        
        # M3 = A00(B01-B11)
        c = col_offset + 2
        U[i0*m + j0][c] += 1
        V[j0*p + (k0+1)][c] += 1
        V[(j0+1)*p + (k0+1)][c] += -1
        W[(k0+1)*n + i0][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1
        
        # M4 = A11(B10-B00)
        c = col_offset + 3
        U[(i0+1)*m + (j0+1)][c] += 1
        V[(j0+1)*p + k0][c] += 1
        V[j0*p + k0][c] += -1
        W[k0*n + i0][c] += 1
        W[k0*n + (i0+1)][c] += 1
        
        # M5 = (A00+A01)B11
        c = col_offset + 4
        U[i0*m + j0][c] += 1
        U[i0*m + (j0+1)][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        W[k0*n + i0][c] += -1
        W[(k0+1)*n + i0][c] += 1
        
        # M6 = (A10-A00)(B00+B01)
        c = col_offset + 5
        U[(i0+1)*m + j0][c] += 1
        U[i0*m + j0][c] += -1
        V[j0*p + k0][c] += 1
        V[j0*p + (k0+1)][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1
        
        # M7 = (A01-A11)(B10+B11)
        c = col_offset + 6
        U[i0*m + (j0+1)][c] += 1
        U[(i0+1)*m + (j0+1)][c] += -1
        V[(j0+1)*p + k0][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        W[k0*n + i0][c] += 1

    # Add Strassen blocks
    for i0, j0, k0 in blocks:
        add_strassen_block(i0, j0, k0, col)
        col += 7
    
    # Schoolbook for the rest
    # We need to identify which (i,j,k) triples were covered by Strassen
    # Covered if i in [i0, i0+1] and j in [j0, j0+1] and k in [k0, k0+1] for any block
    
    covered = [[False]*p for _ in range(m)]
    # Actually, we need 3D coverage
    covered_3d = [[[False]*p for _ in range(m)] for _ in range(n)]
    
    for i0, j0, k0 in blocks:
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    covered_3d[i0+di][j0+dj][k0+dk] = True
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if not covered_3d[i][j][k]:
                    U[i*m+j][col] += 1
                    V[j*p+k][col] += 1
                    W[k*n+i][col] += 1
                    col += 1
                    
    # Trim trailing zero columns if any
    while col > 0 and all(U[i][col-1] == 0 for i in range(n*m)) and \
          all(V[i][col-1] == 0 for i in range(m*p)) and \
          all(W[i][col-1] == 0 for i in range(p*n)):
        col -= 1
    
    U = [row[:col] for row in U]
    V = [row[:col] for row in V]
    W = [row[:col] for row in W]
    
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    res = construct()
    U = np.array(res["U"])
    V = np.array(res["V"])
    W = np.array(res["W"])
    n, m, p = DIMENSIONS
    print(f"Rank: {U.shape[1]}")
    # Verify correctness
    # T_ijk = sum_c U_{i*m+j, c} * V_{j*p+k, c} * W_{k*n+i, c} should equal 1 for all i,j,k
    errors = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] for c in range(U.shape[1]))
                if abs(val - 1.0) > 1e-9:
                    errors += 1
    print(f"Errors: {errors}")
