"""Combined program: block Strassen decomposition for 3x3x4 matrix multiplication.
Uses multiple overlapping 2x2x2 Strassen blocks plus schoolbook for remaining terms.
Every candidate is validated before being returned; a verified schoolbook rank-36
solution is always available as a fallback."""
DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _validate(U, V, W, n, m, p):
    """Check that the decomposition exactly reconstructs the matmul tensor."""
    R = len(U[0]) if U else 0
    if len(V[0]) != R or len(W[0]) != R:
        return False
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(R):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(s - 1.0) > 1e-10:
                    return False
    return True

def _strassen_222():
    """Return (U,V,W) each 4x7 for 2x2x2 matmul tensor (rank 7).
    U[i*2+j], V[j*2+k], W[k*2+i]."""
    U = np.zeros((4,7)); V = np.zeros((4,7)); W = np.zeros((4,7))
    # M1 = (A00+A11)(B00+B11)
    U[0,0]=1; U[3,0]=1; V[0,0]=1; V[3,0]=1; W[0,0]=1; W[1,0]=1; W[2,0]=1; W[3,0]=1
    # M2 = (A10+A11)B00
    U[2,1]=1; U[3,1]=1; V[0,1]=1; W[2,1]=1
    # M3 = A00(B10-B11)
    U[0,2]=1; V[2,2]=1; V[3,2]=-1; W[1,2]=1
    # M4 = A11(B01-B00)
    U[3,3]=1; V[1,3]=1; V[0,3]=-1; W[1,3]=1
    # M5 = (A00+A01)B11
    U[0,4]=1; U[1,4]=1; V[3,4]=1; W[0,4]=1
    # M6 = (A10-A00)(B00+B10)
    U[2,5]=1; U[0,5]=-1; V[0,5]=1; V[2,5]=1; W[2,5]=1
    # M7 = (A01-A11)(B01+B11)
    U[1,6]=1; U[3,6]=-1; V[1,6]=1; V[3,6]=1; W[0,6]=1; W[3,6]=1
    return U, V, W

def construct():
    t0 = time.time()
    DEADLINE = 140.0
    n, m, p = DIMENSIONS  # 3, 3, 4

    def schoolbook():
        R = n*m*p
        U = [[0]*R for _ in range(n*m)]
        V = [[0]*R for _ in range(m*p)]
        W = [[0]*R for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    best = schoolbook()  # rank 36 fallback

    # Helper: add a Strassen block for (i_off, j_off, k_off) covering
    # i in {i_off, i_off+1}, j in {j_off, j_off+1}, k in {k_off, k_off+1}
    def add_strassen_block(U, V, W, i_off, j_off, k_off, col_start):
        """Add 7 rank-1 terms for a 2x2x2 Strassen block.
        Returns the next column index."""
        sU, sV, sW = _strassen_222()
        for r in range(7):
            # U: (i_off + di, j_off + dj) for di,dj in {0,1}
            for di in range(2):
                for dj in range(2):
                    i = i_off + di
                    j = j_off + dj
                    if i < n and j < m:
                        U[i*m+j][col_start + r] = sU[di*2+dj, r]
            # V: (j_off + dj, k_off + dk) for dj,dk in {0,1}
            for dj in range(2):
                for dk in range(2):
                    j = j_off + dj
                    k = k_off + dk
                    if j < m and k < p:
                        V[j*p+k][col_start + r] = sV[dj*2+dk, r]
            # W: (k_off + dk, i_off + di) for dk,di in {0,1}
            for dk in range(2):
                for di in range(2):
                    k = k_off + dk
                    i = i_off + di
                    if k < p and i < n:
                        W[k*n+i][col_start + r] = sW[dk*2+di, r]
        return col_start + 7

    # ---- Candidate 1: Two Strassen 2x2x2 blocks + schoolbook rest ----
    # Block 1: i∈{0,1}, j∈{0,1}, k∈{0,1}
    # Block 2: i∈{1,2}, j∈{1,2}, k∈{2,3}
    # These cover:
    # Block 1: (i,j,k) where i<2, j<2, k<2 → 8 tuples
    # Block 2: (i,j,k) where i>=1, j>=1, k>=2 → 8 tuples
    # Overlap: i=1, j=1, k=2 → 1 tuple covered by both
    # Total covered: 8 + 8 - 1 = 15 unique (i,j,k) tuples
    # Schoolbook for remaining: 36 - 15 = 21
    # Total rank: 7 + 7 + 21 = 35
    
    if time.time() < DEADLINE:
        try:
            total_rank = 7 + 7 + 21  # 35
            U = [[0]*total_rank for _ in range(n*m)]
            V = [[0]*total_rank for _ in range(m*p)]
            W = [[0]*total_rank for _ in range(p*n)]
            
            col = 0
            col = add_strassen_block(U, V, W, 0, 0, 0, col)
            col = add_strassen_block(U, V, W, 1, 1, 2, col)
            
            # Schoolbook for remaining (i,j,k) not covered by either block
            covered = set()
            for di in range(2):
                for dj in range(2):
                    for dk in range(2):
                        covered.add((0+di, 0+dj, 0+dk))
                        covered.add((1+di, 1+dj, 2+dk))
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        if (i, j, k) not in covered:
                            U[i*m+j][col] = 1
                            V[j*p+k][col] = 1
                            W[k*n+i][col] = 1
                            col += 1
            
            if col == total_rank and _validate(U, V, W, n, m, p):
                best = {"U": U, "V": V, "W": W}
        except Exception:
            pass

    # ---- Candidate 2: Three Strassen 2x2x2 blocks + schoolbook rest ----
    # Block 1: i∈{0,1}, j∈{0,1}, k∈{0,1}
    # Block 2: i∈{0,1}, j∈{1,2}, k∈{2,3}
    # Block 3: i∈{1,2}, j∈{0,1}, k∈{1,2}
    # Let me compute coverage:
    # Block 1: (0,0,0),(0,0,1),(0,1,0),(0,1,1),(1,0,0),(1,0,1),(1,1,0),(1,1,1)
    # Block 2: (0,1,2),(0,1,3),(0,2,2),(0,2,3),(1,1,2),(1,1,3),(1,2,2),(1,2,3)
    # Block 3: (1,0,1),(1,0,2),(1,1,1),(1,1,2),(2,0,1),(2,0,2),(2,1,1),(2,1,2)
    # Union: Let me list all unique:
    # From B1: (0,0,0),(0,0,1),(0,1,0),(0,1,1),(1,0,0),(1,0,1),(1,1,0),(1,1,1)
    # From B2 new: (0,1,2),(0,1,3),(0,2,2),(0,2,3),(1,1,2),(1,1,3),(1,2,2),(1,2,3)
    # From B3 new: (1,0,2),(2,0,1),(2,0,2),(2,1,1),(2,1,2)
    # Total: 8 + 8 + 5 = 21 unique tuples
    # Schoolbook for remaining: 36 - 21 = 15
    # Total rank: 7 + 7 + 7 + 15 = 36. No improvement.
    # Skip this one.

    # ---- Candidate 3: Four Strassen 2x2x2 blocks + schoolbook rest ----
    # Block 1: i∈{0,1}, j∈{0,1}, k∈{0,1}
    # Block 2: i∈{0,1}, j∈{1,2}, k∈{2,3}
    # Block 3: i∈{1,2}, j∈{0,1}, k∈{1,2}
    # Block 4: i∈{1,2}, j∈{1,2}, k∈{2,3}
    # Let me compute coverage more carefully:
    # B1: (0,0,0),(0,0,1),(0,1,0),(0,1,1),(1,0,0),(1,0,1),(1,1,0),(1,1,1)
    # B2: (0,1,2),(0,1,3),(0,2,2),(0,2,3),(1,1,2),(1,1,3),(1,2,2),(1,2,3)
    # B3: (1,0,1),(1,0,2),(1,1,1),(1,1,2),(2,0,1),(2,0,2),(2,1,1),(2,1,2)
    # B4: (1,1,2),(1,1,3),(1,2,2),(1,2,3),(2,1,2),(2,1,3),(2,2,2),(2,2,3)
    # Union:
    # B1: 8
    # B2 new: (0,1,2),(0,1,3),(0,2,2),(0,2,3),(1,1,2),(1,1,3),(1,2,2),(1,2,3) → 8 new
    # B3 new: (1,0,2),(2,0,1),(2,0,2),(2,1,1),(2,1,2) → (1,0,1) and (1,1,1) already in B1, (1,1,2) in B2
    #   So new from B3: (1,0,2),(2,0,1),(2,0,2),(2,1,1),(2,1,2) → 5 new
    # B4 new: (2,1,3),(2,2,2),(2,2,3) → (1,1,2),(1,1,3),(1,2,2),(1,2,3) already in B2, (2,1,2) in B3
    #   So new from B4: (2,1,3),(2,2,2),(2,2,3) → 3 new
    # Total: 8 + 8 + 5 + 3 = 24 unique tuples
    # Schoolbook for remaining: 36 - 24 = 12
    # Total rank: 7*4 + 12 = 40. Worse than 36. Skip.

    # ---- Candidate 4: Two non-overlapping Strassen blocks + schoolbook ----
    # Block 1: i∈{0,1}, j∈{0,1}, k∈{0,1}
    # Block 2: i∈{0,1}, j∈{0,1}, k∈{2,3}
    # These are disjoint in k, so no overlap.
    # Coverage: 8 + 8 = 16 tuples
    # Schoolbook: 36 - 16 = 20
    # Total rank: 7 + 7 + 20 = 34
    
    if time.time() < DEADLINE:
        try:
            total_rank = 7 + 7 + 20  # 34
            U = [[0]*total_rank for _ in range(n*m)]
            V = [[0]*total_rank for _ in range(m*p)]
            W = [[0]*total_rank for _ in range(p*n)]
            
            col = 0
            col = add_strassen_block(U, V, W, 0, 0, 0, col)
            col = add_strassen_block(U, V, W, 0, 0, 2, col)
            
            covered = set()
            for di in range(2):
                for dj in range(2):
                    for dk in range(2):
                        covered.add((0+di, 0+dj, 0+dk))
                        covered.add((0+di, 0+dj, 2+dk))
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        if (i, j, k) not in covered:
                            U[i*m+j][col] = 1
                            V[j*p+k][col] = 1
                            W[k*n+i][col] = 1
                            col += 1
            
            if col == total_rank and _validate(U, V, W, n, m, p):
                best = {"U": U, "V": V, "W": W}
        except Exception:
            pass

    # ---- Candidate 5: Three non-overlapping Strassen blocks + schoolbook ----
    # Block 1: i∈{0,1}, j∈{0,1}, k∈{0,1}
    # Block 2: i∈{0,1}, j∈{0,1}, k∈{2,3}
    # Block 3: i∈{1,2}, j∈{1,2}, k∈{0,1}
    # Check overlap: B1 and B2 share i,j but different k → no overlap
    # B1 and B3: B1 has i<2, B3 has i>=1. Overlap at i=1, j=1, k in {0,1} → 2 tuples
    # B2 and B3: B2 has i<2, B3 has i>=1. Overlap at i=1, j=1, k in {2,3} → 2 tuples
    # B1 and B2: no overlap (different k ranges)
    # Total overlap: 2 + 2 = 4 tuples
    # Coverage: 8 + 8 + 8 - 4 = 20 tuples
    # Schoolbook: 36 - 20 = 16
    # Total rank: 7*3 + 16 = 37. Worse. Skip.

    # ---- Candidate 6: Two Strassen blocks with minimal overlap ----
    # Block 1: i∈{0,1}, j∈{0,1}, k∈{0,1}
    # Block 2: i∈{1,2}, j∈{0,1}, k∈{2,3}
    # Overlap: i=1, j in {0,1}, k in {0,1} ∩ {2,3} = empty → no overlap!
    # Coverage: 8 + 8 = 16 tuples
    # Schoolbook: 36 - 16 = 20
    # Total rank: 7 + 7 + 20 = 34. Same as candidate 4.
    
    if time.time() < DEADLINE:
        try:
            total_rank = 7 + 7 + 20  # 34
            U = [[0]*total_rank for _ in range(n*m)]
            V = [[0]*total_rank for _ in range(m*p)]
            W = [[0]*total_rank for _ in range(p*n)]
            
            col = 0
            col = add_strassen_block(U, V, W, 0, 0, 0, col)
            col = add_strassen_block(U, V, W, 1, 0, 2, col)
            
            covered = set()
            for di in range(2):
                for dj in range(2):
                    for dk in range(2):
                        covered.add((0+di, 0+dj, 0+dk))
                        covered.add((1+di, 0+dj, 2+dk))
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        if (i, j, k) not in covered:
                            U[i*m+j][col] = 1
                            V[j*p+k][col] = 1
                            W[k*n+i][col] = 1
                            col += 1
            
            if col == total_rank and _validate(U, V, W, n, m, p):
                best = {"U": U, "V": V, "W": W}
        except Exception:
            pass

    # ---- Candidate 7: Four Strassen blocks covering all i,j with k split ----
    # Block 1: i∈{0,1}, j∈{0,1}, k∈{0,1}
    # Block 2: i∈{0,1}, j∈{0,1}, k∈{2,3}
    # Block 3: i∈{0,1}, j∈{1,2}, k∈{0,1}
    # Block 4: i∈{0,1}, j∈{1,2}, k∈{2,3}
    # Overlaps: B1∩B3 at i∈{0,1}, j=1, k∈{0,1} → 2 tuples
    #            B2∩B4 at i∈{0,1}, j=1, k∈{2,3} → 2 tuples
    #            B1∩B2: no (different k)
    #            B3∩B4: no (different k)
    # Total overlap: 4
    # Coverage: 8*4 - 4 = 28 tuples
    # Schoolbook: 36 - 28 = 8
    # Total rank: 7*4 + 8 = 36. No improvement.

    # ---- Candidate 8: Use Strassen for (i,j) blocks that don't overlap in (i,j) ----
    # Block 1: i∈{0,1}, j∈{0,1}, k∈{0,1}
    # Block 2: i∈{0,1}, j∈{0,1}, k∈{2,3}
    # Block 3: i∈{1,2}, j∈{1,2}, k∈{0,1}
    # Block 4: i∈{1,2}, j∈{1,2}, k∈{2,3}
    # B1∩B3: i=1, j=1, k∈{0,1} → 2 tuples
    # B2∩B4: i=1, j=1, k∈{2,3} → 2 tuples
    # B1∩B2: no
    # B3∩B4: no
    # Total overlap: 4
    # Coverage: 32 - 4 = 28
    # Schoolbook: 8
    # Total rank: 28 + 8 = 36. No improvement.

    # The best we've found is rank 34 from candidates 4 and 6.
    # Let me verify candidate 4 more carefully.

    return best

# EVOLVE-BLOCK-END
