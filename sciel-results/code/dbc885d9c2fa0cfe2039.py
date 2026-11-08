import numpy as np
import time

DIMENSIONS = [3, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    
    # Strategy: Use 4 Strassen blocks of size 2x2x2 for the core
    # Blocks:
    # 1. i0=0, j0=0, k0=0 -> covers i:0-1, j:0-1, k:0-1
    # 2. i0=0, j0=2, k0=0 -> covers i:0-1, j:2-3, k:0-1
    # 3. i0=0, j0=0, k0=2 -> covers i:0-1, j:0-1, k:2-3
    # 4. i0=0, j0=2, k0=2 -> covers i:0-1, j:2-3, k:2-3
    # 
    # This covers all terms where i in {0,1}, j in {0,1,2,3}, k in {0,1,2,3}.
    # Total covered terms: 2 * 4 * 4 = 32.
    # Total terms: 3 * 5 * 5 = 75.
    # Uncovered terms: 75 - 32 = 43.
    # Strassen rank: 4 * 7 = 28.
    # Total rank: 28 + 43 = 71.
    
    # Can we do better?
    # Let's try to cover more with Strassen or use smaller blocks.
    # Or maybe 5 Strassen blocks?
    # If we add a block at (0, 4, 0), it covers i:0-1, j:4-5 (out of bounds), so no.
    # j goes 0..4. j0 can be 0,1,2,3.
    # If j0=3, covers j:3-4.
    # Block (0, 3, 0): covers i:0-1, j:3-4, k:0-1.
    # Block (0, 3, 2): covers i:0-1, j:3-4, k:2-3.
    # These overlap with the previous blocks on j=3.
    # Previous blocks covered j:0-1 and j:2-3.
    # So j=3 is covered. j=4 is NOT covered by previous blocks.
    # So adding blocks for j=3,4 would cover j=4.
    
    # Let's refine the blocks to minimize uncovered terms.
    # We want to cover as much of the 3x5x5 space as possible with 2x2x2 blocks.
    # i dimension: size 3. Blocks can start at i=0 or i=1.
    # If we use i=0, covers i:0,1. Leaves i=2.
    # If we use i=1, covers i:1,2. Leaves i=0.
    # Best to use i=0 to cover 2 out of 3.
    
    # j dimension: size 5.
    # k dimension: size 5.
    
    # Current setup: 4 blocks covering i:0-1, j:0-3, k:0-3.
    # Uncovered:
    # i=0,1: j=4 (all k), k=4 (all j<4).
    #   j=4, k=0..4: 5 terms.
    #   j=0..3, k=4: 4 terms.
    #   Total for i=0,1: 9 terms each. 18 total.
    # i=2: All j, k uncovered. 5*5 = 25 terms.
    # Total uncovered: 18 + 25 = 43.
    # Rank: 28 + 43 = 71.
    
    # Can we cover i=2?
    # Add blocks with i0=1.
    # Block (1, 0, 0): covers i:1-2, j:0-1, k:0-1.
    # This overlaps with block (0,0,0) on i=1, j:0-1, k:0-1.
    # Overlap means we can't just add it. The tensor decomposition must be exact.
    # If we use overlapping blocks, we need to handle the overlap carefully or it won't work directly as simple blocks.
    # Simple block decomposition requires disjoint support in the tensor space (i,j,k).
    # So blocks must be disjoint in (i,j,k).
    
    # Disjoint blocks:
    # i=0,1 vs i=1,2 overlap at i=1. So we can't mix i0=0 and i0=1 blocks.
    # We must choose either i0=0 or i0=1 for all blocks, or partition the i-axis.
    # If we use i0=0, we cover i=0,1. i=2 is left for schoolbook.
    # If we use i0=1, we cover i=1,2. i=0 is left for schoolbook.
    
    # Let's stick to i0=0.
    # Can we cover j=4 or k=4 with Strassen?
    # Block size is 2.
    # j=4 requires j0=3 (covers 3,4).
    # k=4 requires k0=3 (covers 3,4).
    
    # Current blocks:
    # (0,0,0), (0,2,0), (0,0,2), (0,2,2).
    # Covers j: 0,1 and 2,3. So j=4 uncovered.
    # Covers k: 0,1 and 2,3. So k=4 uncovered.
    
    # Add blocks to cover j=4 and k=4?
    # Block (0, 3, 0): covers j:3,4. Overlaps with (0,2,0) on j=3. Not disjoint.
    # Block (0, 4, 0): invalid.
    
    # So with disjoint 2x2x2 blocks, we can only cover j:0-1, 2-3. j=4 must be schoolbook.
    # Same for k=4.
    
    # What if we use different block sizes?
    # 1x1x1 is schoolbook (rank 1).
    # 2x2x2 is Strassen (rank 7).
    # 2x1x1? Rank?
    # Multiplying 2x1 by 1x1.
    # A is 2x1, B is 1x1. C is 2x1.
    # C0 = A0 B0. C1 = A1 B0.
    # Rank 2. Schoolbook is 2. No gain.
    
    # 1x2x1?
    # A 1x2, B 2x1. C 1x1.
    # C = A0 B0 + A1 B1.
    # Rank 2. Schoolbook 2. No gain.
    
    # 1x1x2?
    # A 1x1, B 1x2. C 1x2.
    # C0 = A B0. C1 = A B1.
    # Rank 2. Schoolbook 2. No gain.
    
    # So 2x2x2 is the only small block with gain (7 vs 8).
    
    # Are there larger blocks?
    # 2x2x3?
    # Rank of 2x2x3 multiplication?
    # A 2x2, B 2x3.
    # C 2x3.
    # 6 terms.
    # Strassen-like?
    # If we split B into B_left (2x2) and B_right (2x1).
    # C_left = A B_left (rank 7).
    # C_right = A B_right (rank 4? 2*1*2=4 terms. Rank 4).
    # Total rank 11.
    # Schoolbook: 2*2*3 = 12.
    # Gain of 1.
    
    # 2x3x2?
    # A 2x3, B 3x2.
    # Split A into A_left (2x2), A_right (2x1).
    # C_left = A_left B (rank 7? B is 2x2. 2*2*2=8 terms. Rank 7).
    # C_right = A_right B (rank 4? 2*1*2=4 terms. Rank 4).
    # Total rank 11.
    # Schoolbook: 2*3*2 = 12.
    # Gain of 1.
    
    # 3x2x2?
    # A 3x2, B 2x2.
    # Split A into A_top (2x2), A_bottom (1x2).
    # C_top = A_top B (rank 7).
    # C_bottom = A_bottom B (rank 4).
    # Total rank 11.
    # Schoolbook: 3*2*2 = 12.
    # Gain of 1.
    
    # So 2x2x3, 2x3x2, 3x2x2 all have rank 11 vs schoolbook 12.
    
    # Can we use these to cover the edge cases?
    # We have i=2 uncovered (size 1x5x5).
    # j=4 uncovered for i=0,1 (size 2x1x5).
    # k=4 uncovered for i=0,1 (size 2x5x1).
    
    # Let's re-evaluate the partition.
    # Instead of 4 blocks of 2x2x2, maybe use:
    # 1 block of 2x2x2 (covers i:0-1, j:0-1, k:0-1).
    # Remaining i:0-1, j:2-4, k:0-4.
    # Remaining i:2, j:0-4, k:0-4.
    
    # This is getting complex. The current 71 is likely close to optimal for this simple block strategy.
    
    # Let's check if 70 is possible.
    # 4 Strassen blocks = 28 rank.
    # 43 schoolbook.
    # Total 71.
    
    # If we can save 1 rank in schoolbook? No, schoolbook terms are independent.
    # If we can save 1 rank in Strassen? No, Strassen rank is 7.
    
    # What if we use 5 Strassen blocks?
    # We need disjoint 2x2x2 blocks.
    # i=0,1.
    # j=0,1,2,3. (2 blocks).
    # k=0,1,2,3. (2 blocks).
    # Total 4 blocks.
    # Can we fit a 5th?
    # j=4 is left. k=4 is left.
    # A block needs 2x2 area.
    # j=4 only has width 1.
    # k=4 only has width 1.
    # So no 2x2 block can fit in j=4 or k=4 alone.
    # Unless we overlap? No, must be disjoint.
    
    # So max 4 Strassen blocks for i=0,1.
    
    # What if we use i=1,2 for some blocks?
    # Blocks with i0=1 cover i=1,2.
    # Blocks with i0=0 cover i=0,1.
    # Disjointness:
    # If we use a block at (0,0,0) [i:0-1], we cannot use a block at (1,0,0) [i:1-2] because they overlap at i=1.
    # So we must choose a partition of i.
    # Partition 1: i=0,1 (blocks i0=0). i=2 (schoolbook).
    # Partition 2: i=1,2 (blocks i0=1). i=0 (schoolbook).
    # Partition 3: Mix? No, overlap.
    
    # So we are stuck with 4 blocks max.
    
    # Is there a better way to cover the j=4, k=4 edges?
    # The edges are 2x1x5 and 2x5x1.
    # We can use 2x1x1 blocks (rank 2, schoolbook 2). No gain.
    # 1x2x1 (rank 2). No gain.
    
    # So 71 seems to be the limit for this simple decomposition.
    
    # Let's double check the rank of 2x2x2.
    # It is 7.
    # Schoolbook is 8.
    # Gain 1 per block.
    # 4 blocks -> gain 4.
    # Base schoolbook 75.
    # 75 - 4 = 71.
    
    # Can we do better with non-rectangular blocks?
    # Or a different algorithm?
    # The task asks for "rank-minimal".
    # 71 might not be minimal.
    # But finding a lower rank requires more complex methods (e.g. laser method, etc).
    # Given the constraints, I will output the 71 solution.
    
    blocks = [
        (0, 0, 0),
        (0, 2, 0),
        (0, 0, 2),
        (0, 2, 2)
    ]
    
    num_strassen_blocks = len(blocks)
    total_strassen_rank = num_strassen_blocks * 7
    
    covered_3d = [[[False]*p for _ in range(m)] for _ in range(n)]
    for i0, j0, k0 in blocks:
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    if i0+di < n and j0+dj < m and k0+dk < p:
                        covered_3d[i0+di][j0+dj][k0+dk] = True
    
    schoolbook_terms_count = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if not covered_3d[i][j][k]:
                    schoolbook_terms_count += 1
    
    total_rank = total_strassen_rank + schoolbook_terms_count
    
    U = [[0.0] * total_rank for _ in range(n * m)]
    V = [[0.0] * total_rank for _ in range(m * p)]
    W = [[0.0] * total_rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen_block(i0, j0, k0, col_offset):
        c = col_offset + 0
        U[i0*m + j0][c] += 1
        U[(i0+1)*m + (j0+1)][c] += 1
        V[j0*p + k0][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        W[k0*n + i0][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1
        
        c = col_offset + 1
        U[(i0+1)*m + j0][c] += 1
        U[(i0+1)*m + (j0+1)][c] += 1
        V[j0*p + k0][c] += 1
        W[k0*n + (i0+1)][c] += 1
        W[(k0+1)*n + (i0+1)][c] += -1
        
        c = col_offset + 2
        U[i0*m + j0][c] += 1
        V[j0*p + (k0+1)][c] += 1
        V[(j0+1)*p + (k0+1)][c] += -1
        W[(k0+1)*n + i0][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1
        
        c = col_offset + 3
        U[(i0+1)*m + (j0+1)][c] += 1
        V[(j0+1)*p + k0][c] += 1
        V[j0*p + k0][c] += -1
        W[k0*n + i0][c] += 1
        W[k0*n + (i0+1)][c] += 1
        
        c = col_offset + 4
        U[i0*m + j0][c] += 1
        U[i0*m + (j0+1)][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        W[k0*n + i0][c] += -1
        W[(k0+1)*n + i0][c] += 1
        
        c = col_offset + 5
        U[(i0+1)*m + j0][c] += 1
        U[i0*m + j0][c] += -1
        V[j0*p + k0][c] += 1
        V[j0*p + (k0+1)][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1
        
        c = col_offset + 6
        U[i0*m + (j0+1)][c] += 1
        U[(i0+1)*m + (j0+1)][c] += -1
        V[(j0+1)*p + k0][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        W[k0*n + i0][c] += 1

    for i0, j0, k0 in blocks:
        add_strassen_block(i0, j0, k0, col)
        col += 7
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if not covered_3d[i][j][k]:
                    U[i*m+j][col] += 1
                    V[j*p+k][col] += 1
                    W[k*n+i][col] += 1
                    col += 1
                    
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    res = construct()
    U = np.array(res["U"])
    V = np.array(res["V"])
    W = np.array(res["W"])
    n, m, p = DIMENSIONS
    print(f"Rank: {U.shape[1]}")
    # Verification
    A = np.random.rand(n, m)
    B = np.random.rand(m, p)
    C_true = A @ B
    C_decomp = np.zeros((n, p))
    for r in range(U.shape[1]):
        a_vec = np.array(U[:, r]).reshape(n, m)
        b_vec = np.array(V[:, r]).reshape(m, p)
        w_vec = np.array(W[:, r]).reshape(p, n)
        C_decomp += a_vec @ b_vec * w_vec.T
    if np.allclose(C_true, C_decomp):
        print("Decomposition is CORRECT")
    else:
        print("Decomposition is WRONG")
        print(C_true)
        print(C_decomp)
