import numpy as np
import time
from itertools import product, combinations

DIMENSIONS = [3, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    time_limit = 135  # Leave 15s margin for safety
    
    # Base rank-71 solution: 4 Strassen blocks (28 rank) + 43 schoolbook terms
    # Strassen blocks at positions (0,0,0), (0,2,0), (0,0,2), (0,2,2)
    # Each covers a 2x2x2 subcube
    # Total terms: 3*5*5 = 75, Strassen covers 4*8=32, schoolbook covers 43
    # Total rank = 4*7 + 43 = 71
    
    # Try to find a rank-70 solution by:
    # 1. Using different block placements
    # 2. Merging some schoolbook terms into Strassen-like structures
    # 3. Using half-integer coefficients to merge terms
    
    # First, build the base rank-71 solution
    def build_base_solution():
        blocks = [
            (0, 0, 0),
            (0, 2, 0),
            (0, 0, 2),
            (0, 2, 2)
        ]
        
        num_strassen_blocks = len(blocks)
        total_strassen_rank = num_strassen_blocks * 7
        total_terms = n * m * p
        strassen_terms_count = num_strassen_blocks * 8
        schoolbook_terms_count = total_terms - strassen_terms_count
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
        
        return U, V, W
    
    # Try to find rank-70 by exploring different block placements
    # A 2x2x2 block at (i0,j0,k0) covers 8 terms with rank 7 (saves 1)
    # We need to cover all 75 terms with blocks + singletons
    # Each block saves 1 rank, so we need 5 blocks to get rank 70
    # But blocks must be non-overlapping (or we need to handle overlaps)
    
    # Actually, let's think differently. The base has 4 blocks covering 32 terms
    # with rank 28, plus 43 singletons = rank 71.
    # To get rank 70, we need to save 1 more rank.
    
    # Option 1: Use 5 blocks instead of 4. But 5 blocks cover 40 terms with rank 35,
    # leaving 35 singletons. Total rank = 35 + 35 = 70. But we need non-overlapping
    # 2x2x2 blocks in a 3x5x5 grid.
    
    # Let's find all possible non-overlapping sets of 5 2x2x2 blocks
    # in a 3x5x5 grid.
    
    # A 2x2x2 block at (i0,j0,k0) requires i0+1 < 3, j0+1 < 5, k0+1 < 5
    # So i0 in {0,1}, j0 in {0,1,2,3}, k0 in {0,1,2,3}
    
    all_blocks = []
    for i0 in range(n-1):
        for j0 in range(m-1):
            for k0 in range(p-1):
                all_blocks.append((i0, j0, k0))
    
    def blocks_overlap(b1, b2):
        i1, j1, k1 = b1
        i2, j2, k2 = b2
        # Check if the 2x2x2 cubes overlap
        if abs(i1 - i2) < 2 and abs(j1 - j2) < 2 and abs(k1 - k2) < 2:
            return True
        return False
    
    def find_non_overlapping_blocks(target_count):
        """Find sets of non-overlapping 2x2x2 blocks."""
        results = []
        
        def backtrack(start_idx, current_set):
            if len(current_set) == target_count:
                results.append(list(current_set))
                return
            
            if time.time() - start_time > time_limit - 30:
                return
            
            for i in range(start_idx, len(all_blocks)):
                b = all_blocks[i]
                # Check if b overlaps with any in current_set
                overlap = False
                for cb in current_set:
                    if blocks_overlap(b, cb):
                        overlap = True
                        break
                if not overlap:
                    current_set.append(b)
                    backtrack(i + 1, current_set)
                    current_set.pop()
        
        backtrack(0, [])
        return results
    
    # Try to find 5 non-overlapping blocks
    five_blocks = find_non_overlapping_blocks(5)
    
    if len(five_blocks) > 0:
        # Use the first set of 5 blocks
        blocks = five_blocks[0]
        
        total_strassen_rank = len(blocks) * 7
        total_terms = n * m * p
        strassen_terms_count = len(blocks) * 8
        schoolbook_terms_count = total_terms - strassen_terms_count
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
        
        return {"U": U, "V": V, "W": W}
    
    # If no 5 non-overlapping blocks found, try overlapping blocks with careful handling
    # Or fall back to the base rank-71 solution
    U, V, W = build_base_solution()
    return {"U": U, "V": V, "W": W}
