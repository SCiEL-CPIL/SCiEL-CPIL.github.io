"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import combinations, product

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS  # 3, 5, 5
    
    def add_strassen_block(U, V, W, i0, j0, k0, col_offset):
        """Add a 2x2x2 Strassen decomposition starting at (i0, j0, k0)."""
        # Standard Strassen for 2x2 matrices:
        # M1 = (B11-B22)*(A21+A22)
        # M2 = (A11+A12)*B22
        # M3 = (A11-A22)*(B11+B12)
        # M4 = A11*(B11-B22)
        # M5 = (A12+A22)*(B21-B22)
        # M6 = (A11-A12)*(B11+B21)
        # M7 = (A21+A11)*(B12-B11)
        # C11 = M4+M1+M3+M5
        # C12 = M2+M3
        # C21 = M1+M6
        # C22 = M2+M4+M5+M7
        
        c = col_offset + 0
        # M1: (A21+A22) * (B11-B22)
        U[(i0+1)*m + j0][c] += 1
        U[(i0+1)*m + (j0+1)][c] += 1
        V[j0*p + k0][c] += 1
        V[(j0+1)*p + (k0+1)][c] += -1
        W[k0*n + (i0+1)][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1
        W[k0*n + (i0+1)][c] += 1  # C21 has M1
        W[(k0+1)*n + (i0+1)][c] += 1 # C22 has M1? No, C22 = M2+M4+M5+M7. C21 = M1+M6. C11 = M4+M1+M3+M5. C12 = M2+M3.
        # Let's re-derive carefully.
        # A = [[A11, A12], [A21, A22]]
        # B = [[B11, B12], [B21, B22]]
        # C = AB
        # C11 = A11B11 + A12B21
        # C12 = A11B12 + A12B22
        # C21 = A21B11 + A22B21
        # C22 = A21B12 + A22B22
        
        # M1 = (A11+A12)*(B21-B22) -> contributes to C21? No.
        # Let's use standard Strassen variables:
        # M1 = (A11 + A12) * (B21 - B22)
        # M2 = (A21 + A22) * (B11 + B12)
        # M3 = (A11 - A22) * (B11 + B22)
        # M4 = (A12 + A22) * (B21 - B11)
        # M5 = (A11 + A12) * (B22 - B12)
        # M6 = (A21 - A11) * (B11 + B21)
        # M7 = (A11 + A22) * (B12 - B22)
        # C11 = M1 + M4 - M5 + M7
        # C12 = M2 - M4 - M6 - M7
        # C21 = M1 + M3 + M5 - M6
        # C22 = M2 - M3 + M6 + M7
        
        # Let's just hardcode the 7 terms based on indices.
        # A indices: (i, j). B indices: (j, k). C indices: (i, k).
        # i0, j0, k0 are top-left.
        
        # Reset previous incorrect additions? No, I should just write the correct ones.
        # Since I already added some in the draft above, I will restart the function body.
        
    def add_strassen_block(U, V, W, i0, j0, k0, col_offset):
        # M1 = (A11+A12)*(B21-B22)
        # A11=(i0,j0), A12=(i0,j0+1)
        # B21=(j0+1,k0), B22=(j0+1,k0+1)
        c = col_offset + 0
        U[i0*m + j0][c] += 1
        U[i0*m + (j0+1)][c] += 1
        V[(j0+1)*p + k0][c] += 1
        V[(j0+1)*p + (k0+1)][c] += -1
        W[k0*n + i0][c] += 1   # C11
        W[(k0+1)*n + i0][c] += -1 # C12? No, C12 is (i0, k0+1). C11 is (i0, k0).
        # Wait, C11 = M1 + ... so W[k0*n + i0] += 1
        # C21 = M1 + ... so W[k0*n + (i0+1)] += 1
        W[k0*n + (i0+1)][c] += 1
        
        # M2 = (A21+A22)*(B11+B12)
        # A21=(i0+1, j0), A22=(i0+1, j0+1)
        # B11=(j0, k0), B12=(j0, k0+1)
        c = col_offset + 1
        U[(i0+1)*m + j0][c] += 1
        U[(i0+1)*m + (j0+1)][c] += 1
        V[j0*p + k0][c] += 1
        V[j0*p + (k0+1)][c] += 1
        W[(k0+1)*n + i0][c] += 1 # C12
        W[(k0+1)*n + (i0+1)][c] += 1 # C22
        
        # M3 = (A11-A22)*(B11+B22)
        # A11=(i0,j0), A22=(i0+1, j0+1)
        # B11=(j0,k0), B22=(j0+1, k0+1)
        c = col_offset + 2
        U[i0*m + j0][c] += 1
        U[(i0+1)*m + (j0+1)][c] += -1
        V[j0*p + k0][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        W[k0*n + i0][c] += 1 # C11
        W[k0*n + (i0+1)][c] += 1 # C21
        
        # M4 = (A12+A22)*(B21-B11)
        # A12=(i0,j0+1), A22=(i0+1, j0+1)
        # B21=(j0+1, k0), B11=(j0, k0)
        c = col_offset + 3
        U[i0*m + (j0+1)][c] += 1
        U[(i0+1)*m + (j0+1)][c] += 1
        V[(j0+1)*p + k0][c] += 1
        V[j0*p + k0][c] += -1
        W[k0*n + i0][c] += 1 # C11
        W[(k0+1)*n + i0][c] += -1 # C12
        
        # M5 = (A11+A12)*(B22-B12)
        # A11=(i0,j0), A12=(i0,j0+1)
        # B22=(j0+1, k0+1), B12=(j0, k0+1)
        c = col_offset + 4
        U[i0*m + j0][c] += 1
        U[i0*m + (j0+1)][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        V[j0*p + (k0+1)][c] += -1
        W[k0*n + i0][c] += -1 # C11
        W[k0*n + (i0+1)][c] += 1 # C21
        
        # M6 = (A21-A11)*(B11+B21)
        # A21=(i0+1, j0), A11=(i0, j0)
        # B11=(j0, k0), B21=(j0+1, k0)
        c = col_offset + 5
        U[(i0+1)*m + j0][c] += 1
        U[i0*m + j0][c] += -1
        V[j0*p + k0][c] += 1
        V[(j0+1)*p + k0][c] += 1
        W[(k0+1)*n + i0][c] += -1 # C12
        W[k0*n + (i0+1)][c] += -1 # C21
        
        # M7 = (A11+A22)*(B12-B22)
        # A11=(i0, j0), A22=(i0+1, j0+1)
        # B12=(j0, k0+1), B22=(j0+1, k0+1)
        c = col_offset + 6
        U[i0*m + j0][c] += 1
        U[(i0+1)*m + (j0+1)][c] += 1
        V[j0*p + (k0+1)][c] += 1
        V[(j0+1)*p + (k0+1)][c] += -1
        W[k0*n + i0][c] += 1 # C11
        W[(k0+1)*n + i0][c] += -1 # C12
        W[(k0+1)*n + (i0+1)][c] += 1 # C22
        
    def add_laderman_block(U, V, W, i0, j0, k0, col_offset):
        # Laderman's algorithm for 3x3 matrix multiplication
        # M1 = (A11 + A22) * (B11 + B22)
        # M2 = (A21 + A22 + A31 + A32 + A33) * (B11)
        # M3 = (A11 - A12) * (B11 + B12 + B22)
        # M4 = (A12 + A22 + A32 + A33) * (B12 + B21 + B31)
        # M5 = (A11 - A13) * (B11 + B13 + B22)
        # M6 = (A13 + A23 + A33 - A11) * (B13 + B22 + B31)
        # M7 = (A21 + A31 + A33 - A11) * (B21 + B31 + B12)
        
        c = col_offset + 0
        # M1
        U[i0*m + j0][c] += 1
        U[(i0+1)*m + (j0+1)][c] += 1
        V[j0*p + k0][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        W[k0*n + i0][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1
        W[k0*n + (i0+1)][c] += -1
        W[(k0+1)*n + i0][c] += -1

        c = col_offset + 1
        # M2
        U[(i0+1)*m + j0][c] += 1
        U[(i0+1)*m + (j0+1)][c] += 1
        U[(i0+2)*m + j0][c] += 1
        U[(i0+2)*m + (j0+1)][c] += 1
        U[(i0+2)*m + (j0+2)][c] += 1
        V[j0*p + k0][c] += 1
        W[k0*n + (i0+1)][c] += 1
        W[k0*n + (i0+2)][c] += 1
        W[(k0+1)*n + (i0+1)][c] += -1
        W[(k0+1)*n + (i0+2)][c] += -1
        W[(k0+2)*n + (i0+1)][c] += -1
        W[(k0+2)*n + (i0+2)][c] += -1

        c = col_offset + 2
        # M3
        U[i0*m + j0][c] += 1
        U[i0*m + (j0+1)][c] += -1
        V[j0*p + k0][c] += 1
        V[j0*p + (k0+1)][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        W[k0*n + i0][c] += 1
        W[(k0+1)*n + i0][c] += -1
        W[(k0+1)*n + (i0+1)][c] += 1
        W[(k0+2)*n + i0][c] += -1
        W[(k0+2)*n + (i0+1)][c] += 1

        c = col_offset + 3
        # M4
        U[i0*m + (j0+1)][c] += 1
        U[(i0+1)*m + (j0+1)][c] += 1
        U[(i0+2)*m + (j0+1)][c] += 1
        U[(i0+2)*m + (j0+2)][c] += 1
        V[(j0+1)*p + k0][c] += 1
        V[j0*p + (k0+1)][c] += 1
        V[(j0+2)*p + k0][c] += 1
        W[k0*n + i0][c] += -1
        W[(k0+1)*n + i0][c] += -1
        W[(k0+1)*n + (i0+1)][c] += -1
        W[(k0+2)*n + i0][c] += -1
        W[(k0+2)*n + (i0+1)][c] += -1
        W[(k0+2)*n + (i0+2)][c] += -1

        c = col_offset + 4
        # M5
        U[i0*m + j0][c] += 1
        U[i0*m + (j0+2)][c] += -1
        V[j0*p + k0][c] += 1
        V[j0*p + (k0+2)][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        W[k0*n + i0][c] += 1
        W[(k0+2)*n + i0][c] += -1
        W[(k0+2)*n + (i0+1)][c] += 1
        W[(k0+1)*n + i0][c] += -1
        W[(k0+1)*n + (i0+1)][c] += 1

        c = col_offset + 5
        # M6
        U[i0*m + (j0+2)][c] += 1
        U[(i0+1)*m + (j0+2)][c] += 1
        U[(i0+2)*m + (j0+2)][c] += 1
        U[i0*m + j0][c] += -1
        V[j0*p + (k0+2)][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        V[(j0+2)*p + k0][c] += 1
        W[(k0+2)*n + i0][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1
        W[(k0+2)*n + (i0+2)][c] += 1
        W[(k0+1)*n + i0][c] += -1
        W[(k0+1)*n + (i0+2)][c] += -1
        W[(k0+2)*n + (i0+1)][c] += -1

        c = col_offset + 6
        # M7
        U[(i0+1)*m + j0][c] += 1
        U[(i0+2)*m + j0][c] += 1
        U[(i0+2)*m + (j0+2)][c] += 1
        U[i0*m + j0][c] += -1
        V[(j0+1)*p + k0][c] += 1
        V[(j0+2)*p + k0][c] += 1
        V[j0*p + (k0+1)][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1
        W[(k0+2)*n + (i0+1)][c] += 1
        W[(k0+2)*n + (i0+2)][c] += 1
        W[(k0+1)*n + i0][c] += -1
        W[(k0+1)*n + (i0+2)][c] += -1
        W[(k0+2)*n + i0][c] += -1

    def build_decomposition(blocks):
        """Build decomposition from block placements + schoolbook for rest."""
        covered = [[[False]*p for _ in range(m)] for _ in range(n)]
        total_rank = 0
        
        for block in blocks:
            btype, i0, j0, k0 = block
            if btype == 'strassen':
                for di in range(2):
                    for dj in range(2):
                        for dk in range(2):
                            covered[i0+di][j0+dj][k0+dk] = True
                total_rank += 7
            elif btype == 'larderman':
                for di in range(3):
                    for dj in range(3):
                        for dk in range(3):
                            covered[i0+di][j0+dj][k0+dk] = True
                total_rank += 7
        
        remaining = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if not covered[i][j][k]:
                        remaining += 1
        
        total_rank += remaining
        
        U = [[0] * total_rank for _ in range(n * m)]
        V = [[0] * total_rank for _ in range(m * p)]
        W = [[0] * total_rank for _ in range(p * n)]
        
        col = 0
        for block in blocks:
            btype, i0, j0, k0 = block
            if btype == 'strassen':
                add_strassen_block(U, V, W, i0, j0, k0, col)
                col += 7
            elif btype == 'larderman':
                add_laderman_block(U, V, W, i0, j0, k0, col)
                col += 7
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if not covered[i][j][k]:
                        U[i*m+j][col] += 1
                        V[j*p+k][col] += 1
                        W[k*n+i][col] += 1
                        col += 1
        
        return U, V, W, total_rank
    
    def is_valid_placement(blocks):
        """Check if all blocks are within bounds and non-overlapping."""
        covered = [[[False]*p for _ in range(m)] for _ in range(n)]
        for block in blocks:
            btype, i0, j0, k0 = block
            if btype == 'strassen':
                size = 2
            elif btype == 'larderman':
                size = 3
            else:
                return False
            
            if i0 < 0 or i0+size > n or j0 < 0 or j0+size > m or k0 < 0 or k0+size > p:
                return False
            for di in range(size):
                for dj in range(size):
                    for dk in range(size):
                        if covered[i0+di][j0+dj][k0+dk]:
                            return False
                        covered[i0+di][j0+dj][k0+dk] = True
        return True
    
    def verify(U, V, W):
        """Verify the decomposition is correct."""
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        target = np.zeros((n*m, m*p, p*n), dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    target[i*m+j, j*p+k, k*n+i] = 1
        T_decomp = np.einsum('ar,br,cr->abc', U_np, V_np, W_np)
        return np.allclose(T_decomp, target, atol=1e-10)
    
    best_rank = n * m * p
    best_result = None
    
    # 1. Try 4 Strassen blocks (known good, rank 71)
    base_config = [
        ('strassen', 0, 0, 0),
        ('strassen', 0, 2, 0),
        ('strassen', 0, 0, 2),
        ('strassen', 0, 2, 2),
    ]
    if is_valid_placement(base_config):
        U, V, W, rank = build_decomposition(base_config)
        if verify(U, V, W):
            best_rank = rank
            best_result = {"U": U, "V": V, "W": W}
    
    # 2. Try 1 Laderman + 3 Strassen (target rank 52)
    # Laderman at (0,0,0) covers i in 0..2, j in 0..2, k in 0..2
    # Remaining space for Strassen:
    # i=0,1 (since n=3, Strassen needs 2x2x2)
    # We can place Strassen blocks in:
    # - j in 0..1, k in 3..4 (i=0..1) -> S1: (0,0,3)
    # - j in 3..4, k in 0..1 (i=0..1) -> S2: (0,3,0)
    # - j in 3..4, k in 2..3 (i=0..1) -> S3: (0,3,2)
    # Check if these overlap with Laderman or each other.
    # Laderman: i 0-2, j 0-2, k 0-2
    # S1: i 0-1, j 0-1, k 3-4. Disjoint from L (k disjoint).
    # S2: i 0-1, j 3-4, k 0-1. Disjoint from L (j disjoint). Disjoint from S1 (j disjoint).
    # S3: i 0-1, j 3-4, k 2-3. Disjoint from L (j disjoint). Disjoint from S1 (j disjoint). Disjoint from S2 (k disjoint).
    
    config_52 = [
        ('larderman', 0, 0, 0),
        ('strassen', 0, 0, 3),
        ('strassen', 0, 3, 0),
        ('strassen', 0, 3, 2),
    ]
    if is_valid_placement(config_52):
        U, V, W, rank = build_decomposition(config_52)
        if verify(U, V, W):
            if rank < best_rank:
                best_rank = rank
                best_result = {"U": U, "V": V, "W": W}
    
    # 3. Try other Laderman placements + Strassen
    # Laderman can be at i=0 only (since n=3).
    # j can be 0 or 1 (since m=5, 3x3 needs j, j+2 <= 4).
    # k can be 0, 1, or 2 (since p=5, 3x3 needs k, k+2 <= 4).
    
    laderman_positions = [(0, j0, k0) for j0 in range(m-2) for k0 in range(p-2)]
    strassen_positions = [(i0, j0, k0) for i0 in range(n-1) for j0 in range(m-1) for k0 in range(p-1)]
    
    for l_pos in laderman_positions:
        if time.time() > deadline - 20: break
        i0_l, j0_l, k0_l = l_pos
        # Determine available Strassen positions
        valid_s = []
        for s_pos in strassen_positions:
            i0_s, j0_s, k0_s = s_pos
            # Check non-overlap with Laderman
            # Laderman: i in [i0_l, i0_l+2], j in [j0_l, j0_l+2], k in [k0_l, k0_l+2]
            # Strassen: i in [i0_s, i0_s+1], j in [j0_s, j0_s+1], k in [k0_s, k0_s+1]
            
            # Overlap if intervals intersect
            if not (i0_s+1 < i0_l or i0_s > i0_l+2):
                if not (j0_s+1 < j0_l or j0_s > j0_l+2):
                    if not (k0_s+1 < k0_l or k0_s > k0_l+2):
                        continue # Overlap
            valid_s.append(s_pos)
        
        # Try adding up to 4 Strassen blocks from valid_s
        # Greedy or exhaustive? Let's try exhaustive for small sizes.
        for size in range(4, -1, -1):
            if time.time() > deadline - 10: break
            if len(valid_s) < size: continue
            for combo in combinations(valid_s, size):
                # Check non-overlap between Strassen blocks
                valid = True
                covered_s = set()
                for s_pos in combo:
                    i0_s, j0_s, k0_s = s_pos
                    for di in range(2):
                        for dj in range(2):
                            for dk in range(2):
                                key = (i0_s+di, j0_s+dj, k0_s+dk)
                                if key in covered_s:
                                    valid = False
                                    break
                                covered_s.add(key)
                    if not valid: break
                if not valid: continue
                
                blocks = [('larderman', i0_l, j0_l, k0_l)] + [('strassen', *s) for s in combo]
                rank_est = 7 + size*7 + (n*m*p - 27 - size*8)
                if rank_est >= best_rank: continue
                
                U, V, W, rank = build_decomposition(blocks)
                if verify(U, V, W):
                    if rank < best_rank:
                        best_rank = rank
                        best_result = {"U": U, "V": V, "W": W}
    
    if best_result is None:
        # Fallback to schoolbook
        rank_school = n * m * p
        U_s = [[0] * rank_school for _ in range(n * m)]
        V_s = [[0] * rank_school for _ in range(m * p)]
        W_s = [[0] * rank_school for _ in range(p * n)]
        c2 = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_s[i*m+j][c2] = 1
                    V_s[j*p+k][c2] = 1
                    W_s[k*n+i][c2] = 1
                    c2 += 1
        return {"U": U_s, "V": V_s, "W": W_s}
    
    return best_result
# EVOLVE-BLOCK-END
