"""
Task: Rank-minimal exact tensor decomposition for (2,3,10) matrix multiplication.

Strategy: Split m=3 into j∈{0,1} and j=2.
- For j∈{0,1}: use (2,2,10) sub-problem. Group columns in pairs, apply Strassen per pair.
  5 pairs × 7 Strassen products = 35 rank terms.
- For j=2: use (2,1,10) sub-problem. Schoolbook = 20 rank terms.
Total rank: 55 (down from 60 schoolbook).
"""
import time

DIMENSIONS = [2, 3, 10]

def construct():
    start_time = time.time()
    DEEP_TIME_LIMIT = 135.0
    
    n, m, p = DIMENSIONS  # 2, 3, 10
    
    def verify(U, V, W, n, m, p, max_checks=10):
        import random
        rank = len(U[0])
        for _ in range(max_checks):
            A = [[random.uniform(-1, 1) for _ in range(m)] for _ in range(n)]
            B = [[random.uniform(-1, 1) for _ in range(p)] for _ in range(m)]
            C = [[0.0]*p for _ in range(n)]
            for i in range(n):
                for k in range(p):
                    for j in range(m):
                        C[i][k] += A[i][j] * B[j][k]
            Cc = [[0.0]*p for _ in range(n)]
            for i in range(n):
                for k in range(p):
                    acc = 0.0
                    for r in range(rank):
                        ua = sum(U[i*m+j][r] * A[i][j] for j in range(m))
                        vb = sum(V[j*p+k][r] * B[j][k] for j in range(m))
                        acc += W[k*n+i][r] * ua * vb
                    Cc[i][k] = acc
            for i in range(n):
                for k in range(p):
                    if abs(C[i][k] - Cc[i][k]) > 1e-9:
                        return False
        return True
    
    def make_strassen_2x2x10(n, m, p):
        """
        Construct rank-35 decomposition for (2,2,10) sub-problem.
        Uses Strassen's 7 products per pair of columns. 5 pairs = 35 terms.
        
        Strassen for 2x2:
        M1 = (A00+A11)(B00+B11)
        M2 = (A10+A11)B00
        M3 = A00(B10-B11)
        M4 = A11(B01-B00)
        M5 = (A00+A01)B11
        M6 = (A10-A00)(B00+B10)
        M7 = (A01-A11)(B01+B11)
        
        C00 = M1 + M4 - M5 + M7
        C01 = M3 + M5
        C10 = M2 + M4
        C11 = M1 - M2 + M3 + M6
        
        For (2,2,p) with columns grouped in pairs (k, k+1):
        For each pair, the "B00" is B[0,k], "B01" is B[0,k+1],
        "B10" is B[1,k], "B11" is B[1,k+1].
        """
        # rank = 7 * (p//2) = 35
        rank = 7 * (p // 2)
        U = [[0.0] * rank for _ in range(n * 2)]  # n*2 = 4 rows (i in 0..1, j in 0..1)
        V = [[0.0] * rank for _ in range(2 * p)]  # 2*p = 20 rows (j in 0..1, k in 0..9)
        W = [[0.0] * rank for _ in range(p * n)]  # p*n = 20 rows (k*n+i)
        
        # Strassen patterns: (u_entries, v_entries, w_entries)
        # u_entries: list of (i, j, coeff) for U
        # v_entries: list of (j, k_offset, coeff) for V where k is relative to pair start
        # w_entries: list of (k_offset, i, coeff) for W
        # 
        # For a pair starting at column k0:
        # B[0,k0] = "B00", B[0,k0+1] = "B01", B[1,k0] = "B10", B[1,k0+1] = "B11"
        #
        # M1: U has A00+A11 -> (0,0,1),(1,1,1); V has B00+B11 -> (0,0,1),(1,1,1)
        # M2: U has A10+A11 -> (1,0,1),(1,1,1); V has B00 -> (0,0,1)
        # M3: U has A00 -> (0,0,1); V has B10-B11 -> (1,0,1),(1,1,-1)
        # M4: U has A11 -> (1,1,1); V has B01-B00 -> (0,1,1),(0,0,-1)
        # M5: U has A00+A01 -> (0,0,1),(0,1,1); V has B11 -> (1,1,1)
        # M6: U has A10-A00 -> (1,0,1),(0,0,-1); V has B00+B10 -> (0,0,1),(1,0,1)
        # M7: U has A01-A11 -> (0,1,1),(1,1,-1); V has B01+B11 -> (0,1,1),(1,1,1)
        
        strassen_patterns = [
            # (u_list, v_list, w_list)
            # M1
            ([(0,0,1.0),(1,1,1.0)], [(0,0,1.0),(1,1,1.0)], [(0,0,1.0),(1,1,1.0)]),  # C00+=M1, C11+=M1
            # M2
            ([(1,0,1.0),(1,1,1.0)], [(0,0,1.0)], [(0,0,0.0),(1,0,1.0)]),  # C10+=M2
            # M3
            ([(0,0,1.0)], [(1,0,1.0),(1,1,-1.0)], [(1,0,1.0),(1,1,0.0)]),  # C01+=M3, C11+=M3
            # M4
            ([(1,1,1.0)], [(0,1,1.0),(0,0,-1.0)], [(0,0,1.0),(0,1,0.0),(1,0,1.0)]),  # C00+=M4, C10+=M4
            # M5
            ([(0,0,1.0),(0,1,1.0)], [(1,1,1.0)], [(0,0,-1.0),(1,0,0.0),(1,1,1.0)]),  # C00-=M5, C01+=M5, C11+=M5
            # M6
            ([(1,0,1.0),(0,0,-1.0)], [(0,0,1.0),(1,0,1.0)], [(1,1,1.0)]),  # C11+=M6
            # M7
            ([(0,1,1.0),(1,1,-1.0)], [(0,1,1.0),(1,1,1.0)], [(0,0,1.0),(1,1,1.0)]),  # C00+=M7, C11+=M7
        ]
        
        # Wait, let me re-derive the W entries more carefully.
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        #
        # So W contributions per Strassen product:
        # M1: W[0,0]+=1, W[1,1]+=1  (i.e., C[0,0] and C[1,1])
        # M2: W[1,0]+=1  (C[1,0])
        # M3: W[0,1]+=1, W[1,1]+=1  (C[0,1] and C[1,1])
        # M4: W[0,0]+=1, W[1,0]+=1  (C[0,0] and C[1,0])
        # M5: W[0,0]-=1, W[0,1]+=1, W[1,1]+=1  (C[0,0] subtract, C[0,1] add, C[1,1] add)
        # M6: W[1,1]+=1  (C[1,1])
        # M7: W[0,0]+=1, W[1,1]+=1  (C[0,0] and C[1,1])
        
        # Let me redo:
        # W[k*n+i] where k is column offset within pair (0 or 1), i is row (0 or 1)
        # For pair at columns (k0, k0+1):
        #   C[0,k0] gets: M1, M4, -M5, M7
        #   C[0,k0+1] gets: M3, M5
        #   C[1,k0] gets: M2, M4
        #   C[1,k0+1] gets: M1, -M2, M3, M6
        
        strassen_patterns_v2 = [
            # M1: U={(0,0,1),(1,1,1)}, V={(0,0,1),(1,1,1)}, W={(0,0,1),(1,1,1)}
            ([(0,0,1.0),(1,1,1.0)], [(0,0,1.0),(1,1,1.0)], [(0,0,1.0),(1,1,1.0)]),
            # M2: U={(1,0,1),(1,1,1)}, V={(0,0,1)}, W={(1,0,1)}
            ([(1,0,1.0),(1,1,1.0)], [(0,0,1.0)], [(1,0,1.0)]),
            # M3: U={(0,0,1)}, V={(1,0,1),(1,1,-1)}, W={(0,1,1),(1,1,1)}
            ([(0,0,1.0)], [(1,0,1.0),(1,1,-1.0)], [(0,1,1.0),(1,1,1.0)]),
            # M4: U={(1,1,1)}, V={(0,1,1),(0,0,-1)}, W={(0,0,1),(1,0,1)}
            ([(1,1,1.0)], [(0,1,1.0),(0,0,-1.0)], [(0,0,1.0),(1,0,1.0)]),
            # M5: U={(0,0,1),(0,1,1)}, V={(1,1,1)}, W={(0,0,-1),(0,1,1),(1,1,1)}
            ([(0,0,1.0),(0,1,1.0)], [(1,1,1.0)], [(0,0,-1.0),(0,1,1.0),(1,1,1.0)]),
            # M6: U={(1,0,1),(0,0,-1)}, V={(0,0,1),(1,0,1)}, W={(1,1,1)}
            ([(1,0,1.0),(0,0,-1.0)], [(0,0,1.0),(1,0,1.0)], [(1,1,1.0)]),
            # M7: U={(0,1,1),(1,1,-1)}, V={(0,1,1),(1,1,1)}, W={(0,0,1),(1,1,1)}
            ([(0,1,1.0),(1,1,-1.0)], [(0,1,1.0),(1,1,1.0)], [(0,0,1.0),(1,1,1.0)]),
        ]
        
        col = 0
        for pair_idx in range(p // 2):
            k0 = 2 * pair_idx
            for (u_list, v_list, w_list) in strassen_patterns_v2:
                for (i, j, coeff) in u_list:
                    U[i * 2 + j][col] += coeff
                for (j, k_off, coeff) in v_list:
                    V[j * p + (k0 + k_off)][col] += coeff
                for (k_off, i, coeff) in w_list:
                    W[(k0 + k_off) * n + i][col] += coeff
                col += 1
        
        return U, V, W, rank
    
    def make_schoolbook_2x1x10(n, p):
        """Schoolbook for (2,1,10): rank = 20."""
        rank = n * 1 * p
        U = [[0.0] * rank for _ in range(n * 1)]
        V = [[0.0] * rank for _ in range(1 * p)]
        W = [[0.0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for k in range(p):
                U[i * 1 + 0][col] = 1.0
                V[0 * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
        return U, V, W, rank
    
    # Build the full (2,3,10) decomposition
    # Part 1: (2,2,10) using Strassen pairs -> rank 35
    # Part 2: (2,1,10) using schoolbook -> rank 20
    # Total rank: 55
    
    total_rank = 55
    U_full = [[0.0] * total_rank for _ in range(n * m)]  # 6 rows
    V_full = [[0.0] * total_rank for _ in range(m * p)]  # 30 rows
    W_full = [[0.0] * total_rank for _ in range(p * n)]  # 20 rows
    
    # Part 1: Strassen for j=0,1 (columns 0,1 of A/B)
    U1, V1, W1, r1 = make_strassen_2x2x10(n, 2, p)
    for i in range(n):
        for j in range(2):
            for r in range(r1):
                U_full[i * m + j][r] += U1[i * 2 + j][r]
    for j in range(2):
        for k in range(p):
            for r in range(r1):
                V_full[j * p + k][r] += V1[j * p + k][r]
    for k in range(p):
        for i in range(n):
            for r in range(r1):
                W_full[k * n + i][r] += W1[k * n + i][r]
    
    # Part 2: Schoolbook for j=2 (column 2 of A/B)
    U2, V2, W2, r2 = make_schoolbook_2x1x10(n, p)
    offset = r1
    for i in range(n):
        for r in range(r2):
            U_full[i * m + 2][offset + r] += U2[i * 1 + 0][r]
    for k in range(p):
        for r in range(r2):
            V_full[2 * p + k][offset + r] += V2[0 * p + k][r]
    for k in range(p):
        for i in range(n):
            for r in range(r2):
                W_full[k * n + i][offset + r] += W2[k * n + i][r]
    
    result = {"U": U_full, "V": V_full, "W": W_full}
    
    # Verify
    if not verify(U_full, V_full, W_full, n, m, p, max_checks=10):
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0.0] * rank for _ in range(n * m)]
        V = [[0.0] * rank for _ in range(m * p)]
        W = [[0.0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1.0
                    V[j * p + k][col] = 1.0
                    W[k * n + i][col] = 1.0
                    col += 1
        result = {"U": U, "V": V, "W": W}
    
    elapsed = time.time() - start_time
    if elapsed > DEEP_TIME_LIMIT:
        return result
    
    return result
