"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

import time
import random
import itertools
import numpy as np
from scipy.optimize import milp, Bounds, LinearConstraint
from scipy.sparse import csc_matrix

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    
    # The goal is to find U (n*m, rank), V (m*p, rank), W (p*n, rank)
    # such that T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # Indices: i in [0, n), j in [0, m), k in [0, p)
    # U index: i*m + j
    # V index: j*p + k
    # W index: k*n + i

    # Strategy: Constructive Block Decomposition
    # We partition the multiplication (2x4) * (4x6) -> (2x6)
    # Let A be 2x4, B be 4x6.
    # Split B into two 4x3 blocks: B1 (cols 0-2), B2 (cols 3-5).
    # Compute A*B1 (2x3) and A*B2 (2x3).
    
    # To compute A (2x4) * B1 (4x3):
    # Split A into A1 (2x2, cols 0-1) and A2 (2x2, cols 2-3).
    # Split B1 into B11 (2x3, rows 0-1) and B12 (2x3, rows 2-3).
    # Product = A1*B11 + A2*B12.
    # A1*B11 is (2x2)*(2x3). Rank of (2,2,3) tensor is 7 (Strassen-like).
    # A2*B12 is (2x2)*(2x3). Rank of (2,2,3) tensor is 7.
    # Total for A*B1: 14.
    
    # Similarly for A*B2: 14.
    # Total rank: 28? 
    # Wait, can we do better?
    # The (2,2,3) tensor has rank 7.
    # Is there a better way to split 4x6?
    
    # Let's look at the (2,4,6) tensor directly.
    # Known lower bounds and constructions:
    # (2,4,4) has rank 14? No, (2,4,4) is equivalent to two (2,2,4)s?
    # (2,2,4) has rank 8? No, (2,2,4) can be done with rank 8?
    # Actually, (2,2,2) is 7.
    # (2,2,3) is 7.
    # (2,2,4) is 8? Let's check.
    # If we split 4 into 2+2:
    # A(2x2) * B(2x4) -> C(2x4)
    # B = [B1(2x2), B2(2x2)]
    # C = [A*B1, A*B2]
    # Rank = Rank(2,2,2) + Rank(2,2,2) = 7 + 7 = 14.
    # But we can share? No, A is the same.
    # T_{i,j,k} for k in 0..3.
    # If we use 7 terms for k=0,1 and 7 terms for k=2,3, total 14.
    # Is there a rank 13? 12?
    # Actually, (2,2,4) rank is 8? No.
    # Let's use the known (2,4,6) construction.
    
    # A common construction for (2, m, p) uses the (2,2,2) Strassen kernel.
    # Split m=4 into 2+2, p=6 into 3+3.
    # A = [A1 A2], B = [B1; B2] (top 2 rows, bottom 2 rows)
    # C = A1*B1 + A2*B2? No, B is 4x6.
    # B = [B_top (2x6); B_bottom (2x6)]
    # C = A1 * B_top + A2 * B_bottom.
    
    # Compute A1 (2x2) * B_top (2x6).
    # Split B_top into B1 (2x3) and B2 (2x3).
    # A1*B1: (2,2,3) rank 7.
    # A1*B2: (2,2,3) rank 7.
    # Total for A1*B_top: 14.
    
    # Compute A2 (2x2) * B_bottom (2x6).
    # Similarly: 14.
    
    # Total rank: 28.
    
    # Can we do better?
    # Maybe split p=6 into 2+2+2?
    # A1 (2x2) * B_top (2x6).
    # B_top = [B1(2x2), B2(2x2), B3(2x2)].
    # A1*B1: 7, A1*B2: 7, A1*B3: 7. Total 21. Worse.
    
    # What if we use a (2,3,3) decomposition?
    # Rank of (2,3,3) is 11.
    # Split m=4 into 2+2.
    # A1 (2x2) * B_top (2x6).
    # Split B_top (2x6) into B1 (2x3) and B2 (2x3).
    # A1*B1: (2,2,3) rank 7.
    # A1*B2: (2,2,3) rank 7.
    # Total 14.
    
    # Is there a (2,4,3) decomposition with rank < 14?
    # (2,4,3) tensor.
    # Split m=4 into 2+2.
    # A1(2x2)*B1(2x3) + A2(2x2)*B2(2x3).
    # Rank(2,2,3)=7. Total 14.
    
    # What if we split p=3? No.
    
    # Let's check the rank of (2,4,6).
    # If we use the 28-rank construction, score is 48/28 = 1.71.
    # The inspirations mention score 1.1429 which corresponds to rank 42 (48/42 = 1.1428).
    # So rank 42 is the current best found by inspirations.
    # My 28-rank construction should be better.
    
    # Let's verify the (2,2,3) rank 7 construction.
    # T_{i,j,k} for i,j in 0,1 and k in 0,1,2.
    # We need 7 terms.
    # Standard Strassen for (2,2,2) uses 7 terms.
    # For (2,2,3), we can view it as (2,2,2) + (2,2,1)?
    # (2,2,1) rank is 4? No, (2,2,1) is just matrix mult, rank 4?
    # T_{i,j,k} = A_{i,j} * B_{j,k}.
    # For k=0,1,2.
    # If we use 7 terms for k=0,1 and 4 terms for k=2, total 11.
    # But we can do better.
    # The rank of (2,2,3) is indeed 7?
    # Let's check literature.
    # R(2,2,2)=7.
    # R(2,2,3) <= R(2,2,2) + R(2,2,1) = 7 + 4 = 11.
    # Is R(2,2,3) = 7?
    # If R(2,2,3)=7, then we can compute (2x2)*(2x3) with 7 multiplications.
    # This is equivalent to computing (2x2)*(2x2) and (2x2)*(2x1) with 7?
    # No.
    # Actually, R(2,2,3) is 7.
    # Construction:
    # Let A = [[a,b],[c,d]], B = [[e,f],[g,h]] (2x2 part), C = [[i],[j]] (2x1 part).
    # We want A*B and A*C.
    # This is not standard.
    
    # Let's stick to a safer construction: Rank 42 is known to work (from inspirations).
    # Rank 28 might work if R(2,2,3)=7 is correct.
    # Let's implement the rank 28 construction and verify it.
    # If it fails, fall back to rank 42 or 48.
    
    def build_rank_28():
        # U: 8x28, V: 24x28, W: 12x28
        U = np.zeros((8, 28), dtype=int)
        V = np.zeros((24, 28), dtype=int)
        W = np.zeros((12, 28), dtype=int)
        
        # We need a (2,2,3) rank 7 decomposition.
        # Let's find one.
        # If R(2,2,3) is not 7, this will fail.
        # Actually, R(2,2,3) is 7?
        # Let's assume we can use a known (2,2,3) rank 7 scheme.
        # If not, we can use two (2,2,2) blocks? No.
        
        # Alternative: Use rank 14 for (2,4,3) by using two (2,2,3) blocks of rank 7.
        # If (2,2,3) rank is 11, then (2,4,3) rank is 22.
        # Then (2,4,6) rank is 44.
        # 44 is better than 48, worse than 42? No, 42 is better.
        
        # Let's look at the inspiration "rank-11 2x3x3".
        # If R(2,3,3)=11.
        # Split m=4 into 1+3?
        # A1(1x1)*B1(1x6) + A2(1x3)*B2(3x6)?
        # (1,1,6) rank 6.
        # (1,3,6) rank?
        
        # Let's try to construct a rank 42 solution explicitly, as it is proven in inspirations.
        # Strategy:
        # Split p=6 into 3+3.
        # Compute A(2x4) * B1(4x3) and A(2x4) * B2(4x3).
        # For A(2x4) * B(4x3):
        # Split m=4 into 2+2.
        # A1(2x2)*B11(2x3) + A2(2x2)*B12(2x3).
        # If R(2,2,3) = 7, then total rank for (2,4,3) is 14.
        # Total for (2,4,6) is 28.
        
        # If R(2,2,3) = 11, then total rank for (2,4,3) is 22.
        # Total for (2,4,6) is 44.
        
        # The inspiration says "rank-11 2x3x3".
        # This implies R(2,3,3) = 11.
        # Note R(2,2,3) <= R(2,3,3) = 11.
        # Is R(2,2,3) = 7?
        # According to some sources, R(2,2,3) = 7.
        # Let's try to find the 7 terms.
        # If I can't find them, I will implement the 42-rank solution which uses
        # Strassen (2,2,2) blocks.
        
        # Rank 42 Construction:
        # Split p=6 into 2+2+2.
        # A(2x4) * B1(4x2) + A(2x4) * B2(4x2) + A(2x4) * B3(4x2).
        # For each (2x4)*(4x2):
        # Split m=4 into 2+2.
        # A1(2x2)*B11(2x2) + A2(2x2)*B12(2x2).
        # Each (2,2,2) has rank 7.
        # So each (2,4,2) has rank 14.
        # Total rank: 14 * 3 = 42.
        
        # This is a valid construction.
        # Let's implement this.
        
        def strassen_222():
            # Returns U, V, W for (2,2,2)
            # U: 4x7, V: 4x7, W: 4x7
            # Indices: i,j in 0,1. U[i*2+j], V[j*2+k], W[k*2+i]
            u = [[0]*7 for _ in range(4)]
            v = [[0]*7 for _ in range(4)]
            w = [[0]*7 for _ in range(4)]
            
            # Strassen terms:
            # M1 = (A11+A22)(B11+B22)
            # M2 = (A21+A22)B11
            # M3 = A11(B21-B22)
            # M4 = A22(B11-B21)
            # M5 = (A11+A12)B22
            # M6 = (A21-A11)(B11+B12)
            # M7 = (A12-A22)(B21+B22)
            
            # A indices: 00, 01, 10, 11 -> 0, 1, 2, 3
            # B indices: 00, 01, 10, 11 -> 0, 1, 2, 3
            # C indices: 00, 01, 10, 11 -> 0, 1, 2, 3
            
            # Term 1: (A00+A11)(B00+B11)
            # A: 00(+), 11(+)
            # B: 00(+), 11(+)
            # C: 00(+), 01(-), 10(-), 11(+)
            u[0][0] = 1; u[3][0] = 1
            v[0][0] = 1; v[3][0] = 1
            w[0][0] = 1; w[1][0] = -1; w[2][0] = -1; w[3][0] = 1
            
            # Term 2: (A10+A11)B00
            # A: 10(+), 11(+)
            # B: 00(+)
            # C: 10(+), 11(+)
            u[2][1] = 1; u[3][1] = 1
            v[0][1] = 1
            w[2][1] = 1; w[3][1] = 1
            
            # Term 3: A00(B10-B11)
            # A: 00(+)
            # B: 10(+), 11(-)
            # C: 00(+), 01(-)
            u[0][2] = 1
            v[2][2] = 1; v[3][2] = -1
            w[0][2] = 1; w[1][2] = -1
            
            # Term 4: A11(B00-B10)
            # A: 11(+)
            # B: 00(+), 10(-)
            # C: 10(+), 11(-)
            u[3][3] = 1
            v[0][3] = 1; v[2][3] = -1
            w[2][3] = 1; w[3][3] = -1
            
            # Term 5: (A00+A01)B11
            # A: 00(+), 01(+)
            # B: 11(+)
            # C: 01(+), 11(+)
            u[0][4] = 1; u[1][4] = 1
            v[3][4] = 1
            w[1][4] = 1; w[3][4] = 1
            
            # Term 6: (A10-A00)(B00+B01)
            # A: 10(+), 00(-)
            # B: 00(+), 01(+)
            # C: 10(+), 11(-)
            u[2][5] = 1; u[0][5] = -1
            v[0][5] = 1; v[1][5] = 1
            w[2][5] = 1; w[3][5] = -1
            
            # Term 7: (A01-A11)(B10+B11)
            # A: 01(+), 11(-)
            # B: 10(+), 11(+)
            # C: 01(+), 11(-)
            u[1][6] = 1; u[3][6] = -1
            v[2][6] = 1; v[3][6] = 1
            w[1][6] = 1; w[3][6] = -1
            
            return u, v, w
        
        u_222, v_222, w_222 = strassen_222()
        
        # Build U, V, W for (2,4,6)
        # 3 blocks of (2,4,2)
        # Block 0: k=0,1
        # Block 1: k=2,3
        # Block 2: k=4,5
        
        # Each (2,4,2) is sum of two (2,2,2)s.
        # A1(2x2, rows 0-1 of A) * B11(2x2, rows 0-1 of B, cols k0-k1)
        # A2(2x2, rows 2-3 of A) * B12(2x2, rows 2-3 of B, cols k0-k1)
        
        total_rank = 42
        U = np.zeros((8, total_rank), dtype=int)
        V = np.zeros((24, total_rank), dtype=int)
        W = np.zeros((12, total_rank), dtype=int)
        
        col_offset = 0
        for block_idx in range(3):
            k_start = block_idx * 2
            
            # Sub-block 1: A1 * B11
            # A1 indices: i in 0,1, j in 0,1 -> U indices 0..3
            # B11 indices: j in 0,1, k in 0,1 (relative) -> V indices (0*6+0)..(1*6+1) = 0..7?
            # Wait, V index is j*p + k.
            # j in 0,1. k in k_start, k_start+1.
            # V indices: 0*6+k_start, 0*6+k_start+1, 1*6+k_start, 1*6+k_start+1.
            # W indices: k*n + i.
            # k in k_start, k_start+1. i in 0,1.
            # W indices: k_start*2+0, k_start*2+1, (k_start+1)*2+0, (k_start+1)*2+1.
            
            for t in range(7):
                # Map u_222 (4x7) to U
                # u_222 row 0 -> U row 0 (i=0,j=0)
                # u_222 row 1 -> U row 1 (i=0,j=1)
                # u_222 row 2 -> U row 2 (i=1,j=0)
                # u_222 row 3 -> U row 3 (i=1,j=1)
                for r in range(4):
                    U[r, col_offset + t] = u_222[r][t]
                
                # Map v_222 (4x7) to V
                # v_222 row 0 -> V row 0*6+k_start (j=0, k=k_start)
                # v_222 row 1 -> V row 0*6+k_start+1 (j=0, k=k_start+1)
                # v_222 row 2 -> V row 1*6+k_start (j=1, k=k_start)
                # v_222 row 3 -> V row 1*6+k_start+1 (j=1, k=k_start+1)
                v_rows = [0*6+k_start, 0*6+k_start+1, 1*6+k_start, 1*6+k_start+1]
                for r in range(4):
                    V[v_rows[r], col_offset + t] = v_222[r][t]
                
                # Map w_222 (4x7) to W
                # w_222 row 0 -> W row k_start*2+0 (k=k_start, i=0)
                # w_222 row 1 -> W row k_start*2+1 (k=k_start, i=1)
                # w_222 row 2 -> W row (k_start+1)*2+0 (k=k_start+1, i=0)
                # w_222 row 3 -> W row (k_start+1)*2+1 (k=k_start+1, i=1)
                w_rows = [k_start*2+0, k_start*2+1, (k_start+1)*2+0, (k_start+1)*2+1]
                for r in range(4):
                    W[w_rows[r], col_offset + t] = w_222[r][t]
            
            col_offset += 7
            
            # Sub-block 2: A2 * B12
            # A2 indices: i in 0,1, j in 2,3 -> U indices 4..7
            # B12 indices: j in 2,3, k in 0,1 (relative) -> V indices (2*6+k_start)..
            # W indices: same as above.
            
            for t in range(7):
                for r in range(4):
                    U[4+r, col_offset + t] = u_222[r][t]
                
                v_rows = [2*6+k_start, 2*6+k_start+1, 3*6+k_start, 3*6+k_start+1]
                for r in range(4):
                    V[v_rows[r], col_offset + t] = v_222[r][t]
                
                w_rows = [k_start*2+0, k_start*2+1, (k_start+1)*2+0, (k_start+1)*2+1]
                for r in range(4):
                    W[w_rows[r], col_offset + t] = w_222[r][t]
            
            col_offset += 7
        
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}, total_rank

    # Verify the rank 42 solution
    res, rank = build_rank_28() # Name is misleading, it's 42
    
    # Check validity
    def check_validity(res, rank):
        U = np.array(res['U'])
        V = np.array(res['V'])
        W = np.array(res['W'])
        if U.shape[1] != rank or V.shape[1] != rank or W.shape[1] != rank:
            return False
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(rank):
                        s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if s != 1:
                        return False
        return True

    if check_validity(res, rank):
        return res
    else:
        # Fallback to schoolbook
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
        return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
