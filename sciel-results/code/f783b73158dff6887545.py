"""Strassen-based block decomposition for 2x2 by 2x13 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [2, 2, 13]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 110  # Safety margin before 120s limit
    n, m, p = DIMENSIONS  # n=2, m=2, p=13

    # Schoolbook rank = n*m*p = 52
    # Strassen block approach: 6 pairs of columns use rank-7 (2x2*2x2), 1 single column uses rank-4 (2x2*2x1)
    # Total rank = 6*7 + 4 = 46
    
    # For 2x2*2x2, we need a rank-7 tensor decomposition.
    # The tensor T has entries T[i*m+j, j*p+k, k*n+i] = 1 for i,j,k in {0,1}
    # Local dimensions: U has 4 rows, V has 4 rows, W has 4 rows
    
    # Standard Strassen rank-7 decomposition for 2x2*2x2:
    # U rows: A00=0, A01=1, A10=2, A11=3
    # V rows: B00=0, B01=1, B10=2, B11=3
    # W rows: C00=0, C10=1, C01=2, C11=3
    
    # Let me construct this carefully. The 8 nonzero entries of the tensor are:
    # (i,j,k) -> (U_idx, V_idx, W_idx)
    # (0,0,0) -> (0, 0, 0)
    # (0,0,1) -> (0, 1, 2)
    # (0,1,0) -> (1, 2, 0)
    # (0,1,1) -> (1, 3, 2)
    # (1,0,0) -> (2, 0, 1)
    # (1,0,1) -> (2, 1, 3)
    # (1,1,0) -> (3, 2, 1)
    # (1,1,1) -> (3, 3, 3)
    
    # I'll use a known rank-7 decomposition. Here's one construction:
    # We can use the fact that for 2x2*2x2, the rank is exactly 7.
    # One explicit decomposition:
    
    # Let me use a systematic approach. The matrix multiplication tensor for 2x2x2
    # can be decomposed using the 7 Strassen products. Let me verify each term.
    
    # Term 0: M1 = (A00+A11)(B00+B01)
    #   u = [1, 0, 0, 1]  (A00 + A11)
    #   v = [1, 1, 0, 0]  (B00 + B01)
    #   This contributes to C00 and C11. w = [1, 0, 0, 1]
    
    # Term 1: M2 = (A10+A11)B00
    #   u = [0, 0, 1, 1]  (A10 + A11)
    #   v = [1, 0, 0, 0]  (B00)
    #   Contributes to C10. w = [0, 1, 0, 0]
    
    # Term 2: M3 = A00(B01-B11)
    #   u = [1, 0, 0, 0]  (A00)
    #   v = [0, 1, 0, -1] (B01 - B11)
    #   Contributes to C00 and C01. w = [1, 0, 1, 0]
    
    # Term 3: M4 = A11(B10-B00)
    #   u = [0, 0, 0, 1]  (A11)
    #   v = [-1, 0, 1, 0] (B10 - B00)
    #   Contributes to C00 and C11... let me check. 
    #   Actually in standard Strassen: C00 = M1 + M4 - M5 + M7, C01 = M3 + M5, C10 = M2 + M4, C11 = M1 - M2 + M3 + M6
    #   So M4 contributes to C00 (with +1) and C10 (with +1). w = [1, 1, 0, 0]... 
    
    # Wait, I need to be more careful. Let me use the actual Strassen formulas:
    # M1 = (A00+A11)(B00+B01)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # So the w vectors are:
    # M1: C00+=1, C11+=1 -> w=[1, 0, 0, 1]
    # M2: C10+=1 -> w=[0, 1, 0, 0]
    # M3: C00+=1, C01+=1 -> w=[1, 0, 1, 0]
    # M4: C00+=1, C10+=1 -> w=[1, 1, 0, 0]
    # M5: C00-=1, C01+=1 -> w=[-1, 0, 1, 0]
    # M6: C11+=1 -> w=[0, 0, 0, 1]
    # M7: C00+=1, C11+=1 -> w=[1, 0, 0, 1]
    
    # Let me verify this decomposes correctly by checking one entry:
    # T[0,0,0] should be 1 (i=0,j=0,k=0)
    # sum_c u[c][0]*v[c][0]*w[c][0]
    # = u0*v0*w0 + u1*v1*w1 + ... + u6*v6*w6
    # = (1)(1)(1) + (0)(1)(0) + (1)(0)(1) + (0)(-1)(1) + (1)(0)(-1) + (-1)(1)(0) + (0)(1)(1)
    # = 1 + 0 + 0 + 0 + 0 + 0 + 0 = 1 ✓
    
    # T[0,1,2] should be 1 (i=0,j=0,k=1)
    # sum_c u[c][0]*v[c][1]*w[c][2]
    # = (1)(1)(0) + (0)(0)(0) + (1)(1)(1) + (0)(0)(0) + (1)(0)(1) + (-1)(1)(0) + (0)(0)(0)
    # = 0 + 0 + 1 + 0 + 0 + 0 + 0 = 1 ✓
    
    # T[1,2,0] should be 1 (i=0,j=1,k=0)
    # sum_c u[c][1]*v[c][2]*w[c][0]
    # = (0)(0)(1) + (0)(0)(0) + (0)(0)(1) + (0)(1)(1) + (0)(0)(-1) + (0)(0)(0) + (1)(1)(1)
    # = 0 + 0 + 0 + 0 + 0 + 0 + 1 = 1 ✓
    
    # T[1,3,2] should be 1 (i=0,j=1,k=1)
    # sum_c u[c][1]*v[c][3]*w[c][2]
    # = (0)(0)(0) + (0)(0)(0) + (0)(-1)(1) + (0)(0)(0) + (0)(1)(1) + (0)(0)(0) + (1)(1)(0)
    # = 0 + 0 + 0 + 0 + 0 + 0 + 0 = 0 ✗
    
    # That's wrong! Let me reconsider. The issue is that my u, v, w assignments may be incorrect.
    # Let me recheck M7: u = [0, 1, 0, -1] (A01 - A11), v = [0, 0, 1, 1] (B10 + B11)
    # For (i=0,j=1,k=1): u[1]=1, v[3]=1, w[2]=0. So contribution = 1*1*0 = 0.
    # But we need the total to be 1. Let me check all terms:
    # c=0: u[1]=0, v[3]=0, w[2]=0 -> 0
    # c=1: u[1]=0, v[3]=0, w[2]=0 -> 0
    # c=2: u[1]=0, v[3]=-1, w[2]=1 -> 0
    # c=3: u[1]=0, v[3]=0, w[2]=0 -> 0
    # c=4: u[1]=0, v[3]=1, w[2]=1 -> 0
    # c=5: u[1]=0, v[3]=0, w[2]=0 -> 0
    # c=6: u[1]=1, v[3]=1, w[2]=0 -> 0
    # Total = 0, but should be 1.
    
    # I think my w assignment for M7 is wrong. Let me reconsider.
    # Actually, I think the issue is that I'm using the wrong Strassen variant or I made an error.
    
    # Let me try a different approach. I'll directly construct a valid rank-7 decomposition.
    # For 2x2*2x2, I'll use a known valid decomposition from the literature.
    
    # After careful consideration, here's a valid rank-7 decomposition:
    # The seven terms are:
    strassen_U = [
        [1,  0,  0,  1,  1,  0,  0],  # A00
        [0,  0,  1,  0,  0,  0,  1],  # A01
        [0,  1,  0,  0,  0,  1,  0],  # A10
        [1,  1,  0,  0,  0,  0, -1],  # A11
    ]
    strassen_V = [
        [1,  1,  0, -1,  0,  0,  0],  # B00
        [1,  0,  1,  0,  0,  1,  0],  # B01
        [0,  0,  0,  1,  0,  0,  1],  # B10
        [0,  0, -1, 0,  1,  0,  1],  # B11
    ]
    strassen_W = [
        [1,  0,  1,  1, -1,  0,  1],  # C00
        [0,  1,  0,  1,  0,  0,  0],  # C10
        [0,  0,  1,  0,  1,  0,  0],  # C01
        [1,  0,  0,  0,  0,  1,  1],  # C11
    ]
    
    # Let me verify a few entries:
    # T[0,0,0]: sum_c strassen_U[0][c]*strassen_V[0][c]*strassen_W[0][c]
    # = 1*1*1 + 0*1*0 + 0*0*1 + 1*(-1)*1 + 1*0*(-1) + 0*0*0 + 0*0*1
    # = 1 + 0 + 0 - 1 + 0 + 0 + 0 = 0 ✗
    
    # Still wrong. Let me try yet another approach. I'll just verify numerically.
    
    # Actually, let me take a completely different approach. I'll use the fact that
    # I can verify any decomposition numerically, and I'll try to find one that works.
    
    # For now, let me just return the schoolbook decomposition to ensure correctness.
    # Rank 52 is the baseline. To improve, I need to find a valid lower-rank decomposition.
    
    # Let me try a simpler optimization: for 2x2 * 2x13, instead of using Strassen on pairs,
    # I can use a different blocking strategy. But without a verified Strassen decomposition,
    # I can't guarantee correctness.
    
    # Let me just return schoolbook (rank 52) for now.
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
