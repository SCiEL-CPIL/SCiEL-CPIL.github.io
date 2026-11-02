"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 5, 5]
START_TIME = time.time()
TIME_LIMIT = 120.0
SAFETY_MARGIN = 5.0

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 5, 5
    
    # Start with schoolbook rank 75, then try to reduce via Strassen-style
    # block decomposition on 2x2x2 sub-structures.
    
    # We'll build a decomposition and verify it.
    # Use a known upper bound approach: for M_{<3,5,5>}, 
    # we can use the "3x3x3 = 23" result (Landsberg) combined with padding.
    # But let's try a direct Strassen extension.
    
    # Strategy: decompose 5 = 2+2+1, apply Strassen to 2x2 blocks.
    # For each pair of input indices (i,j) where i in 0..2, j in 0..4,
    # and output k in 0..4, we need T[i,j,k] = sum_j A[i,j]*B[j,k].
    
    # Let's try: partition j into {0,1}, {2,3}, {4}
    # For j in {0,1}: A[i,0:2] * B[0:2, :] -> 2x5 product, rank 4 schoolbook
    # For j in {2,3}: A[i,2:4] * B[2:4, :] -> 2x5 product, rank 4 schoolbook  
    # For j in {4}: A[i,4] * B[4, :] -> 1x5 product, rank 5 schoolbook
    # Total: 3*(4+4+5) = 39? No, that's not right - these share the same A and B.
    
    # Actually the tensor is sum over j of A[i,j]*B[j,k].
    # We need to express this as sum_c U[i,j,c]*V[j,k,c]*W[k,i,c]... 
    # Wait, the convention is W[k*n+i, c] so W index is (k,i).
    
    # Let me just verify the schoolbook works and then try to find a lower rank
    # by checking if any linear dependencies exist.
    
    # For now, return schoolbook as baseline, but let's try a smarter construction.
    
    # Key insight: for 1x5 * 5x5, the bilinear complexity is 5 (can't do better
    # for a single row). But for 3x5 * 5x5, we can share computations.
    
    # Use the following: think of it as 3 independent 1x5*5x5 problems
    # but allow sharing of intermediate products.
    
    # Actually, let's try the direct approach: the bilinear complexity of 
    # M_{<3,5,5>} is at most 3*5 = 15 if we view it as 3 rows each needing
    # 5 mults, but that's not how it works - each "multiplication" in the
    # bilinear sense produces a full row of output.
    
    # Correct view: C[i,k] = sum_j A[i,j]*B[j,k]
    # Bilinear rank = min number of rank-1 tensors whose sum = T
    # where T[i,j,k] = delta (i,j,k) in the multiplication tensor.
    
    # Schoolbook: rank 75.
    # Lower bound: max(n*p, m*p, n*m) = max(15,25,15) = 25.
    # Upper bound via Strassen recursion: M_{<3,5,5>} <= ... 
    
    # Let's try a concrete construction with rank < 75.
    # Use block Strassen: write A as [A1; A2; A3] where Ai is 1x5,
    # B as [B1 B2 B3 B4 B5] where Bj is 5x1.
    # C[i,k] = sum_j A[i,j]*B[j,k]
    
    # Group: j in {0,1}, {2,3}, {4}
    # For group {0,1}: contribution is A[i,0]*B[0,k] + A[i,1]*B[1,k]
    # This is a 1x2 * 2x5 product. Bilinear rank of M_{<1,2,5>} = 10? No.
    # M_{<1,2,5>}: C[k] = A0*B0[k] + A1*B1[k], 5 outputs.
    # Each output needs 2 mults schoolbook, total 10. But can we share?
    # (A0+A1)*(B0[k]+B1[k]) = A0*B0[k] + A0*B1[k] + A1*B0[k] + A1*B1[k]
    # We need A0*B0[k] + A1*B1[k], so:
    # (A0+A1)*(B0[k]+B1[k]) - A0*B1[k] - A1*B0[k]
    # That's 3 mults per k, total 15. Worse than 10.
    
    # So for 1x2*2x5, schoolbook (10) is optimal.
    
    # Total schoolbook: 3 * (2+2+1) * 5 = 3*5*5 = 75. Confirmed.
    
    # To beat 75, we need cross-term sharing between different i values.
    # C[0,k] = sum_j A[0,j]*B[j,k]
    # C[1,k] = sum_j A[1,j]*B[j,k]
    # C[2,k] = sum_j A[2,j]*B[j,k]
    
    # For a fixed j, the contribution to all 3 rows is:
    # [A[0,j], A[1,j], A[2,j]] * B[j,k]
    # This is a 3x1 * 1x5 = 3x5 matrix, rank 1.
    # Bilinear rank of M_{<3,1,5>} = 15 (each of 3*5 entries needs one mult).
    # But wait: C[i,k] += A[i,j]*B[j,k] for fixed j.
    # The tensor for fixed j has T[i,k] = A[i,j]*B[j,k], which is rank 1 in (i,k) space.
    # So for each j, we need 15 mults (3*5), total 5*15 = 75.
    
    # To reduce, we need to combine contributions from different j's.
    # E.g., for j=0,1: 
    # C[i,k] += A[i,0]*B[0,k] + A[i,1]*B[1,k]
    # = (A[i,0]+A[i,1])*(B[0,k]+B[1,k]) - A[i,0]*B[1,k] - A[i,1]*B[0,k]
    # That's 3 "terms" per (i,k), so 3*3*5 = 45 for two j's vs 2*15=30 schoolbook. Worse.
    
    # Alternative: (A[i,0]+A[i,1])*(B[0,k]+B[1,k]) gives cross terms we don't want.
    
    # It seems hard to beat 75 for 3x5x5 with simple Strassen.
    
    # Let me check: is the known lower bound tight? 
    # Lower bound: max(15, 25, 15) = 25.
    # Upper bound: 75 (schoolbook).
    # Best known for small cases: M_{<2,2,2>} = 7, M_{<3,3,3>} <= 23.
    
    # For M_{<3,5,5>}, one approach: embed in M_{<3,6,6>} and use 
    # M_{<3,6,6>} <= M_{<3,6,6>}... not helpful.
    
    # Another: M_{<3,5,5>} <= M_{<3,4,4>} * something? No.
    
    # Let's try: M_{<3,5,5>} can be computed by:
    # Split 5 = 4+1. 
    # A = [A_4 | A_1] where A_4 is 3x4, A_1 is 3x1
    # B = [B_4; B_1] where B_4 is 4x5, B_1 is 1x5
    # C = A_4*B_4 + A_1*B_1
    # A_4*B_4 is 3x4*4x5, A_1*B_1 is 3x1*1x5.
    
    # M_{<3,4,5>} schoolbook = 60, M_{<3,1,5>} schoolbook = 15, total 75.
    
    # Use Strassen for 3x4*4x5? Write 4=2+2:
    # A_4 = [A_a | A_b], B_4 = [B_a; B_b]
    # A_4*B_4 = A_a*B_a + A_b*B_b
    # Each is 3x2*2x5, schoolbook = 30 each, total 60. Same.
    
    # I think for this specific case, 75 might be close to optimal or the
    # best we can easily achieve. Let me just return the schoolbook.
    
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
