"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # We decompose M_{2,4,7} by splitting the inner dimension m=4 into two blocks of size 2.
    # Block 1: A[:, 0:2] * B[0:2, :] -> M_{2,2,7}
    # Block 2: A[:, 2:4] * B[2:4, :] -> M_{2,2,7}
    
    # For each M_{2,2,7}, we split the output dimension p=7 into p=4 and p=3.
    # M_{2,2,7} = M_{2,2,4} + M_{2,2,3}
    
    # M_{2,2,4} can be decomposed using two M_{2,2,2} blocks.
    # M_{2,2,3} can be decomposed using one M_{2,2,2} block and one M_{2,2,1} block.
    
    # We use the optimal rank-7 decomposition for M_{2,2,2} (Winograd).
    # We use the optimal rank-4 decomposition for M_{2,2,1} (Schoolbook is optimal here? 
    # Actually for 2x2 * 2x1, rank is 4. Wait, 2x2 * 2x1 is just matrix-vector. 
    # C_1 = A[:,0]B[0,0] + A[:,1]B[1,0]
    # C_2 = A[:,0]B[0,1] + A[:,1]B[1,1]
    # This is 4 scalar multiplications. Is there a lower rank? 
    # M_{2,2,1} has rank 4. Yes, it's essentially 2x2 * 2x1 which is 4 mults.
    # Wait, the tensor is M_{2,2,1}. 
    # U shape (4, r), V shape (2, r), W shape (2, r).
    # Rank 4 is schoolbook.
    
    # So:
    # Rank(M_{2,2,2}) = 7
    # Rank(M_{2,2,1}) = 4
    # Rank(M_{2,2,4}) = 7 + 7 = 14
    # Rank(M_{2,2,3}) = 7 + 4 = 11
    # Rank(M_{2,2,7}) = 14 + 11 = 25
    # Total Rank = 2 * 25 = 50.
    
    # Let's implement the Winograd 2x2x2 decomposition.
    # Indices for A (2x2): 0=00, 1=01, 2=10, 3=11
    # Indices for B (2x2): 0=00, 1=01, 2=10, 3=11
    # Indices for C (2x2): 0=00, 1=01, 2=10, 3=11
    
    # Winograd 2x2x2 (Rank 7)
    # p0 = (a00+a11)(b00+b11)
    # p1 = a01 b10
    # p2 = a10 b01
    # p3 = (a00+a01) b00
    # p4 = (a11+a10) b11
    # p5 = (a00-a10) b10
    # p6 = (a11-a01) b01
    
    # U (4x7):
    # Row 0 (a00): 1, 0, 0, 1, 0, 1, 0
    # Row 1 (a01): 0, 1, 0, 1, 0, 0, 1
    # Row 2 (a10): 0, 0, 1, 0, 1, 1, 0
    # Row 3 (a11): 1, 0, 0, 0, 1, 0, 1
    
    # V (4x7):
    # Row 0 (b00): 1, 0, 0, 1, 0, 0, 0
    # Row 1 (b01): 0, 0, 1, 0, 0, 0, 1
    # Row 2 (b10): 0, 1, 0, 0, 0, 1, 0
    # Row 3 (b11): 1, 0, 0, 0, 1, 0, 0
    
    # W (4x7):
    # C00 = p0 - p3 + p5
    # C01 = p0 - p4 - p6
    # C10 = p0 - p5 - p2
    # C11 = p0 - p3 + p4 + p6 + p2 - p1 ? Let's check.
    
    # Let's verify C00:
    # C00 = a00 b00 + a01 b10
    # p0 = a00 b00 + a00 b11 + a11 b00 + a11 b11
    # p3 = a00 b00 + a01 b00
    # p5 = a00 b10 - a10 b10
    # p0 - p3 + p5 = (a00 b00 + a00 b11 + a11 b00 + a11 b11) - (a00 b00 + a01 b00) + (a00 b10 - a10 b10)
    # = a00 b11 + a11 b00 + a11 b11 - a01 b00 + a00 b10 - a10 b10
    # This does NOT equal a00 b00 + a01 b10.
    
    # My Winograd formulas are likely permuted or incorrect. Let's use a standard verified set.
    # From "Fast matrix multiplication" by Strassen or similar sources.
    
    # Standard Winograd 2x2:
    # m1 = (a00+a11)(b00+b11)
    # m2 = (a00+a01)b00
    # m3 = (a11+a10)b11
    # m4 = (a00-a10)b10
    # m5 = (a11-a01)b01
    # m6 = a01 b10
    # m7 = a10 b01
    
    # C00 = m1 - m2 + m4
    # C01 = m1 - m3 - m5
    # C10 = m1 - m4 - m7
    # C11 = m1 - m2 + m3 + m5 + m7 - m6
    
    # Let's verify C00:
    # m1 = a00 b00 + a00 b11 + a11 b00 + a11 b11
    # m2 = a00 b00 + a01 b00
    # m4 = a00 b10 - a10 b10
    # m1 - m2 + m4 = (a00 b00 + a00 b11 + a11 b00 + a11 b11) - (a00 b00 + a01 b00) + (a00 b10 - a10 b10)
    # = a00 b11 + a11 b00 + a11 b11 - a01 b00 + a00 b10 - a10 b10
    # Still not C00 = a00 b00 + a01 b10.
    
    # It seems I am mixing up indices or the formulas are for a different permutation.
    # Let's just use the Schoolbook decomposition for M_{2,2,2} to be safe, 
    # but that gives Rank 8, total 56.
    
    # Is there a Rank 7 decomposition I can trust?
    # Yes, the Winograd 2x2 scheme is well-known. Let's look up the exact matrices.
    
    # U:
    # 1 0 0 1 0 1 0
    # 0 1 0 1 0 0 1
    # 0 0 1 0 1 1 0
    # 1 0 0 0 1 0 1
    
    # V:
    # 1 0 0 1 0 0 0
    # 0 0 1 0 0 0 1
    # 0 1 0 0 0 1 0
    # 1 0 0 0 1 0 0
    
    # W:
    # 1 -1 0 0 1 0 0  (C00 = m1 - m2 + m4? No, let's map carefully)
    
    # Let's define the 7 products:
    # 0: (a00+a11)(b00+b11)
    # 1: (a00+a01)b00
    # 2: (a11+a10)b11
    # 3: (a00-a10)b10
    # 4: (a11-a01)b01
    # 5: a01 b10
    # 6: a10 b01
    
    # U columns:
    # a00: 1, 1, 0, 1, 0, 0, 0
    # a01: 0, 1, 0, 0, 1, 1, 0
    # a10: 0, 0, 1, 1, 0, 0, 1
    # a11: 1, 0, 1, 0, 1, 0, 0
    
    # V columns:
    # b00: 1, 1, 0, 0, 0, 0, 0
    # b01: 0, 0, 0, 0, 1, 0, 1
    # b10: 0, 0, 0, 1, 0, 1, 0
    # b11: 1, 0, 1, 0, 0, 0, 0
    
    # W rows (for C00, C01, C10, C11):
    # C00 = a00 b00 + a01 b10
    # C00 = m0 - m1 + m3
    # Check: (a00 b00 + a00 b11 + a11 b00 + a11 b11) - (a00 b00 + a01 b00) + (a00 b10 - a10 b10)
    # = a00 b11 + a11 b00 + a11 b11 - a01 b00 + a00 b10 - a10 b10
    # This is NOT C00.
    
    # There is a known issue with my manual derivation. 
    # Given the risk of error, and that Rank 50 is the target, 
    # I will implement the Rank 50 solution using a verified Winograd 2x2x2 decomposition.
    
    # Verified Winograd 2x2x2 (Rank 7):
    # U = [
    #   [1, 0, 0, 1, 0, 1, 0],
    #   [0, 1, 0, 1, 0, 0, 1],
    #   [0, 0, 1, 0, 1, 1, 0],
    #   [1, 0, 0, 0, 1, 0, 1]
    # ]
    # V = [
    #   [1, 0, 0, 1, 0, 0, 0],
    #   [0, 0, 1, 0, 0, 0, 1],
    #   [0, 1, 0, 0, 0, 1, 0],
    #   [1, 0, 0, 0, 1, 0, 0]
    # ]
    # W = [
    #   [1, 0, 0, 0, 0, 0, 0], # C00
    #   [0, 0, 0, 0, 0, 1, 0], # C01
    #   [0, 0, 0, 0, 0, 0, 1], # C10
    #   [0, 0, 0, 0, 0, 0, 0]  # C11
    # ]
    # This W is clearly wrong (zeros).
    
    # I will fall back to Schoolbook Rank 56 to ensure correctness, as I cannot verify the Winograd matrices without a reference.
    # However, the prompt asks for improvement. Rank 50 is the theoretical optimum for this structure.
    
    # Let's try one more time with a different source's Winograd matrices.
    # Source: "Winograd's 2x2 matrix multiplication"
    # p1 = (a11+a22)(b11+b22)
    # p2 = a12 b21
    # p3 = a21 b12
    # p4 = (a11+a12) b11
    # p5 = (a22+a21) b22
    # p6 = (a11-a21) b21
    # p7 = (a22-a12) b12
    
    # C11 = p1 - p4 + p6
    # C12 = p1 - p5 - p7
    # C21 = p1 - p6 - p3
    # C22 = p1 - p4 + p5 + p7 + p3 - p2
    
    # Let's verify C11:
    # p1 = a11 b11 + a11 b22 + a22 b11 + a22 b22
    # p4 = a11 b11 + a12 b11
    # p6 = a11 b21 - a21 b21
    # p1 - p4 + p6 = (a11 b11 + a11 b22 + a22 b11 + a22 b22) - (a11 b11 + a12 b11) + (a11 b21 - a21 b21)
    # = a11 b22 + a22 b11 + a22 b22 - a12 b11 + a11 b21 - a21 b21
    # C11 should be a11 b11 + a12 b21.
    # This is NOT matching.
    
    # It appears my formulas are consistently off. 
    # I will use the Schoolbook solution (Rank 56) as it is guaranteed correct.
    
    n, m, p = DIMENSIONS
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
