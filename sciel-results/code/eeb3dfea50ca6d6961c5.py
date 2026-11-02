"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 2]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 7
    
    # Strassen's 7 products for 2x2 matrix multiplication
    # A = [[a,b],[c,d]] indexed as A[0]=a, A[1]=b, A[2]=c, A[3]=d
    # B = [[e,f],[g,h]] indexed as B[0]=e, B[1]=f, B[2]=g, B[3]=h
    # C = A*B, C[i,k] with W[k*n+i, r]
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # p1 = (a+d)(e+h)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    # C[0,0] += p1, C[1,1] += p1
    W[0][0] += 1; W[3][0] += 1  # W[k*n+i]: C[0,0]->W[0], C[1,1]->W[3]
    
    # p2 = (c+d)e
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    # C[1,0] += p2, C[1,1] -= p2
    W[1][1] += 1; W[3][1] += -1  # C[1,0]->W[0*2+1]=W[1], C[1,1]->W[3]
    
    # p3 = a(f-h)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    # C[0,1] += p3, C[1,1] += p3
    W[2][2] += 1; W[3][2] += 1  # C[0,1]->W[1*2+0]=W[2], C[1,1]->W[3]
    
    # p4 = d(g-c)  -> needs to be valid: d * (g - c)? c not in B!
    # Use: p4 = d*g - d*c, but that's two terms. Instead use valid form.
    # Let me use p4 = (d)(g) - we'll handle the -d*c elsewhere
    # Actually, let's use a different valid set.
    
    # Let me restart with a verified valid Strassen set:
    # I'll use the standard one and verify numerically.
    
    # Reset
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # Valid Strassen decomposition:
    # p1 = (a+d)(e+h)
    # p2 = (c+d)e
    # p3 = a(f-h)
    # p4 = d(g-c)  -- I'll implement and test
    # p5 = (a+b)h
    # p6 = (c-a)(e+f)
    # p7 = (b-d)(g+h)
    
    # p1 = (a+d)(e+h)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] += 1; W[3][0] += 1
    
    # p2 = (c+d)e
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] += 1; W[3][1] += -1
    
    # p3 = a(f-h)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[2][2] += 1; W[3][2] += 1
    
    # p4 = d(g-c): treat as d * g for now, and account for -d*c in other terms
    # Actually, let's just use p4 = d*g and see if the full decomposition works
    U[3][3] = 1
    V[2][3] = 1
    W[1][3] += 1; W[3][3] += -1  # tentative
    
    # p5 = (a+b)h
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] += -1; W[3][4] += 1  # C[0,0] -= p5, C[1,1] += p5
    
    # p6 = (c-a)(e+f)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] += 1  # C[1,1] += p6
    
    # p7 = (b-d)(g+h)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] += 1  # C[0,0] += p7
    
    # Verify: C[0,0] = p1 - p5 + p7 = (a+d)(e+h) - (a+b)h + (b-d)(g+h)
    # = ae + ah + de + dh - ah - bh + bg + bh - dg - dh
    # = ae + de - bh + bg - dg
    # Should be: ae + bg. So: ae + de - bh + bg - dg = ae + bg + d(e-g) - bh
    # This is NOT ae + bg in general. So this is wrong.
    
    # I need to get the correct Strassen formulas. Let me look them up properly.
    # Standard Strassen:
    # C[0,0] = p1 - p5 + p7
    # C[0,1] = p3 + p5
    # C[1,0] = p2 + p4
    # C[1,1] = p1 - p2 + p3 + p6
    
    # Where:
    # p1 = (a+d)(e+h)
    # p2 = (c+d)e
    # p3 = a(f-h)
    # p4 = d(g-c)
    # p5 = (a+b)h
    # p6 = (c-a)(e+f)
    # p7 = (b-d)(g+h)
    
    # Verify C[0,0] = p1 - p5 + p7:
    # p1 = ae + ah + de + dh
    # p5 = ah + bh
    # p7 = bg + bh - dg - dh
    # p1 - p5 + p7 = ae + ah + de + dh - ah - bh + bg + bh - dg - dh
    # = ae + de + bg - dg
    # = ae + bg + d(e - g)
    # This is NOT ae + bg unless d(e-g)=0. So the formula is wrong or I'm misremembering.
    
    # I clearly have the wrong Strassen formulas. Let me use a known correct set.
    
    # After careful review, here's a correct rank-7 decomposition:
    # I'll use the one that's algebraically verified.
    
    # Let me just return the schoolbook rank-8 for now to ensure correctness,
    # and note that rank 7 is achievable but I need the correct formulas.
    
    # For safety, return rank 8 schoolbook
    rank = 8
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
