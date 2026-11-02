import numpy as np
import time

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    # Total rank 28
    rank = 28
    
    U = [[0] * rank for _ in range(n * m)] # 6 x 28
    V = [[0] * rank for _ in range(m * p)] # 15 x 28
    W = [[0] * rank for _ in range(p * n)] # 10 x 28
    
    col = 0
    
    # Part 1: A[:, 0] * B[0, :] -> C[:, :]
    # A is 2x1, B is 1x5.
    # A indices: 0, 1 (i*m + j = i*3 + 0)
    # B indices: 0..4 (j*p + k = 0*5 + k = k)
    # C indices: k*n + i = 5*i + k? No, W[k*n+i].
    # C is 2x5. Index is k*n + i? 
    # Task says W[k*n+i, r].
    # C_{i,k} = sum A_{i,j} B_{j,k}.
    # Here j=0.
    # C_{i,k} = A_{i,0} B_{0,k}.
    # U row: i*m + 0 = 3i.
    # V row: 0*p + k = k.
    # W row: k*n + i = 5k + i.
    
    for i in range(n):
        for k in range(p):
            U[i * m + 0][col] = 1
            V[0 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
    # col is now 10

    # Part 2: A[:, 1:3] * B[1:3, :] -> C[:, :]
    # A is 2x2, B is 2x5.
    # Split B into B_left (2x2) and B_right (2x3).
    # B_left cols 0,1. B_right cols 2,3,4.
    
    # Part 2a: A[:, 1:3] * B[1:3, 0:2] (2x2 x 2x2)
    # Strassen decomposition.
    # A indices: i*m + j. j in 1,2.
    # B indices: j*p + k. j in 1,2. k in 0,1.
    # C indices: k*n + i. k in 0,1.
    
    # Strassen for A (2x2) and B (2x2).
    # Let A_sub = [[a, b], [c, d]] corresponding to A[0,1], A[0,2], A[1,1], A[1,2].
    # Let B_sub = [[e, f], [g, h]] corresponding to B[1,0], B[1,1], B[2,0], B[2,1].
    
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-i) -- wait, B indices are e,f (row 1) and g,h (row 2).
    # B[1,0]=e, B[1,1]=f, B[2,0]=g, B[2,1]=h.
    # C[0,0] = ae + bg. C[0,1] = af + bh.
    # C[1,0] = ce + dg. C[1,1] = cf + dh.
    
    # Standard Strassen:
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-i) -> d(g - ?) No, B is [[e,f],[g,h]].
    # M4 = d(g - ?)
    # Let's use the explicit 7 terms for 2x2x2.
    # U, V, W contributions for Strassen.
    
    # We need to map the Strassen variables to the global indices.
    # a = A[0,1] -> U index 3*0+1 = 1
    # b = A[0,2] -> U index 3*0+2 = 2
    # c = A[1,1] -> U index 3*1+1 = 4
    # d = A[1,2] -> U index 3*1+2 = 5
    
    # e = B[1,0] -> V index 1*5+0 = 5
    # f = B[1,1] -> V index 1*5+1 = 6
    # g = B[2,0] -> V index 2*5+0 = 10
    # h = B[2,1] -> V index 2*5+1 = 11
    
    # C outputs:
    # C[0,0] -> W index 0*2+0 = 0
    # C[0,1] -> W index 1*2+0 = 2
    # C[1,0] -> W index 0*2+1 = 1
    # C[1,1] -> W index 1*2+1 = 3
    
    # Strassen coefficients:
    # M1 = (a+d)(e+h)
    # U: a=1, d=1. V: e=1, h=1.
    # W: C00 = M1 - M3 - M4 - M6?
    # Let's just use the known rank 7 decomposition.
    # It's easier to hardcode the 7 terms.
    
    # Term 1: (a+d)(e+h)
    U[1][col] += 1; U[5][col] += 1
    V[5][col] += 1; V[11][col] += 1
    W[0][col] += 1; W[2][col] += 1; W[1][col] -= 1; W[3][col] -= 1
    col += 1
    
    # Term 2: (c+d)e
    U[4][col] += 1; U[5][col] += 1
    V[5][col] += 1
    W[1][col] += 1; W[3][col] += 1
    col += 1
    
    # Term 3: a(f-h)
    U[1][col] += 1
    V[6][col] += 1; V[11][col] -= 1
    W[2][col] += 1
    col += 1
    
    # Term 4: d(g-h) -- Wait, standard Strassen M4 is d(g-i) where B is [[e,f],[g,h]]?
    # Let's check C[1,1] = cf + dh.
    # C[0,0] = ae + bg.
    # C[1,0] = ce + dg.
    # C[0,1] = af + bh.
    
    # Let's use a verified Strassen set.
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-i) -> This assumes B[2,1] is i? No.
    # Let's stick to the indices:
    # B = [[e, f], [g, h]]
    # M4 = d(g - ?)
    # Actually, M4 = d(g - ?) is not standard.
    # Standard:
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-i) -> This is for B = [[e, f], [i, j]]?
    # Let's use the one from Wikipedia:
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-i) -- No, B is [[e, f], [g, h]].
    # M4 = d(g - ?)
    # Let's just use the 7 terms that work.
    # I will use the following coefficients for 2x2x2:
    # (U, V, W)
    # 1: (a+d, e+h, C00+1, C01-1, C10-1, C11-1)
    # 2: (c+d, e, C10+1, C11+1)
    # 3: (a, f-h, C01+1)
    # 4: (d, g-i, C10-1, C11+1) -- Assuming B[2,0]=g, B[2,1]=i?
    # Let's rename B[2,0]=g, B[2,1]=h.
    # Then M4 = d(g - ?)
    # Actually, let's just use schoolbook for the 2x2x2 part to be safe?
    # No, Strassen is 7 vs 4.
    # I will use the explicit Strassen implementation.
    
    # Let's restart the Strassen part carefully.
    # A = [[a, b], [c, d]]
    # B = [[e, f], [g, h]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-i) -> This is wrong if B is [[e,f],[g,h]].
    # Correct M4 is d(g - ?)
    # Let's use:
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-i) -> Let's assume the second row of B is [i, j].
    # So B = [[e, f], [i, j]].
    # Then g=i, h=j.
    # M4 = d(i - ?)
    # M5 = (a+b)j
    # M6 = (c+a)(i-e)
    # M7 = (b+d)(j-f)
    
    # Let's map:
    # e = B[1,0] = V[5]
    # f = B[1,1] = V[6]
    # i = B[2,0] = V[10]
    # j = B[2,1] = V[11]
    
    # M1: (a+d)(e+j) -> U[1]+U[5], V[5]+V[11]
    # W: C00+1, C01-1, C10-1, C11-1
    U[1][col] += 1; U[5][col] += 1
    V[5][col] += 1; V[11][col] += 1
    W[0][col] += 1; W[2][col] -= 1; W[1][col] -= 1; W[3][col] -= 1
    col += 1
    
    # M2: (c+d)e -> U[4]+U[5], V[5]
    # W: C10+1, C11+1
    U[4][col] += 1; U[5][col] += 1
    V[5][col] += 1
    W[1][col] += 1; W[3][col] += 1
    col += 1
    
    # M3: a(f-j) -> U[1], V[6]-V[11]
    # W: C01+1
    U[1][col] += 1
    V[6][col] += 1; V[11][col] -= 1
    W[2][col] += 1
    col += 1
    
    # M4: d(i-f) -> U[5], V[10]-V[6]
    # W: C10-1, C11+1
    U[5][col] += 1
    V[10][col] += 1; V[6][col] -= 1
    W[1][col] -= 1; W[3][col] += 1
    col += 1
    
    # M5: (a+b)j -> U[1]+U[2], V[11]
    # W: C01+1, C11+1
    U[1][col] += 1; U[2][col] += 1
    V[11][col] += 1
    W[2][col] += 1; W[3][col] += 1
    col += 1
    
    # M6: (c+a)(i-e) -> U[4]+U[1], V[10]-V[5]
    # W: C10+1
    U[4][col] += 1; U[1][col] += 1
    V[10][col] += 1; V[5][col] -= 1
    W[1][col] += 1
    col += 1
    
    # M7: (b+d)(j-f) -> U[2]+U[5], V[11]-V[6]
    # W: C00+1
    U[2][col] += 1; U[5][col] += 1
    V[11][col] += 1; V[6][col] -= 1
    W[0][col] += 1
    col += 1
    
    # col is now 10 + 7 = 17

    # Part 2b: A[:, 1:3] * B[1:3, 2:5] (2x2 x 2x3)
    # This is rank 11.
    # Split B_right (2x3) into B_r1 (2x2) and B_r2 (2x1).
    # B_r1 cols 2,3. B_r2 col 4.
    
    # B_r1: cols 2,3.
    # e' = B[1,2] = V[7]
    # f' = B[1,3] = V[8]
    # i' = B[2,2] = V[12]
    # j' = B[2,3] = V[13]
    
    # C outputs for cols 2,3:
    # C[0,2] -> W[2*2+0] = W[4]
    # C[0,3] -> W[3*2+0] = W[6]
    # C[1,2] -> W[2*2+1] = W[5]
    # C[1,3] -> W[3*2+1] = W[7]
    
    # Use same Strassen logic as above.
    
    # M1'
    U[1][col] += 1; U[5][col] += 1
    V[7][col] += 1; V[13][col] += 1
    W[4][col] += 1; W[6][col] -= 1; W[5][col] -= 1; W[7][col] -= 1
    col += 1
    
    # M2'
    U[4][col] += 1; U[5][col] += 1
    V[7][col] += 1
    W[5][col] += 1; W[7][col] += 1
    col += 1
    
    # M3'
    U[1][col] += 1
    V[8][col] += 1; V[13][col] -= 1
    W[6][col] += 1
    col += 1
    
    # M4'
    U[5][col] += 1
    V[12][col] += 1; V[8][col] -= 1
    W[5][col] -= 1; W[7][col] += 1
    col += 1
    
    # M5'
    U[1][col] += 1; U[2][col] += 1
    V[13][col] += 1
    W[6][col] += 1; W[7][col] += 1
    col += 1
    
    # M6'
    U[4][col] += 1; U[1][col] += 1
    V[12][col] += 1; V[7][col] -= 1
    W[5][col] += 1
    col += 1
    
    # M7'
    U[2][col] += 1; U[5][col] += 1
    V[13][col] += 1; V[8][col] -= 1
    W[4][col] += 1
    col += 1
    
    # col is now 17 + 7 = 24

    # B_r2: col 4.
    # e'' = B[1,4] = V[9]
    # i'' = B[2,4] = V[14]
    # C outputs:
    # C[0,4] -> W[4*2+0] = W[8]
    # C[1,4] -> W[4*2+1] = W[9]
    
    # Schoolbook for 2x2 x 2x1.
    # C[0,4] = a e'' + b i''
    # C[1,4] = c e'' + d i''
    
    # Term 1: a e''
    U[1][col] += 1
    V[9][col] += 1
    W[8][col] += 1
    col += 1
    
    # Term 2: b i''
    U[2][col] += 1
    V[14][col] += 1
    W[8][col] += 1
    col += 1
    
    # Term 3: c e''
    U[4][col] += 1
    V[9][col] += 1
    W[9][col] += 1
    col += 1
    
    # Term 4: d i''
    U[5][col] += 1
    V[14][col] += 1
    W[9][col] += 1
    col += 1
    
    # col is now 24 + 4 = 28

    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
