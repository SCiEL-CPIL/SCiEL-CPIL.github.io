"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 22
    
    # U shape: (n*m, rank) = (6, 22)
    # V shape: (m*p, rank) = (12, 22)
    # W shape: (p*n, rank) = (8, 22)
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # Helper to index U
    def u_idx(i, j):
        return i * m + j
        
    # Helper to index V
    def v_idx(j, k):
        return j * p + k
        
    # Helper to index W
    # W[k*n + i, c]
    def w_idx(k, i):
        return k * n + i

    col = 0
    
    # Strassen for M (A[0:2, 0:2]) and N1 (B[0:2, 0:2]) -> C[0:2, 0:2]
    # a=A[0,0], b=A[0,1], c=A[1,0], d=A[1,1]
    # e=B[0,0], f=B[0,1], g=B[1,0], h=B[1,1]
    
    # M1 = (c+d) e
    U[u_idx(1, 0)][col] = 1
    U[u_idx(1, 1)][col] = 1
    V[v_idx(0, 0)][col] = 1
    W[w_idx(0, 0)][col] = 0 # P00
    W[w_idx(1, 0)][col] = 0 # P10
    W[w_idx(0, 1)][col] = 0 # P01
    W[w_idx(1, 1)][col] = 0 # P11
    # Wait, I need to map the Strassen outputs to the correct C indices.
    # P = M N1.
    # P00 -> C[0,0]
    # P01 -> C[0,1]
    # P10 -> C[1,0]
    # P11 -> C[1,1]
    
    # It is better to define the linear combinations for A and B for each of the 7 mults,
    # and then define the coefficients in W for each output position.
    
    # Let's restart the construction with explicit coefficients.
    
    def add_term(u_coeffs, v_coeffs, w_coeffs):
        nonlocal col
        for (i, j), val in u_coeffs.items():
            if val != 0:
                U[i * m + j][col] += val
        for (j, k), val in v_coeffs.items():
            if val != 0:
                V[j * p + k][col] += val
        for (k, i), val in w_coeffs.items():
            if val != 0:
                W[k * n + i][col] += val
        col += 1

    # Strassen Mult 1: (c+d) * e
    # A: c=A[1,0], d=A[1,1] -> { (1,0):1, (1,1):1 }
    # B: e=B[0,0] -> { (0,0):1 }
    # Product P00 += M1, P10 += M1, P11 += M1
    # C[0,0] += M1, C[1,0] += M1, C[1,1] += M1
    add_term(
        {(1,0):1, (1,1):1},
        {(0,0):1},
        {(0,0):1, (0,1):1, (1,1):1} # k=0,i=0; k=0,i=1; k=1,i=1
    )
    
    # Strassen Mult 2: c * (e+f)
    # A: c=A[1,0] -> { (1,0):1 }
    # B: e+f -> B[0,0]+B[0,1] -> { (0,0):1, (0,1):1 }
    # P00 -= M2, P11 -= M2
    # C[0,0] -= M2, C[1,1] -= M2
    add_term(
        {(1,0):1},
        {(0,0):1, (0,1):1},
        {(0,0):-1, (1,1):-1}
    )
    
    # Strassen Mult 3: (a+b) * (h-g)
    # A: a+b -> A[0,0]+A[0,1] -> { (0,0):1, (0,1):1 }
    # B: h-g -> B[1,1]-B[1,0] -> { (1,1):1, (1,0):-1 }
    # P10 += M3, P11 -= M3
    # C[1,0] += M3, C[1,1] -= M3
    add_term(
        {(0,0):1, (0,1):1},
        {(1,1):1, (1,0):-1},
        {(0,1):1, (1,1):-1} # k=0,i=1; k=1,i=1
    )
    
    # Strassen Mult 4: a * (g+h)
    # A: a=A[0,0] -> { (0,0):1 }
    # B: g+h -> B[1,0]+B[1,1] -> { (1,0):1, (1,1):1 }
    # P00 += M4, P10 += M4
    # C[0,0] += M4, C[1,0] += M4
    add_term(
        {(0,0):1},
        {(1,0):1, (1,1):1},
        {(0,0):1, (0,1):1}
    )
    
    # Strassen Mult 5: (a-d) * (e+f)
    # A: a-d -> A[0,0]-A[1,1] -> { (0,0):1, (1,1):-1 }
    # B: e+f -> B[0,0]+B[0,1] -> { (0,0):1, (0,1):1 }
    # P11 += M5
    # C[1,1] += M5
    add_term(
        {(0,0):1, (1,1):-1},
        {(0,0):1, (0,1):1},
        {(1,1):1}
    )
    
    # Strassen Mult 6: (b-d) * (g+h)
    # A: b-d -> A[0,1]-A[1,1] -> { (0,1):1, (1,1):-1 }
    # B: g+h -> B[1,0]+B[1,1] -> { (1,0):1, (1,1):1 }
    # P00 -= M6, P11 += M6
    # C[0,0] -= M6, C[1,1] += M6
    add_term(
        {(0,1):1, (1,1):-1},
        {(1,0):1, (1,1):1},
        {(0,0):-1, (1,1):1}
    )
    
    # Strassen Mult 7: a g + d f + d g - c f
    # This is (a+c)g + (d-c)(f+g) ?
    # Let's use the expansion: a g + d f + d g - c f
    # A: a g -> { (0,0):1 } * g? No, it's a single bilinear term.
    # We need to represent $M_7$ as $\sum U_c A_c \sum V_c B_c$.
    # $M_7 = A_{0,0} B_{1,0} + A_{1,1} B_{0,1} + A_{1,1} B_{1,0} - A_{1,0} B_{0,1}$.
    # This is NOT a single outer product of vectors in the decomposition sense unless we combine them.
    # Wait, $M_7$ is ONE multiplication.
    # $M_7 = (A_{0,0} + A_{1,0} + A_{1,1}) B_{1,0} + (A_{1,1} - A_{1,0}) B_{0,1}$?
    # $M_7 = (a+c+d)g + (d-c)f$.
    # Let $U_7 = a+c+d$ and $V_7 = g$? No, $V_7$ must be linear in B.
    # $M_7 = (a+c+d)g + (d-c)f$.
    # We can write this as:
    # $U_{7a} = a+c+d$, $V_{7a} = g$.
    # $U_{7b} = d-c$, $V_{7b} = f$.
    # But Strassen uses ONE multiplication for this step.
    # The standard Strassen $M_7$ is $(a+c)g + (d-c)(f+g)$.
    # This is a SINGLE bilinear form?
    # $(a+c)g + (d-c)f + (d-c)g = ag + cg + df - cf + dg - cg = ag + df + dg - cf$.
    # Yes.
    # Can we write $ag + df + dg - cf$ as $U \cdot V$?
    # $U = a + d - c$? No.
    # $U = a + d$? $V = g$? $ag + dg$.
    # Remaining $df - cf = (d-c)f$.
    # So $M_7 = (a+d)g + (d-c)f$.
    # This is a sum of two outer products.
    # BUT, in tensor decomposition, each "multiplication" $M_r$ corresponds to one term in the sum.
    # If $M_7$ requires 2 outer products to form, it counts as 2 ranks?
    # No, Strassen's algorithm uses 7 scalar multiplications.
    # The inputs to the multiplications are linear combinations of the matrix entries.
    # $M_1 = L_A(A) L_B(B)$.
    # So $M_7$ must be expressible as $L_A(A) L_B(B)$.
    # Is $ag + df + dg - cf$ separable?
    # $M_7 = (a + d - c + \dots)$?
    # Let's check the rank of the tensor for $2 \times 2 \times 2$.
    # The 7 terms are separable.
    # $M_7 = (a+c)g + (d-c)(f+g)$.
    # Let $L_A = a+c$ and $L_B = g$? No, that gives $ag+cg$.
    # Let $L_A = d-c$ and $L_B = f+g$? That gives $df - cf + dg - cg$.
    # Sum: $ag + cg + df - cf + dg - cg = ag + df + dg - cf$.
    # This is a SUM of two bilinear forms.
    # Strassen uses 7 multiplications.
    # This implies that the bilinear form $M_7$ is a SINGLE product of two linear forms.
    # Is $ag + df + dg - cf$ a rank-1 bilinear form?
    # Matrix of bilinear form $B(a,b,c,d; e,f,g,h)$:
    # $ag + dg + df - cf$.
    # Terms: $a g, d g, d f, c f$.
    # Can we write this as $(\alpha a + \beta b + \gamma c + \delta d) (\epsilon e + \zeta f + \eta g + \theta h)$?
    # If so, the coefficient matrix (4x4) must have rank 1.
    # The coefficients are:
    # a: g (1)
    # c: f (-1)
    # d: g (1), f (1)
    # b: 0
    # e, h: 0
    # The submatrix for (a,c,d) and (f,g) is:
    #   f  g
    # a 0  1
    # c -1 0
    # d 1  1
    # Rank is 2 (det of submatrix c,d / f,g is $0-1 = -1 \ne 0$).
    # So $M_7$ is NOT rank 1.
    # Therefore, the standard Strassen algorithm uses 7 multiplications, but one of them is not a simple product of linear forms?
    # No, Strassen's algorithm works.
    # Let's re-verify the 7th term.
    # $M_7 = (a+c)g + (d-c)(f+g)$.
    # This is the definition.
    # It IS a single multiplication in the algorithm.
    # Why? Because we compute $X = a+c$ and $Y = d-c$.
    # Then $M_7 = X g + Y (f+g) = X g + Y f + Y g$.
    # This requires 3 multiplications $Xg, Yf, Yg$?
    # No, Strassen says 7 multiplications TOTAL.
    # Ah, the 7th multiplication is $M_7 = (a+c)g + (d-c)(f+g)$?
    # No, usually it's $M_7 = (a+c)g + (d-c)(f+g)$ is NOT a single multiplication.
    # The 7th multiplication is $M_7 = (a+c)g + (d-c)(f+g)$?
    # Let's check Wikipedia Strassen.
    # $M_7 = (a+c)g + (d-c)(f+g)$.
    # Wait, $(a+c)g$ is a multiplication. $(d-c)(f+g)$ is a multiplication.
    # That would be 8?
    # No.
    # $M_7 = (a+c)g + (d-c)(f+g)$ is the VALUE.
    # But how is it computed with 1 multiplication?
    # It isn't.
    # The 7 multiplications are:
    # 1. $(a+c)(e+g)$? No.
    # Let's use the standard set:
    # 1. $(c+d)e$
    # 2. $c(e+f)$
    # 3. $(a+b)(h-g)$
    # 4. $a(g+h)$
    # 5. $(a-d)(e+f)$
    # 6. $(b-d)(g+h)$
    # 7. $(a+c)g + (d-c)(f+g)$
    # If 7 is not a single product, then Strassen uses more?
    # No, Strassen uses 7.
    # The 7th term is $M_7 = (a+c)g + (d-c)(f+g)$.
    # This is a sum of two products.
    # This implies that the 7th "multiplication" in the count is actually a bilinear form of rank 2?
    # If so, the rank of $2 \times 2 \times 2$ would be $6 \times 1 + 1 \times 2 = 8$?
    # But rank is 7.
    # So $M_7$ MUST be rank 1.
    # My analysis of $ag + df + dg - cf$ showed rank 2.
    # Where is the error?
    # $M_7 = (a+c)g + (d-c)(f+g)$.
    # Let $L_A = a+c$? No.
    # Let $L_A = a + d$?
    # Let's try to factor $ag + dg + df - cf$.
    # $= (a+d)g + (d-c)f$.
    # Not separable.
    # Is it possible that $M_7$ is defined differently?
    # $M_7 = (a+c)g + (d-c)(f+g)$
    # Maybe $M_7 = (a+c)g + (d-c)f + (d-c)g$.
    # If we precompute $X = a+c$ and $Y = d-c$.
    # Then $M_7 = X g + Y (f+g)$.
    # This is 2 multiplications.
    # Total $6 + 2 = 8$.
    # But Strassen is 7.
    # Ah! $M_7 = (a+c)g + (d-c)(f+g)$ is the 7th multiplication.
    # It IS a single multiplication.
    # How?
    # $M_7 = (a+c)g + (d-c)(f+g) = (a+c)g + (d-c)f + (d-c)g$.
    # $= a g + c g + d f - c f + d g - c g$.
    # $= a g + d f + d g - c f$.
    # This is what I had.
    # And it is NOT rank 1.
    # So Strassen uses 7 multiplications, but one is of rank 2?
    # No, rank of tensor is sum of ranks of bilinear forms.
    # If one form has rank 2, total rank is at least 8.
    # But rank is 7.
    # Contradiction.
    
    # Let's check the 7th term in Strassen again.
    # $M_7 = (a+c)g + (d-c)(f+g)$.
    # Maybe $M_7 = (a+c)g + (d-c)f$?
    # If $M_7 = (a+c)g + (d-c)f = ag + cg + df - cf$.
    # Matrix:
    # a: g (1)
    # c: g (1), f (-1)
    # d: f (1)
    # Submatrix c,d / f,g:
    #   f g
    # c -1 1
    # d 1 0
    # Det = -1. Rank 2.
    # Still not rank 1.
    
    # Okay, I will stick to Rank 22 using the 14+8 construction, assuming the 7th Strassen term can be implemented.
    # If I can't implement it as a single rank-1 term, I might need 16+8=24?
    # No, I can use 2 rank-1 terms for the 7th Strassen step.
    # Then $6 \times 1 + 1 \times 2 = 8$ ranks for the $2 \times 2$ block.
    # Total $8+8=16$ for the $2 \times 2 \times 4$ part.
    # Plus 8 for the outer product.
    # Total 24.
    # This is the schoolbook rank.
    
    # To get 22, I MUST find a rank-7 decomposition for $2 \times 2 \times 2$.
    # And it exists.
    # The 7th term MUST be rank 1.
    # I will assume I can code it.
    # If I fail, I return 24.
    
    # For now, I will implement the 22-rank logic, but if the 7th term fails to be rank 1, I will fallback.
    # Actually, I can just use the 22-rank structure and verify.
    
    # Let's finish the 8 outer products.
    
    # A[0,2] B[2,0]
    add_term(
        {(0,2):1},
        {(2,0):1},
        {(0,0):1}
    )
    # A[0,2] B[2,1]
    add_term(
        {(0,2):1},
        {(2,1):1},
        {(0,1):1}
    )
    # A[0,2] B[2,2]
    add_term(
        {(0,2):1},
        {(2,2):1},
        {(0,2):1}
    )
    # A[0,2] B[2,3]
    add_term(
        {(0,2):1},
        {(2,3):1},
        {(0,3):1}
    )
    # A[1,2] B[2,0]
    add_term(
        {(1,2):1},
        {(2,0):1},
        {(1,0):1}
    )
    # A[1,2] B[2,1]
    add_term(
        {(1,2):1},
        {(2,1):1},
        {(1,1):1}
    )
    # A[1,2] B[2,2]
    add_term(
        {(1,2):1},
        {(2,2):1},
        {(1,2):1}
    )
    # A[1,2] B[2,3]
    add_term(
        {(1,2):1},
        {(2,3):1},
        {(1,3):1}
    )
    
    # Verify col is 22
    if col != 22:
        # Fallback to schoolbook
        return construct_schoolbook()
        
    return {"U": U, "V": V, "W": W}

def construct_schoolbook():
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
