<<<<<<< SEARCH
def _try_rank_r(n, m, p, r, tensor):
    """Try to find a rank-r decomposition by fixing U and V and solving for W."""
    if r >= n * m * p:
        # Schoolbook
        U = [[Fraction(0)] * r for _ in range(n * m)]
        V = [[Fraction(0)] * r for _ in range(m * p)]
        W = [[Fraction(0)] * r for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = Fraction(1)
                    V[j * p + k][col] = Fraction(1)
                    W[k * n + i][col] = Fraction(1)
                    col += 1
        return U, V, W
    
    # Fix U and V to be specific matrices, then solve for W
    # U: n*m x r, V: m*p x r, W: p*n x r
    # Equation: for each (a,b,c): sum_col U[a,col]*V[b,col]*W[c,col] = T[a,b,c]
    
    # We need to choose U and V. Let's try a structured approach.
    # For rank 27, we have 3 fewer terms than schoolbook.
    
    # Strategy: Use the schoolbook U and V but with 3 columns removed/combined.
    # This is heuristic, but let's try.
    
    # Alternative: Use a random-like but deterministic pattern.
    # For reproducibility, use a fixed pattern.
    
    # Let's try: U has the same structure as schoolbook but with some entries
    # set to create dependencies.
    
    # For now, return None - we need a better strategy
    return None
=======
def _try_rank_r(n, m, p, r, tensor):
    """Try to find a rank-r decomposition by fixing U and V and solving for W."""
    if r >= n * m * p:
        # Schoolbook
        U = [[Fraction(0)] * r for _ in range(n * m)]
        V = [[Fraction(0)] * r for _ in range(m * p)]
        W = [[Fraction(0)] * r for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = Fraction(1)
                    V[j * p + k][col] = Fraction(1)
                    W[k * n + i][col] = Fraction(1)
                    col += 1
        return U, V, W
    
    # For rank < 30, try to find a decomposition.
    # We'll fix U and V to be specific matrices and solve for W.
    
    # U: (n*m) x r, V: (m*p) x r, W: (p*n) x r
    # The tensor equation: T[a,b,c] = sum_{col=0}^{r-1} U[a,col] * V[b,col] * W[c,col]
    
    # For fixed U and V, this is linear in W.
    # For each (a,b), we have: sum_c' ... wait, no.
    # For each (a,b,c): sum_col U[a,col]*V[b,col]*W[c,col] = T[a,b,c]
    
    # This means: for fixed a,b, the vector (sum_col U[a,col]*V[b,col]*W[c,col])_{c} must equal (T[a,b,c])_{c}
    # So: M_{a,b} * W = T_{a,b} where M_{a,b}[c,col] = U[a,col]*V[b,col]
    
    # So for each (a,b), we have a system: (n*m) * (m*p) systems, each with (p*n) equations and r unknowns (W[c,col] for fixed c... wait)
    
    # Actually, W has shape (p*n) x r. Let's index W[c,col].
    # For fixed (a,b): sum_col U[a,col]*V[b,col]*W[c,col] = T[a,b,c] for each c
    # This is: sum_col (U[a,col]*V[b,col]) * W[:,col] = T[a,b,:]
    # So: sum_col M_{a,b}[col] * W[:,col] = T[a,b,:] where M_{a,b}[col] = U[a,col]*V[b,col]
    
    # This is a linear combination of the columns of W equals T[a,b,:].
    # So the vector T[a,b,:] must be in the span of the columns of W.
    
    # But W is what we're solving for, so this is coupled.
    
    # Better approach: Flatten W into a vector of length (p*n)*r.
    # Then the equation becomes linear in this flattened vector.
    
    # Let's set up the full linear system.
    # Unknowns: W[c,col] for c in [0,p*n), col in [0,r) -> (p*n)*r unknowns
    # Equations: for each (a,b,c): sum_col U[a,col]*V[b,col]*W[c,col] = T[a,b,c]
    # That's (n*m)*(m*p)*(p*n) equations, but many are dependent.
    
    # For a given (a,b), the equations over c are:
    # For each c: sum_col U[a,col]*V[b,col]*W[c,col] = T[a,b,c]
    # This is (p*n) equations in the r unknowns W[c,col] for fixed c... no, W[c,col] varies with c.
    
    # I think I'm overcomplicating. Let me just build the full linear system.
    
    num_unknowns = (p * n) * r
    num_equations = (n * m) * (m * p) * (p * n)
    
    # This is too large. Let's be smarter.
    
    # For fixed (a,b), we have p*n equations:
    # sum_col U[a,col]*V[b,col]*W[c,col] = T[a,b,c] for c=0,...,p*n-1
    # This is a system in the unknowns W[c,col].
    # But W[c,col] appears only in the equation for that specific c.
    # So for each c, we have: sum_col U[a,col]*V[b,col]*W[c,col] = T[a,b,c]
    # This is r unknowns (W[c,0],...,W[c,r-1]) and 1 equation per (a,b,c).
    
    # So for fixed (a,b,c), we have 1 equation in r unknowns.
    # Total: (n*m)*(m*p)*(p*n) equations in (p*n)*r unknowns.
    
    # The system is underdetermined if (p*n)*r > (n*m)*(m*p)*(p*n), which is unlikely.
    # Actually (p*n)*r = 10*27 = 270, and (n*m)*(m*p)*(p*n) = 6*15*10 = 900.
    # So 900 equations in 270 unknowns - overdetermined!
    
    # So we need the system to be consistent. This is a strong constraint.
    
    # Let's build the system and solve it.
    
    # Index unknowns: W[c,col] -> index = c*r + col
    # For each (a,b,c), equation: sum_col U[a,col]*V[b,col]*W[c,col] = T[a,b,c]
    # Coefficient of W[c,col] in this equation: U[a,col]*V[b,col]
    
    # Build A (num_equations x num_unknowns) and b (num_equations)
    # But this is 900 x 270, which is manageable.
    
    A = [[Fraction(0)] * num_unknowns for _ in range(num_equations)]
    b = [Fraction(0)] * num_equations
    
    eq_idx = 0
    for a in range(n * m):
        for b_ in range(m * p):
            for c in range(p * n):
                for col in range(r):
                    coeff = U[a][col] * V[b_][col]
                    if coeff != 0:
                        A[eq_idx][c * r + col] = coeff
                b[eq_idx] = tensor[a, b_, c]
                eq_idx += 1
    
    # Solve Ax = b
    x = _solve_linear_system(A, b)
    if x is None:
        return None
    
    # Reconstruct W
    W = [[Fraction(0)] * r for _ in range(p * n)]
    for c in range(p * n):
        for col in range(r):
            W[c][col] = x[c * r + col]
    
    return U, V, W
>>>>>>> REPLACE

I realize I still need to define U and V before calling _solve_linear_system. Let me fix the structure.

