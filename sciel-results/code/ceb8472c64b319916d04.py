"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 26
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add a term: U[i*m+j] += u, V[j*p+k] += v, W[k*n+i] += w
    def add_term(i, j, k, u_val, v_val, w_val, col):
        U[i * m + j][col] += u_val
        V[j * p + k][col] += v_val
        W[k * n + i][col] += w_val

    col = 0
    
    # --- Strassen 2x2 for top-left block ---
    # A indices: (0,0), (0,1), (1,0), (1,1) -> a, b, c, d
    # B indices: (0,0), (0,1), (1,0), (1,1) -> e, f, g, h
    # C indices: (0,0), (0,1), (1,0), (1,1)
    
    # M1 = (a+d)(e+h)
    # a=A00, d=A11, e=B00, h=B11
    add_term(0, 0, 0, 1, 1, 1, col) # a*e
    add_term(0, 0, 1, 1, 1, 0, col) # a*f (wait, Strassen uses combinations)
    # Actually, it's easier to define the 7 products explicitly as linear combinations in U and V.
    # The decomposition is sum of rank-1 tensors.
    # M1 = (A00+A11)(B00+B11) = A00B00 + A00B11 + A11B00 + A11B11
    # This single multiplication contributes to C00, C01, C10, C11.
    # In the tensor decomposition, each "multiplication" is a column c.
    # U and V contain the coefficients of the linear forms for A and B.
    # W contains the coefficients for C.
    
    # Let's list the 7 Strassen products and their contributions to C.
    # P1 = (A00+A11)(B00+B11)
    # P2 = (A10+A11)B00
    # P3 = A00(B11-B00)
    # P4 = A11(B10-B00)
    # P5 = (A00+A01)B11
    # P6 = (A10-A11)(B01+B11)
    # P7 = (A01-A11)(B00+B10)
    
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    
    # We need to map these to U, V, W.
    # For each product Pk, we define a column.
    # U entries are coefficients of A_ij.
    # V entries are coefficients of B_jk.
    # W entries are coefficients of C_ik.
    
    # P1
    U[0*3+0][col] = 1; U[1*3+1][col] = 1
    V[0*3+0][col] = 1; V[1*3+1][col] = 1
    W[0*3+0][col] = 1; W[1*3+1][col] = 1; W[0*3+1][col] = 0; W[1*3+0][col] = 0
    # Wait, W must be linear.
    # P1 contributes 1 to C00 and 1 to C11.
    # So W[0*3+0] += 1, W[1*3+1] += 1.
    # But Strassen formulas have signs.
    # C00 = P1 + P4 - P5 + P7
    # So for P1, W[0*3+0] = 1.
    # For P1, W[1*3+1] = 1.
    # For P1, W[0*3+1] = 0.
    # For P1, W[1*3+0] = 0.
    # Let's refine W assignments for each Pk.
    
    # Reset W for col 0 (P1)
    W[0*3+0][col] = 1
    W[1*3+1][col] = 1
    # U, V for P1:
    # U[0,0]=1, U[1,1]=1
    # V[0,0]=1, V[1,1]=1
    col += 1
    
    # P2 = (A10+A11)B00
    U[1*3+0][col] = 1; U[1*3+1][col] = 1
    V[0*3+0][col] = 1
    W[1*3+0][col] = 1 # C10 += P2
    W[1*3+1][col] = -1 # C11 -= P2
    col += 1
    
    # P3 = A00(B11-B00)
    U[0*3+0][col] = 1
    V[1*3+1][col] = 1; V[0*3+1][col] = -1
    W[0*3+1][col] = 1 # C01 += P3
    W[1*3+1][col] = 1 # C11 += P3
    col += 1
    
    # P4 = A11(B10-B00)
    U[1*3+1][col] = 1
    V[1*3+0][col] = 1; V[0*3+0][col] = -1
    W[0*3+0][col] = 1 # C00 += P4
    W[1*3+0][col] = 1 # C10 += P4
    col += 1
    
    # P5 = (A00+A01)B11
    U[0*3+0][col] = 1; U[0*3+1][col] = 1
    V[1*3+1][col] = 1
    W[0*3+0][col] = -1 # C00 -= P5
    W[0*3+1][col] = 1 # C01 += P5
    col += 1
    
    # P6 = (A10-A11)(B01+B11)
    U[1*3+0][col] = 1; U[1*3+1][col] = -1
    V[0*3+1][col] = 1; V[1*3+1][col] = 1
    W[1*3+1][col] = 1 # C11 += P6
    col += 1
    
    # P7 = (A01-A11)(B00+B10)
    U[0*3+1][col] = 1; U[1*3+1][col] = -1
    V[0*3+0][col] = 1; V[1*3+0][col] = 1
    W[0*3+0][col] = 1 # C00 += P7
    col += 1
    
    # --- Remaining 19 terms ---
    # 1. u z^T (4 terms)
    # C00 += A02 B20
    U[0*3+2][col] = 1; V[2*3+0][col] = 1; W[0*3+0][col] = 1; col += 1
    # C01 += A02 B21
    U[0*3+2][col] = 1; V[2*3+1][col] = 1; W[0*3+1][col] = 1; col += 1
    # C10 += A12 B20
    U[1*3+2][col] = 1; V[2*3+0][col] = 1; W[1*3+0][col] = 1; col += 1
    # C11 += A12 B21
    U[1*3+2][col] = 1; V[2*3+1][col] = 1; W[1*3+1][col] = 1; col += 1
    
    # 2. A2x2 w (4 terms)
    # C02 = A00 B02 + A01 B12
    U[0*3+0][col] = 1; V[0*3+2][col] = 1; W[0*3+2][col] = 1; col += 1
    U[0*3+1][col] = 1; V[1*3+2][col] = 1; W[0*3+2][col] = 1; col += 1
    # C12 = A10 B02 + A11 B12
    U[1*3+0][col] = 1; V[0*3+2][col] = 1; W[1*3+2][col] = 1; col += 1
    U[1*3+1][col] = 1; V[1*3+2][col] = 1; W[1*3+2][col] = 1; col += 1
    
    # 3. u b (2 terms)
    # C02 += A02 B22
    U[0*3+2][col] = 1; V[2*3+2][col] = 1; W[0*3+2][col] = 1; col += 1
    # C12 += A12 B22
    U[1*3+2][col] = 1; V[2*3+2][col] = 1; W[1*3+2][col] = 1; col += 1
    
    # 4. v^T B2x2 (4 terms)
    # C20 = A20 B00 + A21 B10
    U[2*3+0][col] = 1; V[0*3+0][col] = 1; W[2*3+0][col] = 1; col += 1
    U[2*3+1][col] = 1; V[1*3+0][col] = 1; W[2*3+0][col] = 1; col += 1
    # C21 = A20 B01 + A21 B11
    U[2*3+0][col] = 1; V[0*3+1][col] = 1; W[2*3+1][col] = 1; col += 1
    U[2*3+1][col] = 1; V[1*3+1][col] = 1; W[2*3+1][col] = 1; col += 1
    
    # 5. a z^T (2 terms)
    # C20 += A22 B20
    U[2*3+2][col] = 1; V[2*3+0][col] = 1; W[2*3+0][col] = 1; col += 1
    # C21 += A22 B21
    U[2*3+2][col] = 1; V[2*3+1][col] = 1; W[2*3+1][col] = 1; col += 1
    
    # 6. v^T w (2 terms)
    # C22 = A20 B02 + A21 B12
    U[2*3+0][col] = 1; V[0*3+2][col] = 1; W[2*3+2][col] = 1; col += 1
    U[2*3+1][col] = 1; V[1*3+2][col] = 1; W[2*3+2][col] = 1; col += 1
    
    # 7. a b (1 term)
    # C22 += A22 B22
    U[2*3+2][col] = 1; V[2*3+2][col] = 1; W[2*3+2][col] = 1; col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
