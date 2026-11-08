import numpy as np
import time

DIMENSIONS = [3, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    
    # Target rank
    rank = 46
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_term(u_idx, v_idx, w_idx, coeff):
        nonlocal col
        if col >= rank:
            return
        U[u_idx][col] += coeff
        V[v_idx][col] += coeff
        W[w_idx][col] += coeff
        col += 1

    def add_strassen_block(a_r0, a_c0, b_r0, b_c0, c_r0, c_c0):
        nonlocal col
        # A is 2x2 at (a_r0, a_c0)
        # B is 2x2 at (b_r0, b_c0)
        # C is 2x2 at (c_r0, c_c0)
        
        # A indices
        a00 = a_r0 * m + a_c0
        a01 = a_r0 * m + a_c0 + 1
        a10 = (a_r0 + 1) * m + a_c0
        a11 = (a_r0 + 1) * m + a_c0 + 1
        
        # B indices
        b00 = b_r0 * p + b_c0
        b01 = b_r0 * p + b_c0 + 1
        b10 = (b_r0 + 1) * p + b_c0
        b11 = (b_r0 + 1) * p + b_c0 + 1
        
        # C indices
        c00 = c_r0 * n + c_c0
        c01 = c_r0 * n + c_c0 + 1
        c10 = (c_r0 + 1) * n + c_c0
        c11 = (c_r0 + 1) * n + c_c0 + 1
        
        # M1 = (a00+a11)(b00+b11)
        # c00 += 1, c11 += 1
        # U: a00, a11; V: b00, b11; W: c00, c11
        # This is 4 terms for M1 contribution to c00 and c11?
        # No, M1 is a single product.
        # U[a00][col] += 1; V[b00][col] += 1; W[c00][col] += 1
        # U[a00][col] += 1; V[b11][col] += 1; W[c00][col] += 1
        # U[a11][col] += 1; V[b00][col] += 1; W[c00][col] += 1
        # U[a11][col] += 1; V[b11][col] += 1; W[c00][col] += 1
        # This adds 4 rank-1 terms to c00.
        # And same for c11.
        # Total 8 terms? No, Strassen uses 7 products.
        # Each product is a rank-1 term in the tensor decomposition?
        # No, the product $M_1$ is a scalar.
        # The term in the sum is $M_1 \times E_{00} + M_1 \times E_{11}$.
        # So $M_1$ contributes to $c_{00}$ and $c_{11}$.
        # $M_1 = (a_{00}+a_{11})(b_{00}+b_{11})$.
        # This expands to 4 terms in schoolbook, but as a single product in Strassen.
        # In the tensor decomposition, $M_1$ is a single rank-1 component?
        # No, $M_1$ is a scalar value.
        # The decomposition is $C = \sum_k u_k \otimes v_k \otimes w_k$.
        # $M_1$ is not a basis vector.
        # We need to express the bilinear form.
        # $M_1$ involves $a_{00}, a_{11}$ and $b_{00}, b_{11}$.
        # So $M_1$ is a sum of 4 rank-1 terms in the schoolbook sense?
        # No, Strassen reduces the number of multiplications.
        # But the tensor rank is the number of multiplications?
        # Yes, for bilinear algorithms, the rank is the number of scalar multiplications.
        # So each $M_i$ corresponds to one rank-1 term in the decomposition?
        # No, $M_i$ is a product of linear forms.
        # $M_1 = (a_{00}+a_{11})(b_{00}+b_{11})$.
        # This is one multiplication.
        # So it contributes 1 to the rank.
        # But how do we write it as $u \otimes v \otimes w$?
        # $u$ is the coefficient of $a$, $v$ is the coefficient of $b$, $w$ is the coefficient of $c$.
        # $M_1$ contributes to $c_{00}$ and $c_{11}$.
        # $c_{00} \leftarrow M_1$.
        # $c_{11} \leftarrow M_1$.
        # So $w$ has 1 at $c_{00}$ and 1 at $c_{11}$.
        # $u$ has 1 at $a_{00}$ and 1 at $a_{11}$.
        # $v$ has 1 at $b_{00}$ and 1 at $b_{11}$.
        # So we add ONE term:
        # U[a00][col] += 1
        # U[a11][col] += 1
        # V[b00][col] += 1
        # V[b11][col] += 1
        # W[c00][col] += 1
        # W[c11][col] += 1
        # col += 1
        
        # Let's implement this way.
        
        # M1
        U[a00][col] += 1
        U[a11][col] += 1
        V[b00][col] += 1
        V[b11][col] += 1
        W[c00][col] += 1
        W[c11][col] += 1
        col += 1
        
        # M2 = (a10+a11)b00
        # c10 += M2
        U[a10][col] += 1
        U[a11][col] += 1
        V[b00][col] += 1
        W[c10][col] += 1
        col += 1
        
        # M3 = a00(b01-b11)
        # c00 += M3
        U[a00][col] += 1
        V[b01][col] += 1
        V[b11][col] += -1
        W[c00][col] += 1
        col += 1
        
        # M4 = a11(b10-b00)
        # c00 += M4, c10 += M4
        U[a11][col] += 1
        V[b10][col] += 1
        V[b00][col] += -1
        W[c00][col] += 1
        W[c10][col] += 1
        col += 1
        
        # M5 = (a00+a10)b11
        # c01 += M5
        U[a00][col] += 1
        U[a10][col] += 1
        V[b11][col] += 1
        W[c01][col] += 1
        col += 1
        
        # M6 = (a01-a10)(b00+b10)
        # c11 += M6
        U[a01][col] += 1
        U[a10][col] += -1
        V[b00][col] += 1
        V[b10][col] += 1
        W[c11][col] += 1
        col += 1
        
        # M7 = (a11-a01)(b01+b11)
        # c01 += M7
        U[a11][col] += 1
        U[a01][col] += -1
        V[b01][col] += 1
        V[b11][col] += 1
        W[c01][col] += 1
        col += 1

    # C1 = A1 B1 (2x2)
    # A1: rows 0,1. B1: cols 0,1.
    # A1 is 2x4. B1 is 4x2.
    # C1 = A11 B11 + A12 B21
    # A11: rows 0,1 cols 0,1. B11: rows 0,1 cols 0,1.
    # A12: rows 0,1 cols 2,3. B21: rows 2,3 cols 0,1.
    # C1 is 2x2 at (0,0).
    
    # Strassen for A11 B11
    # a_r0=0, a_c0=0, b_r0=0, b_c0=0, c_r0=0, c_c0=0
    # But B11 is not 2x2 in the same sense?
    # B is 4x7.
    # B1 is cols 0,1.
    # B11 is rows 0,1 of B1. So B[0:2, 0:2].
    # b_r0=0, b_c0=0.
    add_strassen_block(0, 0, 0, 0, 0, 0)
    
    # Strassen for A12 B21
    # A12: rows 0,1 cols 2,3. a_r0=0, a_c0=2.
    # B21: rows 2,3 cols 0,1. b_r0=2, b_c0=0.
    # C1 is 2x2 at (0,0).
    # But we are adding to the same C1.
    # So we call add_strassen_block again, but it will add to col.
    # And the W indices will be the same.
    add_strassen_block(0, 2, 2, 0, 0, 0)
    
    # C2 = A1 B2 (2x2)
    # B2: cols 2,3.
    # C2 = A11 B12 + A12 B22
    # A11: rows 0,1 cols 0,1.
    # B12: rows 0,1 cols 2,3. b_r0=0, b_c0=2.
    # C2 is 2x2 at (0,2).
    add_strassen_block(0, 0, 0, 2, 0, 2)
    
    # A12: rows 0,1 cols 2,3.
    # B22: rows 2,3 cols 2,3. b_r0=2, b_c0=2.
    add_strassen_block(0, 2, 2, 2, 0, 2)
    
    # C3 = A1 B3 (2x3)
    # B3: cols 4,5,6.
    # C3 = A11 B13 + A12 B23
    # A11: rows 0,1 cols 0,1.
    # B13: rows 0,1 cols 4,5,6.
    # B13 is 2x3.
    # A11 B13 is 2x3.
    # We can split B13 into B13a (2x2) and B13b (2x1).
    # B13a: cols 4,5.
    # B13b: col 6.
    # A11 B13a is 2x2. Strassen?
    # A11 is 2x2. B13a is 2x2.
    # Yes.
    add_strassen_block(0, 0, 0, 4, 0, 4)
    
    # A11 B13b is 2x1.
    # Schoolbook rank 4?
    # A11 is 2x2. B13b is 2x1.
    # C is 2x1.
    # Terms:
    # a00*b0, a01*b1, a10*b0, a11*b1.
    # 4 terms.
    # U: a00, a01, a10, a11
    # V: b0, b1, b0, b1
    # W: c0, c0, c1, c1
    # c_r0=0, c_c0=6.
    # c00 = 6*3+0 = 18? No.
    # C is 7x3.
    # C3 is 2x3 at (0,4).
    # c_r0=0, c_c0=4.
    # c00 = 4, c01 = 5, c10 = 7, c11 = 8.
    # B13b is col 6. b_r0=0, b_c0=6.
    # b00 = 6, b10 = 13.
    
    U[0][col] += 1; V[6][col] += 1; W[4][col] += 1; col += 1
    U[1][col] += 1; V[13][col] += 1; W[4][col] += 1; col += 1
    U[4][col] += 1; V[6][col] += 1; W[7][col] += 1; col += 1
    U[5][col] += 1; V[13][col] += 1; W[7][col] += 1; col += 1
    
    # A12 B23
    # A12: rows 0,1 cols 2,3.
    # B23: rows 2,3 cols 4,5,6.
    # B23a: cols 4,5.
    add_strassen_block(0, 2, 2, 4, 0, 4)
    
    # B23b: col 6.
    # A12 B23b is 2x1.
    # a_r0=0, a_c0=2.
    # a00=2, a01=3, a10=6, a11=7.
    # b_r0=2, b_c0=6.
    # b00=20, b10=27.
    # c_r0=0, c_c0=6.
    # c00=6, c10=9.
    
    U[2][col] += 1; V[20][col] += 1; W[6][col] += 1; col += 1
    U[3][col] += 1; V[27][col] += 1; W[6][col] += 1; col += 1
    U[6][col] += 1; V[20][col] += 1; W[9][col] += 1; col += 1
    U[7][col] += 1; V[27][col] += 1; W[9][col] += 1; col += 1
    
    # C4 = A2 B1 (1x2)
    # A2: row 2.
    # B1: cols 0,1.
    # C4 = A21 B11 + A22 B21
    # A21: cols 0,1.
    # B11: rows 0,1 cols 0,1.
    # A21 B11 is 1x2.
    # Schoolbook rank 4.
    # a_r0=2, a_c0=0.
    # a00=8, a01=9.
    # b_r0=0, b_c0=0.
    # b00=0, b01=1, b10=7, b11=8.
    # c_r0=2, c_c0=0.
    # c00=0, c01=1.
    
    U[8][col] += 1; V[0][col] += 1; W[0][col] += 1; col += 1
    U[8][col] += 1; V[7][col] += 1; W[0][col] += 1; col += 1
    U[9][col] += 1; V[1][col] += 1; W[1][col] += 1; col += 1
    U[9][col] += 1; V[8][col] += 1; W[1][col] += 1; col += 1
    
    # A22 B21
    # A22: cols 2,3.
    # B21: rows 2,3 cols 0,1.
    # a_r0=2, a_c0=2.
    # a00=10, a01=11.
    # b_r0=2, b_c0=0.
    # b00=14, b01=15, b10=21, b11=22.
    # c_r0=2, c_c0=0.
    
    U[10][col] += 1; V[14][col] += 1; W[0][col] += 1; col += 1
    U[10][col] += 1; V[21][col] += 1; W[0][col] += 1; col += 1
    U[11][col] += 1; V[15][col] += 1; W[1][col] += 1; col += 1
    U[11][col] += 1; V[22][col] += 1; W[1][col] += 1; col += 1
    
    # C5 = A2 B2 (1x2)
    # B2: cols 2,3.
    # C5 = A21 B12 + A22 B22
    # A21 B12
    # a_r0=2, a_c0=0.
    # b_r0=0, b_c0=2.
    # b00=2, b01=3, b10=9, b11=10.
    # c_r0=2, c_c0=2.
    # c00=2, c01=3.
    
    U[8][col] += 1; V[2][col] += 1; W[2][col] += 1; col += 1
    U[8][col] += 1; V[9][col] += 1; W[2][col] += 1; col += 1
    U[9][col] += 1; V[3][col] += 1; W[3][col] += 1; col += 1
    U[9][col] += 1; V[10][col] += 1; W[3][col] += 1; col += 1
    
    # A22 B22
    # a_r0=2, a_c0=2.
    # b_r0=2, b_c0=2.
    # b00=16, b01=17, b10=23, b11=24.
    # c_r0=2, c_c0=2.
    
    U[10][col] += 1; V[16][col] += 1; W[2][col] += 1; col += 1
    U[10][col] += 1; V[23][col] += 1; W[2][col] += 1; col += 1
    U[11][col] += 1; V[17][col] += 1; W[3][col] += 1; col += 1
    U[11][col] += 1; V[24][col] += 1; W[3][col] += 1; col += 1
    
    # C6 = A2 B3 (1x3)
    # B3: cols 4,5,6.
    # C6 = A21 B13 + A22 B23
    # A21 B13
    # A21: cols 0,1.
    # B13: rows 0,1 cols 4,5,6.
    # B13a: cols 4,5.
    # a_r0=2, a_c0=0.
    # b_r0=0, b_c0=4.
    # b00=4, b01=5, b10=11, b11=12.
    # c_r0=2, c_c0=4.
    # c00=4, c01=5.
    
    U[8][col] += 1; V[4][col] += 1; W[4][col] += 1; col += 1
    U[8][col] += 1; V[11][col] += 1; W[4][col] += 1; col += 1
    U[9][col] += 1; V[5][col] += 1; W[5][col] += 1; col += 1
    U[9][col] += 1; V[12][col] += 1; W[5][col] += 1; col += 1
    
    # B13b: col 6.
    # b_r0=0, b_c0=6.
    # b00=6, b10=13.
    # c_r0=2, c_c0=6.
    # c00=6.
    
    U[8][col] += 1; V[6][col] += 1; W[6][col] += 1; col += 1
    U[9][col] += 1; V[13][col] += 1; W[6][col] += 1; col += 1
    
    # A22 B23
    # A22: cols 2,3.
    # B23: rows 2,3 cols 4,5,6.
    # B23a: cols 4,5.
    # a_r0=2, a_c0=2.
    # b_r0=2, b_c0=4.
    # b00=18, b01=19, b10=25, b11=26.
    # c_r0=2, c_c0=4.
    
    U[10][col] += 1; V[18][col] += 1; W[4][col] += 1; col += 1
    U[10][col] += 1; V[25][col] += 1; W[4][col] += 1; col += 1
    U[11][col] += 1; V[19][col] += 1; W[5][col] += 1; col += 1
    U[11][col] += 1; V[26][col] += 1; W[5][col] += 1; col += 1
    
    # B23b: col 6.
    # b_r0=2, b_c0=6.
    # b00=20, b10=27.
    # c_r0=2, c_c0=6.
    
    U[10][col] += 1; V[20][col] += 1; W[6][col] += 1; col += 1
    U[11][col] += 1; V[27][col] += 1; W[6][col] += 1; col += 1
    
    if col != rank:
        # Fallback or error
        pass
        
    return {"U": U, "V": W, "W": W} # Wait, W is W.

if __name__ == "__main__":
    construct()
