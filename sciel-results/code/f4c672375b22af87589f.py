"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 26
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add a term
    # U_idx: index in A (0..8)
    # V_idx: index in B (0..8)
    # W_idx: index in C (0..8) with convention k*n+i
    # coeff: coefficient for this term
    def add_term(u_idx, v_idx, w_idx, coeff, col):
        U[u_idx][col] += coeff
        V[v_idx][col] += coeff
        W[w_idx][col] += coeff

    col = 0
    
    # Strassen Terms for 2x2 Core (A11, A12, A21, A22) x (B11, B12, B21, B22)
    # A indices: 0, 1, 3, 4
    # B indices: 0, 1, 3, 4
    # C indices for 2x2 block:
    # C11 -> 0
    # C12 -> 3
    # C21 -> 1
    # C22 -> 4
    
    # M1 = (A11 + A22) * (B11 + B22)
    # U: A11(0), A22(4)
    # V: B11(0), B22(4)
    # W: C11(0), C12(3), C21(1), C22(4)
    # Coeffs in P:
    # P11: +1
    # P12: 0
    # P21: 0
    # P22: +1
    
    # We need to distribute M1 into the 4 C entries based on Strassen formula.
    # P11 = M1 + M4 - M5 + M7
    # P12 = M3 + M5
    # P21 = M2 + M4
    # P22 = M1 - M2 + M3 + M6
    
    # Term 1: M1
    # U: A11(0)=1, A22(4)=1
    # V: B11(0)=1, B22(4)=1
    # W: C11(0)=1, C22(4)=1
    add_term(0, 0, 0, 1, col)
    add_term(4, 0, 0, 1, col)
    add_term(0, 4, 0, 1, col)
    add_term(4, 4, 0, 1, col)
    
    add_term(0, 0, 4, 1, col)
    add_term(4, 0, 4, 1, col)
    add_term(0, 4, 4, 1, col)
    add_term(4, 4, 4, 1, col)
    col += 1

    # Term 2: M2 = (A21 + A22) * B11
    # U: A21(3)=1, A22(4)=1
    # V: B11(0)=1
    # W: C21(1)=1, C22(4)=-1
    add_term(3, 0, 1, 1, col)
    add_term(4, 0, 1, 1, col)
    add_term(3, 0, 4, -1, col)
    add_term(4, 0, 4, -1, col)
    col += 1

    # Term 3: M3 = A11 * (B12 - B22)
    # U: A11(0)=1
    # V: B12(1)=1, B22(4)=-1
    # W: C12(3)=1, C22(4)=1
    add_term(0, 1, 3, 1, col)
    add_term(0, 4, 3, -1, col)
    add_term(0, 1, 4, 1, col)
    add_term(0, 4, 4, -1, col)
    col += 1

    # Term 4: M4 = A22 * (B21 - B12)
    # U: A22(4)=1
    # V: B21(3)=1, B12(1)=-1
    # W: C11(0)=1, C21(1)=1
    add_term(4, 3, 0, 1, col)
    add_term(4, 1, 0, -1, col)
    add_term(4, 3, 1, 1, col)
    add_term(4, 1, 1, -1, col)
    col += 1

    # Term 5: M5 = (A11 + A21) * B22
    # U: A11(0)=1, A21(3)=1
    # V: B22(4)=1
    # W: C12(3)=1, C11(0)=-1
    add_term(0, 4, 3, 1, col)
    add_term(3, 4, 3, 1, col)
    add_term(0, 4, 0, -1, col)
    add_term(3, 4, 0, -1, col)
    col += 1

    # Term 6: M6 = (A12 - A22) * (B11 + B21)
    # U: A12(1)=1, A22(4)=-1
    # V: B11(0)=1, B21(3)=1
    # W: C22(4)=1
    add_term(1, 0, 4, 1, col)
    add_term(4, 0, 4, -1, col)
    add_term(1, 3, 4, 1, col)
    add_term(4, 3, 4, -1, col)
    col += 1

    # Term 7: M7 = (A21 - A12) * (B12 + B22)
    # U: A21(3)=1, A12(1)=-1
    # V: B12(1)=1, B22(4)=1
    # W: C11(0)=1
    add_term(3, 1, 0, 1, col)
    add_term(1, 1, 0, -1, col)
    add_term(3, 4, 0, 1, col)
    add_term(1, 4, 0, -1, col)
    col += 1

    # Scalar Terms
    # 1. A13 B31 -> C11
    add_term(2, 6, 0, 1, col); col += 1
    # 2. A13 B32 -> C12
    add_term(2, 7, 3, 1, col); col += 1
    # 3. A23 B31 -> C21
    add_term(5, 6, 1, 1, col); col += 1
    # 4. A23 B32 -> C22
    add_term(5, 7, 4, 1, col); col += 1
    # 5. A11 B13 -> C13
    add_term(0, 2, 6, 1, col); col += 1
    # 6. A12 B23 -> C13
    add_term(1, 5, 6, 1, col); col += 1
    # 7. A13 B33 -> C13
    add_term(2, 8, 6, 1, col); col += 1
    # 8. A21 B13 -> C23
    add_term(3, 2, 7, 1, col); col += 1
    # 9. A22 B23 -> C23
    add_term(4, 5, 7, 1, col); col += 1
    # 10. A23 B33 -> C23
    add_term(5, 8, 7, 1, col); col += 1
    # 11. A31 B11 -> C31
    add_term(6, 0, 2, 1, col); col += 1
    # 12. A32 B21 -> C31
    add_term(7, 3, 2, 1, col); col += 1
    # 13. A33 B31 -> C31
    add_term(8, 6, 2, 1, col); col += 1
    # 14. A31 B12 -> C32
    add_term(6, 1, 5, 1, col); col += 1
    # 15. A32 B22 -> C32
    add_term(7, 4, 5, 1, col); col += 1
    # 16. A33 B32 -> C32
    add_term(8, 7, 5, 1, col); col += 1
    # 17. A31 B13 -> C33
    add_term(6, 2, 8, 1, col); col += 1
    # 18. A32 B23 -> C33
    add_term(7, 5, 8, 1, col); col += 1
    # 19. A33 B33 -> C33
    add_term(8, 8, 8, 1, col); col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
