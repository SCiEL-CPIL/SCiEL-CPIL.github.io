"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import sys

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    # Rank-26 decomposition using Strassen for 2x2 core
    n, m, p = DIMENSIONS
    rank = 26
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add a term
    # U indices: i*m + j
    # V indices: j*p + k
    # W indices: k*n + i
    def add_term(u_coeffs, v_coeffs, w_coeffs, col):
        # u_coeffs: dict { (i,j): val }
        # v_coeffs: dict { (j,k): val }
        # w_coeffs: dict { (k,i): val }
        for (i, j), val in u_coeffs.items():
            U[i * m + j][col] += val
        for (j, k), val in v_coeffs.items():
            V[j * p + k][col] += val
        for (k, i), val in w_coeffs.items():
            W[k * n + i][col] += val

    col = 0
    
    # 1. P0 = (A00 + A11) * (B00 + B11)
    add_term({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, col)
    col += 1
    
    # 2. P1 = (A10 + A11) * B00
    add_term({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}, col)
    col += 1
    
    # 3. P2 = A00 * (B11 - B00)
    add_term({(0,0): 1}, {(1,1): 1, (0,0): -1}, {(1,1): 1}, col)
    col += 1
    
    # 4. P3 = (A11 - A00) * (B01 + B11)
    add_term({(1,1): 1, (0,0): -1}, {(0,1): 1, (1,1): 1}, {(0,0): 1, (1,0): 1, (1,1): 1}, col)
    col += 1
    
    # 5. P4 = (A01 + A11) * (B10 - B00)
    add_term({(0,1): 1, (1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,1): 1}, col)
    col += 1
    
    # 6. P5 = (A10 - A00) * (B11 + B10)
    add_term({(1,0): 1, (0,0): -1}, {(1,1): 1, (1,0): 1}, {(0,0): -1}, col)
    col += 1
    
    # 7. P6 = (A11 + A01) * (B01 - B10)
    add_term({(1,1): 1, (0,1): 1}, {(0,1): 1, (1,0): -1}, {(0,0): 1, (0,1): 1}, col)
    col += 1
    
    # 8. A02 * B02 -> C00
    add_term({(0,2): 1}, {(0,2): 1}, {(0,0): 1}, col)
    col += 1
    
    # 9. A02 * B12 -> C01
    add_term({(0,2): 1}, {(1,2): 1}, {(0,1): 1}, col)
    col += 1
    
    # 10. A12 * B02 -> C10
    add_term({(1,2): 1}, {(0,2): 1}, {(1,0): 1}, col)
    col += 1
    
    # 11. A12 * B12 -> C11
    add_term({(1,2): 1}, {(1,2): 1}, {(1,1): 1}, col)
    col += 1
    
    # 12. A00 * B02 -> C02
    add_term({(0,0): 1}, {(0,2): 1}, {(0,2): 1}, col)
    col += 1
    
    # 13. A01 * B12 -> C02
    add_term({(0,1): 1}, {(1,2): 1}, {(0,2): 1}, col)
    col += 1
    
    # 14. A10 * B02 -> C12
    add_term({(1,0): 1}, {(0,2): 1}, {(1,2): 1}, col)
    col += 1
    
    # 15. A11 * B12 -> C12
    add_term({(1,1): 1}, {(1,2): 1}, {(1,2): 1}, col)
    col += 1
    
    # 16. A02 * B22 -> C02
    add_term({(0,2): 1}, {(2,2): 1}, {(0,2): 1}, col)
    col += 1
    
    # 17. A12 * B22 -> C12
    add_term({(1,2): 1}, {(2,2): 1}, {(1,2): 1}, col)
    col += 1
    
    # 18. A20 * B00 -> C20
    add_term({(2,0): 1}, {(0,0): 1}, {(2,0): 1}, col)
    col += 1
    
    # 19. A21 * B10 -> C20
    add_term({(2,1): 1}, {(1,0): 1}, {(2,0): 1}, col)
    col += 1
    
    # 20. A20 * B01 -> C21
    add_term({(2,0): 1}, {(0,1): 1}, {(2,1): 1}, col)
    col += 1
    
    # 21. A21 * B11 -> C21
    add_term({(2,1): 1}, {(1,1): 1}, {(2,1): 1}, col)
    col += 1
    
    # 22. A22 * B02 -> C20
    add_term({(2,2): 1}, {(0,2): 1}, {(2,0): 1}, col)
    col += 1
    
    # 23. A22 * B12 -> C21
    add_term({(2,2): 1}, {(1,2): 1}, {(2,1): 1}, col)
    col += 1
    
    # 24. A20 * B20 -> C22
    add_term({(2,0): 1}, {(2,0): 1}, {(2,2): 1}, col)
    col += 1
    
    # 25. A21 * B21 -> C22
    add_term({(2,1): 1}, {(2,1): 1}, {(2,2): 1}, col)
    col += 1
    
    # 26. A22 * B22 -> C22
    add_term({(2,2): 1}, {(2,2): 1}, {(2,2): 1}, col)
    col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
