"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
import time
_start_time = time.time()

def construct():
    n, m, p = DIMENSIONS
    
    # Try to implement Laderman's rank-23 algorithm
    rank = 23
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def set_term(u_coeffs, v_coeffs, w_coeffs):
        for idx, c in u_coeffs.items():
            U[idx][0] = c # Placeholder, we need to track columns
        # This approach is flawed for dynamic column tracking in a simple function without state.
        pass
    
    # We will build the matrices directly.
    # Term 1
    U[0][0] = 1; U[4][0] = 1
    V[0][0] = 1; V[4][0] = 1
    W[0][0] = 1; W[4][0] = 1
    
    # Term 2
    U[3][1] = 1; U[4][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[4][1] = -1
    
    # Term 3
    U[0][2] = 1
    V[1][2] = 1; V[4][2] = -1
    W[3][2] = 1; W[4][2] = 1
    
    # Term 4
    U[4][3] = 1
    V[3][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1
    
    # Term 5
    U[0][4] = 1; U[1][4] = 1
    V[4][4] = 1
    W[0][4] = -1; W[3][4] = 1
    
    # Term 6
    U[3][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[4][5] = 1
    
    # Term 7
    U[1][6] = 1; U[4][6] = -1
    V[3][6] = 1; V[4][6] = 1
    W[0][6] = 1
    
    # Border terms 8-22
    # 8
    U[2][7] = 1; V[6][7] = 1; W[0][7] = 1
    # 9
    U[2][8] = 1; V[7][8] = 1; W[3][8] = 1
    # 10
    U[2][9] = 1; V[8][9] = 1; W[6][9] = 1
    # 11
    U[5][10] = 1; V[6][10] = 1; W[1][10] = 1
    # 12
    U[5][11] = 1; V[7][11] = 1; W[4][11] = 1
    # 13
    U[5][12] = 1; V[8][12] = 1; W[7][12] = 1
    # 14
    U[6][13] = 1; V[0][13] = 1; W[2][13] = 1
    # 15
    U[6][14] = 1; V[1][14] = 1; W[5][14] = 1
    # 16
    U[6][15] = 1; V[2][15] = 1; W[8][15] = 1
    # 17
    U[7][16] = 1; V[3][16] = 1; W[2][16] = 1
    # 18
    U[7][17] = 1; V[4][17] = 1; W[5][17] = 1
    # 19
    U[7][18] = 1; V[5][18] = 1; W[8][18] = 1
    # 20
    U[8][19] = 1; V[6][19] = 1; W[2][19] = 1
    # 21
    U[8][20] = 1; V[7][20] = 1; W[5][20] = 1
    # 22
    U[8][21] = 1; V[8][21] = 1; W[8][21] = 1
    
    # Term 23: The missing combination
    # We need to cover aC, bF, dC, eF.
    # aC -> C02 (w=6), dC -> C12 (w=7)
    # bF -> C02 (w=6), eF -> C12 (w=7)
    # Try: U = {0:1, 1:1, 3:1, 4:1}, V = {2:1, 5:1}, W = {6:1, 7:1}?
    # This would be 4*2*2 = 16 terms.
    # U[0]=1, U[1]=1, U[3]=1, U[4]=1
    # V[2]=1, V[5]=1
    # W[6]=1, W[7]=1
    # This adds:
    # A00*B02 -> C02 (valid)
    # A00*B12 -> C02 (invalid)
    # A01*B02 -> C02 (invalid)
    # A01*B12 -> C02 (valid)
    # A10*B02 -> C12 (valid)
    # A10*B12 -> C12 (invalid)
    # A11*B02 -> C12 (invalid)
    # A11*B12 -> C12 (valid)
    
    # So this term adds 1 to all 4 valid positions and 4 spurious positions.
    # Do the spurious positions cancel?
    # Spurious:
    # A00*B12 -> C02? No, A00*B12 is C02? i=0,j=0,k=2 -> C02. Valid!
    # Wait. A00*B12: i=0, j=0. B12: j=1, k=2. j mismatch. Spurious.
    # A01*B02: i=0, j=1. B02: j=0, k=2. j mismatch. Spurious.
    # A10*B12: i=1, j=0. B12: j=1, k=2. j mismatch. Spurious.
    # A11*B02: i=1, j=1. B02: j=0, k=2. j mismatch. Spurious.
    
    # So Term 23 adds 1 to T[0,2,6], T[1,5,6], T[3,2,7], T[4,5,7] (valid)
    # And 1 to T[0,5,6], T[1,2,6], T[3,5,7], T[4,2,7] (spurious).
    
    # Let's check if these spurious terms are cancelled by previous terms.
    # Previous terms did not touch indices 2 or 5 in V (except border terms which are single).
    # Border terms:
    # 16: U[6], V[2], W[8].
    # 19: U[7], V[5], W[8].
    # None of the border terms touch U[0,1,3,4] with V[2,5].
    # So the spurious terms from Term 23 are NOT cancelled.
    # Thus, this Term 23 is invalid.
    
    # Therefore, the simple combination doesn't work.
    # I will fall back to the rank-26 solution which is known to be correct.
    
    rank = 26
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_term(u_idx, v_idx, w_idx, coeff):
        nonlocal col
        U[u_idx][col] += coeff
        V[v_idx][col] += coeff
        W[w_idx][col] += coeff
        col += 1
    
    # 1. Strassen 2x2 core (rank 7)
    # M1
    U[0][col] = 1; U[4][col] = 1
    V[0][col] = 1; V[4][col] = 1
    W[0][col] = 1; W[4][col] = 1
    col += 1
    
    # M2
    U[3][col] = 1; U[4][col] = 1
    V[0][col] = 1
    W[1][col] = 1; W[4][col] = -1
    col += 1
    
    # M3
    U[0][col] = 1
    V[1][col] = 1; V[4][col] = -1
    W[3][col] = 1; W[4][col] = 1
    col += 1
    
    # M4
    U[4][col] = 1
    V[3][col] = 1; V[0][col] = -1
    W[0][col] = 1; W[1][col] = 1
    col += 1
    
    # M5
    U[0][col] = 1; U[1][col] = 1
    V[4][col] = 1
    W[0][col] = -1; W[3][col] = 1
    col += 1
    
    # M6
    U[3][col] = 1; U[0][col] = -1
    V[0][col] = 1; V[1][col] = 1
    W[4][col] = 1
    col += 1
    
    # M7
    U[1][col] = 1; U[4][col] = -1
    V[3][col] = 1; V[4][col] = 1
    W[0][col] = 1
    col += 1
    
    # 2. Border terms (rank 19)
    add_term(2, 6, 0, 1)
    add_term(2, 7, 3, 1)
    add_term(5, 6, 1, 1)
    add_term(5, 7, 4, 1)
    add_term(0, 2, 6, 1)
    add_term(1, 5, 6, 1)
    add_term(3, 2, 7, 1)
    add_term(4, 5, 7, 1)
    add_term(2, 8, 6, 1)
    add_term(5, 8, 7, 1)
    add_term(6, 0, 2, 1)
    add_term(7, 3, 2, 1)
    add_term(6, 1, 5, 1)
    add_term(7, 4, 5, 1)
    add_term(8, 6, 2, 1)
    add_term(8, 7, 5, 1)
    add_term(6, 2, 8, 1)
    add_term(7, 5, 8, 1)
    add_term(8, 8, 8, 1)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
