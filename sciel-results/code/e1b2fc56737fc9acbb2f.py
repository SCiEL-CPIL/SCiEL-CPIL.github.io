"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=4, p=5
    # U shape: (n*m, rank) = (8, rank)
    # V shape: (m*p, rank) = (20, rank)
    # W shape: (p*n, rank) = (10, rank)
    
    rank = 34
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def u_idx(i, j):
        return i * m + j

    def v_idx(j, k):
        return j * p + k

    def w_idx(k, i):
        return k * n + i

    col = 0
    
    def strassen_2x2x2(y00, y01, y10, y11, z00, z01, z10, z11, 
                        w00, w01, w10, w11):
        nonlocal col
        
        # M1: (y00+y11)(z00+z11) -> X00, X11
        U[u_idx(*y00)][col] = 1
        U[u_idx(*y11)][col] = 1
        V[v_idx(*z00)][col] = 1
        V[v_idx(*z11)][col] = 1
        W[w_idx(*w00)][col] += 1
        W[w_idx(*w11)][col] += 1
        col += 1
        
        # M2: (y10+y11)z00 -> X10, -X11
        U[u_idx(*y10)][col] = 1
        U[u_idx(*y11)][col] = 1
        V[v_idx(*z00)][col] = 1
        W[w_idx(*w10)][col] += 1
        W[w_idx(*w11)][col] -= 1
        col += 1
        
        # M3: y00(z01-z11) -> X01, X11
        U[u_idx(*y00)][col] = 1
        V[v_idx(*z01)][col] = 1
        V[v_idx(*z11)][col] = -1
        W[w_idx(*w01)][col] += 1
        W[w_idx(*w11)][col] += 1
        col += 1
        
        # M4: y11(z10-z00) -> X00, X10
        U[u_idx(*y11)][col] = 1
        V[v_idx(*z10)][col] = 1
        V[v_idx(*z00)][col] = -1
        W[w_idx(*w00)][col] += 1
        W[w_idx(*w10)][col] += 1
        col += 1
        
        # M5: (y00+y01)z11 -> -X00, X01
        U[u_idx(*y00)][col] = 1
        U[u_idx(*y01)][col] = 1
        V[v_idx(*z11)][col] = 1
        W[w_idx(*w00)][col] -= 1
        W[w_idx(*w01)][col] += 1
        col += 1
        
        # M6: (y10-y00)(z00+z01) -> X11
        U[u_idx(*y10)][col] = 1
        U[u_idx(*y00)][col] = -1
        V[v_idx(*z00)][col] = 1
        V[v_idx(*z01)][col] = 1
        W[w_idx(*w11)][col] += 1
        col += 1
        
        # M7: (y01-y11)(z10+z11) -> X00
        U[u_idx(*y01)][col] = 1
        U[u_idx(*y11)][col] = -1
        V[v_idx(*z10)][col] = 1
        V[v_idx(*z11)][col] = 1
        W[w_idx(*w00)][col] += 1
        col += 1

    def strassen_2x3x2(y00, y01, y02, y10, y11, y12, 
                       z00, z01, z10, z11, z20, z21,
                       w00, w01, w10, w11):
        nonlocal col
        
        # Split A (2x3) into A1 (2x2) = [y00,y01;y10,y11] and A2 (2x1) = [y02;y12]
        # Split B (3x2) into B1 (2x2) = [z00,z01;z10,z11] and B2 (1x2) = [z20,z21]
        # C = A1*B1 + A2*B2
        # A1*B1 is 2x2x2 (rank 7)
        # A2*B2 is 2x1x2 (rank 4)
        
        # 2x2x2 part: A1*B1
        # y00, y01, y10, y11 and z00, z01, z10, z11
        # w00, w01, w10, w11
        strassen_2x2x2(y00, y01, y10, y11, z00, z01, z10, z11, 
                       w00, w01, w10, w11)
        
        # 2x1x2 part: A2*B2
        # A2 = [y02; y12], B2 = [z20, z21]
        # C_0 += y02*z20, C_1 += y02*z21
        # C_2 += y12*z20, C_3 += y12*z21
        # In our notation: w00, w01, w10, w11 are the full result indices
        
        # Term 1: y02 * z20 -> w00
        U[u_idx(*y02)][col] = 1
        V[v_idx(*z20)][col] = 1
        W[w_idx(*w00)][col] += 1
        col += 1
        
        # Term 2: y02 * z21 -> w01
        U[u_idx(*y02)][col] = 1
        V[v_idx(*z21)][col] = 1
        W[w_idx(*w01)][col] += 1
        col += 1
        
        # Term 3: y12 * z20 -> w10
        U[u_idx(*y12)][col] = 1
        V[v_idx(*z20)][col] = 1
        W[w_idx(*w10)][col] += 1
        col += 1
        
        # Term 4: y12 * z21 -> w11
        U[u_idx(*y12)][col] = 1
        V[v_idx(*z21)][col] = 1
        W[w_idx(*w11)][col] += 1
        col += 1

    def block_2x4x1(y00, y01, y02, y03, y10, y11, y12, y13, 
                    z00, z01, z02, z03,
                    w00, w01):
        nonlocal col
        
        # 2x4x1: rank 8
        # C_0 = y00*z00 + y01*z01 + y02*z02 + y03*z03
        # C_1 = y10*z00 + y11*z01 + y12*z02 + y13*z03
        
        U[u_idx(*y00)][col] = 1
        V[v_idx(*z00)][col] = 1
        W[w_idx(*w00)][col] += 1
        col += 1
        
        U[u_idx(*y01)][col] = 1
        V[v_idx(*z01)][col] = 1
        W[w_idx(*w00)][col] += 1
        col += 1
        
        U[u_idx(*y02)][col] = 1
        V[v_idx(*z02)][col] = 1
        W[w_idx(*w00)][col] += 1
        col += 1
        
        U[u_idx(*y03)][col] = 1
        V[v_idx(*z03)][col] = 1
        W[w_idx(*w00)][col] += 1
        col += 1
        
        U[u_idx(*y10)][col] = 1
        V[v_idx(*z00)][col] = 1
        W[w_idx(*w01)][col] += 1
        col += 1
        
        U[u_idx(*y11)][col] = 1
        V[v_idx(*z01)][col] = 1
        W[w_idx(*w01)][col] += 1
        col += 1
        
        U[u_idx(*y12)][col] = 1
        V[v_idx(*z02)][col] = 1
        W[w_idx(*w01)][col] += 1
        col += 1
        
        U[u_idx(*y13)][col] = 1
        V[v_idx(*z03)][col] = 1
        W[w_idx(*w01)][col] += 1
        col += 1

    # 2x4x4 part: rank 26
    # Split A (2x4) into A1 (2x3) = cols 0,1,2 and A2 (2x1) = col 3
    # Split B (4x4) into B1 (3x4) = rows 0,1,2 and B2 (1x4) = row 3
    # C = A1*B1 + A2*B2
    # A1*B1 is 2x3x4 (rank 22)
    # A2*B2 is 2x1x4 (rank 4)
    
    # 2x3x4 part: A1*B1
    # Split B1 (3x4) into B1a (3x2) = cols 0,1 and B1b (3x2) = cols 2,3
    # A1*B1a is 2x3x2 (rank 11)
    # A1*B1b is 2x3x2 (rank 11)
    
    # A1*B1a: A1 (2x3) = [A00,A01,A02; A10,A11,A12] = [(0,0),(0,1),(0,2),(1,0),(1,1),(1,2)]
    # B1a (3x2) = [B00,B01; B10,B11; B20,B21] = [(0,0),(0,1),(1,0),(1,1),(2,0),(2,1)]
    # Result: C cols 0,1 = w00, w01, w10, w11
    strassen_2x3x2((0,0), (0,1), (0,2), (1,0), (1,1), (1,2),
                   (0,0), (0,1), (1,0), (1,1), (2,0), (2,1),
                   (0,0), (0,1), (1,0), (1,1))
    
    # A1*B1b: A1 (2x3) = [(0,0),(0,1),(0,2),(1,0),(1,1),(1,2)]
    # B1b (3x2) = [B02,B03; B12,B13; B22,B23] = [(0,2),(0,3),(1,2),(1,3),(2,2),(2,3)]
    # Result: C cols 2,3 = w02, w03, w12, w13
    strassen_2x3x2((0,0), (0,1), (0,2), (1,0), (1,1), (1,2),
                   (0,2), (0,3), (1,2), (1,3), (2,2), (2,3),
                   (0,2), (0,3), (1,2), (1,3))
    
    # A2*B2: A2 (2x1) = col 3 of A = [(0,3),(1,3)]
    # B2 (1x4) = row 3 of B = [(3,0),(3,1),(3,2),(3,3)]
    # Result: all 4 cols of C
    # C_0 += A03*B30, C_1 += A03*B31, C_2 += A03*B32, C_3 += A03*B33
    # C_4 += A13*B30, C_5 += A13*B31, C_6 += A13*B32, C_7 += A13*B33
    
    U[u_idx(0,3)][col] = 1
    V[v_idx(3,0)][col] = 1
    W[w_idx(0,0)][col] += 1
    col += 1
    
    U[u_idx(0,3)][col] = 1
    V[v_idx(3,1)][col] = 1
    W[w_idx(1,0)][col] += 1
    col += 1
    
    U[u_idx(0,3)][col] = 1
    V[v_idx(3,2)][col] = 1
    W[w_idx(2,0)][col] += 1
    col += 1
    
    U[u_idx(0,3)][col] = 1
    V[v_idx(3,3)][col] = 1
    W[w_idx(3,0)][col] += 1
    col += 1
    
    U[u_idx(1,3)][col] = 1
    V[v_idx(3,0)][col] = 1
    W[w_idx(0,1)][col] += 1
    col += 1
    
    U[u_idx(1,3)][col] = 1
    V[v_idx(3,1)][col] = 1
    W[w_idx(1,1)][col] += 1
    col += 1
    
    U[u_idx(1,3)][col] = 1
    V[v_idx(3,2)][col] = 1
    W[w_idx(2,1)][col] += 1
    col += 1
    
    U[u_idx(1,3)][col] = 1
    V[v_idx(3,3)][col] = 1
    W[w_idx(3,1)][col] += 1
    col += 1
    
    # 2x4x1 part: rank 8
    # A (2x4) * B_col4 (4x1) = C_col4 (2x1)
    block_2x4x1((0,0), (0,1), (0,2), (0,3), (1,0), (1,1), (1,2), (1,3),
                (0,4), (1,4), (2,4), (3,4),
                (4,0), (4,1))
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
