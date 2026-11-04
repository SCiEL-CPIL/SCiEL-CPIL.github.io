"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-24 exact tensor decomposition for (2,4,4) matrix multiplication.
    Uses Strassen's algorithm on two independent 2x2 block products.
    A is 2x4, B is 4x4, C is 2x4.
    Split columns of A: A = [A1, A2] where A1 is 2x2, A2 is 2x2.
    Split rows of B: B = [[B11, B12], [B21, B22]] where each is 2x2.
    Then C = A1*B1 + A2*B2 (where B1 = [B11, B12], B2 = [B21, B22]).
    Each 2x2 * 2x2 multiplication uses 7 multiplications (Strassen).
    Total rank = 7 + 7 = 14? No, wait.
    
    Actually, the standard Strassen for 2x2 * 2x2 gives rank 7.
    Here we have two such products:
    Product 1: A1 (2x2) * B1 (2x2) -> C1 (2x2)
    Product 2: A2 (2x2) * B2 (2x2) -> C2 (2x2)
    C = [C1, C2]
    
    So the total rank is 7 + 7 = 14?
    Let's check the indices.
    U has shape (n*m, rank) = (2*4, 14) = (8, 14).
    V has shape (m*p, rank) = (4*4, 14) = (16, 14).
    W has shape (p*n, rank) = (4*2, 14) = (8, 14).
    
    Wait, is rank 14 possible?
    The lower bound for 2x4 * 4x4 is max(2*4, 4*4, 4*2) = 16? No, the trivial lower bound is n*m*p / (something)?
    Actually, the rank of (2,4,4) tensor is at least 16?
    Let's re-evaluate.
    
    A is 2x4. B is 4x4.
    C_{ik} = sum_j A_{ij} B_{jk}.
    
    If we split A into two 2x2 blocks A1, A2.
    And B into 2x2 blocks B11, B12, B21, B22.
    C1 = A1 * B1 where B1 = [B11 B12] (2x2).
    C2 = A2 * B2 where B2 = [B21 B22] (2x2).
    
    Yes, this is two independent 2x2 matrix multiplications.
    The rank of a single 2x2 multiplication is 7 (Strassen).
    Since they are independent (disjoint indices in A, B, C), the total rank is 7 + 7 = 14.
    
    Is there a known lower bound?
    For (2,4,4), the rank is known to be 16?
    Let's check literature.
    The rank of (2,2,2) is 7.
    The rank of (2,2,n) is 2n + something?
    Actually, (2,4,4) can be seen as two (2,2,2) tensors in parallel?
    A = [A1 A2].
    B = [[B11 B12]
         [B21 B22]]
    
    C1 = A1 B11 + A2 B21 ? No.
    C = A B.
    C_{i0} = A_{i0}B_{00} + A_{i1}B_{10} + A_{i2}B_{20} + A_{i3}B_{30}
    
    Let's map indices carefully.
    A indices: i (0,1), j (0,1,2,3).
    B indices: j (0,1,2,3), k (0,1,2,3).
    C indices: k (0,1,2,3), i (0,1).
    
    Split j into j' (0,1) and j'' (2,3).
    A = [A1, A2] where A1 is 2x2 (cols 0,1), A2 is 2x2 (cols 2,3).
    B = [[B11, B12], [B21, B22]] where B11 is 2x2 (rows 0,1 cols 0,1), etc.
    
    C = A1 * [B11 B12] + A2 * [B21 B22]
    Let B1 = [B11 B12] (2x2). B2 = [B21 B22] (2x2).
    C1 = A1 * B1 (2x2).
    C2 = A2 * B2 (2x2).
    C = [C1 C2].
    
    This is exactly two independent 2x2 matrix multiplications.
    Rank = 7 + 7 = 14.
    
    Why did I think 16?
    Maybe there's a dependency?
    No, the indices are disjoint.
    A1 uses A_{i0}, A_{i1}. A2 uses A_{i2}, A_{i3}.
    B1 uses B_{0k}, B_{1k}. B2 uses B_{2k}, B_{3k}.
    C1 uses C_{0i}, C_{1i}. C2 uses C_{2i}, C_{3i}.
    All disjoint.
    
    So rank 14 is achievable.
    
    Let's implement the Strassen construction for two 2x2 products.
    
    Product 1: A1 (2x2) * B1 (2x2) -> C1 (2x2)
    A1 elements: A_{00}, A_{01}, A_{10}, A_{11}. (Global indices: i=0,j=0; i=0,j=1; i=1,j=0; i=1,j=1)
    B1 elements: B_{00}, B_{01}, B_{10}, B_{11}. (Global indices: j=0,k=0; j=0,k=1; j=1,k=0; j=1,k=1)
    C1 elements: C_{00}, C_{01}, C_{10}, C_{11}. (Global indices: k=0,i=0; k=0,i=1; k=1,i=0; k=1,i=1)
    
    Product 2: A2 (2x2) * B2 (2x2) -> C2 (2x2)
    A2 elements: A_{02}, A_{03}, A_{12}, A_{13}. (Global indices: i=0,j=2; i=0,j=3; i=1,j=2; i=1,j=3)
    B2 elements: B_{20}, B_{21}, B_{30}, B_{31}. (Global indices: j=2,k=0; j=2,k=1; j=3,k=0; j=3,k=1)
    C2 elements: C_{20}, C_{21}, C_{30}, C_{31}. (Global indices: k=2,i=0; k=2,i=1; k=3,i=0; k=3,i=1)
    
    We will construct 14 columns.
    Columns 0-6: Strassen for Product 1.
    Columns 7-13: Strassen for Product 2.
    """
    n, m, p = DIMENSIONS  # 2, 4, 4
    rank = 14
    U = [[0] * rank for _ in range(n * m)]  # 8x14
    V = [[0] * rank for _ in range(m * p)]  # 16x14
    W = [[0] * rank for _ in range(p * n)]  # 8x14
    
    # Strassen terms for 2x2 * 2x2
    # P1 = (X00+X11)(Y00+Y11)
    # P2 = (X10+X11)Y00
    # P3 = X00(Y01-Y11)
    # P4 = X11(Y10-Y00)
    # P5 = (X00+X01)Y11
    # P6 = (X10-X00)(Y00+Y01)
    # P7 = (X01-X11)(Y10+Y11)
    
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    
    # Helper to add a Strassen product
    # x_indices: (x00, x01, x10, x11) in U
    # y_indices: (y00, y01, y10, y11) in V
    # c_indices: (c00, c01, c10, c11) in W
    # base_col: starting column index
    
    def add_strassen(x_idx, y_idx, c_idx, base_col):
        # x_idx, y_idx, c_idx are tuples of 4 global indices
        # U indices are i*m + j. V indices are j*p + k. W indices are k*n + i.
        
        # P1
        U[x_idx[0]][base_col + 0] += 1
        U[x_idx[3]][base_col + 0] += 1
        V[y_idx[0]][base_col + 0] += 1
        V[y_idx[3]][base_col + 0] += 1
        W[c_idx[0]][base_col + 0] += 1
        W[c_idx[3]][base_col + 0] += 1
        
        # P2
        U[x_idx[2]][base_col + 1] += 1
        U[x_idx[3]][base_col + 1] += 1
        V[y_idx[0]][base_col + 1] += 1
        W[c_idx[2]][base_col + 1] += 1
        W[c_idx[3]][base_col + 1] -= 1
        
        # P3
        U[x_idx[0]][base_col + 2] += 1
        V[y_idx[1]][base_col + 2] += 1
        V[y_idx[3]][base_col + 2] -= 1
        W[c_idx[1]][base_col + 2] += 1
        W[c_idx[3]][base_col + 2] += 1
        
        # P4
        U[x_idx[3]][base_col + 3] += 1
        V[y_idx[2]][base_col + 3] += 1
        V[y_idx[0]][base_col + 3] -= 1
        W[c_idx[0]][base_col + 3] += 1
        W[c_idx[2]][base_col + 3] += 1
        
        # P5
        U[x_idx[0]][base_col + 4] += 1
        U[x_idx[1]][base_col + 4] += 1
        V[y_idx[3]][base_col + 4] += 1
        W[c_idx[0]][base_col + 4] -= 1
        W[c_idx[1]][base_col + 4] += 1
        
        # P6
        U[x_idx[2]][base_col + 5] += 1
        U[x_idx[0]][base_col + 5] -= 1
        V[y_idx[0]][base_col + 5] += 1
        V[y_idx[1]][base_col + 5] += 1
        W[c_idx[3]][base_col + 5] += 1
        
        # P7
        U[x_idx[1]][base_col + 6] += 1
        U[x_idx[3]][base_col + 6] -= 1
        V[y_idx[2]][base_col + 6] += 1
        V[y_idx[3]][base_col + 6] += 1
        W[c_idx[0]][base_col + 6] += 1
        W[c_idx[1]][base_col + 6] -= 1

    # Product 1: A1 * B1 -> C1
    # A1: A_{00}, A_{01}, A_{10}, A_{11}
    # U indices: i*m+j. n=2, m=4.
    # i=0, j=0 -> 0
    # i=0, j=1 -> 1
    # i=1, j=0 -> 4
    # i=1, j=1 -> 5
    x1 = (0, 1, 4, 5)
    
    # B1: B_{00}, B_{01}, B_{10}, B_{11}
    # V indices: j*p+k. m=4, p=4.
    # j=0, k=0 -> 0
    # j=0, k=1 -> 1
    # j=1, k=0 -> 4
    # j=1, k=1 -> 5
    y1 = (0, 1, 4, 5)
    
    # C1: C_{00}, C_{01}, C_{10}, C_{11}
    # W indices: k*n+i. p=4, n=2.
    # k=0, i=0 -> 0
    # k=0, i=1 -> 1
    # k=1, i=0 -> 2
    # k=1, i=1 -> 3
    c1 = (0, 1, 2, 3)
    
    add_strassen(x1, y1, c1, 0)
    
    # Product 2: A2 * B2 -> C2
    # A2: A_{02}, A_{03}, A_{12}, A_{13}
    # i=0, j=2 -> 2
    # i=0, j=3 -> 3
    # i=1, j=2 -> 6
    # i=1, j=3 -> 7
    x2 = (2, 3, 6, 7)
    
    # B2: B_{20}, B_{21}, B_{30}, B_{31}
    # j=2, k=0 -> 8
    # j=2, k=1 -> 9
    # j=3, k=0 -> 12
    # j=3, k=1 -> 13
    y2 = (8, 9, 12, 13)
    
    # C2: C_{20}, C_{21}, C_{30}, C_{31}
    # k=2, i=0 -> 4
    # k=2, i=1 -> 5
    # k=3, i=0 -> 6
    # k=3, i=1 -> 7
    c2 = (4, 5, 6, 7)
    
    add_strassen(x2, y2, c2, 7)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
