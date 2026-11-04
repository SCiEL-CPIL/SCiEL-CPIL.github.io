"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Target rank: 50 (using Strassen-like decomposition)
    # Structure:
    # Split A (2x4) into A1 (2x2) and A2 (2x2)
    # Split B (4x7) into B1 (2x7) and B2 (2x7)
    # C = A1 B1 + A2 B2
    # Each Ai Bi is 2x2x7.
    # 2x2x7 decomposed into 2x2x4 (Rank 14) + 2x2x3 (Rank 11)
    # 2x2x4 decomposed into two 2x2x2 (Rank 7 each)
    # 2x2x3 decomposed into one 2x2x2 (Rank 7) + one 2x2x1 (Rank 4)
    # Total rank per 2x2x7: 14 + 7 + 4 = 25
    # Total rank: 25 + 25 = 50
    
    rank = 50
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_strassen_222(offset_u, offset_v, offset_w, base_col):
        # Adds 7 rank-1 terms for 2x2x2 multiplication
        # A indices: 0,1,2,3 (i*m+j)
        # B indices: 0,1,2,3 (j*p+k)
        # C indices: 0,1,2,3 (k*n+i)
        
        # M1 = (A00+A11)(B00+B11)
        # A: 0, 3. B: 0, 3. C: 0 (k=0,i=0), 3 (k=1,i=1)
        U[offset_u + 0][base_col] = 1
        U[offset_u + 3][base_col] = 1
        V[offset_v + 0][base_col] = 1
        V[offset_v + 3][base_col] = 1
        W[offset_w + 0][base_col] = 1
        W[offset_w + 3][base_col] = 1
        base_col += 1
        
        # M2 = (A10+A11)B00
        # A: 2, 3. B: 0. C: 2 (k=1,i=0), 3 (k=1,i=1)
        U[offset_u + 2][base_col] = 1
        U[offset_u + 3][base_col] = 1
        V[offset_v + 0][base_col] = 1
        W[offset_w + 2][base_col] = 1
        W[offset_w + 3][base_col] = 1
        base_col += 1
        
        # M3 = A00(B01-B11)
        # A: 0. B: 1, -1 at 3. C: 1 (k=0,i=1), 3 (k=1,i=1)
        U[offset_u + 0][base_col] = 1
        V[offset_v + 1][base_col] = 1
        V[offset_v + 3][base_col] = -1
        W[offset_w + 1][base_col] = 1
        W[offset_w + 3][base_col] = 1
        base_col += 1
        
        # M4 = A11(B10-B00)
        # A: 3. B: 2, -1 at 0. C: 0 (k=0,i=0), 2 (k=1,i=0)
        U[offset_u + 3][base_col] = 1
        V[offset_v + 2][base_col] = 1
        V[offset_v + 0][base_col] = -1
        W[offset_w + 0][base_col] = 1
        W[offset_w + 2][base_col] = 1
        base_col += 1
        
        # M5 = (A00+A01)B11
        # A: 0, 1. B: 3. C: 0 (k=0,i=0), 1 (k=0,i=1)
        U[offset_u + 0][base_col] = 1
        U[offset_u + 1][base_col] = 1
        V[offset_v + 3][base_col] = 1
        W[offset_w + 0][base_col] = 1
        W[offset_w + 1][base_col] = 1
        base_col += 1
        
        # M6 = (A10-A00)(B00+B01)
        # A: -1 at 0, 1 at 2. B: 0, 1. C: 1 (k=0,i=1), 3 (k=1,i=1)
        U[offset_u + 0][base_col] = -1
        U[offset_u + 2][base_col] = 1
        V[offset_v + 0][base_col] = 1
        V[offset_v + 1][base_col] = 1
        W[offset_w + 1][base_col] = 1
        W[offset_w + 3][base_col] = 1
        base_col += 1
        
        # M7 = (A01-A11)(B10+B11)
        # A: 1 at 1, -1 at 3. B: 2, 3. C: 0 (k=0,i=0), 2 (k=1,i=0)
        U[offset_u + 1][base_col] = 1
        U[offset_u + 3][base_col] = -1
        V[offset_v + 2][base_col] = 1
        V[offset_v + 3][base_col] = 1
        W[offset_w + 0][base_col] = 1
        W[offset_w + 2][base_col] = 1
        base_col += 1
        
        return base_col

    def add_221(offset_u, offset_v, offset_w, base_col):
        # Adds 4 rank-1 terms for 2x2x1 multiplication
        # A: 0,1,2,3. B: 0,1. C: 0 (k=0,i=0), 1 (k=0,i=1)
        # C0 = A00 B0 + A01 B1
        # C1 = A10 B0 + A11 B1
        
        # T1: A00 B0 -> C0
        U[offset_u + 0][base_col] = 1
        V[offset_v + 0][base_col] = 1
        W[offset_w + 0][base_col] = 1
        base_col += 1
        
        # T2: A01 B1 -> C0
        U[offset_u + 1][base_col] = 1
        V[offset_v + 1][base_col] = 1
        W[offset_w + 0][base_col] = 1
        base_col += 1
        
        # T3: A10 B0 -> C1
        U[offset_u + 2][base_col] = 1
        V[offset_v + 0][base_col] = 1
        W[offset_w + 1][base_col] = 1
        base_col += 1
        
        # T4: A11 B1 -> C1
        U[offset_u + 3][base_col] = 1
        V[offset_v + 1][base_col] = 1
        W[offset_w + 1][base_col] = 1
        base_col += 1
        
        return base_col

    col = 0
    
    # Part 1: A1 (cols 0,1) * B1 (rows 0,1)
    # A1 indices in U: 0,1,4,5
    # B1 indices in V: 0..13 (j=0,1; k=0..6)
    # C indices in W: k*2+i
    
    # 2x2x4 part (k=0..3)
    # Block k=0,1
    # A1 U offset: 0,1,4,5. Map to 0,1,2,3 locally?
    # Local A: 0,1,2,3.
    # U[0]=A00, U[1]=A01, U[4]=A10, U[5]=A11.
    # So offset_u is not contiguous.
    # We need to handle non-contiguous U offsets.
    
    # Let's refactor to pass explicit index mappings.
    
    def add_strassen_222_mapped(u_map, v_map, w_map, base_col):
        # u_map: [u_idx0, u_idx1, u_idx2, u_idx3] for A00, A01, A10, A11
        # v_map: [v_idx0, v_idx1, v_idx2, v_idx3] for B00, B01, B10, B11
        # w_map: [w_idx0, w_idx1, w_idx2, w_idx3] for C00, C01, C10, C11
        
        # M1
        U[u_map[0]][base_col] = 1
        U[u_map[3]][base_col] = 1
        V[v_map[0]][base_col] = 1
        V[v_map[3]][base_col] = 1
        W[w_map[0]][base_col] = 1
        W[w_map[3]][base_col] = 1
        base_col += 1
        
        # M2
        U[u_map[2]][base_col] = 1
        U[u_map[3]][base_col] = 1
        V[v_map[0]][base_col] = 1
        W[w_map[2]][base_col] = 1
        W[w_map[3]][base_col] = 1
        base_col += 1
        
        # M3
        U[u_map[0]][base_col] = 1
        V[v_map[1]][base_col] = 1
        V[v_map[3]][base_col] = -1
        W[w_map[1]][base_col] = 1
        W[w_map[3]][base_col] = 1
        base_col += 1
        
        # M4
        U[u_map[3]][base_col] = 1
        V[v_map[2]][base_col] = 1
        V[v_map[0]][base_col] = -1
        W[w_map[0]][base_col] = 1
        W[w_map[2]][base_col] = 1
        base_col += 1
        
        # M5
        U[u_map[0]][base_col] = 1
        U[u_map[1]][base_col] = 1
        V[v_map[3]][base_col] = 1
        W[w_map[0]][base_col] = 1
        W[w_map[1]][base_col] = 1
        base_col += 1
        
        # M6
        U[u_map[0]][base_col] = -1
        U[u_map[2]][base_col] = 1
        V[v_map[0]][base_col] = 1
        V[v_map[1]][base_col] = 1
        W[w_map[1]][base_col] = 1
        W[w_map[3]][base_col] = 1
        base_col += 1
        
        # M7
        U[u_map[1]][base_col] = 1
        U[u_map[3]][base_col] = -1
        V[v_map[2]][base_col] = 1
        V[v_map[3]][base_col] = 1
        W[w_map[0]][base_col] = 1
        W[w_map[2]][base_col] = 1
        base_col += 1
        
        return base_col

    def add_221_mapped(u_map, v_map, w_map, base_col):
        # T1
        U[u_map[0]][base_col] = 1
        V[v_map[0]][base_col] = 1
        W[w_map[0]][base_col] = 1
        base_col += 1
        
        # T2
        U[u_map[1]][base_col] = 1
        V[v_map[1]][base_col] = 1
        W[w_map[0]][base_col] = 1
        base_col += 1
        
        # T3
        U[u_map[2]][base_col] = 1
        V[v_map[0]][base_col] = 1
        W[w_map[1]][base_col] = 1
        base_col += 1
        
        # T4
        U[u_map[3]][base_col] = 1
        V[v_map[1]][base_col] = 1
        W[w_map[1]][base_col] = 1
        base_col += 1
        
        return base_col

    # Define mappings for Part 1 (j=0,1)
    # A1: i=0,1; j=0,1
    # U indices: i*m+j
    # i=0,j=0 -> 0
    # i=0,j=1 -> 1
    # i=1,j=0 -> 4
    # i=1,j=1 -> 5
    u_map_p1 = [0, 1, 4, 5]
    
    # B1: j=0,1; k=0..6
    # V indices: j*p+k
    # For k=0,1:
    # j=0,k=0 -> 0
    # j=0,k=1 -> 1
    # j=1,k=0 -> 7
    # j=1,k=1 -> 8
    v_map_p1_k01 = [0, 1, 7, 8]
    
    # For k=2,3:
    # j=0,k=2 -> 2
    # j=0,k=3 -> 3
    # j=1,k=2 -> 9
    # j=1,k=3 -> 10
    v_map_p1_k23 = [2, 3, 9, 10]
    
    # For k=4,5:
    # j=0,k=4 -> 4
    # j=0,k=5 -> 5
    # j=1,k=4 -> 11
    # j=1,k=5 -> 12
    v_map_p1_k45 = [4, 5, 11, 12]
    
    # For k=6:
    # j=0,k=6 -> 6
    # j=1,k=6 -> 13
    v_map_p1_k6 = [6, 13]
    
    # C: k=0..6; i=0,1
    # W indices: k*n+i
    # For k=0,1:
    # k=0,i=0 -> 0
    # k=0,i=1 -> 1
    # k=1,i=0 -> 2
    # k=1,i=1 -> 3
    w_map_p1_k01 = [0, 1, 2, 3]
    
    # For k=2,3:
    # k=2,i=0 -> 4
    # k=2,i=1 -> 5
    # k=3,i=0 -> 6
    # k=3,i=1 -> 7
    w_map_p1_k23 = [4, 5, 6, 7]
    
    # For k=4,5:
    # k=4,i=0 -> 8
    # k=4,i=1 -> 9
    # k=5,i=0 -> 10
    # k=5,i=1 -> 11
    w_map_p1_k45 = [8, 9, 10, 11]
    
    # For k=6:
    # k=6,i=0 -> 12
    # k=6,i=1 -> 13
    w_map_p1_k6 = [12, 13]
    
    # Assemble Part 1
    col = add_strassen_222_mapped(u_map_p1, v_map_p1_k01, w_map_p1_k01, col)
    col = add_strassen_222_mapped(u_map_p1, v_map_p1_k23, w_map_p1_k23, col)
    col = add_strassen_222_mapped(u_map_p1, v_map_p1_k45, w_map_p1_k45, col)
    col = add_221_mapped(u_map_p1, v_map_p1_k6, w_map_p1_k6, col)
    
    # Define mappings for Part 2 (j=2,3)
    # A2: i=0,1; j=2,3
    # U indices: i*m+j
    # i=0,j=2 -> 2
    # i=0,j=3 -> 3
    # i=1,j=2 -> 6
    # i=1,j=3 -> 7
    u_map_p2 = [2, 3, 6, 7]
    
    # B2: j=2,3; k=0..6
    # V indices: j*p+k
    # For k=0,1:
    # j=2,k=0 -> 14
    # j=2,k=1 -> 15
    # j=3,k=0 -> 21
    # j=3,k=1 -> 22
    v_map_p2_k01 = [14, 15, 21, 22]
    
    # For k=2,3:
    # j=2,k=2 -> 16
    # j=2,k=3 -> 17
    # j=3,k=2 -> 23
    # j=3,k=3 -> 24
    v_map_p2_k23 = [16, 17, 23, 24]
    
    # For k=4,5:
    # j=2,k=4 -> 18
    # j=2,k=5 -> 19
    # j=3,k=4 -> 25
    # j=3,k=5 -> 26
    v_map_p2_k45 = [18, 19, 25, 26]
    
    # For k=6:
    # j=2,k=6 -> 20
    # j=3,k=6 -> 27
    v_map_p2_k6 = [20, 27]
    
    # W maps are same as Part 1
    w_map_p2_k01 = w_map_p1_k01
    w_map_p2_k23 = w_map_p1_k23
    w_map_p2_k45 = w_map_p1_k45
    w_map_p2_k6 = w_map_p1_k6
    
    # Assemble Part 2
    col = add_strassen_222_mapped(u_map_p2, v_map_p2_k01, w_map_p2_k01, col)
    col = add_strassen_222_mapped(u_map_p2, v_map_p2_k23, w_map_p2_k23, col)
    col = add_strassen_222_mapped(u_map_p2, v_map_p2_k45, w_map_p2_k45, col)
    col = add_221_mapped(u_map_p2, v_map_p2_k6, w_map_p2_k6, col)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
