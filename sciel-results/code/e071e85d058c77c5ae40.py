"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
import time

def construct():
    n, m, p = DIMENSIONS
    # We use a block decomposition.
    # Split the middle dimension m=4 into two blocks of size 2.
    # A is 2x4, B is 4x7.
    # A = [A1; A2] where A1, A2 are 2x2.
    # B = [B1; B2] where B1, B2 are 2x7.
    # C = A1 B1 + A2 B2.
    # Rank of 2x2x7 multiplication.
    # We use a hybrid decomposition for 2x2xp:
    # Split B into B_a (2x2) and B_b (2x(p-2)).
    # Compute A B_a using Strassen (rank 7).
    # Compute A B_b using Schoolbook (rank 4*(p-2)).
    # Total rank for 2x2xp is 7 + 4*(p-2) = 4p - 1.
    # For p=7, rank = 27.
    # Total rank = 27 + 27 = 54.
    
    def get_2x2x7_decomposition(offset_u, offset_v, offset_w):
        """
        Returns U, V, W lists for a 2x2x7 multiplication block.
        U shape (4, rank), V shape (14, rank), W shape (14, rank).
        Indices:
        U: i*m + j. m=2.
        V: j*p + k. p=7.
        W: k*n + i. n=2.
        """
        n_sub, m_sub, p_sub = 2, 2, 7
        rank_sub = 4 * p_sub - 1  # 27
        
        U = [[0] * rank_sub for _ in range(n_sub * m_sub)]
        V = [[0] * rank_sub for _ in range(m_sub * p_sub)]
        W = [[0] * rank_sub for _ in range(p_sub * n_sub)]
        
        # Strassen part for first 2 columns of B (k=0, 1)
        # B_sub is 2x2. A_sub is 2x2.
        # C_sub is 2x2.
        # Strassen terms for 2x2x2:
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A11+A22)(B21-B22)
        # M3 = (A11+A12)(B12-B22)
        # M4 = (A21+A22)(B11-B12)
        # M5 = A11(B12-B22)
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        # Note: B indices in Strassen are (row, col) of the 2x2 block.
        # In our global V/W, the first 2 columns of B correspond to k=0,1.
        # B11 -> V[0,0], B12 -> V[0,1]
        # B21 -> V[1,0], B22 -> V[1,1]
        
        col = 0
        # Term 1
        U[0][col] += 1; U[3][col] += 1
        V[0][col] += 1; V[1][col] += 1
        W[0][col] += 1; W[3][col] += 1 # C11 += M1, C22 += M1
        col += 1
        
        # Term 2
        U[0][col] += 1; U[3][col] += 1
        V[1][col] += 1; V[1][col] -= 1 # B21-B22 -> V[1,0]-V[1,1]
        W[1][col] += 1; W[2][col] -= 1 # C21 += M2, C12 -= M2
        col += 1
        
        # Term 3
        U[0][col] += 1; U[1][col] += 1 # A11+A12
        V[0][col] += 1; V[1][col] -= 1 # B12-B22 -> V[0,1]-V[1,1]
        W[0][col] += 1; W[3][col] -= 1 # C11 += M3, C22 -= M3
        col += 1
        
        # Term 4
        U[2][col] += 1; U[3][col] += 1 # A21+A22
        V[0][col] += 1; V[0][col] -= 1 # B11-B12 -> V[0,0]-V[0,1]
        W[1][col] += 1; W[0][col] -= 1 # C21 += M4, C11 -= M4
        col += 1
        
        # Term 5
        U[0][col] += 1 # A11
        V[0][col] += 1; V[1][col] -= 1 # B12-B22
        W[1][col] += 1 # C12 += M5
        col += 1
        
        # Term 6
        U[2][col] += 1; U[0][col] -= 1 # A21-A11
        V[0][col] += 1; V[0][col] += 1 # B11+B12 -> V[0,0]+V[0,1]
        W[1][col] += 1; W[0][col] += 1 # C21 += M6, C11 += M6
        col += 1
        
        # Term 7
        U[1][col] += 1; U[3][col] -= 1 # A12-A22
        V[1][col] += 1; V[1][col] += 1 # B21+B22 -> V[1,0]+V[1,1]
        W[1][col] += 1; W[3][col] += 1 # C21 += M7, C22 += M7
        col += 1
        
        # Schoolbook part for remaining p-2 columns (k=2..6)
        # A is 2x2, B is 2x(p-2).
        # For each k in 2..6:
        # C1k = A11 B1k + A12 B2k
        # C2k = A21 B1k + A22 B2k
        for k in range(2, p_sub):
            # Term for A11 B1k
            U[0][col] = 1
            V[k][col] = 1 # V index: row 0, col k-0? 
            # V is flattened B. B is 2x7.
            # B[j, k]. V[j*p + k].
            # Here j=0 (first row of B block), k is global col index.
            V[0 * p_sub + k][col] = 1
            W[k * n_sub + 0][col] = 1 # W[k*n + i]. i=0.
            col += 1
            
            # Term for A12 B2k
            U[1][col] = 1
            V[1 * p_sub + k][col] = 1
            W[k * n_sub + 0][col] = 1
            col += 1
            
            # Term for A21 B1k
            U[2][col] = 1
            V[0 * p_sub + k][col] = 1
            W[k * n_sub + 1][col] = 1
            col += 1
            
            # Term for A22 B2k
            U[3][col] = 1
            V[1 * p_sub + k][col] = 1
            W[k * n_sub + 1][col] = 1
            col += 1
            
        return U, V, W

    rank = 54
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Block 1: A1 (rows 0-1 of A, cols 0-1 of A? No, A is 2x4. Rows 0-1, Cols 0-1)
    # A1 corresponds to A[0:2, 0:2].
    # B1 corresponds to B[0:2, 0:7].
    # U indices for Block 1: i in 0..1, j in 0..1. U[i*4 + j].
    # V indices for Block 1: j in 0..1 (local), k in 0..6. V[j*7 + k].
    # W indices for Block 1: k in 0..6, i in 0..1. W[k*2 + i].
    
    U1, V1, W1 = get_2x2x7_decomposition(0, 0, 0)
    for i in range(4):
        for c in range(27):
            U[i*2 + (i % 2) + 2*(i//2)][c] += U1[i][c] # Wait, U1 is 4x27. U is 8x54.
            # Mapping: U1 row r (0..3) corresponds to A1[i,j] where i=r//2, j=r%2.
            # In global U, index is i*m + j = i*4 + j.
            # So U[ i*4 + j ][c] += U1[r][c].
    
    # Let's do the mapping carefully.
    # U1[r][c] where r=i*2+j.
    # Global U index: i*4 + j.
    # For Block 1 (Top A, Top B):
    # i is 0..1. j is 0..1.
    for i in range(2):
        for j in range(2):
            r = i*2 + j
            global_u_idx = i*4 + j
            for c in range(27):
                U[global_u_idx][c] += U1[r][c]
                
    # V1[v_idx][c] where v_idx = j_local*7 + k.
    # Global V index: j_global*7 + k.
    # For Block 1, j_global is 0..1.
    for j_local in range(2):
        for k in range(7):
            v_idx = j_local*7 + k
            global_v_idx = j_local*7 + k
            for c in range(27):
                V[global_v_idx][c] += V1[v_idx][c]
                
    # W1[w_idx][c] where w_idx = k*2 + i.
    # Global W index: k*2 + i.
    # For Block 1, i is 0..1, k is 0..6.
    for k in range(7):
        for i in range(2):
            w_idx = k*2 + i
            global_w_idx = k*2 + i
            for c in range(27):
                W[global_w_idx][c] += W1[w_idx][c]
                
    # Block 2: A2 (rows 0-1 of A, cols 2-3 of A).
    # A2 corresponds to A[0:2, 2:4].
    # B2 corresponds to B[2:4, 0:7].
    # U indices: i in 0..1, j in 2..3.
    # V indices: j in 2..3 (local 0..1), k in 0..6.
    # W indices: same as Block 1 (i, k).
    
    U2, V2, W2 = get_2x2x7_decomposition(0, 0, 0)
    for i in range(2):
        for j_local in range(2):
            j_global = j_local + 2
            r = i*2 + j_local
            global_u_idx = i*4 + j_global
            for c in range(27):
                U[global_u_idx][c+27] += U2[r][c]
                
    for j_local in range(2):
        j_global = j_local + 2
        for k in range(7):
            v_idx = j_local*7 + k
            global_v_idx = j_global*7 + k
            for c in range(27):
                V[global_v_idx][c+27] += V2[v_idx][c]
                
    for k in range(7):
        for i in range(2):
            w_idx = k*2 + i
            global_w_idx = k*2 + i
            for c in range(27):
                W[global_w_idx][c+27] += W2[w_idx][c]
                
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
