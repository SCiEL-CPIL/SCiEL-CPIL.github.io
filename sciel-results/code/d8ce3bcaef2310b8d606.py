"""Matrix multiplication 2,4,4 via Strassen blocks."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We split the 4x4 B matrix into 2x2 blocks and the 2x4 A matrix into 2x2 blocks.
    # A is 2x4, split into A_left (2x2) and A_right (2x2).
    # B is 4x4, split into B_TL, B_TR, B_BL, B_BR (each 2x2).
    # C = A * B is 2x4, split into C_left (2x2) and C_right (2x2).
    
    # C_left = A_left * B_TL + A_right * B_BL
    # C_right = A_left * B_TR + A_right * B_BR
    
    # We can use Strassen for each 2x2x2 multiplication.
    # Strassen 2x2x2 has rank 7.
    # We have 4 multiplications: 
    # 1. A_left * B_TL -> contributes to C_left
    # 2. A_right * B_BL -> contributes to C_left
    # 3. A_left * B_TR -> contributes to C_right
    # 4. A_right * B_BR -> contributes to C_right
    # Total rank = 4 * 7 = 28.
    
    # Let's define the Strassen decomposition for 2x2x2.
    # Input matrices X (2x2), Y (2x2), output Z = X*Y (2x2).
    # Indices for X, Y, Z are 0,1,2,3 in row-major order.
    # X = [[x0, x1], [x2, x3]]
    # Y = [[y0, y1], [y2, y3]]
    # Z = [[z0, z1], [z2, z3]]
    
    # Strassen factors:
    # M1 = (x0 + x3) * (y0 + y3)
    # M2 = (x2 + x3) * y0
    # M3 = x0 * (y1 - y3)
    # M4 = x3 * (y2 - y0)
    # M5 = (x0 + x1) * y3
    # M6 = (x2 - x0) * (y0 + y1)
    # M7 = (x1 - x3) * (y2 + y3)
    
    # Z0 = M1 + M4 - M5 + M7
    # Z1 = M3 + M5
    # Z2 = M2 + M4
    # Z3 = M1 - M2 + M3 + M6
    
    # U factors (for X), V factors (for Y), W factors (for Z)
    # We need to map these to the global indices.
    
    # Global A is 2x4. Row-major index: i*4 + j.
    # A_left: i in [0,1], j in [0,1] -> global index i*4 + j
    # A_right: i in [0,1], j in [2,3] -> global index i*4 + j
    # Local indices for A_left: 0->0, 1->1, 2->4, 3->5
    # Local indices for A_right: 0->2, 1->3, 2->6, 3->7
    
    # Global B is 4x4. Row-major index: i*4 + j.
    # B_TL: i in [0,1], j in [0,1] -> global index i*4 + j
    # B_TR: i in [0,1], j in [2,3] -> global index i*4 + j
    # B_BL: i in [2,3], j in [0,1] -> global index i*4 + j
    # B_BR: i in [2,3], j in [2,3] -> global index i*4 + j
    # Local indices for B_TL: 0->0, 1->1, 2->4, 3->5
    # Local indices for B_TR: 0->2, 1->3, 2->6, 3->7
    # Local indices for B_BL: 0->8, 1->9, 2->12, 3->13
    # Local indices for B_BR: 0->10, 1->11, 2->14, 3->15
    
    # Global C is 2x4. Row-major index: i*4 + j.
    # C_left: i in [0,1], j in [0,1] -> global index i*4 + j
    # C_right: i in [0,1], j in [2,3] -> global index i*4 + j
    # Local indices for C_left: 0->0, 1->1, 2->4, 3->5
    # Local indices for C_right: 0->2, 1->3, 2->6, 3->7
    
    # Note: The task specifies W[k*n+i, r].
    # Here n=2, p=4. W has shape (p*n, rank) = (8, rank).
    # Wait, the task says W shape is (p*n, rank).
    # And the convention is W[k*n+i, r].
    # Let's re-read carefully: "output factor follows the official trace convention W[k*n+i,r]"
    # Usually, tensor T[i,j,k] = sum_r U[i,r] V[j,r] W[k,r]
    # Here i is row of A (0..n-1), j is col of A / row of B (0..m-1), k is col of B (0..p-1).
    # A is n x m. B is m x p. C is n x p.
    # T[i,j,k] is the coefficient of A[i,j] * B[j,k] in C[i,k].
    # So T[i,j,k] = delta_{i,k} ? No.
    # C[i,k] = sum_j A[i,j] B[j,k].
    # So the tensor T has indices i (from A), j (shared), k (from B).
    # T[i,j,k] = 1 if it's a valid term in the sum, else 0.
    # Actually, for matrix multiplication, the tensor is just the structure of the multiplication.
    # T[i,j,k] is 1 if the term A[i,j]*B[j,k] appears in C[i,k]. It always does.
    # So T[i,j,k] = 1 for all i,j,k? No, that's not right.
    # The tensor represents the bilinear map (A, B) -> C.
    # C_{i,k} = sum_{j} A_{i,j} B_{j,k}.
    # So the coefficient of A_{i,j} B_{j,k} in C_{i,k} is 1.
    # The tensor T is of size n x m x p.
    # T[i, j, k] = 1 if k is the column index in C corresponding to row i and column j of A/B?
    # No. The output C is n x p.
    # The term A[i,j] * B[j,k] contributes to C[i,k].
    # So the tensor T has non-zero entries at (i, j, k) corresponding to the contribution to C[i,k].
    # But the decomposition T = U V W means T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r].
    # Wait, U has shape (n*m, rank). Index is i*m + j.
    # V has shape (m*p, rank). Index is j*p + k.
    # W has shape (p*n, rank). Index is k*n + i.
    
    # Let's check the W index. W[k*n+i, r].
    # n=2, p=4.
    # i in 0..1, k in 0..3.
    # Index = k*2 + i.
    # If k=0, i=0 -> 0 (C[0,0])
    # If k=0, i=1 -> 1 (C[1,0])
    # If k=1, i=0 -> 2 (C[0,1])
    # If k=1, i=1 -> 3 (C[1,1])
    # If k=2, i=0 -> 4 (C[0,2])
    # ...
    # So W index corresponds to the linear index of C[k, i]?
    # C is n x p. C[i, k].
    # Usually linear index is i*p + k.
    # Here it is k*n + i. This is column-major index for C.
    # C[i,k] is at position k*n + i.
    
    # Let's verify with the schoolbook code:
    # W[k * n + i][column] = 1
    # Yes.
    
    # So, for each Strassen block, we need to map the local Z indices to global W indices.
    # Z is 2x2. Local index z_idx in 0..3.
    # Z is a submatrix of C.
    # If Z is C_left (i in 0..1, k in 0..1):
    # Local 0 (Z[0,0]) -> C[0,0] -> W[0*2+0] = 0
    # Local 1 (Z[0,1]) -> C[0,1] -> W[1*2+0] = 2
    # Local 2 (Z[1,0]) -> C[1,0] -> W[0*2+1] = 1
    # Local 3 (Z[1,1]) -> C[1,1] -> W[1*2+1] = 3
    # Note: Strassen Z indices are row-major for Z.
    # Z[0,0]=0, Z[0,1]=1, Z[1,0]=2, Z[1,1]=3.
    # So:
    # local 0 -> global W 0
    # local 1 -> global W 2
    # local 2 -> global W 1
    # local 3 -> global W 3
    
    # If Z is C_right (i in 0..1, k in 2..3):
    # Local 0 (Z[0,0]) -> C[0,2] -> W[2*2+0] = 4
    # Local 1 (Z[0,1]) -> C[0,3] -> W[3*2+0] = 6
    # Local 2 (Z[1,0]) -> C[1,2] -> W[2*2+1] = 5
    # Local 3 (Z[1,1]) -> C[1,3] -> W[3*2+1] = 7
    
    # Similarly for U (A) and V (B).
    # U index is i*m + j. m=4.
    # A_left (j in 0..1):
    # Local 0 (A[0,0]) -> U[0*4+0] = 0
    # Local 1 (A[0,1]) -> U[0*4+1] = 1
    # Local 2 (A[1,0]) -> U[1*4+0] = 4
    # Local 3 (A[1,1]) -> U[1*4+1] = 5
    # A_right (j in 2..3):
    # Local 0 (A[0,2]) -> U[0*4+2] = 2
    # Local 1 (A[0,3]) -> U[0*4+3] = 3
    # Local 2 (A[1,2]) -> U[1*4+2] = 6
    # Local 3 (A[1,3]) -> U[1*4+3] = 7
    
    # V index is j*p + k. p=4.
    # B_TL (j in 0..1, k in 0..1):
    # Local 0 (B[0,0]) -> V[0*4+0] = 0
    # Local 1 (B[0,1]) -> V[0*4+1] = 1
    # Local 2 (B[1,0]) -> V[1*4+0] = 4
    # Local 3 (B[1,1]) -> V[1*4+1] = 5
    # B_TR (j in 0..1, k in 2..3):
    # Local 0 (B[0,2]) -> V[0*4+2] = 2
    # Local 1 (B[0,3]) -> V[0*4+3] = 3
    # Local 2 (B[1,2]) -> V[1*4+2] = 6
    # Local 3 (B[1,3]) -> V[1*4+3] = 7
    # B_BL (j in 2..3, k in 0..1):
    # Local 0 (B[2,0]) -> V[2*4+0] = 8
    # Local 1 (B[2,1]) -> V[2*4+1] = 9
    # Local 2 (B[3,0]) -> V[3*4+0] = 12
    # Local 3 (B[3,1]) -> V[3*4+1] = 13
    # B_BR (j in 2..3, k in 2..3):
    # Local 0 (B[2,2]) -> V[2*4+2] = 10
    # Local 1 (B[2,3]) -> V[2*4+3] = 11
    # Local 2 (B[3,2]) -> V[3*4+2] = 14
    # Local 3 (B[3,3]) -> V[3*4+3] = 15

    # Strassen U (for X) coefficients:
    # M1: x0+1, x3+1
    # M2: x2+1, x3+1
    # M3: x0+1
    # M4: x3+1
    # M5: x0+1, x1+1
    # M6: x2+1, x0-1
    # M7: x1+1, x3-1
    
    # Strassen V (for Y) coefficients:
    # M1: y0+1, y3+1
    # M2: y0+1
    # M3: y1+1, y3-1
    # M4: y2+1, y0-1
    # M5: y3+1
    # M6: y0+1, y1+1
    # M7: y2+1, y3+1

    # Strassen W (for Z) coefficients:
    # M1: z0+1, z3+1
    # M2: z2+1, z3-1
    # M3: z1+1, z3+1
    # M4: z0+1, z2+1
    # M5: z0-1, z1+1
    # M6: z3+1
    # M7: z0+1, z3-1
    
    # Let's define the blocks.
    # Block 1: A_left * B_TL -> C_left
    # Block 2: A_right * B_BL -> C_left
    # Block 3: A_left * B_TR -> C_right
    # Block 4: A_right * B_BR -> C_right
    
    # We will create U, V, W as lists of lists.
    # Dimensions:
    # U: 8 x 28
    # V: 16 x 28
    # W: 8 x 28
    
    rank = 28
    U = [[0] * rank for _ in range(8)]
    V = [[0] * rank for _ in range(16)]
    W = [[0] * rank for _ in range(8)]
    
    # Strassen factors defined as lists of (index, coeff)
    # U_factors[r] = list of (local_idx, coeff)
    # V_factors[r] = list of (local_idx, coeff)
    # W_factors[r] = list of (local_idx, coeff)
    
    U_strassen = [
        [(0, 1), (3, 1)],  # M1
        [(2, 1), (3, 1)],  # M2
        [(0, 1)],          # M3
        [(3, 1)],          # M4
        [(0, 1), (1, 1)],  # M5
        [(2, 1), (0, -1)], # M6
        [(1, 1), (3, -1)]  # M7
    ]
    
    V_strassen = [
        [(0, 1), (3, 1)],  # M1
        [(0, 1)],          # M2
        [(1, 1), (3, -1)], # M3
        [(2, 1), (0, -1)], # M4
        [(3, 1)],          # M5
        [(0, 1), (1, 1)],  # M6
        [(2, 1), (3, 1)]   # M7
    ]
    
    W_strassen = [
        [(0, 1), (3, 1)],  # M1
        [(2, 1), (3, -1)], # M2
        [(1, 1), (3, 1)],  # M3
        [(0, 1), (2, 1)],  # M4
        [(0, -1), (1, 1)], # M5
        [(3, 1)],          # M6
        [(0, 1)]           # M7
    ]
    
    # Mapping functions
    # local_idx in 0..3 for 2x2 block
    # block_type: 'left' or 'right' for A/C, 'TL', 'TR', 'BL', 'BR' for B
    
    def map_A(local_idx, side):
        if side == 'left':
            return [0, 1, 4, 5][local_idx]
        else: # right
            return [2, 3, 6, 7][local_idx]
            
    def map_B(local_idx, block):
        if block == 'TL':
            return [0, 1, 4, 5][local_idx]
        elif block == 'TR':
            return [2, 3, 6, 7][local_idx]
        elif block == 'BL':
            return [8, 9, 12, 13][local_idx]
        else: # BR
            return [10, 11, 14, 15][local_idx]
            
    def map_C(local_idx, side):
        if side == 'left':
            return [0, 2, 1, 3][local_idx]
        else: # right
            return [4, 6, 5, 7][local_idx]
            
    # Blocks configuration: (A_side, B_block, C_side)
    blocks = [
        ('left', 'TL', 'left'),
        ('right', 'BL', 'left'),
        ('left', 'TR', 'right'),
        ('right', 'BR', 'right')
    ]
    
    col = 0
    for a_side, b_block, c_side in blocks:
        for r in range(7):
            # Fill U
            for local_idx, coeff in U_strassen[r]:
                global_idx = map_A(local_idx, a_side)
                U[global_idx][col] += coeff
                
            # Fill V
            for local_idx, coeff in V_strassen[r]:
                global_idx = map_B(local_idx, b_block)
                V[global_idx][col] += coeff
                
            # Fill W
            for local_idx, coeff in W_strassen[r]:
                global_idx = map_C(local_idx, c_side)
                W[global_idx][col] += coeff
                
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
