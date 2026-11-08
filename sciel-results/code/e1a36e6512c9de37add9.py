import numpy as np
import time

def construct():
    n, m, p = 4, 5, 5
    start_time = time.time()
    deadline = start_time + 100  # Safety margin

    # Dimensions
    dim_a = n * m  # 20
    dim_b = m * p  # 25
    dim_c = p * n  # 20

    # Rank components
    # 1. 4x4x4 Strassen: Rank 49
    # 2. 4x4x1 (Matrix-Vector right mult): Rank 16
    # 3. 4x1x4 (Vector-Matrix left mult): Rank 16
    # 4. 4x1x1 (Scalar mult): Rank 4
    rank_444 = 49
    rank_441 = 16
    rank_414 = 16
    rank_411 = 4
    total_rank = rank_444 + rank_441 + rank_414 + rank_411  # 85

    # Initialize factor matrices
    U = np.zeros((dim_a, total_rank), dtype=float)
    V = np.zeros((dim_b, total_rank), dtype=float)
    W = np.zeros((dim_c, total_rank), dtype=float)

    # --- Helper: Generate Strassen 2x2 factors ---
    # Returns U_2x2 (4x7), V_2x2 (4x7), W_2x2 (4x7)
    # where U, V, W map 2x2 matrices to rank-7 intermediates
    def get_strassen_2x2():
        u = np.zeros((4, 7))
        v = np.zeros((4, 7))
        w = np.zeros((4, 7))
        
        # M1 = (a00+a11)(b00+b11) -> c00 += M1, c11 += M1
        u[0,0]=1; u[3,0]=1; v[0,0]=1; v[3,0]=1; w[0,0]=1; w[3,0]=1
        # M2 = (a10+a11)b00 -> c10 += M2, c11 -= M2
        u[2,1]=1; u[3,1]=1; v[0,1]=1; w[2,1]=1; w[3,1]=-1
        # M3 = a00(b01-b11) -> c01 += M3, c11 += M3
        u[0,2]=1; v[1,2]=1; v[3,2]=-1; w[1,2]=1; w[3,2]=1
        # M4 = a11(b10-b00) -> c10 += M4, c00 += M4
        u[3,3]=1; v[2,3]=1; v[0,3]=-1; w[0,3]=1; w[2,3]=1
        # M5 = (a00+a01)b11 -> c01 += M5, c00 -= M5
        u[0,4]=1; u[1,4]=1; v[3,4]=1; w[0,4]=-1; w[1,4]=1
        # M6 = (a10-a00)(b00+b01) -> c11 += M6
        u[0,5]=-1; u[2,5]=1; v[0,5]=1; v[1,5]=1; w[3,5]=1
        # M7 = (a01-a11)(b10+b11) -> c00 += M7
        u[1,6]=1; u[3,6]=-1; v[2,6]=1; v[3,6]=1; w[0,6]=1
        
        return u, v, w

    # --- Helper: Build 4x4x4 factors from 2x2x2 factors ---
    # We treat 4x4 blocks as 2x2 blocks of 2x2 sub-matrices.
    # The tensor product of the 2x2 Strassen algorithm with itself gives the 4x4 algorithm.
    # Rank = 7 * 7 = 49.
    def get_444_factors():
        u2, v2, w2 = get_strassen_2x2()
        # U_444 is (16, 49). It maps a 4x4 matrix (vec size 16) to 49 intermediates.
        # The structure is U_444[i, j, k] = U_2x2[i, j] * U_2x2[k, l]? No.
        # The indices for 4x4 multiplication are (i, j, k) where i is row of A (0..3), j is col of A/row of B (0..3), k is col of B (0..3).
        # Wait, standard tensor indices:
        # A is 4x4. B is 4x4. C is 4x4.
        # Tensor T_{i, j, k} = 1 if A_{i,j} B_{j,k} contributes to C_{i,k}.
        # Here the middle index is j (0..3).
        # The Strassen algorithm for 4x4 splits into 2x2 blocks.
        # Let A = [[A00, A01], [A10, A11]] where each is 2x2.
        # The rank-7 algorithm for 2x2 uses 7 products.
        # For 4x4, we apply the 2x2 algorithm to the block structure.
        # This results in 7 * 7 = 49 multiplications.
        
        # We need to map the 4x4 matrix indices (0..15) to the Kronecker structure.
        # Vectorization order: row-major.
        # A_4x4 index a = i * 4 + j.
        # In block terms: i = 2*I + i_loc, j = 2*J + j_loc.
        # a = (2*I + i_loc)*4 + (2*J + j_loc) = 8*I + 4*i_loc + 2*J + j_loc.
        
        # The Kronecker product of the factors works if we reorder the basis vectors appropriately.
        # However, it's easier to just use the explicit Kronecker product of the factor matrices 
        # if the vectorization is consistent with the Kronecker product definition.
        # np.kron(U2, U2) gives a matrix of size (4*4) x (7*7) = 16x49.
        # Does this match the index mapping?
        # Let's verify one entry.
        # U_444[a, k_prod] where k_prod corresponds to (k_outer, k_inner).
        # The product is U2[I, k_outer] * U2[i_loc, k_inner]?
        # Actually, the standard Kronecker product $U \otimes U$ satisfies:
        # $(U \otimes U)_{(i, i'), (j, j')} = U_{i, j} U_{i', j'}$.
        # This matches the block structure of 4x4 multiplication if we view the 4x4 matrix as a 2x2 matrix of 2x2 blocks.
        
        u44 = np.kron(u2, u2)
        v44 = np.kron(v2, v2)
        w44 = np.kron(w2, w2)
        return u44, v44, w44

    # --- Build the full decomposition ---
    
    # Block 1: A_L (4x4) * B_TL (4x4) -> C_L (4x4)
    # A_L is A[0:4, 0:4]. In vectorization (row-major 4x5), these are rows 0..3, cols 0..3.
    # Indices in A (dim 20):
    # Row i (0..3), Col j (0..3). Index = i*5 + j.
    # We need to map the 4x4 vector (16 elements) to these 20 positions.
    
    # Let's create the mapping matrices.
    # M_A_L: (16, 20) extracts A_L from A.
    # M_B_TL: (16, 25) extracts B_TL from B.
    # M_C_L: (16, 20) inserts C_L into C.
    
    # It is easier to construct U, V, W directly by assigning values to the correct columns.
    
    u44, v44, w44 = get_444_factors()
    
    # Indices for A_L (4x4 part of 4x5)
    # A[i, j] for i in 0..3, j in 0..3.
    # Vector index in U: row is i*5 + j.
    # Vector index in u44: row is i*4 + j.
    
    # Indices for B_TL (4x4 part of 5x5 top-left)
    # B[j, k] for j in 0..3, k in 0..3.
    # Vector index in V: row is j*5 + k.
    # Vector index in v44: row is j*4 + k.
    
    # Indices for C_L (4x4 part of 4x5 left)
    # C[i, k] for i in 0..3, k in 0..3.
    # Vector index in W: row is k*4 + i (Note: W index is p*n = 5*4=20. C is 4x5. 
    # The problem states W[k*n+i, r]. So W index for C[i,k] is k*4 + i.
    # Wait, standard tensor T_{A,B,C} where C = AB.
    # C[i,k] depends on A[i,j] and B[j,k].
    # W factor index corresponds to C element.
    # C is 4x5. Element (i,k) has index k*4 + i in the W vector?
    # Let's check the convention: W[k*n+i, r]. Here n=4, p=5.
    # The output matrix is 4x5. Let's call its rows i and cols k.
    # The index in the flattened vector is usually row-major: i*p + k.
    # BUT the prompt says: "output factor follows the official trace convention W[k*n+i,r]".
    # This implies the index is $k \times n + i$.
    # So for C[i, k], the W row index is $k \times 4 + i$.
    # Let's verify with schoolbook.
    # Schoolbook term for C[i,k] is A[i,j] B[j,k].
    # U row: i*m+j = i*5+j.
    # V row: j*p+k = j*5+k.
    # W row: k*n+i = k*4+i.
    # Yes.
    
    # So, for C_L (4x4), i in 0..3, k in 0..3.
    # W index: k*4 + i.
    # Vector index in w44: row is i*4 + k? 
    # In the 4x4 multiplication tensor T_{i,j,k} = delta_{j,j} delta_{i,i} delta_{k,k}? No.
    # The tensor for matrix mult M_{n,p,q} has entries 1 at (i,j, k) if they form a valid chain i->j->k.
    # Here the indices are (RowA, ColA/RowB, ColB).
    # So for 4x4x4, the W index corresponds to (RowC, ColC).
    # The vectorization of W_4x4 usually matches the vectorization of C.
    # If we use np.kron, we need to ensure the vectorization of the 4x4 block C_L matches the indices.
    # The row index in w44 is $i_{loc} \times 4 + k_{loc}$? Or $k_{loc} \times 4 + i_{loc}$?
    # Standard Kronecker product $W \otimes W$ assumes consistent vectorization.
    # If $W_{2x2}$ is vectorized row-major (index $r \times 2 + c$), then $W_{4x4}$ vectorized row-major is $(R \times 2 + r) \times 4 + (C \times 2 + c)$.
    # This equals $R \times 8 + r \times 4 + C \times 2 + c$.
    # Our W index for C_L[i,k] is $k \times 4 + i$.
    # This is NOT row-major ($i \times 4 + k$). It is column-major ($k \times 4 + i$).
    # So we need to be careful with the mapping.
    
    # Let's explicitly build the mappings to be safe.
    
    # 1. A_L (4x4) to U rows
    # U rows for A_L are i*5+j for i,j in 0..3.
    # u44 rows are i*4+j for i,j in 0..3.
    # We will add u44 to U[:, col_offset] by mapping indices.
    
    # 2. B_TL (4x4) to V rows
    # V rows for B_TL are j*5+k for j,k in 0..3.
    # v44 rows are j*4+k for j,k in 0..3.
    
    # 3. C_L (4x4) to W rows
    # W rows for C_L are k*4+i for k,i in 0..3.
    # w44 rows are i*4+k for i,k in 0..3 (row-major).
    # We need to permute the rows of w44 or map them.
    # Mapping: w44_row = i*4+k. Target W_row = k*4+i.
    
    def map_44_to_full(u44, v44, w44, U, V, W, offset):
        # Map u44 (16x49) to U (20x49) at offset
        for r44 in range(16):
            i, j = divmod(r44, 4)
            r_full = i * 5 + j
            U[r_full, offset:offset+49] += u44[r44, :]
            
        # Map v44 (16x49) to V (25x49) at offset
        for r44 in range(16):
            j, k = divmod(r44, 4)
            r_full = j * 5 + k
            V[r_full, offset:offset+49] += v44[r44, :]
            
        # Map w44 (16x49) to W (20x49) at offset
        # w44 row r44 corresponds to C[i, k] with i, k = divmod(r44, 4).
        # Target W row is k * 4 + i.
        for r44 in range(16):
            i, k = divmod(r44, 4)
            r_full = k * 4 + i
            W[r_full, offset:offset+49] += w44[r44, :]

    # --- Part 1: A_L * B_TL -> C_L ---
    # Offset 0, Rank 49
    map_44_to_full(u44, v44, w44, U, V, W, 0)

    # --- Part 2: A_L (4x4) * B_TR (4x1) -> C_R (4x1) ---
    # B_TR is B[j, 4] for j in 0..3.
    # A_L is A[i, j] for i,j in 0..3.
    # C_R is C[i, 4] for i in 0..3.
    # This is a 4x4 times 4x1 multiplication.
    # Rank is 4*4=16.
    # Term: A[i, j] * B[j, 4] contributes to C[i, 4].
    # We can use the standard basis for this.
    # For each i in 0..3, j in 0..3, there is a term.
    # U[i*5+j, col] = 1
    # V[j*5+4, col] = 1
    # W[4*4+i, col] = 1  (C[i,4] index is 4*4+i = 16+i)
    
    offset = 49
    for i in range(4):
        for j in range(4):
            U[i*5 + j, offset] = 1.0
            V[j*5 + 4, offset] = 1.0
            W[4*4 + i, offset] = 1.0
            offset += 1
            
    # --- Part 3: A_R (4x1) * B_BL (1x4) -> C_L (4x4) ---
    # A_R is A[i, 4] for i in 0..3.
    # B_BL is B[4, k] for k in 0..3.
    # C_L is C[i, k] for i,k in 0..3.
    # Term: A[i, 4] * B[4, k] contributes to C[i, k].
    # Rank 16.
    # U[i*5+4, col] = 1
    # V[4*5+k, col] = 1
    # W[k*4+i, col] = 1
    
    for i in range(4):
        for k in range(4):
            U[i*5 + 4, offset] = 1.0
            V[4*5 + k, offset] = 1.0
            W[k*4 + i, offset] = 1.0
            offset += 1

    # --- Part 4: A_R (4x1) * B_BR (1x1) -> C_R (4x1) ---
    # A_R is A[i, 4].
    # B_BR is B[4, 4].
    # C_R is C[i, 4].
    # Term: A[i, 4] * B[4, 4] contributes to C[i, 4].
    # Rank 4.
    # U[i*5+4, col] = 1
    # V[4*5+4, col] = 1
    # W[4*4+i, col] = 1
    
    for i in range(4):
        U[i*5 + 4, offset] = 1.0
        V[4*5 + 4, offset] = 1.0
        W[4*4 + i, offset] = 1.0
        offset += 1

    # Final check
    if offset != total_rank:
        # Should not happen
        pass

    # Convert to lists for the return format if required, or return numpy arrays?
    # The interface says "factor matrices ... with integer/half-integer entries".
    # The seed program returns lists of lists.
    # I will return lists of lists to be safe, as the evaluator might expect that type.
    # Or I can check if numpy arrays are accepted. The seed used `[[0]*rank for ...]`.
    # Let's convert.
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
