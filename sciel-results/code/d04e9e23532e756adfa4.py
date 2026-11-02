import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [4, 5, 6]
n, m, p = DIMENSIONS
TOTAL_TIME_LIMIT = 120.0

def construct():
    start_time = time.time()
    
    # Strategy:
    # Decompose 4x5 * 5x6 into blocks.
    # A (4x5) = [A1 (4x4), A2 (4x1)]
    # B (5x6) = [B1 (4x6); B2 (1x6)]
    # C = A1*B1 + A2*B2
    
    # 1. A1*B1: 4x4 * 4x6
    #    Split B1 into B1a (4x4) and B1b (4x2)
    #    A1*B1a: 4x4 * 4x4. Use Rank 49 algorithm.
    #    A1*B1b: 4x4 * 4x2. Use Schoolbook (Rank 32).
    #    Total for A1*B1: 49 + 32 = 81.
    
    # 2. A2*B2: 4x1 * 1x6. Use Schoolbook (Rank 24).
    
    # Total Rank: 81 + 24 = 105.
    
    # We need the Rank 49 decomposition for 4x4x4.
    # This is a known result (e.g., from Strassen's 1969 paper or subsequent improvements).
    # The standard Rank 49 algorithm for 4x4x4 uses 49 multiplications.
    # I will construct this using a known set of coefficients.
    # Since hardcoding 49x3 matrices of complex coefficients is error-prone without a reference,
    # I will use a verified construction if possible, or fall back to a safer, slightly higher rank if 49 fails verification.
    # However, the prompt implies we should find a rank-minimal exact decomposition.
    # Rank 49 is the best known for 4x4x4.
    
    # Let's try to construct the 4x4x4 rank 49 decomposition.
    # If I cannot guarantee the coefficients, I can use a rank 48 or 49 algorithm if I find one.
    # Actually, a simpler approach:
    # The Rank 49 algorithm is complex to hardcode from memory.
    # Is there a simpler block structure?
    
    # Alternative: Use Strassen (Rank 7) for 2x2x2 blocks?
    # 4x4 * 4x4 can be split into four 2x2 blocks.
    # C11 = A11B11 + A12B21
    # C12 = A11B12 + A12B22
    # C21 = A21B11 + A22B21
    # C22 = A21B12 + A22B22
    # This requires 8 multiplications of 2x2 matrices if done naively? No, that's 8 multiplications of 2x2 blocks.
    # Each 2x2 * 2x2 multiplication has rank 7 (Strassen).
    # So 4 blocks * 7 = 28? No, the blocks are not independent.
    # Strassen's algorithm for 4x4x4 (using 2x2 blocks) actually uses 7 * 4 = 28 multiplications?
    # Wait, Strassen's algorithm for 2x2x2 uses 7 multiplications.
    # If we apply Strassen to 4x4x4 by viewing it as 2x2 blocks of 2x2 matrices:
    # We perform 7 "multiplications" where each "multiplication" is a 2x2 * 2x2 matrix multiplication.
    # Each 2x2 * 2x2 multiplication has rank 7.
    # So total rank = 7 * 7 = 49.
    # Yes! This is the standard way to derive Rank 49 for 4x4x4.
    
    # So, the strategy is:
    # 1. Implement Strassen's algorithm for 2x2x2 (Rank 7).
    # 2. Use it to build a 4x4x4 decomposition of Rank 49.
    # 3. Use this 4x4x4 decomposition for the A1*B1a part.
    # 4. Use Schoolbook for the rest.
    
    # Let's implement Strassen 2x2x2 first.
    # Inputs: A (2x2), B (2x2).
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # We need to represent this as a tensor decomposition.
    # The tensor T for 2x2x2 has rank 7.
    # We can generate the U, V, W matrices for this rank 7 decomposition.
    
    def get_strassen_2x2_decomp():
        # Returns U (4x7), V (4x7), W (4x7) for 2x2 * 2x2 -> 2x2
        # Indices:
        # A: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
        # B: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
        # C: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
        
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # M1 = (a+d)(e+h)
        # U[:, 0]: a, 0, 0, d -> [1, 0, 0, 1]
        # V[:, 0]: e, 0, 0, h -> [1, 0, 0, 1]
        # W[:, 0]: C11, 0, 0, C22 -> [1, 0, 0, 1]
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        
        # M2 = (c+d)e
        # U[:, 1]: 0, 0, c, d -> [0, 0, 1, 1]
        # V[:, 1]: e, 0, 0, 0 -> [1, 0, 0, 0]
        # W[:, 1]: 0, 0, C21, C22 -> [0, 0, 1, 1] (Note: C21 gets M2, C22 gets -M2? 
        # Formula: C21 = M2 + M4, C22 = M1 - M2 + M3 + M6.
        # So W for M2 contributes to C21 (+1) and C22 (-1).
        # W[:, 1]: [0, 0, 1, -1]
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1; W[3, 1] = -1
        
        # M3 = a(f-h)
        # U[:, 2]: a, 0, 0, 0 -> [1, 0, 0, 0]
        # V[:, 2]: 0, f, 0, -h -> [0, 1, 0, -1]
        # W[:, 2]: C12, C22 -> [0, 1, 0, 1] (C12 = M3+M5, C22 = ... +M3)
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1; W[3, 2] = 1
        
        # M4 = d(g-e)
        # U[:, 3]: 0, 0, 0, d -> [0, 0, 0, 1]
        # V[:, 3]: -e, 0, g, 0 -> [-1, 0, 1, 0]
        # W[:, 3]: C11, C21 -> [1, 0, 1, 0] (C11 = ... +M4, C21 = M2 + M4)
        U[3, 3] = 1
        V[0, 3] = -1; V[2, 3] = 1
        W[0, 3] = 1; W[2, 3] = 1
        
        # M5 = (a+b)h
        # U[:, 4]: a, b, 0, 0 -> [1, 1, 0, 0]
        # V[:, 4]: 0, 0, 0, h -> [0, 0, 0, 1]
        # W[:, 4]: C11, C12 -> [-1, 1, 0, 0] (C11 = ... -M5, C12 = M3 + M5)
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1; W[1, 4] = 1
        
        # M6 = (c-a)(e+f)
        # U[:, 5]: -a, 0, c, 0 -> [-1, 0, 1, 0]
        # V[:, 5]: e, f, 0, 0 -> [1, 1, 0, 0]
        # W[:, 5]: C22 -> [0, 0, 0, 1]
        U[0, 5] = -1; U[2, 5] = 1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1
        
        # M7 = (b-d)(g+h)
        # U[:, 6]: 0, b, 0, -d -> [0, 1, 0, -1]
        # V[:, 6]: 0, 0, g, h -> [0, 0, 1, 1]
        # W[:, 6]: C11 -> [1, 0, 0, 0]
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1
        
        return U, V, W
    
    # Now build 4x4x4 Rank 49 using Strassen 2x2x2.
    # A (4x4) is split into 2x2 blocks:
    # A11 A12
    # A21 A22
    # B (4x4) is split into 2x2 blocks:
    # B11 B12
    # B21 B22
    
    # Strassen steps for 4x4 (treating blocks as scalars in the meta-algorithm):
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # Each "Multiplication" M_k is a 2x2 * 2x2 matrix multiplication.
    # So for each M_k, we invoke the Rank 7 decomposition.
    # Total rank = 7 * 7 = 49.
    
    # We need to map the indices.
    # A indices: 0-3 for 2x2 block 11, 4-7 for 12, 8-11 for 21, 12-15 for 22.
    # Actually, standard row-major:
    # A[0,0] A[0,1] A[0,2] A[0,3]
    # A[1,0] A[1,1] A[1,2] A[1,3]
    # A[2,0] A[2,1] A[2,2] A[2,3]
    # A[3,0] A[3,1] A[3,2] A[3,3]
    
    # Block A11: indices 0,1,4,5
    # Block A12: indices 2,3,6,7
    # Block A21: indices 8,9,12,13
    # Block A22: indices 10,11,14,15
    
    # Similarly for B and C.
    
    def get_4x4x4_decomp():
        U = np.zeros((16, 49))
        V = np.zeros((16, 49))
        W = np.zeros((16, 49))
        
        U_s, V_s, W_s = get_strassen_2x2_decomp()
        
        # Define block index mappings
        # For a 2x2 block at (br, bc) in a 4x4 matrix:
        # Indices are:
        # (2*br + 0)*4 + (2*bc + 0)
        # (2*br + 0)*4 + (2*bc + 1)
        # (2*br + 1)*4 + (2*bc + 0)
        # (2*br + 1)*4 + (2*bc + 1)
        
        def get_block_indices(br, bc):
            return [
                (2*br + 0)*4 + (2*bc + 0),
                (2*br + 0)*4 + (2*bc + 1),
                (2*br + 1)*4 + (2*bc + 0),
                (2*br + 1)*4 + (2*bc + 1)
            ]
        
        # Strassen coefficients for the 2x2 block level
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # We need to express each M_k as a linear combination of A-blocks times B-blocks.
        # This is tricky because M_k is a matrix product of linear combinations.
        # E.g., M1 = (A11+A22)(B11+B22) = A11B11 + A11B22 + A22B11 + A22B22.
        # This is NOT a single matrix multiplication of two linear combinations in the tensor sense?
        # Wait. The Strassen algorithm for 2x2x2 reduces the number of multiplications.
        # When applied to 4x4x4, each "multiplication" is a 2x2x2 multiplication.
        # The inputs to these 2x2x2 multiplications are linear combinations of the 2x2 blocks of A and B.
        
        # Let's define the 7 input matrices for the meta-Strassen step.
        # X1 = A11 + A22
        # X2 = A21 + A22
        # X3 = A11
        # X4 = A22
        # X5 = A11 + A12
        # X6 = A21 - A11
        # X7 = A12 - A22
        
        # Y1 = B11 + B22
        # Y2 = B11
        # Y3 = B12 - B22
        # Y4 = B21 - B11
        # Y5 = B22
        # Y6 = B11 + B12
        # Y7 = B21 + B22
        
        # The 7 products are Z1 = X1 Y1, ..., Z7 = X7 Y7.
        # Each Zk is a 2x2 matrix.
        # The output blocks are:
        # C11 = Z1 + Z4 - Z5 + Z7
        # C12 = Z3 + Z5
        # C21 = Z2 + Z4
        # C22 = Z1 - Z2 + Z3 + Z6
        
        # We need to construct the tensor decomposition.
        # For each k in 1..7, we have a 2x2x2 multiplication Zk = Xk Yk.
        # We use the Rank 7 decomposition for this.
        # The U, V, W matrices for the 4x4x4 case will be formed by:
        # 1. For each k, take the U_s, V_s, W_s from Strassen 2x2.
        # 2. Map the local 2x2 indices of Xk to the global 4x4 indices.
        # 3. Map the local 2x2 indices of Yk to the global 4x4 indices.
        # 4. Map the local 2x2 indices of Zk to the global 4x4 indices (with coefficients from the output formula).
        
        # Let's iterate through the 7 meta-steps.
        
        # Define the linear combinations for X and Y blocks.
        # X_k is a 2x2 matrix formed by linear combos of A-blocks.
        # A-blocks are 2x2 matrices.
        # So X_k[i,j] = sum_{br,bc} coeff[k][br,bc] * A[br,bc][i,j]
        
        # We will build the U, V, W matrices by accumulating contributions.
        
        # Helper to add a block contribution
        def add_block_contrib(U_global, V_global, W_global, 
                             U_local, V_local, W_local,
                             x_block_coeffs, y_block_coeffs, z_block_coeffs,
                             offset_col):
            # x_block_coeffs: dict {(br,bc): coeff}
            # y_block_coeffs: dict {(br,bc): coeff}
            # z_block_coeffs: dict {(br,bc): coeff}
            
            # For each of the 7 local ranks:
            for r in range(7):
                col = offset_col + r
                
                # U part:
                # U_local[i, r] is the coefficient for A_local[i].
                # A_local[i] corresponds to the (i_row, i_col) element in the 2x2 block.
                # We need to map this to the global A index.
                # But X_k is a sum of blocks.
                # X_k[i_row, i_col] = sum_{br,bc} coeff_x[br,bc] * A_block[br,bc][i_row, i_col]
                # So the coefficient for global A index corresponding to block (br,bc) and local pos (i_row, i_col) is:
                # coeff_x[br,bc] * U_local[i_row*2 + i_col, r]
                
                for (br, bc), cx in x_block_coeffs.items():
                    if cx == 0: continue
                    local_indices = get_block_indices(br, bc)
                    # local_indices are [0, 1, 2, 3] corresponding to (0,0), (0,1), (1,0), (1,1)
                    # U_local is 4x7. Row index 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
                    for li, glob_idx in enumerate(local_indices):
                        U_global[glob_idx, col] += cx * U_local[li, r]
                
                # V part:
                for (br, bc), cy in y_block_coeffs.items():
                    if cy == 0: continue
                    local_indices = get_block_indices(br, bc)
                    for li, glob_idx in enumerate(local_indices):
                        V_global[glob_idx, col] += cy * V_local[li, r]
                
                # W part:
                # W_local[j, r] is the coefficient for C_local[j].
                # C_local[j] corresponds to the (j_row, j_col) element in the 2x2 block Zk.
                # Zk contributes to C blocks with coefficients.
                # C_block[br,bc][j_row, j_col] += coeff_z[br,bc] * Zk[j_row, j_col]
                for (br, bc), cz in z_block_coeffs.items():
                    if cz == 0: continue
                    local_indices = get_block_indices(br, bc)
                    for li, glob_idx in enumerate(local_indices):
                        W_global[glob_idx, col] += cz * W_local[li, r]
        
        # Define the X and Y block coefficients for each of the 7 steps.
        # Blocks: 11:(0,0), 12:(0,1), 21:(1,0), 22:(1,1)
        
        # Step 1: M1 = (A11+A22)(B11+B22)
        x1 = {(0,0): 1, (1,1): 1}
        y1 = {(0,0): 1, (1,1): 1}
        # Output: C11 += Z1, C22 += Z1
        z1 = {(0,0): 1, (1,1): 1}
        add_block_contrib(U, V, W, U_s, V_s, W_s, x1, y1, z1, 0)
        
        # Step 2: M2 = (A21+A22)B11
        x2 = {(1,0): 1, (1,1): 1}
        y2 = {(0,0): 1}
        # Output: C21 += Z2, C22 -= Z2
        z2 = {(1,0): 1, (1,1): -1}
        add_block_contrib(U, V, W, U_s, V_s, W_s, x2, y2, z2, 7)
        
        # Step 3: M3 = A11(B12-B22)
        x3 = {(0,0): 1}
        y3 = {(0,1): 1, (1,1): -1}
        # Output: C12 += Z3, C22 += Z3
        z3 = {(0,1): 1, (1,1): 1}
        add_block_contrib(U, V, W, U_s, V_s, W_s, x3, y3, z3, 14)
        
        # Step 4: M4 = A22(B21-B11)
        x4 = {(1,1): 1}
        y4 = {(1,0): 1, (0,0): -1}
        # Output: C11 += Z4, C21 += Z4
        z4 = {(0,0): 1, (1,0): 1}
        add_block_contrib(U, V, W, U_s, V_s, W_s, x4, y4, z4, 21)
        
        # Step 5: M5 = (A11+A12)B22
        x5 = {(0,0): 1, (0,1): 1}
        y5 = {(1,1): 1}
        # Output: C11 -= Z5, C12 += Z5
        z5 = {(0,0): -1, (0,1): 1}
        add_block_contrib(U, V, W, U_s, V_s, W_s, x5, y5, z5, 28)
        
        # Step 6: M6 = (A21-A11)(B11+B12)
        x6 = {(1,0): 1, (0,0): -1}
        y6 = {(0,0): 1, (0,1): 1}
        # Output: C22 += Z6
        z6 = {(1,1): 1}
        add_block_contrib(U, V, W, U_s, V_s, W_s, x6, y6, z6, 35)
        
        # Step 7: M7 = (A12-A22)(B21+B22)
        x7 = {(0,1): 1, (1,1): -1}
        y7 = {(1,0): 1, (1,1): 1}
        # Output: C11 += Z7
        z7 = {(0,0): 1}
        add_block_contrib(U, V, W, U_s, V_s, W_s, x7, y7, z7, 42)
        
        return U, V, W
    
    # Verify the 4x4x4 decomposition
    U_44, V_44, W_44 = get_4x4x4_decomp()
    
    # Quick verification
    # C = A * B
    # C[i,k] = sum_j A[i,j] B[j,k]
    # Decomposition: C[i,k] = sum_r U[i,j?] ... wait.
    # The tensor is T = sum_{i,j,k} C_{ik} A_{ij} B_{jk}.
    # Decomposition: T = sum_r U_{ij}^r V_{jk}^r W_{ik}^r.
    # Here U is (n*m, rank), V is (m*p, rank), W is (p*n, rank).
    # My U_44 is (16, 49). Index is i*4+j?
    # get_block_indices returns i*4+j.
    # So U_44[row, col] is U_{i,j}^r.
    # V_44[row, col] is V_{j,k}^r.
    # W_44[row, col] is W_{k,i}^r?
    # Let's check W indices.
    # In get_strassen_2x2_decomp, W indices are 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1).
    # This corresponds to C[0,0], C[0,1], C[1,0], C[1,1].
    # In the 4x4 case, W_44[row, col] corresponds to C[i,k] where i,k are derived from the block index.
    # get_block_indices(br, bc) returns indices for C_block[br,bc].
    # The index is i*4+k.
    # So W_44[row, col] is W_{k,i}^r?
    # The task contract says: W[k*n+i, r].
    # My W_44 row index is i*4+k.
    # So I need to transpose the row index of W_44 to k*4+i.
    
    # Let's create W_44_transposed.
    W_44_t = np.zeros_like(W_44)
    for i in range(4):
        for k in range(4):
            W_44_t[k*4 + i, :] = W_44[i*4 + k, :]
    
    # Verify
    errors = 0
    for i in range(4):
        for j in range(4):
            for k in range(4):
                val = np.dot(np.dot(U_44[i*4+j], V_44[j*4+k]), W_44_t[k*4+i])
                # This val should be 1 if (i,j,k) is a valid term, 0 otherwise?
                # No, the decomposition is exact.
                # C[i,k] = sum_j A[i,j] B[j,k]
                # In the tensor form, the coefficient of A[i,j] B[j,k] in C[i,k] is 1.
                # The decomposition approximates this tensor.
                # So for any i,j,k, sum_r U[i,j,r] V[j,k,r] W[k,i,r] should be 1 if we are looking at the term A[i,j]B[j,k] contributing to C[i,k]?
                # Actually, the tensor T has components T[i,j,k] = 1 if the term A[i,j]B[j,k] is in the sum for C[i,k].
                # Since C[i,k] = sum_j A[i,j]B[j,k], T[i,j,k] = 1 for all i,j,k.
                # So the sum should be 1 for all i,j,k.
                if abs(val - 1.0) > 1e-9:
                    errors += 1
    
    if errors > 0:
        # Fallback to Schoolbook for 4x4x4 (Rank 64)
        # This ensures correctness.
        U_44 = np.zeros((16, 64))
        V_44 = np.zeros((16, 64))
        W_44_t = np.zeros((16, 64))
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    U_44[i*4+j, col] = 1
                    V_44[j*4+k, col] = 1
                    W_44_t[k*4+i, col] = 1
                    col += 1
        rank_44 = 64
    else:
        rank_44 = 49
    
    # Now build the full 4x5 * 5x6 decomposition.
    # A (4x5) = [A1 (4x4), A2 (4x1)]
    # B (5x6) = [B1 (4x6); B2 (1x6)]
    # C = A1*B1 + A2*B2
    
    # A1*B1: 4x4 * 4x6
    # B1 (4x6) = [B1a (4x4), B1b (4x2)]
    # A1*B1a: 4x4 * 4x4 (Rank 49)
    # A1*B1b: 4x4 * 4x2 (Rank 32, Schoolbook)
    
    # A2*B2: 4x1 * 1x6 (Rank 24, Schoolbook)
    
    # Total Rank: 49 + 32 + 24 = 105.
    
    rank_total = rank_44 + 32 + 24
    U = np.zeros((n*m, rank_total))
    V = np.zeros((m*p, rank_total))
    W = np.zeros((p*n, rank_total))
    
    col_offset = 0
    
    # 1. A1*B1a (4x4 * 4x4)
    # A1 indices: A[i, j] for j in 0..3. Global A index i*5+j.
    # B1a indices: B[j, k] for j in 0..3, k in 0..3. Global B index j*6+k.
    # C indices: C[i, k] for i in 0..3, k in 0..3. Global C index k*4+i.
    
    for r in range(rank_44):
        # U
        for idx_u in range(16):
            val = U_44[idx_u, r]
            if val != 0:
                i, j = divmod(idx_u, 4)
                U[i*5 + j, col_offset + r] = val
        # V
        for idx_v in range(16):
            val = V_44[idx_v, r]
            if val != 0:
                j, k = divmod(idx_v, 4)
                V[j*6 + k, col_offset + r] = val
        # W
        for idx_w in range(16):
            val = W_44_t[idx_w, r]
            if val != 0:
                k, i = divmod(idx_w, 4)
                W[k*4 + i, col_offset + r] = val
    col_offset += rank_44
    
    # 2. A1*B1b (4x4 * 4x2)
    # Schoolbook Rank 32.
    # A1 indices: A[i, j] for j in 0..3.
    # B1b indices: B[j, k] for j in 0..3, k in 4..5.
    # C indices: C[i, k] for i in 0..3, k in 4..5.
    
    for i in range(4):
        for j in range(4):
            for k in range(2):
                k_global = 4 + k
                U[i*5 + j, col_offset] = 1
                V[j*6 + k_global, col_offset] = 1
                W[k_global*4 + i, col_offset] = 1
                col_offset += 1
    
    # 3. A2*B2 (4x1 * 1x6)
    # Schoolbook Rank 24.
    # A2 indices: A[i, 4].
    # B2 indices: B[4, k] for k in 0..5.
    # C indices: C[i, k].
    
    for i in range(4):
        for k in range(6):
            U[i*5 + 4, col_offset] = 1
            V[4*6 + k, col_offset] = 1
            W[k*4 + i, col_offset] = 1
            col_offset += 1
            
    # Convert to lists for the return format
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

if __name__ == "__main__":
    start = time.time()
    res = construct()
    end = time.time()
    print(f"Time taken: {end - start}")
    
    U = np.array(res['U'])
    V = np.array(res['V'])
    W = np.array(res['W'])
    
    errors = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(np.dot(U[i*m+j], V[j*p+k]), W[k*n+i])
                if abs(val - 1.0) > 1e-9:
                    errors += 1
    if errors == 0:
        print("Verification PASSED")
    else:
        print(f"Verification FAILED with {errors} errors")
